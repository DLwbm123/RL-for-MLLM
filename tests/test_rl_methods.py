"""Formula, tool-boundary and native tiny-Qwen/PEFT engineering checks; no clinical data."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from src.objectives import advantages
from src.rl_methods import (PRIORITY,visurf_advantages,zvp_advantages,completion_loss,
                            perception_loss,axpo_advantages,author_advantages,crossmodal_values,cfpo_attention)
from src.rl_observation import (parse_observation,tool_prefix,tool_span,validate_observation_support,active_reward,random_patch_mask,
                               defacto_views,parse_evidence,defacto_reward)
from src.rl_methods_run import UpdateGroup,probability_audit,native_sampling,query_masks
from scripts.run_rl_methods import validate_spec


CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/rl_methods.json').read_text())
GT=[[100,100,300,300]];CONTROL=[[600,600,800,800]]


def test_visurf_label_sample_and_smoothing():
    a=visurf_advantages([0,0,0,0],1)
    assert (a[:-1]<0).all() and a[-1]>0 and abs(float(a.mean()))<1e-6
    mixed=visurf_advantages([0,1,0,1],1)
    assert mixed[-1]==0 and torch.allclose(mixed,advantages(torch.tensor([0.,1.,0.,1.,.5])))
    assert torch.equal(visurf_advantages([1,1,1,1],1),torch.zeros(5))
    with pytest.raises(ValueError):visurf_advantages([0,float('nan')],1)


def test_zvp_exact_entropy_branches():
    entropy=[torch.tensor([.2,.6]),torch.tensor([.1,.5,.4])]
    mixed=zvp_advantages([0,1],entropy,.1)
    assert torch.equal(mixed[0],-torch.ones(2)) and torch.equal(mixed[1],torch.ones(3))
    correct=zvp_advantages([1,1],entropy,.1)
    assert torch.allclose(correct[0],.1*entropy[0])
    wrong=zvp_advantages([0,0],entropy,.1)
    assert torch.allclose(wrong[0],torch.tensor([-.04,0]))
    assert zvp_advantages([0,0],[torch.ones(1),torch.ones(1)],.1)[0].item()==0
    with pytest.raises(ValueError):zvp_advantages([.5,.5],entropy,.1)


def test_completion_mask_detaches_advantage_and_gradient_direction():
    x=torch.tensor([-.5,-.7,-.8],requires_grad=True);a=torch.tensor([-1.,-1.,-1.],requires_grad=True)
    loss,_=completion_loss(x,x.detach(),x.detach(),a,[True,False,True],.2,.01)
    loss.backward();assert x.grad[0]>0 and x.grad[2]>0 and x.grad[1]==0 and a.grad is None
    with pytest.raises(ValueError):completion_loss(x,x.detach(),x.detach(),a,[True],.2,.01)
    with pytest.raises(FloatingPointError):completion_loss(x*float('nan'),x.detach(),x.detach(),0.,[True]*3,.2,0.)


def test_visurf_sequence_gradient_has_no_reference_kl():
    x=torch.tensor([-1.,-2.],requires_grad=True)
    loss,stats=completion_loss(x,x.detach(),torch.tensor([-100.,-101.]),1.,[True,True],.2,.01,'visurf')
    loss.backward()
    assert loss.item()==-2. and x.grad.tolist()==[-1.,-1.] and stats['reference_kl']==0


def test_zvp_appendix_clip_and_no_kl():
    x=torch.tensor([-1.],requires_grad=True);old=torch.tensor([-1.-np.log(1.5)],requires_grad=True)
    ref=torch.tensor([-101.],requires_grad=True)
    loss,stats=completion_loss(x,old,ref,1.,[True],.2,.01,'rl_zvp')
    assert loss.item()==pytest.approx(-1.28) and stats['reference_kl']==0
    loss.backward()
    assert x.grad.item()==0 and old.grad is None and ref.grad is None


@pytest.mark.parametrize('method',['rl_zvp','papo','cfpo'])
def test_batch_token_reduction_and_detached_entropy(method):
    parameter=torch.tensor(0.,requires_grad=True)
    m=SimpleNamespace(score=lambda inputs,features,ids: {'tokens':parameter.expand(len(ids))-1.})
    engine=UpdateGroup(m,method,CFG,nullcontext,lambda:None,np.random.default_rng(1))
    h=[torch.ones(1,requires_grad=True),torch.full((3,),2.,requires_grad=True)]
    a=zvp_advantages([1.,1.],h,.1)
    samples=[{'tokens':[3]*n,'old':torch.full((n,),-1.),'reference':torch.full((n,),-101.),
              'corrupted_old':torch.full((n,),-1.,requires_grad=True),
              'mask':torch.ones(n,dtype=torch.bool),'alignment':{'max_error':0.}} for n in (1,3)]
    g={'inputs':{},'features':None,'samples':samples,'advantages':a,'view':'pos'}
    stats=engine.backward([g])
    assert [t['weight'] for t in stats['terms']]==[.25,.75]
    assert parameter.grad.item()==pytest.approx(-.225 if method=='papo' else -.175)
    assert all(v.grad is None for v in h) and all(s['corrupted_old'].grad is None for s in samples)


def test_old_policy_gate_checks_whole_batch_before_gradient():
    parameter=torch.tensor(0.,requires_grad=True)
    m=SimpleNamespace(score=lambda inputs,features,ids: {'tokens':parameter.expand(len(ids))-1.})
    engine=UpdateGroup(m,'visurf',CFG,nullcontext,lambda:None,np.random.default_rng(1))
    samples=[{'tokens':[3],'old':torch.tensor([v]),'mask':torch.ones(1,dtype=torch.bool)} for v in (-1.,-2.)]
    with pytest.raises(RuntimeError,match='Old/current'):
        engine.backward([{'inputs':{},'features':None,'samples':samples}])
    assert parameter.grad is None


@pytest.mark.parametrize('method',['papo','cfpo'])
def test_author_dual_clipping_and_reference_stabilization(method):
    x=torch.tensor([-1.],requires_grad=True);old=torch.tensor([-1.-np.log(4.)])
    loss,_=completion_loss(x,old,x.detach(),-1.,[True],.2,0.,method)
    assert loss.item()==pytest.approx(3.)  # Author negative-advantage dual cap.
    loss,_=completion_loss(x,old,torch.tensor([-101.]),1.,[True],.2,.01,method)
    assert loss.item()==pytest.approx(-1.2,abs=1e-6)  # 1.3 upper clip and .01 * clamped k3=10.


@pytest.mark.parametrize('method',['papo','cfpo'])
def test_perception_author_golden_value_and_cached_gradient(method):
    # Pinned author compute_kl(-1, -3, low_var_kl) = exp(-2)+2-1.
    f=torch.tensor([-1.],requires_grad=True);c=torch.tensor([-3.],requires_grad=True)
    value,_=perception_loss(f,c,method,.1,.03,.02)
    assert value.item()==pytest.approx(-.0235335283237,abs=1e-7)
    value.backward()
    assert f.grad.item()==pytest.approx(-.116466471676,abs=1e-7) and c.grad is None


@pytest.mark.parametrize('method',['papo','cfpo'])
@pytest.mark.parametrize('pair',[(-1.,-101.),(-101.,-1.)])
def test_author_perception_extreme_finite_inputs(method,pair):
    f=torch.tensor([pair[0]],requires_grad=True);c=torch.tensor([pair[1]],requires_grad=True)
    loss,stats=perception_loss(f,c,method,.02,0.,0.)
    assert loss.item()==pytest.approx(-.2) and stats['perception_k3']==10.
    loss.backward();assert torch.isfinite(f.grad).all() and c.grad is None


@pytest.mark.parametrize('method,eps',[('papo',1e-6),('cfpo',1e-6),('defacto',1e-4)])
def test_author_sample_standard_deviation(method,eps):
    # Four Bernoulli outcomes have sample std sqrt(1/3), rather than population std .5.
    actual=author_advantages([0,1,0,1],method)
    expected=.5/(3**-.5+eps)
    assert actual.tolist()==pytest.approx([-expected,expected,-expected,expected])
    assert torch.equal(author_advantages([1,1,1,1],method),torch.zeros(4))


def test_cfpo_per_head_value_only_intervention():
    attention=torch.zeros(1,2,4,4);attention[:,:,0,0]=1;attention[:,:,1,:2]=.5
    attention[0,0,2]=torch.tensor([.9,.05,.05,0]);attention[0,1,2]=torch.tensor([.1,.8,.1,0]);attention[:,:,3,:]=.25
    values=torch.arange(16.).reshape(1,2,4,2).requires_grad_();before=values.detach().clone()
    out,n=crossmodal_values(attention,values,[True,True,False,False],[False,False,True,False],0.)
    assert n==2 and torch.equal(out[:,:,[0,1,3]],(attention@values)[:,:,[0,1,3]])
    assert torch.equal(values,before) and not torch.equal(out[:,:,2],(attention@values)[:,:,2])
    out.sum().backward();assert torch.isfinite(values.grad).all()


def test_cfpo_author_scalar_prior_and_pooled_valid_statistics():
    attention=torch.zeros(1,2,4,4)
    attention[0,0,2:,:2]=torch.tensor([.6,.3]);attention[0,1,2:,:2]=.05
    values=torch.tensor([[[[1.,3.],[5.,7.],[0.,0.],[0.,0.]],[[10.,20.],[30.,40.],[0.,0.],[0.,0.]]]])
    # Author medium GMM: pooled mean .25, sample std sqrt(.41/7); only the two .6 edges exceed it.
    out,count=crossmodal_values(attention,values,[True,True,False,False],[False,False,True,True],1.)
    assert count==2
    assert torch.allclose(out[0,0,2:],torch.tensor([[3.9,4.5],[3.9,4.5]]))
    assert torch.allclose(out[0,1,2:],torch.tensor([[2.,3.],[2.,3.]]))
    tiny=attention.clone();tiny[:]=1e-9
    assert crossmodal_values(tiny,values,[True,True,False,False],[False,False,True,True],2.)[1]==0


def test_axpo_recovered_prefix_and_trigger():
    a,b=axpo_advantages([0,0,0,0],[2,2,None,None],{0:[0,1,0,0]})
    assert a.tolist()==pytest.approx([3**.5,0.,0.,0.]) and b[0].tolist()==pytest.approx([-3**-.5,3**.5,-3**-.5,-3**-.5])
    # A successful no-tool original keeps its original advantage; each selected prefix uses its own Eq.4 replacement.
    a,_=axpo_advantages([0,0,1,0],[2,2,None,None],{0:[0,1]})
    assert a.tolist()==pytest.approx([1.,-3**-.5,3**.5,-3**-.5])
    a,_=axpo_advantages([0,0,0,0],[2,2,None,None],{0:[0,1],1:[1,0]})
    assert a.tolist()==pytest.approx([3**.5,3**.5,0.,0.])
    with pytest.raises(ValueError):axpo_advantages([0,1],[2,2],{0:[0,1]})
    with pytest.raises(ValueError):axpo_advantages([0,0],[None,None],{0:[0,1]})


def axpo_fixture(confidences,cfg):
    engine=UpdateGroup(SimpleNamespace(),'axpo',cfg,nullcontext,lambda:None,np.random.default_rng(2));groups=[];calls=[]
    for confidence in confidences:
        samples=[]
        for pair in confidence:
            samples.append({'tokens':[1,2,3,4],'generated':[-1.]*4,'mask':torch.ones(4,dtype=torch.bool),
                'prefix':1 if pair else None,'tool_span':(1,3) if pair else None,'correct':0.,
                'old':torch.tensor([np.log(pair[0]),np.log(pair[1]),np.log(pair[1]),-3.]) if pair else torch.full((4,),-1.)})
        groups.append({'samples':samples,'original_count':len(samples),'advantages':[0.]*len(samples),'selected_prefixes':[],
                       'case':(None,[],(),''),'inputs':{},'features':None,'view':'pos'})
    def collect(*args):
        calls.append(args[-1]);return {'tokens':[1,2,3,4],'generated':[-1.]*4,'mask':torch.ones(4,dtype=torch.bool),
                                      'correct':float(len(calls)%2)}
    engine.collect=collect
    return engine,groups,calls


def test_axpo_ranking_uses_tool_tokens_instead_of_thinking():
    cfg=json.loads(json.dumps(CFG))
    engine,groups,calls=axpo_fixture([[ (.9,.1),(.2,.8),None,None]]+[[None]*4]*3,cfg)
    engine.resample(groups)
    assert groups[0]['selected_prefixes']==[0] and len(calls)==4
    assert groups[0]['advantages'][:4]==pytest.approx([3**.5,0.,0.,0.])


def test_axpo_global_budget_is_breadth_first_and_never_rounded_up():
    cfg=json.loads(json.dumps(CFG));cfg['axpo'].update(continuations=2,extra_rollout_ratio=.75)
    engine,groups,calls=axpo_fixture([[(.9,.1),(.8,.2),(.7,.3),(.6,.4)]]*2,cfg)
    engine.resample(groups)
    assert [g['selected_prefixes'] for g in groups]==[[0,1],[0]] and len(calls)==6
    assert len(calls)<=.75*8
    engine,groups,calls=axpo_fixture([[(.9,.1)]*4],CFG)
    engine.resample(groups)
    assert not calls and not groups[0]['selected_prefixes'] and groups[0]['advantages']==[0.]*4


def test_axpo_complete_call_span_excludes_thinking_and_eos():
    tokenizer=Tokenizer();prefix='<think>abc</think><tool_call>';body='[[100,100,300,300]]</tool_call>'
    tokens=tokenizer.encode(prefix+body)+[2]
    assert tool_span(tokens,tokenizer)==(len('<think>abc</think>'),len(prefix+body))
    assert tool_prefix(tokens,tokenizer)==len(prefix)
    thinking='<think>literal <tool_call> text</think><tool_call>'
    assert tool_prefix(tokenizer.encode(thinking+body),tokenizer)==len(thinking)
    assert tool_span(tokenizer.encode(prefix+'[['),tokenizer) is None
    assert tool_span(tokenizer.encode('<think>abc</think><answer>no</answer>'),tokenizer) is None


def test_axpo_opening_tag_token_can_include_whitespace_but_not_action():
    pieces={10:'<think>x</think>',11:'<tool_call>\n',12:'[[100,100,300,300]]',13:'</tool_call>',14:'<tool_call>['}
    tokenizer=SimpleNamespace(decode=lambda ids,**kwargs: ''.join(pieces.get(int(i),'') for i in ids))
    assert tool_prefix([10,11,12,13,2],tokenizer)==2
    assert tool_span([10,11,12,13,2],tokenizer)==(1,4)
    assert tool_prefix([10,14,13,2],tokenizer) is None


def test_observation_bounds_and_positive_joint_success():
    action=parse_observation('<think>x</think><tool_call>[[100,100,300,300]]</tool_call>',2)
    assert action['valid'] and active_reward(action,GT,'yes',.01,.5)['correct']==1
    assert active_reward(action,GT,'no',.01,.5)['correct']==0
    assert active_reward(parse_observation('<think>x</think><answer>yes</answer>',2),GT,'yes',.01,.5)['correct']==0
    assert not parse_observation('<think>x</think><tool_call>python:exec("bad")</tool_call>',2)['valid']
    assert not parse_observation('<think>x</think><tool_call>[[100,100,300,300]]</tool_call>',2,True)['valid']
    assert not parse_observation('<think>x</think><tool_call>[]</tool_call>',2)['valid']


def test_active_appendix_heuristic_weights_overlap_and_coverage():
    boxes=GT+[[250,100,450,300]]  # IoU=1/7: permitted by the paper's .3 threshold.
    action={'valid':True,'tool':True,'boxes':boxes,'answer':None}
    assert active_reward(action,boxes,'yes',.01,.5)['reward']==5.
    action['boxes']=GT
    result=active_reward(action,GT+[[120,100,320,300]],'yes',.01,.5)
    assert result['coverage']==1. and result['correct']==0 and result['reward']==4.
    assert active_reward({'valid':False},GT,None,.01,.5)['reward']==0.


@pytest.mark.parametrize('method',['active_o3','axpo'])
def test_crop_cap_rejected_before_any_sampling(method):
    calls=[];engine=UpdateGroup(SimpleNamespace(),method,CFG,nullcontext,lambda:calls.append(1),np.random.default_rng(2))
    gt=GT+CONTROL+[[400,400,500,500]]
    with pytest.raises(ValueError,match='strict success is impossible'):engine.groups(None,gt,'q')
    with pytest.raises(ValueError,match='strict success is impossible'):engine.batch_groups([(None,GT,'q'),(None,gt,'q')])
    assert calls==[]
    validate_observation_support([],2);validate_observation_support(GT+CONTROL,2)
    validate_observation_support([[1100,1100,1400,1400]],2)  # Manifest geometry is in pixels before normalization.


def test_defacto_evidence_label_guard_and_controls():
    image=Image.new('RGB',(100,100),'white')
    with pytest.raises(PermissionError):defacto_views(image,GT,CONTROL,False)
    with pytest.raises(ValueError):defacto_views(image,GT,GT,True)
    with pytest.raises(ValueError):defacto_views(image,GT,[[500,500,600,600]],True)
    with pytest.raises(ValueError):defacto_views(image,GT+GT,CONTROL+CONTROL,True)
    views=defacto_views(image,GT,CONTROL,True)
    assert [v[2] for v in views]==['yes','unknown','yes']
    assert views[1][1].getpixel((20,20))==(0,0,0) and views[1][1].getpixel((70,70))==(255,255,255)
    assert len(defacto_views(image,[],[],False))==1
    assert defacto_reward('{"answer":"unknown","boxes":[]}',GT,CONTROL,'cf')['correct']==1
    assert defacto_reward('{"answer":"no","boxes":[]}',GT,CONTROL,'cf')['correct']==0
    assert defacto_reward('{"answer":"yes","boxes":[[100,100,300,300]]}',GT,CONTROL,'pos')['correct']==1
    assert not parse_evidence('{"answer":"no","answer":"yes","boxes":[]}')['valid']
    assert not parse_evidence('{"answer":[],"boxes":[]}')['valid']


def test_defacto_pixel_masks_preserve_disjoint_equal_area_controls():
    image=Image.new('RGB',(10,10),'white')
    # Normalized disjoint equal-area boxes become overlapping pixels at a fractional edge.
    gt=[[100,100,250,300]];controls=[[250,100,400,300]]
    with pytest.raises(ValueError,match='pixel'):defacto_views(image,gt,controls,True)
    # Equal normalized areas can rasterize to 4 versus 9 pixels at different phases.
    with pytest.raises(ValueError,match='pixel'):
        defacto_views(image,[[100,100,300,300]],[[650,650,850,850]],True)
    views=defacto_views(image,GT,CONTROL,True)
    masks=[np.any(np.asarray(v[1])!=np.asarray(image),axis=-1) for v in views[1:]]
    assert masks[0].sum()==masks[1].sum()==4 and not (masks[0]&masks[1]).any()


def test_papo_raw_pixel_bernoulli_mask_can_keep_all_or_mask_all():
    image=Image.new('RGB',(29,15),'white')
    rng=SimpleNamespace(random=lambda:1.)
    out,ids=random_patch_mask(image,14,rng,.6)
    assert ids==[] and np.asarray(out).min()==255
    out,ids=random_patch_mask(image,14,SimpleNamespace(random=lambda:0.),.6)
    assert ids==list(range(6)) and np.asarray(out).max()==0 and np.asarray(image).min()==255
    draws=iter([.1,.9,.9,.9,.9,.1])
    out,ids=random_patch_mask(image,14,SimpleNamespace(random=lambda:next(draws)),.6)
    assert ids==[0,5] and out.getpixel((13,13))==(0,0,0) and out.getpixel((14,13))==(255,255,255)
    assert out.getpixel((28,14))==(0,0,0)


@pytest.mark.parametrize('answer,view,expected',[
    ('yes','pos',1.4),('unknown','pos',-.1),('unknown','cf',1.2),('no','cf',-.4),('yes','cf',-1.3)])
def test_defacto_author_reward_coefficients(answer,view,expected):
    text=json.dumps({'answer':answer,'boxes':GT if answer=='yes' else []})
    assert defacto_reward(text,GT,CONTROL,view)['reward']==pytest.approx(expected)
    wrong=defacto_reward(json.dumps({'answer':'yes','boxes':CONTROL}),GT,CONTROL,'pos')
    assert wrong['coherence']==-1 and wrong['reward']==pytest.approx(1.) and wrong['correct']==0


def test_defacto_rewards_are_independent_of_format_success():
    value=defacto_reward('{"answer":"yes","boxes":[]}',GT,CONTROL,'pos')
    assert value['answer_reward']==1 and value['format_reward']==0 and value['reward']==pytest.approx(.9) and value['correct']==0
    assert defacto_reward('not json',GT,CONTROL,'cf')['reward']==pytest.approx(-.6)
    assert defacto_reward('{"answer":"no","boxes":[]}',[],[],'pos')['correct']==1


def test_probability_and_authorization_fail_closed():
    probability_audit([-.2,-.3],torch.tensor([-.2,-.3]),CFG)
    with pytest.raises(RuntimeError):probability_audit([-.1,-.3],torch.tensor([-.2,-.3]),CFG)
    with pytest.raises(ValueError):native_sampling({'do_sample':True,'top_p':.95})
    with pytest.raises(PermissionError):validate_spec({},CFG)


class Tokenizer:
    eos_token_id=2
    def encode(self,text,**kwargs):return [ord(c)+4 for c in text]
    def decode(self,ids,skip_special_tokens=True):
        return ''.join(chr(int(i)-4) for i in ids if 4<=int(i)<252)
    def convert_tokens_to_ids(self,value):return 255 if value=='<|im_end|>' else None


@pytest.fixture(params=['eager','sdpa'])
def native(tmp_path,request):
    from transformers import Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    from src.model import Model
    from src.medevidence_p4_run import add_reference,current_digest,reference
    torch.manual_seed(27)
    cfg=Qwen2_5_VLConfig(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,rope_scaling={'type':'mrope','mrope_section':[2,1,1]},
        image_token_id=254,video_token_id=251,vision_start_token_id=252,vision_end_token_id=253,bos_token_id=1,eos_token_id=2,pad_token_id=0,
        vision_config={'depth':1,'hidden_size':32,'intermediate_size':64,'num_heads':4,'out_hidden_size':32,
                       'patch_size':14,'temporal_patch_size':2,'spatial_merge_size':2,'fullatt_block_indexes':[0]})
    cfg._attn_implementation=request.param
    base=Qwen2_5_VLForConditionalGeneration(cfg)
    assert next(base.parameters()).device.type=='cpu'
    m=Model.__new__(Model);m.visual=base.visual
    m.model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    for n,p in m.model.named_parameters():
        if 'lora_B' in n:p.data.normal_(0,.03)
    m.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});m.model.enable_input_require_grads()
    m.processor=SimpleNamespace(tokenizer=Tokenizer());m.forward_count=m.score_count=m.vision_count=m.generated_tokens=m.generation_calls=0;m.generating=False
    m.set_training(True);m.visual.requires_grad_(False)
    m.model.save_pretrained(tmp_path/'initial')
    c=SimpleNamespace(references={'COV':{'checkpoint_path':str(tmp_path/'initial'),'identity':{'adapter_digest':current_digest(m)}}},reference_calls=0)
    add_reference(c,m)
    pixels=torch.randn(16,1176)
    def prompt(image,task='A',question=None):
        inputs={'input_ids':torch.tensor([[1,252,254,254,254,254,253,14,15,255,16]]),
                'attention_mask':torch.ones(1,11,dtype=torch.long),'image_grid_thw':torch.tensor([[1,4,4]]),
                'pixel_values':pixels*float(np.asarray(image).mean()/255)}
        return inputs,(2,2)
    m.prompt=prompt
    return m,lambda:reference(c,m),c


def test_native_entropy_generation_and_reference(native):
    m,ref,c=native;image=Image.new('RGB',(28,28),'white');inp,_=m.prompt(image);feat=m.encode(inp)
    score=m.score(inp,feat,ids=[30,31,2],include_entropy=True)
    assert score['length']==3 and (score['entropy']>0).all() and not torch.allclose(score['entropy'],-score['tokens'])
    with ref():frozen=m.score(inp,feat,ids=[30,31,2])['tokens']
    assert not frozen.requires_grad and torch.allclose(frozen,score['tokens']) and c.reference_calls==1
    _,tokens=m.generate(inp,feat,diagnostic=True,sampling_config=CFG['sampling']|{'max_new_tokens':4})
    native_sampling(m.last_generation_settings)
    with torch.no_grad():replayed=m.score(inp,feat,ids=tokens)['tokens']
    stats=probability_audit(m.last_generation_logps,replayed,CFG)
    assert stats['max_error']<1e-5 and not torch.cuda.is_initialized()


def test_native_cfpo_attention_restore_and_checkpoint_gradient(native):
    m,ref,_=native;inp,_=m.prompt(Image.new('RGB',(28,28),'white'));feat=m.encode(inp)
    image=inp['input_ids'][0]==254;query=torch.zeros(11,dtype=torch.bool);query[7:9]=True
    expected=m.score(inp,feat,ids=[30,31,2])['tokens'].detach()
    with cfpo_attention(m.model,image,query,False):actual=m.score(inp,feat,ids=[30,31,2])['tokens']
    assert torch.allclose(actual,expected,atol=1e-6)
    with pytest.raises(RuntimeError):
        with cfpo_attention(m.model,image,query,True,sigma=0.):raise RuntimeError('fixture')
    assert torch.allclose(m.score(inp,feat,ids=[30,31,2])['tokens'],expected)
    with cfpo_attention(m.model,image,query,True,sigma=0.) as stats:
        (-m.score(inp,feat,ids=[30,31,2])['mean']).backward()
    assert stats['forwards']>=4 and stats['salient_edges']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.model.parameters() if p.requires_grad)


def test_native_cfpo_cached_teacher_checkpoint_gradient_matches_direct(native):
    m,_,_=native;inp,_=m.prompt(Image.new('RGB',(28,28),'white'));feat=m.encode(inp)
    image,query=query_masks(m,inp)
    assert query.tolist()==[False]*6+[True]*5
    def context(enabled):return cfpo_attention(m.model,image,query,enabled,sigma=0.)
    trainable=[p for p in m.model.parameters() if p.requires_grad]
    def grads():return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in trainable])
    m.model.gradient_checkpointing_disable()
    with torch.no_grad(),context(True):c=m.score(inp,feat,ids=[30,31,2])['tokens']
    c=c.detach().requires_grad_(True);f=m.score(inp,feat,ids=[30,31,2])['tokens']
    perception_loss(f,c,'cfpo',.1,0.,0.)[0].backward();expected=grads().clone()
    assert c.grad is None
    m.model.zero_grad(set_to_none=True)
    m.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    f=m.score(inp,feat,ids=[30,31,2])['tokens'];perception_loss(f,c,'cfpo',.1,0.,0.)[0].backward()
    assert expected.abs().sum()>0 and torch.allclose(grads(),expected,atol=1e-7,rtol=1e-4)


@pytest.mark.parametrize('method',PRIORITY)
def test_all_method_group_integration_with_native_scores(native,method):
    # Scripted actions test control flow; the separate native-generation test checks sampling alignment.
    m,ref,c=native;image=Image.new('RGB',(28,28),'white');cfg=json.loads(json.dumps(CFG));cfg['sampling']['max_new_tokens']=100
    cfg['cfpo']['sigma']=0.;index=0
    texts=['[]','[[100,100,300,300]]','[]','[[100,100,300,300]]']
    if method in ('active_o3','axpo'):
        texts=['<think>x</think><tool_call>[[500,500,700,700]]</tool_call>']*(16 if method=='axpo' else 4)
        texts+=['<think>x</think><tool_call>[[100,100,300,300]]</tool_call>',
                '<think>x</think><tool_call>[[500,500,700,700]]</tool_call>']*2
        if method=='active_o3':texts=texts[4:8]
    if method=='defacto':
        texts=['{"answer":"yes","boxes":[[100,100,300,300]]}','{"answer":"no","boxes":[]}']*2
        texts+=['{"answer":"unknown","boxes":[]}','{"answer":"no","boxes":[]}']*2
        texts+=['{"answer":"yes","boxes":[[100,100,300,300]]}','{"answer":"no","boxes":[]}']*2
    def scripted(inputs,features,**kwargs):
        nonlocal index
        text=texts[index];index+=1;tokens=m.processor.tokenizer.encode(text)+[2]
        prefix_length=inputs['input_ids'].shape[1]-11
        if prefix_length:tokens=tokens[prefix_length:]
        with torch.no_grad():m.last_generation_logps=m.score(inputs,features,ids=tokens)['tokens'].tolist()
        m.last_generation_settings=cfg['sampling']|{'do_sample':True}
        return m.processor.tokenizer.decode(tokens),tokens
    m.generate=scripted
    engine=UpdateGroup(m,method,cfg,ref,lambda:None,np.random.default_rng(3))
    # Observer contract is separately exercised through its frozen reference context.
    if method in ('active_o3','axpo'):
        actual=engine.frozen_answer(image,'Is the target visible?');assert actual in ('yes','no')
        engine.frozen_answer=lambda image,question:'yes'
    case=(image,GT,'Is the target visible?',CONTROL,True)
    groups=engine.batch_groups([case]*4) if method=='axpo' else engine.groups(*case)
    if method=='axpo':
        assert engine.extra_rollouts==4 and sum(len(g['samples']) for g in groups)==20
        selected=next(g for g in groups if g['selected_prefixes'])
        source=selected['selected_prefixes'][0];n=selected['samples'][source]['prefix']
        assert not selected['samples'][source]['mask'][n:].any()
        assert all(not s['mask'][:n].any() for s in selected['samples'][4:])
        assert tool_prefix(selected['samples'][source]['tokens'],m.processor.tokenizer)==n
        assert all(a==0 for i,a in enumerate(selected['advantages'][:4]) if i!=source)
    if method in ('papo','cfpo'):
        assert all(not s['corrupted_old'].requires_grad for g in groups for s in g['samples'])
        cached_count=m.score_count
    stats=engine.backward(groups)
    if method in ('papo','cfpo'):assert m.score_count-cached_count==2*len(groups[0]['samples'])
    if method in ('visurf','rl_zvp','defacto'):assert all(t['reference_kl']==0 for t in stats['terms'])
    if method=='axpo':
        assert sum(t['weight']==.25 for t in stats['terms'])==5  # One prefix + four branches, Eq.5 added once.
        assert sum(t['weight']==.0625 for t in stats['terms'])==15  # Unselected ordinary GRPO.
    assert np.isfinite(stats['mean_loss']) and stats['completions']>=4
    trained=[p for p in m.model.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in trained)
    assert not any(p.grad is not None for n,p in m.model.named_parameters() if '.reference.' in n or '.visual.' in n)
    opt=torch.optim.AdamW(trained,lr=1e-3);before=[p.detach().clone() for p in trained];opt.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,trained)) and not torch.cuda.is_initialized()
