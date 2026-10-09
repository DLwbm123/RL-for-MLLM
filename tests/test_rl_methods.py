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
                            perception_loss,axpo_advantages,crossmodal_values,cfpo_attention)
from src.rl_observation import (parse_observation,tool_prefix,active_reward,random_patch_mask,
                               defacto_views,parse_evidence,defacto_reward)
from src.rl_methods_run import UpdateGroup,probability_audit,native_sampling
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


@pytest.mark.parametrize('method',['papo','cfpo'])
def test_perception_formula_and_joint_partial_gradient(method):
    f=torch.tensor([-.4,-1.2],requires_grad=True);c=torch.tensor([-.9,-.8],requires_grad=True)
    value,_=perception_loss(f,c,method,.1,.03,.02)
    d=c-f if method=='papo' else f-c
    assert torch.allclose(value,-.1*(d.exp()-d-1).mean()-.03*f.mean()-.02*c.mean())
    expected=torch.autograd.grad(value,(f,c))
    lf,_=perception_loss(f,c.detach(),method,.1,.03,.02)
    lc,_=perception_loss(f.detach(),c,method,.1,.03,.02)
    actual=(torch.autograd.grad(lf,f)[0],torch.autograd.grad(lc,c)[0])
    assert all(torch.allclose(a,b) for a,b in zip(expected,actual))
    if method=='papo':assert torch.isfinite(perception_loss(torch.tensor([-100.]),torch.tensor([0.]),method,.1,0.,0.)[0])


def test_cfpo_per_head_value_only_intervention():
    attention=torch.zeros(1,2,4,4);attention[:,:,0,0]=1;attention[:,:,1,:2]=.5
    attention[0,0,2]=torch.tensor([.9,.05,.05,0]);attention[0,1,2]=torch.tensor([.1,.8,.1,0]);attention[:,:,3,:]=.25
    values=torch.arange(16.).reshape(1,2,4,2).requires_grad_();before=values.detach().clone()
    out,n=crossmodal_values(attention,values,[True,True,False,False],[False,False,True,False],0.)
    assert n==2 and torch.equal(out[:,:,[0,1,3]],(attention@values)[:,:,[0,1,3]])
    assert torch.equal(values,before) and not torch.equal(out[:,:,2],(attention@values)[:,:,2])
    out.sum().backward();assert torch.isfinite(values.grad).all()


def test_axpo_recovered_prefix_and_trigger():
    a,b=axpo_advantages([0,0,0,0],[2,2,None,None],{0:[0,1,0,0]})
    assert a[0]>0 and (a[1:]<0).all() and b[0][1]>0
    with pytest.raises(ValueError):axpo_advantages([0,1],[2,2],{0:[0,1]})
    with pytest.raises(ValueError):axpo_advantages([0,0],[None,None],{0:[0,1]})


def test_observation_bounds_and_positive_joint_success():
    action=parse_observation('<think>x</think><tool_call>[[100,100,300,300]]</tool_call>',2)
    assert action['valid'] and active_reward(action,GT,'yes',.01,.5)['correct']==1
    assert active_reward(action,GT,'no',.01,.5)['correct']==0
    assert active_reward(parse_observation('<think>x</think><answer>yes</answer>',2),GT,'yes',.01,.5)['correct']==0
    assert not parse_observation('<think>x</think><tool_call>python:exec("bad")</tool_call>',2)['valid']
    assert not parse_observation('<think>x</think><tool_call>[[100,100,300,300]]</tool_call>',2,True)['valid']
    assert not parse_observation('<think>x</think><tool_call>[]</tool_call>',2)['valid']


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
    out,ids=random_patch_mask(image,(2,2),np.random.default_rng(2),.6)
    assert len(ids)==3 and np.any(np.asarray(out)==255) and image.getpixel((20,20))==(255,255,255)


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


@pytest.fixture
def native(tmp_path):
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
    base=Qwen2_5_VLForConditionalGeneration(cfg);cfg._attn_implementation='eager'
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


def test_native_cfpo_paired_checkpoint_gradient_matches_joint(native):
    m,_,_=native;inp,_=m.prompt(Image.new('RGB',(28,28),'white'));feat=m.encode(inp)
    image=inp['input_ids'][0]==254;query=torch.zeros(11,dtype=torch.bool);query[7:9]=True
    def context(enabled):return cfpo_attention(m.model,image,query,enabled,sigma=0.)
    trainable=[p for p in m.model.parameters() if p.requires_grad]
    def grads():return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in trainable])
    m.model.gradient_checkpointing_disable()
    with context(False):f=m.score(inp,feat,ids=[30,31,2])['tokens']
    with context(True):c=m.score(inp,feat,ids=[30,31,2])['tokens']
    perception_loss(f,c,'cfpo',.1,.03,0.)[0].backward();expected=grads().clone()
    m.model.zero_grad(set_to_none=True)
    m.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    with torch.no_grad(),context(True):c=m.score(inp,feat,ids=[30,31,2])['tokens']
    with context(False):
        f=m.score(inp,feat,ids=[30,31,2])['tokens'];perception_loss(f,c,'cfpo',.1,.03,0.)[0].backward()
    with context(True):
        c=m.score(inp,feat,ids=[30,31,2])['tokens'];perception_loss(f.detach(),c,'cfpo',.1,.03,0.)[0].backward()
    assert expected.abs().sum()>0 and torch.allclose(grads(),expected,atol=1e-7,rtol=1e-4)


@pytest.mark.parametrize('method',PRIORITY)
def test_all_method_group_integration_with_native_scores(native,method):
    # Scripted actions test control flow; the separate native-generation test checks sampling alignment.
    m,ref,c=native;image=Image.new('RGB',(28,28),'white');cfg=json.loads(json.dumps(CFG));cfg['sampling']['max_new_tokens']=100
    cfg['cfpo']['sigma']=0.;index=0
    texts=['[]','[[100,100,300,300]]','[]','[[100,100,300,300]]']
    if method in ('active_o3','axpo'):
        texts=['<think>x</think><tool_call>[[500,500,700,700]]</tool_call>']*4
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
    groups=engine.groups(image,GT,'Is the target visible?',CONTROL,True)
    if method=='axpo':
        assert engine.extra_rollouts==4 and len(groups[0]['samples'])==8
        source=groups[0]['selected_prefixes'][0];n=groups[0]['samples'][source]['prefix']
        assert not groups[0]['samples'][source]['mask'][n:].any()
        assert all(not s['mask'][:n].any() for s in groups[0]['samples'][4:])
        assert tool_prefix(groups[0]['samples'][source]['tokens'],m.processor.tokenizer)==n
    stats=engine.backward(groups)
    assert np.isfinite(stats['mean_loss']) and stats['completions']>=4
    trained=[p for p in m.model.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in trained)
    assert not any(p.grad is not None for n,p in m.model.named_parameters() if '.reference.' in n or '.visual.' in n)
    opt=torch.optim.AdamW(trained,lr=1e-3);before=[p.detach().clone() for p in trained];opt.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,trained)) and not torch.cuda.is_initialized()
