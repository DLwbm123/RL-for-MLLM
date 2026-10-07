"""Native P6 calibration, three fixed training branches, and separated answer evaluation."""
from collections import Counter
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.data import QUESTIONS
from src.v2 import DevelopmentData,save
from src.model import Model,load_frozen_adapter
from src.experiment import seed_all
from src.medevidence_run import tensor_digest
from src.medevidence_p2 import enrich,distribution
from src.medevidence_p2_run import RNG,set_RNG,evaluating,BudgetStop
from src.medevidence import normalized
from src.medevidence_p4 import reward,policy_terms,extended_summary
from src.medevidence_p4_run import parameters,current_digest,reference_digest,gradient,add_reference,reference,bind_classification_labels
from src.medevidence_p5 import prompt
from src.medevidence_p6 import training_advantages,common_coefficient


def read(p):return json.loads(Path(p).read_text())


class Context:
    def __init__(self):
        self.started=time.time();self.base=Path(os.environ['PILOT_ROOT']);self.root=self.base/'outputs';self.p=self.root/'protocol'
        self.cfg=read(self.base/'code/configs/medevidence_p6.json');self.stage=os.environ['JOB_STAGE'];self.deadline=float(os.environ['JOB_DEADLINE'])
        auth=read(self.root/'authorization.json')
        assert auth['gpu_authorized'] and auth['prior_GPU_seconds']==self.cfg['prior_GPU_seconds']
        self.dest=self.root/self.stage;self.dest.mkdir();self.stopped=False
        signal.signal(signal.SIGUSR1,lambda *_:setattr(self,'stopped',True))
        self.sets=read(self.p/'sets.json');self.plan=read(self.p/'schedule.json');self.contracts=read(self.p/'contracts.json');self.sizes=read(self.p/'sizes.json')
        self.refs=read(self.p/'references.json');self.references={'COV':self.refs['INIT']};self.reference_calls=0;self.sampling_state=None
        self.frame=pd.read_csv(self.p/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
        allowed=self.sets['dev']+self.sets['calibration'] if self.stage.startswith('eval_') else self.sets['train']
        self.frame=self.frame[self.frame.image_id.isin(allowed)];self.rows=self.frame.set_index('image_id').to_dict('index')
        for r in self.rows.values():r['boxes']=json.loads(r['boxes'])
        self.data=DevelopmentData(os.environ['DATA_ROOT'],self.frame,self.dest/'data_access.jsonl');self.data.install_guard();self.cache={}
        self.model_path=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct';identity=read(self.p/'base_identity.json')
        for name,v in identity['base']['files'].items():
            st=(self.model_path/name).stat();assert (st.st_size,st.st_mtime_ns)==(v['bytes'],v['mtime_ns'])
        self.identity=identity;self.tick()

    def tick(self):
        if self.stopped or time.time()>self.deadline-60:raise BudgetStop('P6 stage budget reserve')

    def model(self,name='INIT',trainable=False):
        self.tick();seed_all(self.cfg['seed'])
        cp=Path(self.refs['INIT']['checkpoint_path']) if name=='INIT' else self.root/name/'step_0064'
        expected=self.refs['INIT']['identity'] if name=='INIT' else read(cp/'identity.json')
        m=Model(self.model_path,cp,trainable,576,1024)
        assert current_digest(m)==expected['adapter_digest']
        assert m.processor.tokenizer.eos_token_id==self.identity['eos_id']
        if trainable:assert sum(p.numel() for p in parameters(m))==5046272
        bind_classification_labels(m,['no','yes']);return m

    def encode(self,m,k,task='L'):
        self.tick()
        if k not in self.cache:
            im=self.data.image(k).resize(tuple(self.sizes[k]),Image.Resampling.BICUBIC)
            QUESTIONS['L']=prompt('normalized',self.sizes[k]);inp,grid=m.prompt(im,'L')
            assert [grid[1]*28,grid[0]*28]==self.sizes[k]
            self.cache[k]=(im,inp,m.encode(inp))
        im,inp,feat=self.cache[k]
        if task=='A':
            QUESTIONS['A']='Does this chest radiograph contain lung opacity suspicious for pneumonia? Answer with exactly one label: no or yes.'
            inp,_=m.prompt(im,'A')
        return inp,feat


def optimizer(c,m):
    v=c.cfg['optimizer'];return torch.optim.AdamW(parameters(m),lr=v['lr'],betas=tuple(v['betas']),eps=v['eps'],weight_decay=v['weight_decay'])


def initial_identity(c,m,opt):
    return {'adapter_digest':current_digest(m),'optimizer_digest':tensor_digest(opt.state_dict()),'RNG_digest':tensor_digest(RNG()),
            'trainable_names':[n for n,p in m.model.named_parameters() if p.requires_grad],'optimizer_state_empty':not bool(opt.state),
            'schedule_position':0,'initialization':'P5 normalized step256','lr':c.cfg['optimizer']['lr'],
            'training_mode':m.model.training,'visual_training_mode':m.visual.training}


def supervised(c,m,k,scale):
    inp,feat=c.encode(m,k);contract=c.contracts[k];v=m.score(inp,feat,answer=contract['text'],include_eos=True)
    assert v['ids'].tolist()==contract['ids'];loss=-v['mean']
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SFT loss')
    (loss*scale).backward();return float(loss.detach())


def check_generation(settings,cfg):
    for key in ('temperature','top_p','top_k'):
        if settings[key]!=cfg['RL'][key]:raise ValueError('Sampling transform differs from frozen policy: '+key)
    if not settings['do_sample'] or settings['num_beams']!=1 or settings['repetition_penalty']!=1.:
        raise ValueError('Unsupported probability transform')
    for key in ('forced_bos_token_id','forced_eos_token_id','bad_words_ids','suppress_tokens','begin_suppress_tokens','constraints','force_words_ids'):
        if settings.get(key):raise ValueError('Forced/suppressed sampling support: '+key)
    if settings.get('min_p') not in (None,0.) or settings.get('typical_p',1.)!=1. or settings.get('no_repeat_ngram_size',0)!=0:
        raise ValueError('Unexpected generation filter')
    if settings.get('min_length',0)!=0 or settings.get('min_new_tokens') not in (None,0):raise ValueError('Minimum length changes EOS support')


def rollout(c,m,k,audit=False):
    inp,feat=c.encode(m,k);main=RNG();samples=[];settings=c.cfg['RL']
    if c.sampling_state is None:
        seed_all(c.cfg['sampling_seed']);c.sampling_state=RNG()
    set_RNG(c.sampling_state)
    try:
        for _ in range(4):
            c.tick();text,tokens=m.generate(inp,feat,max_tokens=c.cfg['max_new_tokens'],sample=True,diagnostic=audit,
                sampling_config={x:settings[x] for x in ('temperature','top_p','top_k')}|{'max_new_tokens':c.cfg['max_new_tokens']})
            check_generation(m.last_generation_settings,c.cfg)
            if not tokens:raise ValueError('Empty completion token sequence')
            truncated=len(tokens)>=c.cfg['max_new_tokens'] and tokens[-1]!=m.processor.tokenizer.eos_token_id
            samples.append({'text':text,'tokens':tokens,'truncated':truncated,
                            'generation_logps':m.last_generation_logps[:] if audit else None})
        c.sampling_state=RNG()
    finally:set_RNG(main)
    gt=normalized(c.rows[k]['boxes'],c.rows[k]['width'],c.rows[k]['height'])
    with evaluating(m):
        for r in samples:
            r['reward']=reward(r['text'],gt,r['truncated']);r['old']=m.score(inp,feat,ids=r['tokens'])['tokens'].detach()
            if not torch.isfinite(r['old']).all():raise FloatingPointError('Nonfinite old policy')
            if audit:
                delta=(r['old'].cpu()-torch.tensor(r['generation_logps'])).abs()
                r['generation_replay_error_max']=float(delta.max());r['generation_replay_error_mean']=float(delta.mean())
                if delta.max()>c.cfg['generation_replay_max_logprob_error'] or delta.mean()>c.cfg['generation_replay_mean_logprob_error']:
                    raise ValueError('Generation/replay log probabilities exceed frozen bf16 tolerance')
    with reference(c,m):
        for r in samples:
            r['reference']=m.score(inp,feat,ids=r['tokens'])['tokens'].detach()
            if audit and not torch.allclose(r['old'],r['reference'],atol=c.cfg['reference_initial_logprob_atol'],rtol=0):
                raise ValueError('Initial reference policy probabilities disagree')
    return samples


def rl_backward(c,m,k,samples,branch,scale,coefficient,beta,check_old=False):
    positive=bool(c.rows[k]['boxes']);adv=training_advantages([r['reward'] for r in samples],positive,branch);stats=[];inp,feat=c.encode(m,k)
    for r,a in zip(samples,adv):
        c.tick();current=m.score(inp,feat,ids=r['tokens'])['tokens']
        if check_old and not torch.allclose(current.detach(),r['old'],atol=c.cfg['old_current_logprob_atol'],rtol=c.cfg['old_current_logprob_rtol']):
            raise ValueError('Sampling-time old policy replay mismatch')
        objective,kl=policy_terms(current,r['old'],r['reference'],a,torch.ones_like(current,dtype=torch.bool),c.cfg['RL']['epsilon'])
        loss=(-coefficient*objective+beta*kl)*scale/4
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite RL objective')
        loss.backward();stats.append((float(objective.detach()),float(kl.detach())))
    all_wrong=not positive and all(r['reward']==-1 for r in samples)
    if all_wrong:assert torch.equal(adv,torch.zeros(4) if branch=='B_GRPO' else -torch.ones(4))
    return {'positive':positive,'all_wrong_negative':all_wrong,'all_correct_negative':not positive and all(r['reward']==1 for r in samples),
            'zero_advantage_group':bool(torch.all(adv==0)),'advantage_mean':float(adv.mean()),'advantage_abs_mean':float(adv.abs().mean()),
            'reward_mean':float(np.mean([r['reward'] for r in samples])),'J_GRPO':float(np.mean([v[0] for v in stats])),
            'KL':float(np.mean([v[1] for v in stats])),'rollouts':4,'completion_tokens':sum(len(r['tokens']) for r in samples)}


def assert_frozen(m):
    if any(p.grad is not None for n,p in m.model.named_parameters() if '.visual.' in n or '.reference.' in n):
        raise RuntimeError('Frozen module has gradients')


def preflight(c):
    m=c.model(trainable=True);add_reference(c,m);seed_all(c.cfg['seed']);opt=optimizer(c,m)
    initial=initial_identity(c,m,opt);state=RNG();opt.zero_grad(set_to_none=True);timing={'SFT':[],'rollout':[],'B_GRPO':[],'C_NEGABS':[]}
    save(c.dest/'initial_identity.json',initial)
    for ids in c.plan[:4]:
        t=time.time()
        for k in ids:supervised(c,m,k,1/16)
        timing['SFT'].append(time.time()-t)
    norms={'SFT':float(torch.linalg.vector_norm(gradient(m)))};opt.zero_grad(set_to_none=True);groups={};checks=[]
    for ids in c.plan[:4]:
        t=time.time()
        for k in ids:
            groups[k]=rollout(c,m,k,True)
            checks.extend({'max':r['generation_replay_error_max'],'mean':r['generation_replay_error_mean']} for r in groups[k])
        timing['rollout'].append(time.time()-t)
    for branch in ('B_GRPO','C_NEGABS'):
        opt.zero_grad(set_to_none=True)
        for ids in c.plan[:4]:
            t=time.time()
            for k in ids:rl_backward(c,m,k,groups[k],branch,1/16,1.,0.,True)
            timing[branch].append(time.time()-t)
        norms[branch]=float(torch.linalg.vector_norm(gradient(m)));assert_frozen(m)
    coefficient=common_coefficient(norms['SFT'],norms['B_GRPO'],norms['C_NEGABS'])
    opt.zero_grad(set_to_none=True);set_RNG(state);c.sampling_state=None
    assert initial_identity(c,m,opt)==initial and reference_digest(m)==initial['adapter_digest']
    means={k:float(np.mean(v)) for k,v in timing.items()}
    estimates={'A_SFT':means['SFT'],**{k:means['SFT']+means['rollout']+means[k] for k in ('B_GRPO','C_NEGABS')}}
    save(c.dest/'summary.json',{'status':'completed','optimizer_updates':0,'calibration_patients':16,'calibration_rollouts':64,'gradient_norms':norms,
        'lambda':coefficient,'initial_RL_to_SFT_ratios':{k:coefficient*norms[k]/norms['SFT'] for k in ('B_GRPO','C_NEGABS')},
        'generation_replay_error_max':max(v['max'] for v in checks),'generation_replay_error_mean_max':max(v['mean'] for v in checks),
        'old_current_alignment_checked':True,'initial_reference_probability_alignment_checked':True,'reference_calls_verified':c.reference_calls,
        'reference_adapter_unchanged':True,'model_optimizer_RNG_restored':True,'frozen_gradients_absent':True,'mean_step_seconds':estimates,
        'component_seconds':means,'effective_generation_settings':m.last_generation_settings,'data_access':dict(c.data.access_counts),'test_pixels_read':0,'runtime_seconds':time.time()-c.started})


def train(c):
    plan=read(c.root/'training_plan.json');assert plan['steps']==c.cfg['steps']==64
    m=c.model(trainable=True);is_rl=c.stage!='A_SFT'
    if is_rl:add_reference(c,m)
    seed_all(c.cfg['seed']);opt=optimizer(c,m);initial=initial_identity(c,m,opt);save(c.dest/'initial_identity.json',initial)
    expected=read(c.root/'preflight/initial_identity.json');assert initial==expected
    completed=0;rows=[];counts=Counter();reason=None
    def checkpoint(step):
        cp=c.dest/f'step_{step:04d}';cp.mkdir();m.model.save_pretrained(cp,safe_serialization=True,selected_adapters=['default'])
        torch.save(opt.state_dict(),cp/'optimizer.pt');torch.save(RNG(),cp/'rng.pt')
        if c.sampling_state is not None:torch.save(c.sampling_state,cp/'sampling_rng.pt')
        save(cp/'identity.json',{'adapter_digest':current_digest(m),'steps':step,'optimizer_digest':tensor_digest(opt.state_dict()),'RNG_digest':tensor_digest(RNG())})
    try:
        checkpoint(0)
        with (c.dest/'training.jsonl').open('x') as f,(c.dest/'groups.jsonl').open('x') as trace:
            for ids in c.plan:
                c.tick();start=time.time();opt.zero_grad(set_to_none=True);losses=[];groups=[]
                for k in ids:
                    samples=rollout(c,m,k) if is_rl else None
                    losses.append(supervised(c,m,k,.25))
                    if is_rl:
                        stat=rl_backward(c,m,k,samples,c.stage,.25,plan['lambda'],c.cfg['RL']['beta'],True);groups.append(stat)
                        trace.write(json.dumps({'image_id':k,'step':completed+1,**stat})+'\n');trace.flush()
                assert_frozen(m);norm=torch.nn.utils.clip_grad_norm_(parameters(m),c.cfg['gradient_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                opt.step();completed+=1;counts.update(ids)
                r={'step':completed,'patients':ids,'L_NLL':float(np.mean(losses)),'gradient_norm_pre_clip':float(norm),'seconds':time.time()-start,
                   'rollouts':sum(x['rollouts'] for x in groups),'all_wrong_negative_groups':sum(x['all_wrong_negative'] for x in groups),
                   'all_correct_negative_groups':sum(x['all_correct_negative'] for x in groups),'zero_advantage_groups':sum(x['zero_advantage_group'] for x in groups),
                   'mean_reward':float(np.mean([x['reward_mean'] for x in groups])) if groups else None,
                   'J_GRPO':float(np.mean([x['J_GRPO'] for x in groups])) if groups else None,'KL':float(np.mean([x['KL'] for x in groups])) if groups else None}
                rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
                if completed%8==0:print(json.dumps({k:v for k,v in r.items() if k!='patients'}),flush=True)
        assert counts==Counter(k for ids in c.plan for k in ids)
        if is_rl:assert reference_digest(m)==initial['adapter_digest']
        checkpoint(64);status='completed'
    except Exception as exc:
        status='stopped_budget' if isinstance(exc,BudgetStop) else 'failed';reason=str(exc)
        opt.zero_grad(set_to_none=True)
        save(c.dest/'failure.json',{'error_type':type(exc).__name__,'reason':reason,'steps':completed})
    save(c.dest/'summary.json',{'status':status,'reason':reason,'steps':completed,'planned_steps':64,'SFT_exposures':sum(counts.values()),
        'distinct_training_patients':len(counts),'exposure_histogram':dict(Counter(counts.values())),'rollouts':sum(r['rollouts'] for r in rows),
        'all_wrong_negative_groups':sum(r['all_wrong_negative_groups'] for r in rows),'all_correct_negative_groups':sum(r['all_correct_negative_groups'] for r in rows),
        'zero_advantage_groups':sum(r['zero_advantage_groups'] for r in rows),'lambda':plan['lambda'] if is_rl else 0,'beta':c.cfg['RL']['beta'] if is_rl else 0,
        'statistics':{key:distribution([r[key] for r in rows if r[key] is not None]) for key in ('L_NLL','gradient_norm_pre_clip','seconds','mean_reward','J_GRPO','KL')},
        'reference_calls_verified':c.reference_calls,'data_access':dict(c.data.access_counts),'test_pixels_read':0,
        'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'runtime_seconds':time.time()-c.started})


def answer_digest(m):
    return tensor_digest({n.replace('.answer.','.default.'):p for n,p in m.model.named_parameters() if 'lora_' in n and '.answer.' in n})


def evaluate(c):
    name=c.stage[5:]
    if name!='INIT':assert read(c.root/name/'summary.json')['status']=='completed'
    m=c.model(name,False);load_frozen_adapter(m.model,c.refs['ANSWER']['checkpoint_path'],'answer')
    assert answer_digest(m)==c.refs['ANSWER']['identity']['adapter_digest']
    m.model.set_adapter('default');m.model.requires_grad_(False);m.set_training(False)
    result={'status':'completed','classifier':'separate frozen P3 COV','test_pixels_read':0};initial_default=current_digest(m)
    with torch.no_grad():
        for subset in ('dev','calibration'):
            records=[]
            with (c.dest/(subset+'.jsonl')).open('x') as f:
                for k in c.sets[subset]:
                    c.tick();inp,feat=c.encode(m,k);text,tokens=m.generate(inp,feat,max_tokens=c.cfg['max_new_tokens'],sample=False)
                    r={'image_id':k,'case_id':c.rows[k]['case_id'],'loc_text':text,'truncated':len(tokens)>=c.cfg['max_new_tokens'] and tokens[-1]!=m.processor.tokenizer.eos_token_id}
                    if subset=='dev':
                        ai,_=c.encode(m,k,'A');m.model.set_adapter('answer');m.model.requires_grad_(False)
                        scores=m.classes(ai,feat)[0].float();r.update(p=float(scores.softmax(-1)[1]),score_no=float(scores[0]),score_yes=float(scores[1]))
                        m.model.set_adapter('default');m.model.requires_grad_(False)
                    r=enrich(r,c.rows[k]);records.append(r);f.write(json.dumps(r)+'\n');f.flush()
            result[subset]={'patients':len(records),'metrics':extended_summary(records)}
    assert current_digest(m)==initial_default and answer_digest(m)==c.refs['ANSWER']['identity']['adapter_digest']
    result.update(answer_adapter_verified_unchanged=True,localization_adapter_unchanged=True,data_access=dict(c.data.access_counts),runtime_seconds=time.time()-c.started)
    save(c.dest/'summary.json',result)


def main():
    c=None
    try:
        c=Context()
        if c.stage=='preflight':preflight(c)
        elif c.stage in c.cfg['branches']:train(c)
        elif c.stage.startswith('eval_'):evaluate(c)
        else:raise ValueError('Unknown stage')
    except Exception as exc:
        if c:save(c.dest/'summary.json',{'status':'stopped_budget' if isinstance(exc,BudgetStop) else 'failed','reason':str(exc),'exception_type':type(exc).__name__,'test_pixels_read':0,'runtime_seconds':time.time()-c.started})
        raise


if __name__=='__main__':main()
