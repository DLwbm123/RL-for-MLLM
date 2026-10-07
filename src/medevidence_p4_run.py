"""Native P4 workers: natural diagnostics, full SFT and gated geometry GRPO."""
from collections import Counter
from contextlib import contextmanager
from functools import partial
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
import torch
from src.data import QUESTIONS,sha
from src.model import Model,load_frozen_adapter
from src.v2 import DevelopmentData,save
from src.experiment import seed_all,optimizer_snapshot
from src.medevidence_run import tensor_digest
from src.medevidence_p2_run import RNG,set_RNG,evaluating,measure,BudgetStop
from src.medevidence import normalized,parse_boxes,matching
from src.medevidence_p4 import reward,advantages,policy_terms,best_single,extended_summary
from src.medevidence_p2 import distribution


def read(p):return json.loads(Path(p).read_text())
def bind_classification_labels(model,labels):
    if list(labels)!=['no','yes']:raise ValueError('P4 requires explicit frozen no/yes candidate order')
    model.classes=partial(model.classes,labels=list(labels))
def current_digest(m):return tensor_digest({n:p for n,p in m.model.named_parameters() if 'lora_' in n and '.default.' in n})
def reference_digest(m):return tensor_digest({n.replace('.reference.','.default.'):p for n,p in m.model.named_parameters() if 'lora_' in n and '.reference.' in n})
def parameters(m):return [p for p in m.model.parameters() if p.requires_grad]
def gradient(m):return torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for p in parameters(m)])


class Context:
    def __init__(self):
        self.started=time.time();self.stop=False;signal.signal(signal.SIGUSR1,lambda *_:setattr(self,'stop',True))
        self.base=Path(os.environ['PILOT_ROOT']);self.root=self.base/'outputs';self.code=self.base/'code';self.p=self.root/'protocol'
        self.stage=os.environ['JOB_STAGE'];self.deadline=float(os.environ['JOB_DEADLINE']);self.cfg=read(self.code/'configs/medevidence_p4.json');auth=read(self.root/'authorization.json')
        if not auth['gpu_authorized'] or auth['gpu_hours_limit']!=3 or auth['allowed_gpus']!=[0,1,2]:raise PermissionError('Current P4 resource receipt required before CUDA')
        if self.cfg['max_new_tokens']>256 or self.cfg['answer_coefficient']!=0 or self.cfg['optimizer']['lr']!=5e-6:raise PermissionError('Frozen P4 task/length/lr')
        lock=read(self.p/('evaluation_fix_lock.json' if self.stage.endswith('_corrected') else 'lock.json'))
        for path,digest in lock['code_hashes'].items():
            if sha((self.code/path).read_bytes())!=digest:raise ValueError('Frozen code changed '+path)
        for path,digest in lock['protocol_hashes'].items():
            if sha((self.p/path).read_bytes())!=digest:raise ValueError('Frozen protocol changed '+path)
        self.dest=self.root/self.stage
        if self.dest.exists():raise FileExistsError('No automatic retry/repeated stage')
        self.dest.mkdir();self.frame=pd.read_csv(self.p/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str});self.rows=self.frame.set_index('image_id').to_dict('index')
        for r in self.rows.values():r['boxes']=json.loads(r['boxes'])
        self.plan=read(self.p/'schedule.json');self.selected=read(self.p/'selected_patients.json');self.sets=read(self.p/'diagnostic_sets.json')
        self.contracts=read(self.p/'token_contracts.json');self.images=read(self.p/'selected_images.json');self.references=read(self.p/'references.json');self.identity=read(self.p/'identity.json')
        self.model_path=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
        for n,v in self.identity['base']['files'].items():
            st=(self.model_path/n).stat()
            if (st.st_size,st.st_mtime_ns)!=(v['bytes'],v['mtime_ns']):raise ValueError('Base identity changed')
        for k,v in self.images.items():
            st=(Path(os.environ['DATA_ROOT'])/self.rows[k]['image_path']).stat()
            if (st.st_size,st.st_mtime_ns)!=(v['bytes'],v['mtime_ns']):raise ValueError('Image identity changed')
        self.data=DevelopmentData(os.environ['DATA_ROOT'],self.frame[self.frame.image_id.isin(self.images)],self.dest/'data_access.jsonl');self.data.install_guard()
        QUESTIONS.update(self.cfg['prompts']);self.sampling_state=None;self.reference_calls=0;self.tick()

    def tick(self):
        if self.stop or time.time()>=self.deadline-90:raise BudgetStop('P4 stage/shared budget reserve')

    def model(self,name='COV',trainable=False):
        self.tick();seed_all(17)
        if name in self.references:cp=Path(self.references[name]['checkpoint_path']);identity=self.references[name]['identity']
        else:
            plan=read(self.root/'training_plan.json');cp=self.root/name/f"step_{plan['steps']:04d}";identity=read(cp/'identity.json')
        m=Model(self.model_path,cp,trainable,self.cfg['min_visual_tokens'],self.cfg['max_visual_tokens'])
        if current_digest(m)!=identity['adapter_digest']:raise ValueError('Actual initialization/final adapter identity mismatch')
        if m.processor.tokenizer.eos_token_id!=self.identity['eos_id'] or {x:m.processor.tokenizer.encode(x,add_special_tokens=False) for x in ('no','yes')}!=self.identity['label_ids']:raise ValueError('Tokenizer identity')
        if trainable and (sum(p.numel() for p in parameters(m))!=5046272 or any('lora_' not in n or '.visual.' in n for n,p in m.model.named_parameters() if p.requires_grad)):raise ValueError('Language q/v LoRA scope changed')
        bind_classification_labels(m,self.cfg['label_order'])
        self.current_model_identity=identity['adapter_digest'];self.tick();return m


def optimizer(m,c):
    o=c.cfg['optimizer'];return torch.optim.AdamW(parameters(m),lr=o['lr'],betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'])


def initial_identity(m,opt):
    return {'adapter_digest':current_digest(m),'optimizer_digest':tensor_digest(opt.state_dict()),'RNG_digest':tensor_digest(RNG()),'schedule_position':0,
            'lr':opt.param_groups[0]['lr'],'trainable_modules':[n for n,p in m.model.named_parameters() if p.requires_grad],'optimizer_reset':True,'optimizer_state_empty':not bool(opt.state),'initialization':'P3 COV step256'}


def add_reference(c,m):
    before=current_digest(m);state=RNG();flags={n:p.requires_grad for n,p in m.model.named_parameters()}
    load_frozen_adapter(m.model,c.references['COV']['checkpoint_path'],'reference')
    m.model.set_adapter('default')
    for n,p in m.model.named_parameters():p.requires_grad_(flags.get(n,False))
    set_RNG(state)
    if current_digest(m)!=before or reference_digest(m)!=c.references['COV']['identity']['adapter_digest']:raise ValueError('Independent reference adapter identity')


@contextmanager
def reference(c,m):
    state=RNG();mode=m.model.training;flags={n:p.requires_grad for n,p in m.model.named_parameters()}
    versions={n:(p._version,id(p.grad),p.grad._version if p.grad is not None else None) for n,p in m.model.named_parameters() if '.default.' in n}
    m.model.set_adapter('reference');m.model.requires_grad_(False);m.set_training(False)
    try:
        with torch.no_grad():yield
    finally:
        m.model.set_adapter('default')
        for n,p in m.model.named_parameters():p.requires_grad_(flags[n])
        m.set_training(mode);set_RNG(state)
        after={n:(p._version,id(p.grad),p.grad._version if p.grad is not None else None) for n,p in m.model.named_parameters() if '.default.' in n}
        if versions!=after or tensor_digest(RNG())!=tensor_digest(state):raise RuntimeError('Reference modified train adapter/gradients/RNG')
        if any(p.grad is not None for n,p in m.model.named_parameters() if '.reference.' in n):raise RuntimeError('Reference gradient')
        c.reference_calls+=1


def encode_patient(c,m,k):
    c.tick();inp,_=m.prompt(c.data.image(k),'L');return inp,m.encode(inp)


def sft(c,m,k,inp,feat,scale):
    result=m.score(inp,feat,answer=c.rows[k]['loc_target'],include_eos=True)
    if result['ids'].tolist()!=c.contracts[k]['L']['target_ids']:raise ValueError('SFT exact target/EOS identity')
    loss=-result['mean']
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SFT loss')
    (loss*scale).backward();return float(loss.detach())


def sampling(c,m,inp,feat,count,settings):
    main=RNG()
    if c.sampling_state is None:
        seed_all(settings.get('rng_seed',c.cfg['RL']['sampling_rng_seed']));c.sampling_state=RNG()
    set_RNG(c.sampling_state);out=[]
    try:
        for _ in range(count):
            c.tick();text,tokens=m.generate(inp,feat,max_tokens=c.cfg['max_new_tokens'],sample=True,sampling_config={k:settings[k] for k in ('temperature','top_p','top_k')}|{'max_new_tokens':c.cfg['max_new_tokens']})
            truncated=len(tokens)>=c.cfg['max_new_tokens'] and tokens[-1]!=m.processor.tokenizer.eos_token_id
            if not tokens:raise ValueError('Empty token completion; cannot define token mean')
            out.append({'loc_text':text,'tokens':tokens,'truncated':truncated,'effective_generation_settings':m.last_generation_settings})
        c.sampling_state=RNG()
    finally:set_RNG(main)
    return out


def rollout(c,m,k,inp,feat):
    cfg=c.cfg['RL'];samples=sampling(c,m,inp,feat,4,cfg);gt=normalized(c.rows[k]['boxes'],c.rows[k]['width'],c.rows[k]['height'])
    with evaluating(m):
        for r in samples:
            r['reward']=reward(r['loc_text'],gt,r['truncated']);r['old']=m.score(inp,feat,ids=r['tokens'])['tokens'].detach()
            if not torch.isfinite(r['old']).all():raise FloatingPointError('Nonfinite old-policy logprobs')
    with reference(c,m):
        for r in samples:r['reference']=m.score(inp,feat,ids=r['tokens'])['tokens'].detach()
    return samples


def rl_backward(c,m,inp,feat,samples,scale,coefficient,beta,check_old=False):
    adv=advantages([r['reward'] for r in samples]);values=[]
    for r,a in zip(samples,adv):
        c.tick();current=m.score(inp,feat,ids=r['tokens'])['tokens']
        if check_old and not torch.allclose(current.detach(),r['old'],atol=.015625,rtol=.015625):raise ValueError('Old-policy token alignment failed')
        objective,kl=policy_terms(current,r['old'],r['reference'],a,torch.ones_like(current,dtype=torch.bool),c.cfg['RL']['epsilon'])
        loss=(-coefficient*objective+beta*kl)*scale/4
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite combined RL/KL loss')
        loss.backward();values.append((float(objective.detach()),float(kl.detach())))
    return {'mean_reward':float(np.mean([r['reward'] for r in samples])),'reward_std':float(np.std([r['reward'] for r in samples])),
            'J_GRPO':float(np.mean([x[0] for x in values])),'KL':float(np.mean([x[1] for x in values])),
            'completion_tokens':sum(len(r['tokens']) for r in samples),'rollouts':4,'zero_advantage':bool(torch.all(adv==0))}


def frozen_diagnostic(c):
    if c.stage.startswith('D1_'):
        name=c.stage[3:];m=c.model(name);records=[]
        with evaluating(m),(c.dest/'predictions.jsonl').open('x') as out:
            for k in c.sets['D1']:
                r=measure(c,m,k,'train_D1',False,False,False);records.append(r);out.write(json.dumps(r)+'\n');out.flush()
        summary={'status':'completed','patients':len(records),'metrics':extended_summary(records),'optimizer_updates':0,'test_pixels_read':0,'runtime_seconds':time.time()-c.started}
    else:
        m=c.model();settings=c.cfg[c.stage];groups=[]
        with evaluating(m),(c.dest/'predictions.jsonl').open('x') as out:
            for k in c.sets[c.stage]:
                inp,feat=encode_patient(c,m,k);gt=normalized(c.rows[k]['boxes'],c.rows[k]['width'],c.rows[k]['height'])
                samples=sampling(c,m,inp,feat,settings['completions'],settings)
                for r in samples:
                    boxes,error=parse_boxes(r['loc_text']);r.update(reward=reward(r['loc_text'],gt,r['truncated']),invalid=bool(error),valid_empty=not error and not r['truncated'] and not boxes,valid_nonempty=not error and not r['truncated'] and bool(boxes))
                group={'image_id':k,'case_id':c.rows[k]['case_id'],'gt':gt,'positive':bool(gt),'outputs':samples,'reward_variance':float(np.var([r['reward'] for r in samples])),'reward_std':float(np.std([r['reward'] for r in samples]))}
                if c.stage=='D2':
                    text,tokens=m.generate(inp,feat,max_tokens=c.cfg['max_new_tokens'],sample=False)
                    group['greedy']={'loc_text':text,'truncated':len(tokens)>=c.cfg['max_new_tokens'] and tokens[-1]!=m.processor.tokenizer.eos_token_id}
                groups.append(group);out.write(json.dumps(group)+'\n');out.flush()
                print(json.dumps({'stage':c.stage,'groups':len(groups),'runtime_seconds':time.time()-c.started}),flush=True)
        positive=[g for g in groups if g['positive']];negative=[g for g in groups if not g['positive']];all_outputs=[r for g in groups for r in g['outputs']]
        summary={'status':'completed','patients':len(groups),'rollouts':len(all_outputs),'sampling_rng_digest':tensor_digest(c.sampling_state),'sampling_settings':settings,
                 'all_empty_group_count':sum(all(r['valid_empty'] for r in g['outputs']) for g in groups),'invalid_output_count':sum(r['invalid'] for r in all_outputs),
                 'truncated_output_count':sum(r['truncated'] for r in all_outputs),'runtime_seconds':time.time()-c.started,'optimizer_updates':0,'test_pixels_read':0}
        if c.stage=='D2':
            quality=[best_single(g['outputs'],g['gt']) for g in positive]
            summary.update(positive_patients=len(positive),negative_patients=len(negative),positive_best_of8_success=sum(q['strict_success'] for q in quality),
                positive_greedy_success=sum(best_single([g['greedy']],g['gt'])['strict_success'] for g in positive),best_single_IoU=distribution([q['best_iou'] for q in quality]),
                positive_all_empty_groups=sum(all(r['valid_empty'] for r in g['outputs']) for g in positive),negative_single_nonempty=sum(r['valid_nonempty'] for g in negative for r in g['outputs']),
                negative_completion_denominator=len(negative)*8,negative_any_nonempty=sum(any(r['valid_nonempty'] for r in g['outputs']) for g in negative),
                negative_patient_denominator=len(negative),GT_best_of8_scope='candidate diagnostic only; not formal inference')
        else:
            summary.update(positive_distinguishable_fraction=sum(g['reward_std']>1e-6 for g in positive)/len(positive),
                strata={name:{'patients':len(gs),'mean_reward_variance':float(np.mean([g['reward_variance'] for g in gs])),
                             'distinguishable_count':sum(g['reward_std']>1e-6 for g in gs),'all_same_reward_count':sum(g['reward_std']<=1e-6 for g in gs),
                             'all_empty_count':sum(all(r['valid_empty'] for r in g['outputs']) for g in gs),'rewards':distribution([r['reward'] for g in gs for r in g['outputs']]),
                             'best_matched_regions_mean':float(np.mean([max(matching(g['gt'],parse_boxes(r['loc_text'])[0] if not r['truncated'] else None)['matches'] for r in g['outputs']) for g in gs])),
                             'invalid_outputs':sum(r['invalid'] for g in gs for r in g['outputs']),'truncated_outputs':sum(r['truncated'] for g in gs for r in g['outputs'])}
                        for name,gs in [('single',[g for g in positive if len(g['gt'])==1]),('multi',[g for g in positive if len(g['gt'])>1]),('negative',negative)]})
    save(c.dest/'summary.json',summary)


def preflight(c):
    eligible=read(c.root/'gate.json')['passed'];m=c.model(trainable=True)
    if eligible:add_reference(c,m)
    seed_all(17);opt=optimizer(m,c);initial=initial_identity(m,opt);rng=RNG();sampling_initial=None
    save(c.dest/'initial_identity.json',initial);timings=[];opt.zero_grad(set_to_none=True)
    for ids in c.plan[:4]:
        start=time.time()
        for k in ids:
            inp,feat=encode_patient(c,m,k);sft(c,m,k,inp,feat,1/16)
        timings.append(time.time()-start)
    gs=gradient(m);norm_s=float(torch.linalg.vector_norm(gs));del gs
    if not np.isfinite(norm_s) or norm_s<=0:raise FloatingPointError('Disconnected/nonfinite SFT gradient')
    opt.zero_grad(set_to_none=True);rl_times=[];norm_rl=None;coefficient=None;rl_status='not_started_gate_failed';old_checked=False
    if eligible:
        before_opt=tensor_digest(opt.state_dict())
        for ids in c.plan[:4]:
            start=time.time()
            for k in ids:
                inp,feat=encode_patient(c,m,k);samples=rollout(c,m,k,inp,feat)
                if sampling_initial is None:
                    main=RNG();seed_all(c.cfg['RL']['sampling_rng_seed']);sampling_initial=RNG();set_RNG(main)
                rl_backward(c,m,inp,feat,samples,1/16,1.,0.,True);old_checked=True
            rl_times.append(time.time()-start)
        gr=gradient(m);norm_rl=float(torch.linalg.vector_norm(gr));del gr
        if not np.isfinite(norm_rl) or norm_rl<1e-8:rl_status='blocked_gradient_calibration'
        else:coefficient=min(1.,.25*norm_s/norm_rl);rl_status='eligible'
        if tensor_digest(opt.state_dict())!=before_opt or reference_digest(m)!=initial['adapter_digest']:raise ValueError('Reference/optimizer state changed during calibration')
    if any(p.grad is not None for p in m.visual.parameters()):raise RuntimeError('Frozen visual/merger gradient')
    opt.zero_grad(set_to_none=True);set_RNG(rng);c.sampling_state=sampling_initial
    assert initial_identity(m,opt)==initial
    save(c.dest/'summary.json',{'status':'completed','RL_status':rl_status,'optimizer_updates':0,'initial_identity':initial,'SFT_gradient_norm':norm_s,'negative_GRPO_gradient_norm':norm_rl,
        'lambda':coefficient,'actual_initial_gradient_ratio':coefficient*norm_rl/norm_s if coefficient is not None else None,'lambda_cap_reached':coefficient==1 if coefficient is not None else None,
        'mean_SFT_update_seconds':float(np.mean(timings)),'mean_GRL_update_seconds':float(np.mean(timings)+np.mean(rl_times)) if eligible else None,
        'old_policy_alignment_checked':old_checked,'reference_calls_verified':c.reference_calls,'model_optimizer_RNG_restored_exact':True,'visual_merger_gradient_absent':True,
        'sampling_rng_initial_digest':tensor_digest(sampling_initial) if sampling_initial is not None else None,'calibration_rollouts':64 if eligible else 0,'runtime_seconds':time.time()-c.started,'test_pixels_read':0})


def train(c):
    plan=read(c.root/'training_plan.json')
    lock=read(c.root/'training_plan_lock.json')
    if sha((c.root/'training_plan.json').read_bytes())!=lock['sha256']:raise ValueError('Formal plan changed')
    for n,h in lock['gate_and_calibration_hashes'].items():
        if sha((c.root/n).read_bytes())!=h:raise ValueError('Gate/calibration evidence changed')
    is_rl=c.stage=='FULL-GRL'
    if is_rl and (not read(c.root/'gate.json')['passed'] or not plan['RL_enabled']):raise PermissionError('Natural gate/calibration/budget did not permit RL')
    m=c.model(trainable=True)
    if is_rl:add_reference(c,m)
    seed_all(17);opt=optimizer(m,c);initial=initial_identity(m,opt);assert initial==read(c.root/'preflight/summary.json')['initial_identity']
    save(c.dest/'initial_identity.json',initial);counts=Counter();completed=0;rows=[];status='running';error=None
    def checkpoint(step,path=None):
        dest=path or c.dest/f'step_{step:04d}';optimizer_snapshot(m,opt,dest,step)
        save(dest/'identity.json',{'adapter_digest':current_digest(m),'optimizer_digest':tensor_digest(opt.state_dict()),'RNG_digest':tensor_digest(RNG()),'schedule_position':step,'lr':opt.param_groups[0]['lr']})
        if c.sampling_state is not None:torch.save(c.sampling_state,dest/'sampling_rng.pt')
    try:
        checkpoint(0)
        with (c.dest/'train.jsonl').open('x') as out:
            for ids in c.plan[:plan['steps']]:
                c.tick();start=time.time();opt.zero_grad(set_to_none=True);losses=[];stats=[];before=(m.forward_count,m.vision_count)
                for k in ids:
                    inp,feat=encode_patient(c,m,k);samples=rollout(c,m,k,inp,feat) if is_rl else None
                    losses.append(sft(c,m,k,inp,feat,.25))
                    if is_rl:stats.append(rl_backward(c,m,inp,feat,samples,.25,plan['lambda'],.01))
                if any(p.grad is not None for n,p in m.model.named_parameters() if '.visual.' in n or '.reference.' in n):raise RuntimeError('Frozen visual/reference gradient')
                norm=torch.nn.utils.clip_grad_norm_(parameters(m),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                opt.step();completed+=1;counts.update(ids)
                record={'step':completed,'patients':ids,'L_NLL':float(np.mean(losses)),'gradient_norm_pre_clip':float(norm),'clipped':bool(norm>1),
                        'update_seconds':time.time()-start,'forward_calls':m.forward_count-before[0],'vision_calls':m.vision_count-before[1],'rollouts':sum(r['rollouts'] for r in stats),
                        'J_GRPO':float(np.mean([r['J_GRPO'] for r in stats])) if stats else None,'KL':float(np.mean([r['KL'] for r in stats])) if stats else None,
                        'mean_reward':float(np.mean([r['mean_reward'] for r in stats])) if stats else None,'zero_advantage_groups':sum(r['zero_advantage'] for r in stats),
                        'completion_tokens':sum(r['completion_tokens'] for r in stats),'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3}
                record['total_objective']=record['L_NLL']-(plan['lambda']*record['J_GRPO'] if is_rl else 0)+(.01*record['KL'] if is_rl else 0)
                if not np.isfinite(record['total_objective']):raise FloatingPointError('Nonfinite total objective')
                rows.append(record);out.write(json.dumps(record)+'\n');out.flush();print(json.dumps({k:v for k,v in record.items() if k!='patients'}),flush=True)
                if completed in (64,128,256):checkpoint(completed)
        assert counts==Counter(k for ids in c.plan[:plan['steps']] for k in ids)
        if is_rl and reference_digest(m)!=initial['adapter_digest']:raise ValueError('Fixed reference drift')
        status='completed'
    except Exception as exc:
        status='stopped_budget' if isinstance(exc,BudgetStop) else 'failed';error=str(exc);opt.zero_grad(set_to_none=True);checkpoint(completed,c.dest/f'stopped_step_{completed:04d}')
    save(c.dest/'summary.json',{'status':status,'reason':error,'steps':completed,'planned_steps':plan['steps'],'SFT_exposures':sum(counts.values()),'patient_exposures':dict(counts),
        'exposure_histograms':{name:dict(Counter(counts[k] for k in pool)) for name,pool in c.selected.items()},'rollouts':sum(r['rollouts'] for r in rows),
        'runtime_seconds':time.time()-c.started,'clip_steps':sum(r['clipped'] for r in rows),'peak_memory_gib':max((r['peak_memory_gib'] for r in rows),default=0),
        'lambda':plan['lambda'] if is_rl else 0,'beta':.01 if is_rl else 0,'reference_calls_verified':c.reference_calls,'test_pixels_read':0})


def evaluate(c):
    name=c.stage[5:].removesuffix('_corrected')
    if name!='COV' and read(c.root/name/'summary.json')['status']!='completed':raise PermissionError('No preregistered final checkpoint')
    m=c.model(name);records=[]
    with evaluating(m),(c.dest/'predictions.jsonl').open('x') as out:
        for k in c.sets['dev']:
            record=measure(c,m,k,'development_P4_shared_cap',True,False,False);records.append(record);out.write(json.dumps(record)+'\n');out.flush()
            print(json.dumps({'stage':c.stage,'patients':len(records),'runtime_seconds':time.time()-c.started}),flush=True)
    save(c.dest/'summary.json',{'status':'completed','patients':len(records),'metrics':extended_summary(records),'max_new_tokens':c.cfg['max_new_tokens'],'greedy':True,'candidate_labels':c.cfg['label_order'],'candidate_token_ids':self_label_ids(m),'test_pixels_read':0,'runtime_seconds':time.time()-c.started})


def self_label_ids(m):return {label:m.processor.tokenizer.encode(label,add_special_tokens=False) for label in ('no','yes')}


def main():
    c=None
    try:
        c=Context()
        if c.stage in ('D1_COV','D1_COV-A','D2','D3'):frozen_diagnostic(c)
        elif c.stage=='preflight':preflight(c)
        elif c.stage in ('FULL-SFT','FULL-GRL'):train(c)
        elif c.stage.startswith('eval_'):evaluate(c)
        else:raise PermissionError('Unknown P4 stage')
    except Exception as exc:
        if c:save(c.dest/'summary.json',{'status':'stopped_budget' if isinstance(exc,BudgetStop) else 'failed','reason':str(exc),'exception_type':type(exc).__name__,'test_pixels_read':0,'runtime_seconds':time.time()-c.started})
        raise


if __name__=='__main__':main()
