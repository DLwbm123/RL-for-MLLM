"""Frozen matched coverage branches and inference-only crop diagnosis."""
from collections import Counter
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
import torch
from src.data import QUESTIONS,sha
from src.model import Model
from src.experiment import seed_all,optimizer_snapshot
from src.medevidence_run import adapter_digest,tensor_digest,restore
from src.medevidence_p2_run import RNG,set_RNG,evaluating,measure,BudgetStop
from src.medevidence_p2 import summarize
from src.medevidence_p3 import local_metrics
from src.v2 import DevelopmentData,save


class Context:
    def __init__(self):
        self.started=time.time();self.stop=False;signal.signal(signal.SIGUSR1,lambda *_:setattr(self,'stop',True))
        self.base=Path(os.environ['PILOT_ROOT']);self.code=self.base/'code';self.root=self.base/'outputs';self.p=self.root/'protocol'
        self.stage=os.environ['JOB_STAGE'];self.deadline=float(os.environ['JOB_DEADLINE']);self.cfg=json.loads((self.code/'configs/medevidence_p3.json').read_text())
        auth=json.loads((self.root/'authorization.json').read_text())
        if not auth['gpu_authorized'] or auth['gpu_hours_limit']!=3 or auth['max_concurrent_gpus']!=3:raise PermissionError('Own pilot3 resource authorization required')
        if self.stage not in self.cfg['stages'] or self.cfg['rl_enabled'] or self.cfg['answer_coefficients']!={'COV':0.,'COV-A':1.}:raise PermissionError('Frozen branch/task contract')
        lock=json.loads((self.p/'lock.json').read_text())
        for relative,h in lock['code_hashes'].items():
            if sha((self.code/relative).read_bytes())!=h:raise ValueError('Frozen code/config changed '+relative)
        for name,h in lock['protocol_hashes'].items():
            if sha((self.p/name).read_bytes())!=h:raise ValueError('Frozen private protocol changed '+name)
        self.dest=self.root/self.stage
        if self.dest.exists():raise FileExistsError('No repeated pilot3 stage attempt')
        self.dest.mkdir();self.frame=pd.read_csv(self.p/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
        self.rows=self.frame.set_index('image_id').to_dict('index')
        for r in self.rows.values():r['boxes']=json.loads(r['boxes'])
        self.plan=json.loads((self.p/'schedule.json').read_text());self.selected=json.loads((self.p/'selected_patients.json').read_text())
        self.contracts=json.loads((self.p/'token_contracts.json').read_text());self.images=json.loads((self.p/'selected_images.json').read_text())
        self.reference=json.loads((self.p/'reference.json').read_text());self.identity=json.loads((self.p/'identity.json').read_text())
        self.pairs=json.loads((self.p/'local_pairs.json').read_text())['pairs'];self.model_path=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
        for name,info in self.identity['base']['files'].items():
            st=(self.model_path/name).stat()
            if (st.st_size,st.st_mtime_ns)!=(info['bytes'],info['mtime_ns']):raise ValueError('Base model identity changed')
        for k,info in self.images.items():
            st=(Path(os.environ['DATA_ROOT'])/self.rows[k]['image_path']).stat()
            if (st.st_size,st.st_mtime_ns)!=(info['bytes'],info['mtime_ns']):raise ValueError('Image identity changed')
        self.data=DevelopmentData(os.environ['DATA_ROOT'],self.frame,self.dest/'data_access.jsonl');self.data.install_guard()
        QUESTIONS.update(self.cfg['prompts']);self.tick()

    def tick(self):
        if self.stop or time.time()>=self.deadline-90:raise BudgetStop('Frozen stage/shared budget')

    def model(self,reference='M1',trainable=False):
        self.tick();seed_all(17)
        path=Path(self.reference['checkpoint_path']) if reference=='M1' else self.root/reference/'step_0256'
        expected=self.reference['identity'] if reference=='M1' else json.loads((path/'identity.json').read_text())
        m=Model(self.model_path,path,trainable,self.cfg['min_visual_tokens'],self.cfg['max_visual_tokens'])
        if adapter_digest(m)!=expected['adapter_digest']:raise ValueError('Actual adapter identity mismatch')
        if m.processor.tokenizer.eos_token_id!=self.identity['eos_id'] or {x:m.processor.tokenizer.encode(x,add_special_tokens=False) for x in ('no','yes')}!=self.identity['label_ids']:raise ValueError('Tokenizer identity mismatch')
        if trainable:
            params={n:p for n,p in m.model.named_parameters() if p.requires_grad}
            if sum(p.numel() for p in params.values())!=5046272 or any('lora_' not in n or '.visual.' in n for n in params):raise ValueError('Trainable scope changed')
        self.current_model_identity=expected['adapter_digest'];self.tick();return m


def optimizer(m,c):
    oc=c.cfg['optimizer'];return torch.optim.AdamW([p for p in m.model.parameters() if p.requires_grad],lr=oc['lr'],betas=tuple(oc['betas']),eps=oc['eps'],weight_decay=oc['weight_decay'])


def initial_identity(m,opt):
    return {'adapter_digest':adapter_digest(m),'optimizer_digest':tensor_digest(opt.state_dict()),'RNG_digest':tensor_digest(RNG()),
            'schedule_position':0,'lr':opt.param_groups[0]['lr'],'trainable_modules':[n for n,p in m.model.named_parameters() if p.requires_grad],
            'optimizer_reset':True,'optimizer_state_empty':not bool(opt.state),'initialization':'original pilot1 M1 final'}


def patient(c,m,k,alpha,backward=True,calculate_answer=True):
    c.tick();row=c.rows[k];im=c.data.image(k);inp,_=m.prompt(im,'L');features=m.encode(inp);start=time.time()
    score=m.score(inp,features,answer=row['loc_target'],include_eos=True)
    if score['ids'].tolist()!=c.contracts[k]['L']['target_ids']:raise ValueError('L token/EOS alignment')
    loc=-score['mean'];loc_eos=-float(score['tokens'][-1].detach())
    if not torch.isfinite(loc):raise FloatingPointError('Nonfinite L loss')
    if backward:(loc/4).backward()
    loc_elapsed=time.time()-start;answer=None;answer_eos=None;answer_elapsed=0.
    if calculate_answer:
        c.tick();inp,_=m.prompt(im,'A');start=time.time()
        with torch.set_grad_enabled(backward and alpha!=0):
            score=m.score(inp,features,answer=row['pathology'],include_eos=True);answer=-score['mean'];answer_eos=-float(score['tokens'][-1].detach())
            if score['ids'].tolist()!=c.contracts[k]['A']['target_ids']:raise ValueError('A token/EOS alignment')
            if not torch.isfinite(answer):raise FloatingPointError('Nonfinite A loss')
            if backward and alpha:(alpha*answer/4).backward()
        answer_elapsed=time.time()-start
    return {'image_id':k,'positive':bool(row['boxes']),'L_NLL':float(loc.detach()),'A_NLL':float(answer.detach()) if answer is not None else None,
            'L_EOS_NLL':loc_eos,'A_EOS_NLL':answer_eos,'L_seconds':loc_elapsed,'A_seconds':answer_elapsed,'backward_calls':int(backward)*(1+int(bool(alpha) and calculate_answer))}


def gradient(m):return torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for p in m.model.parameters() if p.requires_grad])


def preflight(c):
    m=c.model(trainable=True);opt=optimizer(m,c);seed_all(17);initial=initial_identity(m,opt);state=RNG();ids=c.plan[0]
    save(c.dest/'initial_identity.json',initial);samples=[]
    for calculate in (False,True):
        set_RNG(state);opt.zero_grad(set_to_none=True)
        values=[patient(c,m,k,0,calculate_answer=calculate) for k in ids]
        samples.append(gradient(m))
    relative=float(torch.linalg.vector_norm(samples[1]-samples[0])/torch.linalg.vector_norm(samples[0]));maximum=float((samples[1]-samples[0]).abs().max());reference_max=float(samples[0].abs().max())
    tolerance=c.cfg['numeric_tolerance']
    if relative>tolerance['gradient_relative_L2'] or maximum>tolerance['gradient_scaled_max_absolute']*reference_max:raise ValueError('COV A diagnostic altered L gradients outside frozen bf16 tolerance')
    set_RNG(state);opt.zero_grad(set_to_none=True);answer_start=time.time()
    for k in ids:
        c.tick();im=c.data.image(k);inp,_=m.prompt(im,'A');feat=m.encode(inp);score=m.score(inp,feat,answer=c.rows[k]['pathology'],include_eos=True);(-score['mean']/4).backward()
    answer_backward_seconds=(time.time()-answer_start)/len(ids)
    answer_norm=float(torch.linalg.vector_norm(gradient(m)))
    if not np.isfinite(answer_norm) or answer_norm<=0 or any(p.grad is not None for p in m.visual.parameters()):raise RuntimeError('A disconnected/nonfinite or visual gradient')
    opt.zero_grad(set_to_none=True);set_RNG(state)
    assert initial_identity(m,opt)==initial
    optimizer_snapshot(m,opt,c.dest/'restore_check',0);restore(m,opt,c.dest/'restore_check');assert initial_identity(m,opt)==initial
    eval_times=[]
    with evaluating(m):
        for k in ids:
            start=time.time();measure(c,m,k,'preflight',True,False,False);eval_times.append(time.time()-start)
    assert initial_identity(m,opt)==initial
    save(c.dest/'summary.json',{'status':'completed','optimizer_updates':0,'initial_identity':initial,'L_grad_relative_L2_error':relative,'L_grad_max_absolute_error':maximum,
        'L_grad_reference_max':reference_max,'A_only_LoRA_gradient_norm':answer_norm,'visual_merger_gradient_absent':True,'restore_exact':True,'RNG_restored_exact':True,
        'numeric_tolerance':tolerance,'mean_L_forward_backward_seconds':float(np.mean([v['L_seconds'] for v in values])),
        'mean_A_forward_seconds':float(np.mean([v['A_seconds'] for v in values])),'mean_A_forward_backward_seconds':answer_backward_seconds,'mean_A_L_eval_case_seconds':float(np.mean(eval_times)),
        'actual_loss_formula':'sum4 L_NLL/4 + alpha*sum4 A_NLL/4; no divide2, no extra accumulation division','runtime_seconds':time.time()-c.started})


def train(c):
    alpha=c.cfg['answer_coefficients'][c.stage];m=c.model(trainable=True);opt=optimizer(m,c);seed_all(17);params=[p for p in m.model.parameters() if p.requires_grad]
    initial=initial_identity(m,opt);expected=json.loads((c.root/'preflight/summary.json').read_text())['initial_identity'];assert initial==expected
    save(c.dest/'initial_identity.json',initial);counts=Counter();completed=0;status='running'
    def checkpoint(step):
        c.tick();path=c.dest/f'step_{step:04d}';optimizer_snapshot(m,opt,path,step)
        save(path/'identity.json',{'adapter_digest':adapter_digest(m),'optimizer_digest':tensor_digest(opt.state_dict()),'RNG_digest':tensor_digest(RNG()),'schedule_position':step,'lr':opt.param_groups[0]['lr']})
    try:
        checkpoint(0)
        with (c.dest/'train.jsonl').open('x') as out:
            for i,ids in enumerate(c.plan):
                c.tick();start=time.time();before=(m.forward_count,m.vision_count);opt.zero_grad(set_to_none=True)
                values=[patient(c,m,k,alpha) for k in ids]
                if any(p.grad is not None for p in m.visual.parameters()):raise RuntimeError('Visual/merger gradient')
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm) or not any(p.grad is not None for p in params):raise FloatingPointError('Nonfinite/disconnected gradient')
                opt.step();completed=i+1;counts.update(ids)
                grouped=lambda key,label:float(np.mean([v[key] for v in values if v['positive']==label]))
                entry={'step':completed,'alpha':alpha,'L_NLL':float(np.mean([v['L_NLL'] for v in values])),'A_NLL':float(np.mean([v['A_NLL'] for v in values])),
                       **{task+'_'+label+'_NLL':grouped(task+'_NLL',label=='positive') for task in ('L','A') for label in ('positive','negative')},
                       'L_EOS_NLL':float(np.mean([v['L_EOS_NLL'] for v in values])),'A_EOS_NLL':float(np.mean([v['A_EOS_NLL'] for v in values])),
                       'gradient_norm_pre_clip':float(norm),'gradient_clipped':bool(norm>1),'forward_calls':m.forward_count-before[0],'vision_calls':m.vision_count-before[1],
                       'backward_calls':sum(v['backward_calls'] for v in values),'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'update_seconds':time.time()-start,'patients':values}
                entry['total_objective']=entry['L_NLL']+alpha*entry['A_NLL'];out.write(json.dumps(entry)+'\n');out.flush()
                save(c.dest/'status.json',{'status':'running','step':completed,'planned_steps':256,'elapsed_seconds':time.time()-c.started})
                print(json.dumps({k:v for k,v in entry.items() if k!='patients'}),flush=True)
                if completed in c.cfg['checkpoints']:checkpoint(completed)
        assert counts==Counter(k for ids in c.plan for k in ids);status='completed_diagnostic'
    except BudgetStop:
        opt.zero_grad(set_to_none=True);optimizer_snapshot(m,opt,c.dest/f'stopped_step_{completed:04d}',completed);status='stopped_budget'
    save(c.dest/'summary.json',{'status':status,'steps':completed,'planned_steps':256,'alpha':alpha,'actual_patient_exposures':dict(counts),'total_L_exposures':sum(counts.values()),
                             'runtime_seconds':time.time()-c.started,'optimizer_reset':True,'test_pixels_read':0})


def evaluate(c):
    name=c.stage[5:];summary=json.loads((c.root/name/'summary.json').read_text());assert summary['status']=='completed_diagnostic' and summary['steps']==256
    m=c.model(name);records=[]
    with evaluating(m),(c.dest/'predictions.jsonl').open('x') as out:
        for k,r in sorted(c.rows.items(),key=lambda x:x[1]['case_id']):
            if r['split']!='validation':continue
            record=measure(c,m,k,'development',True,False,False);records.append(record);out.write(json.dumps(record)+'\n');out.flush()
            print(json.dumps({'stage':c.stage,'patients':len(records),'elapsed_seconds':time.time()-c.started}),flush=True)
    save(c.dest/'summary.json',{'status':'completed','patients':len(records),'metrics':summarize(records),'test_pixels_read':0,'runtime_seconds':time.time()-c.started})


def local(c):
    name=c.stage[6:];m=c.model(name);records=[]
    with evaluating(m),(c.dest/'predictions.jsonl').open('x') as out:
        for index,pair in enumerate(c.pairs):
            for view in pair['views']:
                for label,key,box_key in ((1,'positive','positive_box'),(0,'negative','negative_box')):
                    c.tick();k=pair[key];im=c.data.image(k).crop(view[box_key]);inp,grid=m.prompt(im,'C');features=m.encode(inp);scores,details=m.classes(inp,features,labels=['no','yes'])
                    record={'cluster':pair['positive'],'image_id':k,'case_id':c.rows[k]['case_id'],'y':label,'scale':view['scale'],'box_index':view['box_index'],
                            'score_no':float(scores[0]),'score_yes':float(scores[1]),'p':float(scores.float().softmax(-1)[1]),
                            'candidate_lengths':[v['length'] for v in details],'score_definition':'mean token logprob without EOS; candidate softmax support, not calibrated probability',
                            'processor_grid':list(grid),'cache_identity':{'adapter_digest':c.current_model_identity,'image_identity':c.images[k],'task':'C','scale':view['scale'],'box_index':view['box_index']}}
                    records.append(record);out.write(json.dumps(record)+'\n');out.flush()
            print(json.dumps({'stage':c.stage,'pairs':index+1,'elapsed_seconds':time.time()-c.started}),flush=True)
    save(c.dest/'summary.json',{'status':'completed','metrics':local_metrics(records),'optimizer_updates':0,'test_pixels_read':0,'runtime_seconds':time.time()-c.started})


def main():
    context=None
    try:
        context=Context()
        if context.stage=='preflight':preflight(context)
        elif context.stage in ('COV','COV-A'):train(context)
        elif context.stage.startswith('eval_'):evaluate(context)
        elif context.stage.startswith('local_'):local(context)
    except Exception as error:
        if context:
            save(context.dest/'summary.json',{'status':'stopped_budget' if isinstance(error,BudgetStop) else 'failed','exception_type':type(error).__name__,'reason':str(error),'runtime_seconds':time.time()-context.started})
        raise


if __name__=='__main__':main()
