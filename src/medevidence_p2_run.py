"""Bounded single-GPU frozen diagnostics and one localization-only probe."""
from contextlib import contextmanager
from pathlib import Path
from copy import deepcopy
from collections import Counter
import json
import os
import random
import signal
import time
import numpy as np
import pandas as pd
import torch
from src.data import QUESTIONS, sha
from src.model import Model
from src.experiment import seed_all, optimizer_snapshot
from src.medevidence import parse_boxes
from src.medevidence_p2 import enrich, summarize, token_loss_parts
from src.medevidence_run import adapter_digest, tensor_digest, restore
from src.v2 import DevelopmentData, save


class BudgetStop(Exception):pass


def RNG():
    return [random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()]


def set_RNG(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])


@contextmanager
def evaluating(model):
    state=RNG();before=model.model.training;model.set_training(False)
    try:
        with torch.no_grad():yield
    finally:
        model.set_training(before);set_RNG(state)
        if tensor_digest(RNG())!=tensor_digest(state):raise RuntimeError('Evaluation changed RNG')


class Context:
    def __init__(self):
        self.started=time.time();self.stop=False;signal.signal(signal.SIGUSR1,lambda *_:setattr(self,'stop',True))
        self.base=Path(os.environ['PILOT_ROOT']);self.root=self.base/'outputs';self.code=self.base/'code';self.p=self.root/'protocol'
        self.stage=os.environ['JOB_STAGE'];self.deadline=float(os.environ['JOB_DEADLINE'])
        auth=json.loads((self.root/'authorization.json').read_text())
        if not auth.get('gpu_authorized') or auth['gpu_hours_limit']!=2 or not 1<=auth['max_concurrent_gpus']<=8:raise PermissionError('New pilot GPU authorization missing')
        self.cfgpath=self.code/'configs/medevidence_p2.json';self.cfg=json.loads(self.cfgpath.read_text())
        if self.cfg['max_concurrent_gpus']!=auth['max_concurrent_gpus'] or not self.cfg['single_gpu_per_job']:raise PermissionError('Authorized resource contract differs')
        lock=json.loads((self.p/'lock.json').read_text())
        if sha(self.cfgpath.read_bytes())!=lock['config_sha256']:raise ValueError('Frozen config changed')
        for name,h in lock['protocol_hashes'].items():
            if sha((self.p/name).read_bytes())!=h:raise ValueError('Frozen protocol changed '+name)
        for name,h in lock['source_hashes'].items():
            if sha((self.code/name).read_bytes())!=h:raise ValueError('Frozen source changed '+name)
        if self.cfg['training_task']!='L' or self.cfg['rl_enabled'] or self.cfg['steps']!=256:raise PermissionError('Unexpected training objective')
        self.dest=self.root/self.stage
        if (self.dest/'summary.json').exists() or (self.dest/'predictions.jsonl').exists() or (self.dest/'train.jsonl').exists():raise FileExistsError('No repeated attempt')
        self.dest.mkdir(exist_ok=True)
        self.frame=pd.read_csv(self.p/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
        self.rows=self.frame.set_index('image_id').to_dict('index')
        for r in self.rows.values():r['boxes']=json.loads(r['boxes'])
        self.sets=json.loads((self.p/'sets.json').read_text());self.plan=json.loads((self.p/'schedule.json').read_text())
        self.contracts=json.loads((self.p/'token_contracts.json').read_text());self.images=json.loads((self.p/'selected_images.json').read_text())
        self.references=json.loads((self.p/'reference_checkpoints.json').read_text());self.identity=json.loads((self.p/'reference_identity.json').read_text())
        self.model_path=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
        for name,info in self.identity['base']['files'].items():
            st=(self.model_path/name).stat()
            if (st.st_size,st.st_mtime_ns)!=(info['bytes'],info['mtime_ns']):raise ValueError('Base identity changed')
        for k,info in self.images.items():
            st=(Path(os.environ['DATA_ROOT'])/self.rows[k]['image_path']).stat()
            if (st.st_size,st.st_mtime_ns)!=(info['bytes'],info['mtime_ns']):raise ValueError('Selected image identity changed')
        self.data=DevelopmentData(os.environ['DATA_ROOT'],self.frame,self.dest/'data_access.jsonl');self.data.install_guard()
        QUESTIONS.update(self.cfg['prompts']);self.tick()

    def tick(self):
        if self.stop or time.time()>=self.deadline-90:raise BudgetStop('Frozen stage/shared budget')

    def model(self,reference,trainable=False):
        self.tick();seed_all(self.cfg['seed'])
        path=Path(self.references[reference]['checkpoint_path']) if reference in self.references else self.root/'L1_loc_probe/step_0256'
        expected=self.references[reference]['identity'] if reference in self.references else json.loads((path/'identity.json').read_text())
        m=Model(self.model_path,path,trainable,self.cfg['min_visual_tokens'],self.cfg['max_visual_tokens'])
        if adapter_digest(m)!=expected['adapter_digest']:raise ValueError('Actual adapter identity changed')
        if m.processor.tokenizer.eos_token_id!=self.identity['eos_id']:raise ValueError('Tokenizer EOS changed')
        if trainable:
            params={n:p for n,p in m.model.named_parameters() if p.requires_grad}
            if sum(p.numel() for p in params.values())!=5046272 or any('lora_' not in n or '.visual.' in n for n in params):raise ValueError('Trainable scope changed')
        self.current_model_identity=expected['adapter_digest'];self.tick();return m


def sequence_score(model,inputs,features,target,contract,tolerance):
    start=time.time()
    full=model.score(inputs,features,answer=target,include_eos=True)
    bare=model.score(inputs,features,answer=target,include_eos=False)
    if full['ids'].tolist()!=contract['target_ids']:raise ValueError('Actual target token IDs differ from frozen CPU encoding')
    delta=float((full['tokens'][:-1]-bare['tokens']).abs().max())
    if not torch.allclose(full['tokens'][:-1],bare['tokens'],atol=tolerance['token_logprob_atol'],rtol=tolerance['token_logprob_rtol']):raise ValueError('EOS-independent token alignment mismatch')
    def serialize(result):return {'ids':result['ids'].tolist(),'length':result['length'],'sum_logprob':float(result['sum']),
                                  'mean_logprob':float(result['mean']),'token_logprobs':result['tokens'].tolist(),'EOS_included':result['eos_included']}
    return full,{'including_EOS':serialize(full),'excluding_EOS':serialize(bare),'prefix_logprob_max_error':delta,
                 'token_parts':token_loss_parts(full['tokens'].tolist(),contract),'elapsed_seconds':time.time()-start}


def generate(model,inputs,features,contract,cfg,early):
    support=[];handle=None
    if early:
        def tap(module,args,out):
            if not model.generating or len(support)>=cfg['early_generation_support_positions']:return
            logps=torch.log_softmax(out.logits[0,-1].detach().float(),-1);values,ids=torch.topk(logps,cfg['early_generation_top_k'])
            entry={'position':len(support),'top_ids':ids.tolist(),'top_logprobs':values.tolist(),
                   'EOS_logprob':float(logps[model.processor.tokenizer.eos_token_id]),'support_context':'actual free-generation prefix; raw logits before generation processors'}
            if contract and contract['common_prefix_length'] is not None:
                entry['GT_divergence_token_logprob_in_actual_context']=float(logps[contract['GT_divergence_token']])
                entry['empty_divergence_token_logprob_in_actual_context']=float(logps[contract['empty_divergence_token']])
            support.append(entry)
        handle=model.model.get_base_model().register_forward_hook(tap)
    try:text,tokens=model.generate(inputs,features,max_tokens=cfg['max_new_tokens'],sample=False,diagnostic=early)
    finally:
        if handle:handle.remove()
    for entry in support:
        entry['selected_token_id']=tokens[entry['position']]
        entry['selected_token_text']=model.processor.tokenizer.decode([entry['selected_token_id']],skip_special_tokens=False)
    return text,tokens,support,deepcopy(model.last_generation_settings)


def measure(c,m,k,set_name,classify,teacher,early):
    begin=time.time();c.tick();row=c.rows[k];im=c.data.image(k);loc,grid=m.prompt(im,'L');features=m.encode(loc)
    contract=c.contracts.get(k);text,tokens,support,settings=generate(m,loc,features,contract,c.cfg,early)
    truncated=len(tokens)>=c.cfg['max_new_tokens'] and (not tokens or tokens[-1]!=m.processor.tokenizer.eos_token_id)
    record={'image_id':k,'case_id':row['case_id'],'loc_text':text,'truncated':truncated,'loc_token_ids':tokens,
            'EOS_positions':[i for i,t in enumerate(tokens) if t==m.processor.tokenizer.eos_token_id],'generated_length':len(tokens),
            'early_token_support':support,'effective_generation_settings':settings,'set':set_name,'processor_grid':list(grid),
            'cache_identity':{'adapter_digest':c.current_model_identity,'task':'original_L','image_identity':c.images[k] if k in c.images else {'image_path_private':row['image_path'],'source_manifest_locked':True}}}
    if classify:
        c.tick();inputs,_=m.prompt(im,'A');scores=m.classes(inputs,features)[0].float();record.update(p=float(scores.softmax(-1)[1]),score_no=float(scores[0]),score_yes=float(scores[1]))
    if teacher:
        c.tick();_,target=sequence_score(m,loc,features,row['loc_target'],contract,c.cfg['numeric_tolerance'])
        record['teacher_GT']=target
        if row['boxes']:
            from src.medevidence_p2 import token_contract
            empty_contract=token_contract(m.processor.tokenizer,'[]')
            c.tick();_,empty=sequence_score(m,loc,features,'[]',empty_contract,c.cfg['numeric_tolerance'])
            record['teacher_empty']=empty;j=contract['common_prefix_length'];gt_lp=target['including_EOS']['token_logprobs'][j];empty_lp=empty['including_EOS']['token_logprobs'][j]
            record['divergence']={'position':j,'common_prefix_ids':contract['common_prefix_ids'],'GT_token':contract['GT_divergence_token'],
                'empty_token':contract['empty_divergence_token'],'GT_logprob':gt_lp,'empty_logprob':empty_lp,'GT_minus_empty_logprob':gt_lp-empty_lp,
                'free_generation_entered_shared_prefix':tokens[:j]==contract['common_prefix_ids'],
                'free_next_matches_GT':len(tokens)>j and tokens[j]==contract['GT_divergence_token'],
                'free_next_matches_empty':len(tokens)>j and tokens[j]==contract['empty_divergence_token'],
                'conditional_scope':'teacher logits share exact prompt and actual answer-token prefix; two specific strings, not total nonempty probability'}
    record=enrich(record,row);record['elapsed_seconds']=time.time()-begin;return record


def evaluate_sets(c,m,path,classify=True,teacher=True,early=True):
    all_records=[]
    with evaluating(m),path.open('x') as out:
        for name,ids in c.sets.items():
            for k in ids:
                record=measure(c,m,k,name,classify,teacher,early);all_records.append(record)
                out.write(json.dumps(record)+'\n');out.flush()
                if len(all_records)==c.cfg['d1_throughput_first_cases']:
                    save(path.with_name(path.stem+'_first8_throughput.json'),{'patients':len(all_records),
                        'mean_case_seconds':float(np.mean([r['elapsed_seconds'] for r in all_records])),
                        'mean_teacher_GT_seconds':float(np.mean([r['teacher_GT']['elapsed_seconds'] for r in all_records])) if teacher else None,
                        'fixed_patient_order':True,'optimizer_updates':0})
                print(json.dumps({'stage':c.stage,'checkpoint_eval':path.stem,'patients':len(all_records),'elapsed_seconds':time.time()-c.started}),flush=True)
    return {name:summarize([r for r in all_records if r['set']==name]) for name in c.sets}


def frozen(c):
    stage=c.stage[3:];m=c.model(stage)
    save(c.dest/'initial_identity.json',{'adapter_digest':adapter_digest(m),'reference':stage})
    values=evaluate_sets(c,m,c.dest/'predictions.jsonl')
    save(c.dest/'summary.json',{'status':'completed','patients':64,'sets':values,'optimizer_updates':0,
        'model_identity':c.current_model_identity,'runtime_seconds':time.time()-c.started,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'test_pixels_read':0})


def train(c):
    m=c.model('M1',True);params=[p for p in m.model.parameters() if p.requires_grad];oc=c.cfg['optimizer']
    opt=torch.optim.AdamW(params,lr=oc['lr'],betas=tuple(oc['betas']),eps=oc['eps'],weight_decay=oc['weight_decay'])
    seed_all(c.cfg['seed']);assert not opt.state
    initial=adapter_digest(m);initial_RNG=RNG();initial_opt=tensor_digest(opt.state_dict())
    # One gradient check, no optimizer update, using only the fixed first four Fit patients.
    opt.zero_grad(set_to_none=True);m.set_training(True)
    for k in c.plan[0]:
        c.tick();inp,_=m.prompt(c.data.image(k),'L');features=m.encode(inp)
        scored=m.score(inp,features,answer=c.rows[k]['loc_target'],include_eos=True)
        if scored['ids'].tolist()!=c.contracts[k]['target_ids']:raise ValueError('Teacher token alignment')
        (-scored['mean']/4).backward()
    norm=torch.nn.utils.clip_grad_norm_(params,c.cfg['gradient_clip'])
    if not torch.isfinite(norm) or norm<=0 or any(p.grad is not None for p in m.visual.parameters()):raise RuntimeError('Disconnected/nonfinite gradient or visual update')
    assert adapter_digest(m)==initial and tensor_digest(opt.state_dict())==initial_opt
    opt.zero_grad(set_to_none=True);set_RNG(initial_RNG)
    save(c.dest/'gradient_check.json',{'status':'passed','optimizer_updates':0,'LoRA_gradient_norm':float(norm),'visual_merger_gradient_absent':True,'RNG_restored_exact':True,'adapter_unchanged':True})
    curves={};completed=0;counts=Counter();status='running'
    def checkpoint(step):
        c.tick();dest=c.dest/f'step_{step:04d}';optimizer_snapshot(m,opt,dest,step)
        identity={'adapter_digest':adapter_digest(m),'optimizer_digest':tensor_digest(opt.state_dict()),'RNG_digest':tensor_digest(RNG()),'schedule_position':step,'lr':opt.param_groups[0]['lr']}
        save(dest/'identity.json',identity);c.current_model_identity=identity['adapter_digest']
        if step==0:
            state=RNG();restore(m,opt,dest)
            assert tensor_digest(RNG())==tensor_digest(state) and not opt.state and tensor_digest(opt.state_dict())==initial_opt
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(m.model,load_file(str(dest/'adapter_model.safetensors')))
            assert adapter_digest(m)==initial
            save(c.dest/'initial_identity.json',{**identity,'optimizer_reset':True,'RNG_reset_seed':17,'weight_restore_exact':True,'optimizer_state_empty':True,'initial_reference':'M1 final'})
            source=c.root/'D1_M1/predictions.jsonl'
            cached=[json.loads(v) for v in source.read_text().splitlines()]
            assert len(cached)==64 and all(r['cache_identity']['adapter_digest']==initial for r in cached)
            assert {r['image_id'] for r in cached}==set(c.sets['Fit32']+c.sets['Ref32'])
            (dest/'predictions.jsonl').write_text(source.read_text())
            curve={name:summarize([r for r in cached if r['set']==name]) for name in c.sets}
            save(dest/'cache_receipt.json',{'source':'D1_M1 fixed M1 outputs','checkpoint_identity_equal':True,'tasks':['A','L','teacher_diagnostic'],'patients':64})
        else:
            curve=evaluate_sets(c,m,dest/'predictions.jsonl',classify=step==256,teacher=True,early=step==256)
        curves[str(step)]=curve;save(c.dest/'curves.json',curves)
        if tensor_digest(RNG())!=identity['RNG_digest']:raise RuntimeError('Checkpoint evaluation changed RNG')
    try:
        checkpoint(0)
        with (c.dest/'train.jsonl').open('x') as log:
            for index,ids in enumerate(c.plan):
                c.tick();begin=time.time();opt.zero_grad(set_to_none=True);patients=[]
                for k in ids:
                    c.tick();row=c.rows[k];inp,_=m.prompt(c.data.image(k),'L');features=m.encode(inp)
                    score=m.score(inp,features,answer=row['loc_target'],include_eos=True)
                    if score['ids'].tolist()!=c.contracts[k]['target_ids']:raise ValueError('Actual loss tokens changed')
                    loss=-score['mean']
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite NLL')
                    (loss/4).backward();counts[k]+=1
                    patients.append({'image_id':k,'positive':bool(row['boxes']),'NLL':float(loss.detach()),'token_parts':token_loss_parts(score['tokens'].detach().tolist(),c.contracts[k])})
                if any(p.grad is not None for p in m.visual.parameters()):raise RuntimeError('Visual/merger gradient')
                norm=torch.nn.utils.clip_grad_norm_(params,c.cfg['gradient_clip'])
                if not torch.isfinite(norm) or not any(p.grad is not None for p in params):raise FloatingPointError('Missing/nonfinite LoRA gradient')
                opt.step();completed=index+1
                entry={'step':completed,'task':'L','optimizer_updates':completed,'patient_records':patients,'NLL':float(np.mean([v['NLL'] for v in patients])),
                       'positive_NLL':float(np.mean([v['NLL'] for v in patients if v['positive']])),
                       'negative_NLL':float(np.mean([v['NLL'] for v in patients if not v['positive']])),
                       'gradient_norm_pre_clip':float(norm),'update_seconds':time.time()-begin,'elapsed_seconds':time.time()-c.started}
                log.write(json.dumps(entry)+'\n');log.flush();save(c.dest/'status.json',{'status':'running','step':completed,'planned_steps':256,'pid':os.getpid()})
                print(json.dumps({k:v for k,v in entry.items() if k!='patient_records'}),flush=True)
                if completed in c.cfg['checkpoints']:checkpoint(completed)
        assert counts==Counter({k:32 for k in c.sets['Fit32']});status='completed_diagnostic'
    except BudgetStop:
        opt.zero_grad(set_to_none=True);optimizer_snapshot(m,opt,c.dest/f'stopped_step_{completed:04d}',completed);status='stopped_budget'
    save(c.dest/'summary.json',{'status':status,'steps':completed,'planned_steps':256,'optimizer_reset':True,'task':'L',
        'actual_patient_exposures':dict(counts),'total_exposures':sum(counts.values()),'curves':curves,
        'runtime_seconds':time.time()-c.started,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'test_pixels_read':0})


def development(c):
    summary=json.loads((c.root/'L1_loc_probe/summary.json').read_text())
    if summary['status']!='completed_diagnostic' or summary['steps']!=256:raise PermissionError('Fixed final L1 incomplete')
    m=c.model('L1_final');records=[];status='running'
    try:
        with evaluating(m),(c.dest/'predictions.jsonl').open('x') as out:
            for k,row in sorted(c.rows.items(),key=lambda item:item[1]['case_id']):
                if row['split']!='validation':continue
                record=measure(c,m,k,'development',classify=True,teacher=False,early=False);records.append(record)
                out.write(json.dumps(record)+'\n');out.flush()
                print(json.dumps({'stage':c.stage,'patients':len(records),'elapsed_seconds':time.time()-c.started}),flush=True)
        status='completed'
    except BudgetStop:status='stopped_budget'
    save(c.dest/'summary.json',{'status':status,'patients':len(records),'planned_patients':256,'metrics':summarize(records),
        'runtime_seconds':time.time()-c.started,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'test_pixels_read':0})


def main():
    context=None
    try:
        context=Context()
        if context.stage in ('D1_M1','D1_W','D1_M2'):frozen(context)
        elif context.stage=='L1_loc_probe':train(context)
        elif context.stage=='eval_L1':development(context)
        else:raise PermissionError('Unknown/unauthorized stage')
    except Exception as error:
        if context is not None:
            save(context.dest/'summary.json',{'status':'stopped_budget' if isinstance(error,BudgetStop) else 'failed','exception_type':type(error).__name__,'reason':str(error),'runtime_seconds':time.time()-context.started})
        raise


if __name__=='__main__':main()
