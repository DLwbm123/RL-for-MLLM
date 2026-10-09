"""Neutral-stdin entry point; RUN_SPEC must name a newly authorized private protocol."""
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from src.model import Model
from src.experiment import seed_all
from src.v2 import DevelopmentData
from src.medevidence import normalized
from src.medevidence_p4_run import add_reference,reference,current_digest
from src.rl_methods import PRIORITY
from src.rl_methods_run import UpdateGroup,native_sampling


def validate_spec(spec,cfg):
    if spec.get('registered') is not True or spec.get('gpu_authorized') is not True or not spec.get('protocol_id'):
        raise PermissionError('A new registered protocol and GPU authorization are required')
    if spec.get('execution_mode') not in ('preflight','train') or spec.get('method') not in PRIORITY:
        raise ValueError('Explicit supported stage/method required')
    for key in ('deadline_unix','budget_seconds','reserve_seconds','learning_rate'):
        value=spec.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive finite '+key+' required')
    if spec['reserve_seconds']>=spec['budget_seconds']:raise ValueError('Reserve exceeds budget')
    if spec['deadline_unix']<=time.time()+spec['reserve_seconds']:raise TimeoutError('Protocol deadline already exhausted')
    if type(spec.get('seed')) is not int:raise ValueError('Explicit seed required')
    whitelist=spec.get('train_ids');schedule=spec.get('schedule')
    if not isinstance(whitelist,list) or not whitelist or len(set(whitelist))!=len(whitelist):raise ValueError('Unique training whitelist required')
    if not isinstance(schedule,list) or not schedule or any(not isinstance(step,list) or not step or not set(step)<=set(whitelist) for step in schedule):
        raise PermissionError('Fixed schedule must stay inside the training whitelist')
    if not spec.get('evaluation_criteria',{}).get('positive_evidence_success') or not spec['evaluation_criteria'].get('negative_false_positive'):
        raise PermissionError('Explicit positive-evidence and negative false-positive criteria required')
    if cfg['execution_mode']!='implementation_only':raise ValueError('Unexpected engineering configuration')
    native_sampling(cfg['sampling']|{'do_sample':True})


def main():
    code=Path(__file__).resolve().parents[1];cfg=json.loads((code/'configs/rl_methods.json').read_text())
    spec=json.loads(Path(os.environ['RUN_SPEC']).read_text());validate_spec(spec,cfg)
    # GPU selection precedes the first CUDA operation; no automatic retry or protocol reuse.
    os.environ['CUDA_VISIBLE_DEVICES']=str(spec['gpu_index'])
    dest=Path(spec['output_root']);dest.mkdir(parents=True,exist_ok=False)
    (dest/'run_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    started=time.time();deadline=min(spec['deadline_unix'],started+spec['budget_seconds'])
    def tick():
        if time.time()>=deadline-spec['reserve_seconds']:raise TimeoutError('Registered budget reserve reached')
    completed=0;status='failed';reason=None;model=None;opt=None
    try:
        tick()
        frame=pd.read_csv(spec['manifest'],keep_default_na=False,dtype={'case_id':str,'image_id':str})
        frame=frame[frame.image_id.isin(spec['train_ids'])].copy()
        if set(frame.image_id)!=set(spec['train_ids']) or set(frame.split)!={'train'}:raise PermissionError('Train-only identities required')
        data=DevelopmentData(spec['data_root'],frame,dest/'data_access.jsonl');data.install_guard()
        rows=frame.set_index('image_id').to_dict('index');seed_all(spec['seed']);tick()
        model=Model(spec['model_root'],spec['initial_adapter'],True)
        ctx=SimpleNamespace(references={'COV':{'checkpoint_path':spec['initial_adapter'],'identity':{'adapter_digest':current_digest(model)}}},reference_calls=0)
        add_reference(ctx,model)
        trainable=[p for p in model.model.parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n or '.visual.' in n for n,p in model.model.named_parameters() if p.requires_grad):
            raise ValueError('Only language LoRA parameters may train')
        opt=torch.optim.AdamW(trainable,lr=spec['learning_rate'],weight_decay=0.)
        plan=spec['schedule'] if spec['execution_mode']=='train' else spec['schedule'][:1]
        with (dest/'updates.jsonl').open('x') as out:
            for ids in plan:
                tick();opt.zero_grad(set_to_none=True);records=[]
                for key in ids:
                    row=rows[key];image=data.image(key);gt=normalized(json.loads(row['boxes']),image.width,image.height)
                    controls=json.loads(row.get('control_boxes','[]'))
                    engine=UpdateGroup(model,spec['method'],cfg,lambda:reference(ctx,model),tick,np.random.default_rng(spec['seed']+completed*len(ids)+len(records)))
                    groups=engine.groups(image,gt,spec['question'],controls,row.get('complete_evidence') in (True,'true','True'))
                    records.append(engine.backward(groups,scale=1/len(ids)))
                if any(p.grad is not None for n,p in model.model.named_parameters() if '.visual.' in n or '.reference.' in n):
                    raise RuntimeError('Frozen visual/reference gradient')
                norm=torch.nn.utils.clip_grad_norm_(trainable,cfg['gradient_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                if not any(p.grad is not None for p in trainable):raise RuntimeError('Disconnected training objective')
                tick()
                if spec['execution_mode']=='train':opt.step();completed+=1
                record={'step':completed,'gradient_norm':float(norm),'groups':records}
                out.write(json.dumps(record)+'\n');out.flush()
                print(json.dumps({'step':completed,'gradient_norm':float(norm),'stage':spec['execution_mode']}),flush=True)
        if spec['execution_mode']=='train':model.save(dest/'final_adapter')
        status='completed'
    except Exception as exc:
        reason=str(exc);status='stopped_budget' if isinstance(exc,TimeoutError) else 'failed'
        if model is not None and completed:model.save(dest/'stopped_adapter')
        raise
    finally:
        if opt is not None:opt.zero_grad(set_to_none=True)
        (dest/'summary.json').write_text(json.dumps({'status':status,'reason':reason,'method':spec['method'],
            'stage':spec['execution_mode'],'optimizer_updates':completed,'runtime_seconds':time.time()-started,
            'test_pixels_read':0,'interpretation':'Project adaptation; evaluation is a separate registered stage.'},indent=2)+'\n')


if __name__=='__main__':main()
