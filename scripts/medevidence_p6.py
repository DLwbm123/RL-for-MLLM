"""P6 preparation and a cumulative-budget controller; neutral stdin worker entry."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scripts.medevidence_p5 import read,save


def prepare():
    from collections import Counter
    import numpy as np
    import pandas as pd
    from transformers import AutoProcessor
    from src.data import smart_size
    from src.medevidence_p5 import target,token_parts
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';p=out/'protocol';p.mkdir();code=base/'code'
    cfg=read(code/'configs/medevidence_p6.json');old=base.parent/'medevidence_p4/outputs';p5=base.parent/'medevidence_p5/outputs'
    ledger=read(p5/'gpu_ledger.json');assert not ledger['active']
    assert abs(ledger['prior_GPU_seconds']+ledger['new_charged_seconds']-cfg['prior_GPU_seconds'])<1e-9
    assert read(p5/'fit_normalized/summary.json')['status']=='completed'
    frame=pd.read_csv(old/'protocol/manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str});rows=frame.set_index('image_id').to_dict('index')
    for r in rows.values():r['boxes']=json.loads(r['boxes'])
    oldpools=read(old/'protocol/selected_patients.json');cal=sum(read(p5/'protocol/selection.json')['calibration'].values(),[])
    pools={k:sorted(set(v)-set(cal)) for k,v in oldpools.items()};train=sum(pools.values(),[])
    dev=sorted(k for k,r in rows.items() if r['split']=='validation')
    assert {k:len(v) for k,v in pools.items()}=={'single':55,'multi':62,'negative':224}
    assert len(train)==341 and len(cal)==64 and len(dev)==256
    assert not set(train)&set(cal) and not (set(train)|set(cal))&set(dev)
    selected=train+cal+dev;assert len({rows[k]['case_id'] for k in selected})==661
    rng={k:np.random.default_rng(cfg['seed']) for k in pools};queues={k:[] for k in pools}
    def take(name):
        if not queues[name]:queues[name]=[str(x) for x in rng[name].permutation(pools[name])]
        return queues[name].pop(0)
    plan=[[take(k) for k in ('single','multi','negative','negative')] for _ in range(64)]
    assert len({k for batch in plan[:4] for k in batch})==16
    tokenizer=AutoProcessor.from_pretrained(Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct',local_files_only=True,use_fast=False).tokenizer
    sizes={};contracts={}
    for k in selected:
        r=rows[k];h,w=smart_size(r['height'],r['width']);sizes[k]=[w,h]
        text=target(r['boxes'],r['width'],r['height'],'normalized',(w,h));ids,_=token_parts(tokenizer,text)
        assert len(ids)+(16 if k in train else 0)<=cfg['max_new_tokens'];contracts[k]={'text':text,'ids':ids}
    refs={'INIT':{'checkpoint_path':str(p5/'fit_normalized/step_0256'),'identity':read(p5/'fit_normalized/step_0256/identity.json')},
          'ANSWER':read(old/'protocol/references.json')['COV']}
    frame[frame.image_id.isin(selected)].to_csv(p/'manifest.csv',index=False)
    for name,value in [('sets.json',{'train':train,'calibration':cal,'dev':dev,'pools':pools}),('schedule.json',plan),('sizes.json',sizes),
                       ('contracts.json',contracts),('references.json',refs),('base_identity.json',read(old/'protocol/identity.json')),('folds.json',read(old/'protocol/folds.json'))]:save(p/name,value)
    # Preserve the previous image-identity audit; only a cheap stat check is needed.
    images=read(old/'protocol/selected_images.json')
    for k in selected:
        st=(Path(os.environ['DATA_ROOT'])/rows[k]['image_path']).stat();v=images[k]
        assert (st.st_size,st.st_mtime_ns)==(v['bytes'],v['mtime_ns'])
    import runpy,torch
    runpy.run_path(str(code/'tests/test_medevidence_p6.py'))['main']()
    assert not torch.cuda.is_initialized()
    counts=Counter(k for ids in plan for k in ids)
    receipt={'status':'passed','train_pool':{k:len(v) for k,v in pools.items()},'calibration_patients':64,'development_patients':256,
             'new_update_calibration_overlap':0,'planned_steps':64,'SFT_exposures_per_branch':256,'RL_rollouts_per_branch':1024,
             'distinct_scheduled_train_patients':len(counts),'exposure_histogram':dict(Counter(counts.values())),
             'max_target_length':max(len(v['ids']) for v in contracts.values()),'prior_GPU_seconds':cfg['prior_GPU_seconds'],
             'source_commit':read(out/'authorization.json')['source_commit'],'test_pixels_read':0,'unit_tests':'passed'}
    save(p/'CPU_checks.json',receipt);save(code/'reports/medevidence_p6_preparation.json',receipt);print(json.dumps(receipt),flush=True)


def controller():
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';code=base/'code';cfg=read(code/'configs/medevidence_p6.json')
    assert read(out/'protocol/CPU_checks.json')['status']=='passed'
    if (out/'gpu_ledger.json').exists():raise FileExistsError('P6 already attempted; no budget reset')
    jobs={};records=[];status={};prior=cfg['prior_GPU_seconds'];limit=cfg['combined_GPU_limit_seconds']
    def persist(state='running'):
        now=time.time();charged=sum(r['charged_seconds'] for r in records)+sum(now-j['started'] for j in jobs.values())
        save(out/'gpu_ledger.json',{'status':state,'prior_GPU_seconds':prior,'new_charged_seconds':charged,'combined_GPU_hours':(prior+charged)/3600,
             'limit_seconds':limit,'records':records,'active':[{k:v for k,v in j.items() if k not in ('proc','log')} for j in jobs.values()]})
        save(out/'stage_status.json',status)
    def launch(stage,gpu,quota,reserve=0):
        if stage in status:raise FileExistsError('Repeated P6 stage')
        memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        if gpu not in cfg['allowed_gpus'] or gpu in [j['gpu'] for j in jobs.values()]:raise PermissionError('GPU slot violation')
        if memory[gpu]<cfg['minimum_free_memory_mib']:
            status[stage]={'status':'blocked','reason':'insufficient_free_memory'};persist();return
        available=limit-prior-sum(r['charged_seconds'] for r in records)-sum(j['quota_seconds'] for j in jobs.values())-reserve
        if quota>available:
            status[stage]={'status':'stopped_budget','reason':'cannot preserve frozen quota and remaining evaluation'};persist();return
        started=time.time();env=os.environ.copy();env.update(JOB_STAGE=stage,JOB_DEADLINE=str(started+quota),CUDA_VISIBLE_DEVICES=str(gpu),DATASET_NAME='rsna')
        log=(base/'logs'/(stage+'.log')).open('x')
        proc=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
        proc.stdin.write(b'from src.medevidence_p6_run import main;main()\n');proc.stdin.close()
        jobs[stage]={'proc':proc,'log':log,'started':started,'deadline':started+quota,'quota_seconds':quota,'gpu':gpu,'pid':proc.pid,'signaled':False}
        status[stage]={'status':'running'};persist();print(json.dumps({'launched':stage,'gpu':gpu,'pid':proc.pid}),flush=True)
    def collect(evaluate_training=False):
        while jobs:
            for stage,j in list(jobs.items()):
                proc=j['proc'];now=time.time()
                if proc.poll() is None:
                    if now>=j['deadline']-30 and not j['signaled']:proc.send_signal(signal.SIGUSR1);j['signaled']=True
                    if now>=j['deadline']:
                        proc.terminate()
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    else:continue
                j['log'].close();p=out/stage/'summary.json';summary=read(p) if p.exists() else {'status':'failed','reason':'worker exited without summary'}
                state=summary['status']
                if proc.returncode and state=='completed':state='failed'
                records.append({'stage':stage,'status':state,'exit_code':proc.returncode,'charged_seconds':time.time()-j['started']})
                status[stage]={'status':state,'reason':summary.get('reason'),'steps':summary.get('steps')};del jobs[stage]
            if evaluate_training:
                occupied={j['gpu'] for j in jobs.values()}
                for name,gpu in zip(cfg['branches'],cfg['allowed_gpus']):
                    stage='eval_'+name
                    if status.get(name,{}).get('status')=='completed' and stage not in status and gpu not in occupied:
                        pending=sum(1 for n in cfg['branches'] if 'eval_'+n not in status and n!=name)
                        launch(stage,gpu,cfg['evaluation_quota_seconds'],reserve=pending*cfg['evaluation_quota_seconds']+cfg['exit_reserve_seconds'])
                        occupied.add(gpu)
            persist()
            if jobs:time.sleep(2)
    def finish(state):
        persist(state)
        from scripts.report_medevidence_p6 import main
        main()
    persist();launch('preflight',0,cfg['preflight_quota_seconds'],reserve=3*cfg['evaluation_quota_seconds']+cfg['exit_reserve_seconds'])
    launch('eval_INIT',1,cfg['evaluation_quota_seconds'],reserve=3*cfg['evaluation_quota_seconds']+cfg['exit_reserve_seconds']);collect()
    if any(status.get(s,{}).get('status')!='completed' for s in ('preflight','eval_INIT')):
        for name in cfg['branches']:status[name]={'status':'blocked','reason':'native preflight or initialization evaluation incomplete'}
        finish('blocked');return
    pre=read(out/'preflight/summary.json');remaining=limit-prior-sum(r['charged_seconds'] for r in records)
    quotas={name:pre['mean_step_seconds'][name]*64*cfg['throughput_safety_factor']+cfg['training_fixed_overhead_seconds'] for name in cfg['branches']}
    reserved=3*cfg['evaluation_quota_seconds']+cfg['exit_reserve_seconds']
    plan={'status':'frozen_before_formal_updates','steps':64,'lambda':pre['lambda'],'beta':cfg['RL']['beta'],'training_quotas_seconds':quotas,
          'remaining_GPU_seconds':remaining,'reserved_evaluation_and_exit_seconds':reserved,'fits_budget':sum(quotas.values())+reserved<=remaining}
    save(out/'training_plan.json',plan);save(code/'reports/medevidence_p6_training_plan.json',plan)
    if not plan['fits_budget']:
        for name in cfg['branches']:status[name]={'status':'stopped_budget','reason':'all three64step branches plus final evaluation do not fit'}
        finish('stopped_budget');return
    for name,gpu in zip(cfg['branches'],cfg['allowed_gpus']):launch(name,gpu,quotas[name],reserve=reserved)
    collect(evaluate_training=True)
    needed=cfg['branches']+['eval_'+name for name in cfg['branches']]
    finish('completed' if all(status.get(s,{}).get('status')=='completed' for s in needed) else 'incomplete')


if __name__=='__main__':{'prepare':prepare,'controller':controller}[os.environ['P6_ACTION']]()
