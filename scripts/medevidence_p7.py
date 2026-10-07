"""CPU preparation and bounded, detached GPU supervision for P7."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
from src.v2 import save


def read(p):return json.loads(Path(p).read_text())


def prepare():
    import pandas as pd
    import torch
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';p=out/'protocol';p.mkdir()
    cfg=read(base/'code/configs/medevidence_p7.json');old=base.parent/'medevidence_p4/outputs/protocol';prior=read(base.parent/'medevidence_p6/outputs/gpu_ledger.json')
    assert not prior['active'] and abs(prior['prior_GPU_seconds']+prior['new_charged_seconds']-cfg['prior_GPU_seconds'])<1e-8
    frame=pd.read_csv(old/'manifest.csv',keep_default_na=False,dtype={'image_id':str,'case_id':str})
    assert set(frame.split)=={'train','validation'} and not frame.case_id.duplicated().any()
    cal=sum(read(base.parent/'medevidence_p5/outputs/protocol/selection.json')['calibration'].values(),[])
    train=sorted(set(frame.loc[frame.split=='train','image_id'])-set(cal));dev=sorted(frame.loc[frame.split=='validation','image_id'])
    assert len(train)==960 and len(cal)==64 and len(dev)==256 and not (set(train)&set(cal) or set(train)&set(dev) or set(cal)&set(dev))
    sizes={}
    for name,ids in [('train',train),('calibration',cal),('dev',dev)]:
        rows=frame[frame.image_id.isin(ids)];sizes[name]={'patients':len(rows),'positive':int((rows.pathology=='yes').sum()),'negative':int((rows.pathology=='no').sum())}
    assert sizes['train']['positive']==117 and sizes['calibration']['positive']==32
    for row in frame.to_dict('records'):
        path=(Path(os.environ['DATA_ROOT'])/row['image_path']).resolve();assert path.is_relative_to(Path(os.environ['DATA_ROOT']).resolve()) and path.is_file()
        boxes=json.loads(row['boxes']);assert bool(boxes)==(row['pathology']=='yes')
        for x1,y1,x2,y2 in boxes:assert 0<=x1<x2<=row['width'] and 0<=y1<y2<=row['height']
    frame.to_csv(p/'manifest.csv',index=False);save(p/'sets.json',{'train':train,'calibration':cal,'dev':dev})
    # Reuse the completed CPU test receipt instead of repeating the same native model load.
    assert read(out/'native_cpu_checks.json')['status']=='passed' and not torch.cuda.is_initialized()
    receipt={'status':'passed','patients':sizes,'train_cal_dev_overlap':0,'test_pixels_read':0,'detector_weight_load':'passed_CPU',
        'synthetic_actor_tests':'passed','prior_GPU_seconds':cfg['prior_GPU_seconds'],'source_commit':read(out/'authorization.json')['source_commit']}
    save(p/'CPU_checks.json',receipt);save(base/'code/reports/medevidence_p7_preparation.json',receipt);print(json.dumps(receipt),flush=True)


def controller():
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';cfg=read(base/'code/configs/medevidence_p7.json')
    assert read(out/'protocol/CPU_checks.json')['status']=='passed'
    if (out/'gpu_ledger.json').exists():raise FileExistsError('P7 already attempted; no automatic retry')
    memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
    gpu=next((g for g in cfg['allowed_gpus'] if memory[g]>=cfg['minimum_free_memory_mib']),None)
    if gpu is None:raise RuntimeError('Insufficient authorized GPU memory; no launch')
    quota=cfg['combined_GPU_limit_seconds']-cfg['prior_GPU_seconds']-cfg['exit_reserve_seconds'];started=time.time();deadline=started+quota
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JOB_DEADLINE=str(deadline),DATASET_NAME='rsna')
    with (base/'logs/worker.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,cwd=base/'code',env=env,start_new_session=True)
        proc.stdin.write(b'from src.medevidence_p7 import worker;worker()\n');proc.stdin.close();signaled=False
        def ledger(status):
            charged=time.time()-started
            save(out/'gpu_ledger.json',{'status':status,'prior_GPU_seconds':cfg['prior_GPU_seconds'],'new_charged_seconds':charged,
                'combined_GPU_hours':(cfg['prior_GPU_seconds']+charged)/3600,'limit_seconds':cfg['combined_GPU_limit_seconds'],
                'active':[{'pid':proc.pid,'gpu':gpu,'deadline':deadline}] if status=='running' else [],'exit_code':proc.returncode})
        ledger('running');print(json.dumps({'worker_pid':proc.pid,'GPU':gpu,'quota_seconds':quota}),flush=True)
        while proc.poll() is None:
            if time.time()>deadline-60 and not signaled:proc.send_signal(signal.SIGUSR1);signaled=True
            if time.time()>deadline:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            ledger('running');time.sleep(5)
    if not (out/'FINAL.json').exists():
        progress=read(out/'progress.json') if (out/'progress.json').exists() else {}
        save(out/'FINAL.json',{'status':'failed','reason':'worker exited without final receipt','stage':progress.get('stage'),
            'detector_steps':progress.get('steps',0),'test_pixels_read':None,'language_model_updates':0,'scope':cfg['scope']})
    result=read(out/'FINAL.json')
    if proc.returncode and result['status']=='completed':result['status']='failed';result['reason']='nonzero worker exit';save(out/'FINAL.json',result)
    ledger(result['status']);from src.medevidence_p7 import report
    report()


if __name__=='__main__':{'prepare':prepare,'controller':controller}[os.environ['P7_ACTION']]()
