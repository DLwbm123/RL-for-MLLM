"""Shared three-hour ledger; fixed matched branches, at most three single-GPU jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from src.v2 import save
from scripts.medevidence_p2_pipeline import charged_seconds


def main():
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';code=base/'code';auth=json.loads((root/'authorization.json').read_text())
    if not auth['gpu_authorized'] or auth['gpu_hours_limit']!=3 or auth['max_concurrent_gpus']!=3:raise PermissionError('Own pilot3 GPU authorization')
    if (root/'gpu_ledger.json').exists():raise FileExistsError('No duplicate pilot or reset budget')
    active={};records=[];training_started=False;status=json.loads((root/'stage_status.json').read_text());limit=10800
    quotas={'preflight':500,'COV':2600,'COV-A':3700,'eval_COV':800,'eval_COV-A':800,'local_M1':700,'local_COV':700,'local_COV-A':700}
    def persist(state):
        used=charged_seconds(records,[j['receipt'] for j in active.values()],time.time())
        save(root/'gpu_ledger.json',{'status':state,'gpu_hours':used/3600,'charged_seconds':used,'limit_seconds':limit,'records':records,'active':[j['receipt'] for j in active.values()],'controller_pid':os.getpid(),'max_concurrent_gpus':3})
        save(root/'stage_status.json',status)
    def launch(stage,gpu):
        if stage in status and status[stage].get('status') in ('running','completed','completed_diagnostic'):raise PermissionError('Duplicate stage')
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in free.splitlines()}
        if gpu not in auth['allowed_gpus'] or len(active)>=3:raise PermissionError('Resource contract')
        if memory[gpu]<24000:status[stage]={'status':'blocked_GPU_memory'};persist('running');return
        remaining=limit-sum(r['charged_seconds'] for r in records)-sum(j['receipt']['quota_seconds'] for j in active.values())-100
        quota=min(quotas[stage],remaining)
        if quota<180:status[stage]={'status':'stopped_budget'};persist('running');return
        start=time.time();deadline=start+quota;env=os.environ.copy();env.update(PILOT_ROOT=str(base),JOB_STAGE=stage,JOB_DEADLINE=str(deadline),CUDA_VISIBLE_DEVICES=str(gpu),DATASET_NAME='rsna',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',TMPDIR=str(base/'tmp'),FORBIDDEN_ARGV_TERMS='wangbomin,medevidence,mllm,qwen,sft,grpo')
        log=(base/'logs'/f'{stage}.log').open('x');process=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
        process.stdin.write(("import sys;sys.path.insert(0,"+repr(str(code))+");from src.medevidence_p3_run import main;main()\n").encode());process.stdin.close()
        receipt={'stage':stage,'gpu':gpu,'pid':process.pid,'started':start,'deadline':deadline,'quota_seconds':quota}
        active[stage]={'process':process,'log':log,'receipt':receipt,'signaled':False};status[stage]={'status':'running','pid':process.pid};persist('running');print(json.dumps({'launched':stage,'gpu':gpu,'pid':process.pid}),flush=True)
    def collect():
        while active:
            for stage,job in list(active.items()):
                process=job['process'];r=job['receipt'];now=time.time()
                if process.poll() is None:
                    if now>=r['deadline']-45 and not job['signaled']:process.send_signal(signal.SIGUSR1);job['signaled']=True
                    if now>=r['deadline']:
                        process.terminate()
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                    else:continue
                job['log'].close();path=root/stage/'summary.json';summary=json.loads(path.read_text()) if path.exists() else {'status':'failed','reason':'Worker exited before summary'}
                if process.returncode and summary['status'] in ('completed','completed_diagnostic'):summary={'status':'failed','reason':'Exit contradicts summary'}
                records.append({**r,'charged_seconds':time.time()-r['started'],'exit_code':process.returncode,'status':summary['status']})
                status[stage]={'status':summary['status'],'steps':summary.get('steps'),'patients':summary.get('patients'),'reason':summary.get('reason')};del active[stage]
            if training_started:
                occupied={j['receipt']['gpu'] for j in active.values()}
                for stage,parent,gpu in (('eval_COV','COV',0),('eval_COV-A','COV-A',1),('local_COV','COV',2),('local_COV-A','COV-A',2)):
                    if stage not in status and status.get(parent,{}).get('status')=='completed_diagnostic' and gpu not in occupied:
                        launch(stage,gpu);occupied.add(gpu)
            persist('running')
            if active:time.sleep(2)
    def finish(state):
        persist(state)
        from scripts.report_medevidence_p3 import main as report
        report()
    persist('running');launch('preflight',0);collect()
    if status.get('preflight',{}).get('status')!='completed':finish('blocked_engineering');return
    preflight=json.loads((root/'preflight/summary.json').read_text())
    estimates={'COV':1024*(preflight['mean_L_forward_backward_seconds']+preflight['mean_A_forward_seconds'])*1.5+120,
               'COV-A':1024*(preflight['mean_L_forward_backward_seconds']+preflight['mean_A_forward_backward_seconds'])*1.5+120}
    plan={'status':'frozen_before_any_optimizer_update','quotas_seconds':quotas,'training_estimates_seconds':estimates,'limit_seconds':limit,'safety_margin_seconds':300,'max_concurrent_gpus':3,'single_gpu_per_job':True,'no_optimizer_smoke':True,'source_throughput':'fixed first4 scheduled patients, no outcomes used for selection'}
    save(root/'budget_plan.json',plan)
    if any(estimates[k]>quotas[k] for k in estimates):status['training']={'status':'stopped_budget_before_training'};finish('stopped_budget');return
    training_started=True
    launch('COV',0);launch('COV-A',1);launch('local_M1',2);collect()
    for name in ('COV','COV-A'):
        for prefix in ('eval_','local_'):
            status.setdefault(prefix+name,{'status':'blocked_incomplete_training'})
    matched=all(status.get(k,{}).get('status') in ('completed','completed_diagnostic') for k in ('COV','COV-A','eval_COV','eval_COV-A'))
    status['training']={'status':'completed_diagnostic' if matched else 'incomplete','matched_comparison_available':matched}
    all_required=matched and all(status.get('local_'+k,{}).get('status')=='completed' for k in ('M1','COV','COV-A'))
    finish('completed_diagnostic' if all_required else 'incomplete')


if __name__=='__main__':main()
