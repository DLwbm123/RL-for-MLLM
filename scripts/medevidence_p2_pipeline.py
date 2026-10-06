"""Three fixed single-GPU diagnoses, one training job, shared two-hour ledger."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from src.v2 import save


def charged_seconds(records, active, now):
    return sum(r['charged_seconds'] for r in records)+sum(now-r['started'] for r in active)


def main():
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';code=base/'code'
    auth=json.loads((root/'authorization.json').read_text())
    if not auth.get('gpu_authorized') or auth['gpu_hours_limit']!=2 or auth['max_concurrent_gpus']!=3:raise PermissionError('Own parallel GPU authorization required')
    if (root/'gpu_ledger.json').exists():raise FileExistsError('No repeated pilot or reset budget')
    records=[];active={};ledger=root/'gpu_ledger.json';limit=7200
    stage_status=json.loads((root/'stage_status.json').read_text())
    def persist(status):
        receipts=[job['receipt'] for job in active.values()]
        used=charged_seconds(records,receipts,time.time())
        save(ledger,{'status':status,'limit_seconds':limit,'charged_seconds':used,'gpu_hours':used/3600,'records':records,
                    'active':receipts,'controller_pid':os.getpid(),'max_concurrent_gpus':3})
        save(root/'stage_status.json',stage_status)
    def launch(stage,gpu,quota):
        if gpu not in auth['allowed_gpus'] or len(active)>=3:raise PermissionError('GPU resource contract')
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        available={int(v.split(',')[0]):int(v.split(',')[1]) for v in free.splitlines()}
        if available[gpu]<24000:
            stage_status[stage]={'status':'blocked_GPU_memory','reason':'Existing processes retained'};persist('running');return
        reserved=sum(v['receipt']['quota_seconds'] for v in active.values())
        quota=min(quota,limit-sum(r['charged_seconds'] for r in records)-reserved-100)
        if quota<180:
            stage_status[stage]={'status':'stopped_budget','reason':'No sufficient frozen/shared allocation'};persist('running');return
        started=time.time();deadline=started+quota;env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),DATASET_NAME='rsna',PILOT_ROOT=str(base),JOB_STAGE=stage,JOB_DEADLINE=str(deadline),
                   DATA_ROOT=os.environ['DATA_ROOT'],MODEL_ROOT=os.environ['MODEL_ROOT'],HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                   FORBIDDEN_ARGV_TERMS='wangbomin,medevidence,mllm,qwen,sft,grpo',TMPDIR=str(base/'tmp'),OMP_NUM_THREADS='4')
        log=(base/'logs'/f'{stage}.log').open('x')
        process=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
        process.stdin.write(("import sys;sys.path.insert(0,"+repr(str(code))+");from src.medevidence_p2_run import main;main()\n").encode());process.stdin.close()
        receipt={'stage':stage,'gpu':gpu,'pid':process.pid,'started':started,'quota_seconds':quota,'deadline':deadline}
        active[stage]={'process':process,'log':log,'receipt':receipt,'signaled':False}
        stage_status[stage]={'status':'running','pid':process.pid};persist('running')
        print(json.dumps({'launched':stage,'pid':process.pid,'gpu':gpu,'quota_seconds':quota}),flush=True)
    def collect():
        while active:
            for stage,job in list(active.items()):
                process=job['process'];receipt=job['receipt'];now=time.time()
                if process.poll() is None:
                    if now>=receipt['deadline']-45 and not job['signaled']:process.send_signal(signal.SIGUSR1);job['signaled']=True
                    if now>=receipt['deadline']:
                        process.terminate()
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                    else:continue
                job['log'].close();path=root/stage/'summary.json'
                summary=json.loads(path.read_text()) if path.exists() else {'status':'failed','reason':'Worker exited before stage summary'}
                if process.returncode and summary.get('status') in ('completed','completed_diagnostic'):summary={'status':'failed','reason':'Exit contradicts summary'}
                records.append({**receipt,'charged_seconds':time.time()-receipt['started'],'exit_code':process.returncode,'status':summary['status']})
                stage_status[stage]={'status':summary['status'],'patients':summary.get('patients'),'steps':summary.get('steps'),'output_relative':stage,'reason':summary.get('reason')}
                del active[stage]
            persist('running')
            if active:time.sleep(2)
    def finish(status):
        stage_status['D1']={'status':'completed' if all(stage_status.get(v,{}).get('status')=='completed' for v in ('D1_M1','D1_W','D1_M2')) else 'partial',
                            'patients_per_completed_checkpoint':64}
        persist(status)
        from scripts.report_medevidence_p2 import main as report
        report()
    persist('running')
    references=json.loads((root/'protocol/reference_checkpoints.json').read_text())
    for stage,gpu in zip(('D1_M1','D1_W','D1_M2'),auth['diagnostic_gpus']):
        info=references.get(stage[3:],{})
        if info.get('status')=='blocked_checkpoint' or not info.get('checkpoint_path'):
            stage_status[stage]={'status':'blocked_checkpoint','reason':'Prespecified checkpoint missing'};continue
        launch(stage,gpu,800)
    collect()
    if stage_status.get('D1_M1',{}).get('status')!='completed':
        stage_status['L1_loc_probe']={'status':'blocked_D1_M1','reason':'No validated frozen M1 diagnosis'};finish('incomplete');return
    throughput=json.loads((root/'D1_M1/predictions_first8_throughput.json').read_text())
    # Freeze allocation before any optimizer update. Slack only extends final evaluation.
    plan={'status':'frozen_before_L1','nominal_quotas_seconds':{'D1_M1':800,'D1_W':800,'D1_M2':800,'L1_loc_probe':4000,'eval_L1':700},
          'global_safety_margin_seconds':100,'first8_D1_M1':throughput,'L1_priority':'256 fixed steps plus Fit/Ref evaluation at0/32/64/128/256',
          'evaluation_slack_policy':'Final development evaluation may use unused completed reservations; no pending diagnosis or training remains; no added stage or update',
          'estimate_L1_seconds':256*4*throughput['mean_teacher_GT_seconds']*3*1.5+4*64*throughput['mean_case_seconds']*1.5+180,
          'no_optimizer_smoke':True,'max_concurrent_gpus':3,'single_gpu_per_job':True,'limit_seconds':limit}
    save(root/'budget_plan.json',plan)
    if plan['estimate_L1_seconds']>4000:
        stage_status['L1_loc_probe']={'status':'stopped_budget_before_training','reason':'Fixed first cases indicate protected complete probe cannot fit its allocation'};finish('stopped_budget');return
    launch('L1_loc_probe',auth['training_gpu'],4000);collect()
    result=stage_status['L1_loc_probe']['status']
    if result=='completed_diagnostic':
        remaining=limit-sum(v['charged_seconds'] for v in records)-100
        if remaining>=180:launch('eval_L1',auth['training_gpu'],remaining);collect()
        else:stage_status['eval_L1']={'status':'skipped_budget','reason':'Preserved fixed probe; optional final development evaluation has no allocation'}
    else:stage_status['eval_L1']={'status':'blocked_incomplete_L1','reason':'No completed fixed step256'}
    finish('completed_diagnostic' if result=='completed_diagnostic' else 'incomplete')


if __name__=='__main__':main()
