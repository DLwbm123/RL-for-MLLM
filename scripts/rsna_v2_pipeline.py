"""Bounded independent workers: <=2 GPUs, each trial single-GPU, total <=6 GPU hours."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from src.v2 import require_safe_config,require_stage,save,STAGES


def main():
    root=Path(os.environ['OUTPUT_ROOT']);cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());require_safe_config(cfg)
    authorization=json.loads(Path(os.environ['V2_GPU_AUTH_FILE']).read_text())
    gpus=os.environ.get('V2_GPU_INDICES',os.environ.get('CUDA_VISIBLE_DEVICES','')).split(',')
    if not 1<=len(gpus)<=cfg['max_concurrent_gpus'] or len(set(gpus))!=len(gpus) or any(not g.isdigit() for g in gpus):raise PermissionError('Invalid authorized GPU list')
    if authorization.get('authorized') is not True or authorization.get('max_gpu_hours')!=6 or authorization.get('max_concurrent_gpus',1)<len(gpus):
        raise PermissionError('Explicit v2 GPU authorization missing')
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'launch_receipt.json').exists():raise FileExistsError('Already launched; no automatic retries')
    statuses=json.loads((root/'stage_status.json').read_text());budget=json.loads((root/'budget.json').read_text())
    if budget['limit_seconds']!=cfg['gpu_hours_limit']*3600:raise ValueError('Budget mismatch')
    save(root/'launch_receipt.json',{'pid':os.getpid(),'started':time.time(),'authorization':authorization,'gpus':gpus,'status':'running'})
    logs=root/'logs';logs.mkdir(exist_ok=True)
    def set_status(stage,status,reason,**fields):
        statuses[stage].update(status=status,reason=reason,**fields);save(root/'stage_status.json',statuses)
    def execute(jobs,use_gpu=True):
        """Jobs are (logical stage, optional model filter). Independent workers share an upper-bound GPU reservation."""
        if not jobs:return []
        remaining=budget['limit_seconds']-budget['consumed_seconds']
        if use_gpu and remaining<=40:
            for stage,_ in jobs:set_status(stage,'stopped_budget','Cumulative GPU budget exhausted')
            return [False]*len(jobs)
        available=[]
        if use_gpu:
            for gpu in gpus:
                free=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=30000:available.append(gpu)
            if len(available)<len(jobs):
                if not available:
                    for stage,_ in jobs:set_status(stage,'blocked','Insufficient GPU memory margin')
                    return [False]*len(jobs)
                # Preserve independent experiments; serialize when fewer GPUs remain available.
                answer=[]
                for job in jobs:answer.extend(execute([job],True))
                return answer
        active=[]
        allowance=(remaining/len(jobs)-5) if use_gpu else 3600
        for i,(stage,model_filter) in enumerate(jobs):
            require_stage(stage,statuses,training=stage in ['P1','SFT-N','SFT-B'])
            if stage in ['P1','SFT-N','SFT-B'] and any(a['stage']==stage for a in budget['attempts']):raise PermissionError('No training retry authorized')
            started=time.time();deadline=started+max(1,allowance);tag=stage+('_'+model_filter if model_filter else '')
            attempt={'stage':stage,'model_filter':model_filter,'started':started,'finished':None,'wall_seconds':0.,'gpu_seconds':0.,'gpu':available[i] if use_gpu else None,'exit_code':None,'deadline':deadline}
            budget['attempts'].append(attempt);save(root/'budget.json',budget)
            set_status(stage,'running','Independent bounded worker(s) executing')
            env=os.environ.copy();env.update(V2_STAGE=stage,V2_MODEL_FILTER=model_filter or '',V2_DEADLINE=str(deadline),CUDA_VISIBLE_DEVICES=available[i] if use_gpu else '')
            script="from src.v2_run import main,BudgetStop\ntry: main()\nexcept BudgetStop: raise SystemExit(124)"
            log=(logs/(tag+'.log')).open('x')
            worker=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            worker.stdin.write(script.encode());worker.stdin.close();attempt['pid']=worker.pid
            active.append((stage,model_filter,worker,attempt,log));save(root/'budget.json',budget)
        outcomes=[]
        for stage,model_filter,worker,attempt,log in active:
            try:code=worker.wait(timeout=max(.1,attempt['deadline']-time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGTERM)
                try:worker.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
                code=124
            finally:log.close()
            elapsed=time.time()-attempt['started']
            # A completed child reaped after another child is conservatively charged until reaping.
            gpu_seconds=elapsed if use_gpu else 0.
            attempt.update(finished=time.time(),wall_seconds=elapsed,gpu_seconds=gpu_seconds,exit_code=code)
            budget['consumed_seconds']+=gpu_seconds;save(root/'budget.json',budget)
            suffix='_'+model_filter if model_filter and model_filter!='aggregate' else ''
            path=root/stage/('summary'+suffix+'.json');result=json.loads(path.read_text()) if path.exists() else {}
            state=result.get('status','failed') if code==0 else 'stopped_budget' if code==124 or result.get('status')=='stopped_budget' else 'failed'
            reason=result.get('reason','worker completed; scientific criteria reported separately' if code==0 else 'worker failed; no automatic retry')
            if model_filter and model_filter!='aggregate':
                statuses[stage].setdefault('workers',{})[model_filter]={'status':state,'reason':reason,'wall_seconds':elapsed,'gpu_seconds':gpu_seconds,'exit_code':code}
                if state!='completed':set_status(stage,state,reason)
                else:save(root/'stage_status.json',statuses)
            else:
                set_status(stage,state,reason,actual_samples=result.get('actual_samples',0),steps=result.get('steps',0),runtime_seconds=elapsed,scientific_passed=result.get('scientific_passed'),exit_code=code)
            outcomes.append(state=='completed')
        return outcomes
    def model_stage(stage,names):
        for i in range(0,len(names),len(gpus)):
            outcomes=execute([(stage,n) for n in names[i:i+len(gpus)]])
            if not all(outcomes):return
        execute([(stage,'aggregate')],use_gpu=False)
    def candidates():
        p=root/'working_candidates.json';return json.loads(p.read_text())['models'] if p.exists() else []
    try:
        model_stage('P0',['B0','B1'])
        if statuses['P0']['status']=='completed':
            execute([('P1',None)])
            if statuses['P1']['status']=='completed' and statuses['P1'].get('scientific_passed') is True:
                jobs=[('SFT-N',None),('SFT-B',None)]
                for i in range(0,2,len(gpus)):execute(jobs[i:i+len(gpus)])
                if all(statuses[s]['status']=='completed' for s in ['SFT-N','SFT-B']):execute([('P2-summary',None)],use_gpu=False)
                else:set_status('P2-summary','blocked','Both P2 experiments must complete before candidate selection')
            else:
                for stage in ['SFT-N','SFT-B','P2-summary']:set_status(stage,'blocked','P1 did not pass; P2 cannot start')
            model_stage('P3',['B0','B1']+candidates());model_stage('P4',['B1']+candidates())
        else:
            for stage in ['P1','SFT-N','SFT-B','P2-summary','P3','P4']:set_status(stage,'blocked','P0 score/identity/determinism not complete')
        set_status('R0','blocked',cfg['R0']['reason'])
    finally:
        receipt=json.loads((root/'launch_receipt.json').read_text());receipt.update(status='exited',finished=time.time());save(root/'launch_receipt.json',receipt)
        from scripts.report_rsna_v2 import render
        render(root)

if __name__=='__main__':main()
