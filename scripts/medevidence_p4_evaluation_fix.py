"""Uniform evaluator correction permitted by P4; preserve original attempts and ledger."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from src.v2 import save
from scripts.medevidence_p4 import report,read


def main():
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';ledger=read(root/'gpu_ledger.json');status=read(root/'stage_status.json')
    if ledger['active'] or (root/'evaluation_fix_launch.json').exists():raise FileExistsError('Wait for original controller; no correction repeat')
    if not (root/'evaluation_error.json').exists():raise PermissionError('No evidenced evaluator error')
    if read(root/'FULL-SFT/summary.json')['status']!='completed':raise PermissionError('Training not complete')
    if any(r['stage'].endswith('_corrected') for r in ledger['records']):raise FileExistsError('No second correction attempt')
    save(root/'gpu_ledger_before_evaluation_fix.json',ledger);save(root/'stage_status_before_evaluation_fix.json',status)
    active={};limit=10800.;save(root/'evaluation_fix_launch.json',{'pid':os.getpid(),'started':time.time(),'original_cumulative_seconds':ledger['charged_seconds']})
    for name in ('COV','FULL-SFT'):status['eval_'+name]={'status':'invalidated_classification','reason':'wrong default candidates; original receipts preserved'}
    def persist(state='running_evaluation_fix'):
        total=sum(r['charged_seconds'] for r in ledger['records'])+sum(time.time()-j['receipt']['started'] for j in active.values())
        ledger.update(status=state,charged_seconds=total,gpu_hours=total/3600,active=[j['receipt'] for j in active.values()],evaluation_fix_controller_pid=os.getpid())
        save(root/'gpu_ledger.json',ledger);save(root/'stage_status.json',status)
    for name,gpu in [('COV',0),('FULL-SFT',1)]:
        stage='eval_'+name+'_corrected'
        if (root/stage).exists():raise FileExistsError('No corrected stage reuse')
        memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        if memory[gpu]<24000:raise RuntimeError('GPU memory unavailable')
        remaining=limit-sum(r['charged_seconds'] for r in ledger['records'])-sum(j['receipt']['quota_seconds'] for j in active.values())-120
        if remaining<540:raise RuntimeError('Original cumulative budget cannot preserve corrected evaluation and exit')
        start=time.time();env=os.environ.copy();env.update(JOB_STAGE=stage,JOB_DEADLINE=str(start+540),CUDA_VISIBLE_DEVICES=str(gpu),DATASET_NAME='rsna',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',TMPDIR=str(base/'tmp'),FORBIDDEN_ARGV_TERMS='wangbomin,medevidence,mllm,qwen,sft,grpo')
        log=(base/'logs'/f'{stage}.log').open('x');p=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=base/'code',start_new_session=True)
        p.stdin.write(("import sys;sys.path.insert(0,"+repr(str(base/'code'))+");from src.medevidence_p4_run import main;main()\n").encode());p.stdin.close()
        receipt={'stage':stage,'gpu':gpu,'pid':p.pid,'started':start,'deadline':start+540,'quota_seconds':540};active[stage]={'process':p,'log':log,'receipt':receipt,'signaled':False};status[stage]={'status':'running'};persist()
    while active:
        for stage,j in list(active.items()):
            p=j['process'];r=j['receipt'];now=time.time()
            if p.poll() is None:
                if now>=r['deadline']-45 and not j['signaled']:p.send_signal(signal.SIGUSR1);j['signaled']=True
                if now>=r['deadline']:
                    p.terminate()
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                else:continue
            j['log'].close();path=root/stage/'summary.json';s=read(path) if path.exists() else {'status':'failed','reason':'no worker summary'}
            state=s['status'] if p.returncode==0 else 'failed'
            ledger['records'].append({**r,'status':state,'exit_code':p.returncode,'charged_seconds':time.time()-r['started']});status[stage]={'status':state,'patients':s.get('patients'),'reason':s.get('reason')};del active[stage]
        persist()
        if active:time.sleep(2)
    complete=all(status[k]['status']=='completed' for k in ('eval_COV_corrected','eval_FULL-SFT_corrected'));persist('completed' if complete else 'failed');report()


if __name__=='__main__':main()
