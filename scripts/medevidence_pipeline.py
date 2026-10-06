"""Bounded background chain. Each child receives its neutral entry over stdin."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def main():
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';code=base/'code';cfg=json.loads((code/'configs/medevidence_p1.json').read_text())
    ledger=root/'gpu_ledger.json';mode=os.environ.get('PIPELINE_ACTION','formal')
    previous=json.loads(ledger.read_text()) if ledger.exists() else {'limit_seconds':21600,'records':[]}
    if cfg['max_concurrent_gpus']!=3 or cfg['gpu_hours_limit']!=6:raise PermissionError('Authorized budget differs')
    active={};records=previous['records'];gpus=[2,3,4]
    def used():return sum(r['charged_seconds'] for r in records)+sum(time.time()-r['started'] for r in active.values())
    def persist(status):write(ledger,{'limit_seconds':21600,'charged_seconds':used(),'gpu_hours':used()/3600,'records':records,
                                    'active':{k:{n:v for n,v in r.items() if n not in ('process','log')} for k,r in active.items()},'status':status,'controller_pid':os.getpid()})
    def launch(stage,gpu,quota):
        if stage in active or any(r['stage']==stage for r in records):raise FileExistsError('No automatic attempt repetition')
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        available={int(x.split(',')[0]):int(x.split(',')[1]) for x in free.splitlines()}
        if available[gpu]<56000:raise RuntimeError('Insufficient current GPU memory; existing processes retained')
        started=time.time();deadline=started+quota;env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),DATASET_NAME='rsna',OUTPUT_ROOT=str(root),RUN_CONFIG=str(code/'configs/medevidence_p1.json'),
                   DATA_ROOT=str(Path(os.environ['DATA_ROOT'])/'RSNA_Pneumonia'),MODEL_ROOT=os.environ['MODEL_ROOT'],
                   FORBIDDEN_ARGV_TERMS='wangbomin,medevidence,mllm,qwen,grpo,sft',JOB_STAGE=stage,JOB_DEADLINE=str(deadline),
                   TMPDIR=str(base/'tmp'),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        log=(base/'logs'/f'{stage}.log').open('w');p=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
        entry="import sys;sys.path.insert(0,"+repr(str(code))+ ");from src.medevidence_run import main;main()\n"
        p.stdin.write(entry.encode());p.stdin.close();active[stage]={'process':p,'log':log,'stage':stage,'gpu':gpu,'pid':p.pid,'started':started,'quota':quota,'deadline':deadline}
        persist('running');print(json.dumps({'launched':stage,'pid':p.pid,'gpu':gpu,'quota_seconds':quota}),flush=True)
    def collect():
        for stage,r in list(active.items()):
            p=r['process']
            if p.poll() is None and time.time()>=r['deadline']-45:
                p.send_signal(signal.SIGUSR1)
            if p.poll() is None and time.time()>=r['deadline']:
                p.terminate();p.wait(timeout=10)
            if p.poll() is None:continue
            r['log'].close();summary=root/stage/'summary.json'
            result=json.loads(summary.read_text()) if summary.exists() else {'status':'failed','reason':'Worker exited before summary','exit_code':p.returncode}
            if p.returncode and result.get('status')=='completed':result={'status':'failed','reason':'Exit code contradicts summary'}
            records.append({'stage':stage,'gpu':r['gpu'],'pid':p.pid,'started':r['started'],'charged_seconds':time.time()-r['started'],'exit_code':p.returncode,'status':result['status']})
            del active[stage];persist('running')
    def wait_all():
        while active:collect();time.sleep(3)
    if mode=='smoke':
        launch('smoke',2,min(1200,21600-used()));wait_all();persist('smoke_finished');return
    smoke=root/'smoke/summary.json'
    if not smoke.exists() or json.loads(smoke.read_text())['status']!='completed':raise RuntimeError('Completed engineering smoke required')
    rows=[json.loads(line) for line in (root/'smoke/train.jsonl').read_text().splitlines()]
    # The first positive smoke has extra gradient checks; exclude only these documented diagnostic overheads.
    rate_rows=rows[1:];mean=lambda field:sum(r[field] for r in rate_rows)/len(rate_rows)
    full,aux=mean('full_seconds'),mean('aux_seconds')
    pair=sum(r['pair_seconds']-r.get('dep_gradient_audit_seconds',0.) for r in rate_rows)/len(rate_rows)
    eval_total=cfg['budget_reserve_evaluation_seconds'];remaining=21600-used()-120
    estimates={'M0':512*full*1.3+180,'W':256*(full+aux)*1.3+180,
               'M1':256*(full+aux+pair)*1.3+180,'M2':256*(full+aux+pair)*1.3+180}
    positive=any(v['positive'] and v['split']=='train' for v in json.loads((root/'protocol/views.json').read_text())['pairs'].values())
    stages=['W','M1','M2'] if positive else ['W','M1']
    core=sum(estimates[k] for k in stages)+eval_total
    include_m0=core+estimates['M0']<=remaining
    if include_m0:stages.append('M0')
    if core>remaining:
        write(root/'budget_plan.json',{'status':'stopped_budget_before_formal','reason':'Smoke throughput cannot fit complete matched branches and reserved evaluation','estimated_seconds':estimates,'remaining_seconds':remaining})
        persist('stopped_budget');return
    spare=max(0,remaining-eval_total-sum(estimates[k] for k in stages));quotas={k:estimates[k]+spare*estimates[k]/sum(estimates[x] for x in stages) for k in stages}
    write(root/'budget_plan.json',{'status':'frozen_after_smoke_before_formal','stage_quotas_seconds':quotas,'evaluation_reserved_seconds':eval_total,'estimates':estimates,
                                  'evaluation_policy':'Each evaluation quota fixed at launch; release only unused completed training reservations to evaluation, max2400 seconds per model; protect other pending evaluation minimum reservations and all pending training quotas',
                                  'M0':'planned' if include_m0 else 'skipped_budget_to_protect_matched_branches','M2':'planned' if positive else 'blocked_no_valid_evidence_pairs',
                                  'shared_remaining_seconds':remaining,'full_seconds':full,'aux_seconds':aux,'pair_seconds':pair})
    launch('W',3,quotas['W'])
    if include_m0:launch('M0',2,quotas['M0'])
    started_branches=False;eval_launched=set();failed=False
    while active:
        collect();statuses={r['stage']:r['status'] for r in records}
        if statuses.get('W')=='completed' and not started_branches:
            launch('M1',3,quotas['M1'])
            if positive:launch('M2',4,quotas['M2'])
            started_branches=True
        elif statuses.get('W') in ('failed','stopped_budget'):
            failed=True
        available=[g for g in gpus if all(r['gpu']!=g for r in active.values())]
        eval_models=[m for m in ('M1','M2','M0') if m in stages]
        for model in eval_models:
            if available and statuses.get(model)=='completed' and model not in eval_launched:
                active_reserved=sum(max(0,r['deadline']-time.time()) for r in active.values())
                unlaunched_training=sum(quotas[k] for k in stages if k not in active and k not in statuses)
                pending_other=sum(m not in eval_launched and m!=model for m in eval_models)*eval_total/len(eval_models)
                free=21600-used()-active_reserved-unlaunched_training-pending_other-120
                evaluation_quota=min(2400,free)
                if evaluation_quota>=eval_total/len(eval_models):
                    launch('eval_'+model,available.pop(0),evaluation_quota);eval_launched.add(model)
        persist('running');time.sleep(3)
    statuses={r['stage']:r['status'] for r in records}
    final='completed' if all(statuses.get(m)=='completed' and statuses.get('eval_'+m)=='completed' for m in ('M1','M2')) and (not include_m0 or statuses.get('eval_M0')=='completed') else 'incomplete'
    persist(final)
    if all(statuses.get('eval_'+m)=='completed' for m in ('M1','M2')):
        from scripts.report_medevidence_p1 import main as report
        report()


if __name__=='__main__':main()
