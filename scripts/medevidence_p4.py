"""One-attempt P4 preparation, budget controller and public aggregate delivery."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import numpy as np
from src.data import sha
from src.v2 import save
from src.medevidence_p4 import schedule, shuffled_control, gate, extended_summary, paired_counts


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]


def prepare():
    import pandas as pd
    from transformers import AutoProcessor
    from safetensors.torch import load_file
    from src.medevidence import normalized,parse_boxes,matching
    from src.medevidence_p2 import token_contract
    from src.medevidence_run import tensor_digest
    from src.v2 import DevelopmentData
    base=Path(os.environ['PILOT_ROOT']);code=base/'code';root=base/'outputs';p=root/'protocol';p.mkdir()
    old=base.parent/'medevidence_p3/outputs';oldlock=read(old/'protocol/lock.json')
    for name,h in oldlock['protocol_hashes'].items():assert sha((old/'protocol'/name).read_bytes())==h,'P3 private protocol changed '+name
    for name,h in oldlock['code_hashes'].items():assert sha((code/name).read_bytes())==h,'Historical source changed '+name
    cfg=read(code/'configs/medevidence_p4.json');frame=pd.read_csv(old/'protocol/manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
    assert len(frame)==1280 and not frame.case_id.duplicated().any() and not frame.image_id.duplicated().any()
    rows=frame.set_index('image_id').to_dict('index')
    for r in rows.values():
        r['boxes']=json.loads(r['boxes']);assert bool(r['boxes'])==(r['pathology']=='yes')
        parsed,error=parse_boxes(r['loc_target']);assert not error and parsed==normalized(r['boxes'],r['width'],r['height']) and matching(parsed,parsed)['strict']
    pools,plan=schedule(rows,read(old/'protocol/selected_patients.json')['negative'])
    train=set(k for pool in pools.values() for k in pool);dev=sorted(k for k,r in rows.items() if r['split']=='validation')
    assert len(train)==405 and len(dev)==256 and not train&set(dev)
    assert {rows[k]['case_id'] for k in train}.isdisjoint({rows[k]['case_id'] for k in dev})
    historic={name:lines(old/f'eval_{name}/predictions.jsonl') for name in ('COV','COV-A')}
    assert all(set(r['image_id'] for r in rs)==set(dev) for rs in historic.values())
    expected={'COV':(1,2,11),'COV-A':(3,3,23)}
    for name,rs in historic.items():
        s=extended_summary(rs);assert (s['single_strict_success_count'],s['matched_regions'],s['negative']['valid_nonempty'])==expected[name]
    refs={}
    for name in ('COV','COV-A'):
        cp=old/name/'step_0256';identity=read(cp/'identity.json');weights=load_file(str(cp/'adapter_model.safetensors'))
        weights={k.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight'):v for k,v in weights.items()}
        assert tensor_digest(weights)==identity['adapter_digest'];del weights
        refs[name]={'checkpoint_path':str(cp),'identity':identity}
    identity=read(old/'protocol/identity.json');model=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    for n,v in identity['base']['files'].items():
        st=(model/n).stat();assert (st.st_size,st.st_mtime_ns)==(v['bytes'],v['mtime_ns'])
    tokenizer=AutoProcessor.from_pretrained(model,local_files_only=True,use_fast=False).tokenizer
    assert tokenizer.eos_token_id==identity['eos_id'] and {x:tokenizer.encode(x,add_special_tokens=False) for x in ('no','yes')}==identity['label_ids']
    contracts={k:{'L':token_contract(tokenizer,rows[k]['loc_target'])} for k in sorted(train|set(dev))}
    lengths=[len(contracts[k]['L']['target_ids']) for k in sorted(train)];cap=max(80,max(lengths)+16)
    cfg.update(max_new_tokens=cap,longest_training_target_including_EOS=max(lengths))
    save(code/'configs/medevidence_p4.json',cfg)
    data=DevelopmentData(os.environ['DATA_ROOT'],frame[frame.image_id.isin(train|set(dev))],root/'CPU_data_access.jsonl');data.install_guard()
    try:(Path(os.environ['DATA_ROOT'])/'sealed_test_probe.png').read_bytes()
    except PermissionError:pass
    else:raise AssertionError('Test guard not installed')
    images={};max_error=0.
    for k in sorted(train|set(dev)):
        r=rows[k];im=data.image(k);assert im.size==(r['width'],r['height']);path=Path(os.environ['DATA_ROOT'])/r['image_path'];st=path.stat()
        images[k]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':sha(path.read_bytes()),'size':list(im.size)}
        for b,nb in zip(sorted(r['boxes']),normalized(r['boxes'],r['width'],r['height'])):
            for j,size in enumerate((r['width'],r['height'],r['width'],r['height'])):
                error=abs(b[j]-nb[j]*size/1000);assert error<=size/2000+1e-9;max_error=max(max_error,error)
    choose=lambda ids,n,seed=17:[str(k) for k in np.random.default_rng(seed).permutation(sorted(ids))[:n]]
    negative_dev=[k for k in dev if not rows[k]['boxes']];single_dev=[k for k in dev if len(rows[k]['boxes'])==1]
    assert len(single_dev)==25 and len(negative_dev)==219
    sets={'D1':pools['single']+choose(pools['negative'],64),'D2':single_dev+choose(negative_dev,32),
          'D3':choose(pools['single'],16)+choose(pools['multi'],16)+choose(pools['negative'],32),'dev':dev}
    for filename,value in [('selected_patients.json',pools),('schedule.json',plan),('diagnostic_sets.json',sets),('token_contracts.json',contracts),('selected_images.json',images),('references.json',refs),('identity.json',identity)]:save(p/filename,value)
    (p/'manifest.csv').write_text((old/'protocol/manifest.csv').read_text());(p/'folds.json').write_text((old/'protocol/folds.json').read_text())
    counts=Counter(k for ids in plan for k in ids);oldsets=read(base.parent/'medevidence_p2/outputs/protocol/sets.json')
    audit={'status':'passed' if cap<=256 else 'blocked_target_length','patients':405,'strata':{k:len(v) for k,v in pools.items()},'same_P3_negatives':True,
           'train_dev_patient_overlap':0,'development_patients':256,'train_GT_regions':sum(len(rows[k]['boxes']) for k in train),'all_GT_retained':True,'crop_dep_filter_used':False,
           'maximum_target_tokens_including_EOS':max(lengths),'max_new_tokens':cap,'historical_cap':80,'target_lengths_histogram':dict(Counter(lengths)),
           'plan256_exposures':1024,'exposure_histograms':{name:dict(Counter(counts[k] for k in pool)) for name,pool in pools.items()},
           'old_Ref_overlap':sum(k in train for k in oldsets['Ref32']),'old_Ref_not_untrained_control':True,'image_identity_checks':len(images),'GT_roundtrip_max_pixels':max_error,
           'test_access_guard_passed':True,'test_pixels_read':0,'base_and_COV_adapter_identity_exact':True,'historical_results_not_relabelled_as_reassessment':True}
    save(p/'CPU_checks.json',audit);save(code/'reports/medevidence_p4_data_audit.json',audit)
    import runpy
    runpy.run_path(str(code/'tests/test_medevidence_p4.py'))['main']();audit['synthetic_unit_tests']='passed';save(p/'CPU_checks.json',audit);save(code/'reports/medevidence_p4_data_audit.json',audit)
    names=list(oldlock['code_hashes'])+['src/medevidence_p4.py','src/medevidence_p4_run.py','scripts/medevidence_p4.py','tests/test_medevidence_p4.py','configs/medevidence_p4.json','reports/medevidence_p4_protocol.md']
    lock={'code_hashes':{n:sha((code/n).read_bytes()) for n in names},'protocol_hashes':{n.name:sha(n.read_bytes()) for n in p.iterdir() if n.is_file()},
          'reference_commit':cfg['reference_commit'],'phase':'before_first_P4_GPU_worker','GPU_hours_limit':3.,'max_new_tokens':cap,'initialization':'P3 COV step256','formal_steps':'pending_train_only_throughput_256_or128'}
    save(p/'lock.json',lock);save(code/'protocol/medevidence_p4/freeze_summary.json',lock)
    save(root/'stage_status.json',{'CPU':{'status':audit['status']}})
    print(json.dumps({'CPU':audit,'freeze':'completed'}))


def controller():
    from scripts.medevidence_p2_pipeline import charged_seconds
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';code=base/'code';status=read(root/'stage_status.json');active={};records=[];limit=10800.
    if (root/'gpu_ledger.json').exists():raise FileExistsError('No second attempt or budget reset')
    auth=read(root/'authorization.json')
    if not auth['gpu_authorized'] or auth['gpu_hours_limit']!=3:raise PermissionError('Current P4 GPU authority required')
    def persist(state='running'):
        used=charged_seconds(records,[j['receipt'] for j in active.values()],time.time())
        save(root/'gpu_ledger.json',{'status':state,'charged_seconds':used,'gpu_hours':used/3600,'limit_seconds':limit,'records':records,'active':[j['receipt'] for j in active.values()],'controller_pid':os.getpid()})
        save(root/'stage_status.json',status)
    def launch(stage,gpu,quota,reserve=1200.):
        if stage in status:raise FileExistsError('No repeated stage')
        if gpu not in auth['allowed_gpus'] or len(active)>=3 or gpu in [j['receipt']['gpu'] for j in active.values()]:raise PermissionError('GPU contract')
        memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        if memory[gpu]<24000:status[stage]={'status':'blocked','reason':'insufficient_free_GPU_memory'};persist();return
        available=limit-sum(r['charged_seconds'] for r in records)-sum(j['receipt']['quota_seconds'] for j in active.values())-reserve
        quota=min(quota,available)
        if quota<180:status[stage]={'status':'stopped_budget','reason':'cannot preserve required evaluation and exit'};persist();return
        started=time.time();deadline=started+quota;env=os.environ.copy();env.update(JOB_STAGE=stage,JOB_DEADLINE=str(deadline),CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',TMPDIR=str(base/'tmp'),FORBIDDEN_ARGV_TERMS='wangbomin,medevidence,mllm,qwen,sft,grpo')
        log=(base/'logs'/f'{stage}.log').open('x');proc=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
        proc.stdin.write(("import sys;sys.path.insert(0,"+repr(str(code))+");from src.medevidence_p4_run import main;main()\n").encode());proc.stdin.close()
        receipt={'stage':stage,'gpu':gpu,'pid':proc.pid,'started':started,'deadline':deadline,'quota_seconds':quota};active[stage]={'process':proc,'log':log,'receipt':receipt,'signaled':False};status[stage]={'status':'running'};persist();print(json.dumps({'launched':stage,'gpu':gpu,'pid':proc.pid}),flush=True)
    def collect(diagnostics=False,training=False):
        while active:
            for stage,j in list(active.items()):
                proc=j['process'];r=j['receipt'];now=time.time()
                if proc.poll() is None:
                    if now>=r['deadline']-45 and not j['signaled']:proc.send_signal(signal.SIGUSR1);j['signaled']=True
                    if now>=r['deadline']:
                        proc.terminate()
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    else:continue
                j['log'].close();p=root/stage/'summary.json';s=read(p) if p.exists() else {'status':'failed','reason':'worker exited before summary'}
                if proc.returncode and s['status']=='completed':s={'status':'failed','reason':'exit contradicts completion summary'}
                records.append({**r,'status':s['status'],'exit_code':proc.returncode,'charged_seconds':time.time()-r['started']});status[stage]={'status':s['status'],'steps':s.get('steps'),'reason':s.get('reason')};del active[stage]
            occupied={j['receipt']['gpu'] for j in active.values()}
            if diagnostics:
                for stage,parent,gpu,quota in [('D3','D1_COV',0,900),('eval_COV','D1_COV-A',1,540)]:
                    if stage not in status and status.get(parent,{}).get('status')=='completed' and gpu not in occupied:launch(stage,gpu,quota);occupied.add(gpu)
            if training:
                for name,gpu in [('FULL-SFT',0),('FULL-GRL',1)]:
                    stage='eval_'+name
                    if stage not in status and status.get(name,{}).get('status')=='completed' and gpu not in occupied:launch(stage,gpu,540,reserve=120);occupied.add(gpu)
            persist()
            if active:time.sleep(2)
    def finish(state):persist(state);report()
    persist()
    if status['CPU']['status']!='passed':finish('blocked');return
    for stage,gpu,quota in [('D1_COV',0,400),('D1_COV-A',1,400),('D2',2,1620)]:launch(stage,gpu,quota)
    collect(diagnostics=True)
    required=('D1_COV','D1_COV-A','D2','D3','eval_COV')
    if any(status.get(k,{}).get('status')!='completed' for k in required):finish('blocked');return
    groups=[g for g in lines(root/'D2/predictions.jsonl') if g['positive']]
    d4=shuffled_control([g['outputs'] for g in groups],[g['gt'] for g in groups]);save(root/'D4.json',d4)
    decision=gate(read(root/'D2/summary.json'),read(root/'D3/summary.json'),d4);save(root/'gate.json',decision);print(json.dumps({'RL_gate':decision}),flush=True)
    if not decision['passed']:status['FULL-GRL']={'status':'not_started_gate_failed','reason':'frozen natural sampling gates failed'}
    launch('preflight',0,800);collect()
    if status.get('preflight',{}).get('status')!='completed':finish('blocked');return
    pre=read(root/'preflight/summary.json');rl_enabled=decision['passed'] and pre['RL_status']=='eligible'
    if decision['passed'] and not rl_enabled:status['FULL-GRL']={'status':'blocked','reason':pre['RL_status']}
    remaining=limit-sum(r['charged_seconds'] for r in records);selected=None
    for steps in (256,128):
        estimates={'FULL-SFT':pre['mean_SFT_update_seconds']*steps*1.4+180}
        if rl_enabled:estimates['FULL-GRL']=pre['mean_GRL_update_seconds']*steps*1.4+180
        if sum(estimates.values())+540*(1+int(rl_enabled))+240<=remaining:selected=(steps,estimates);break
    if selected is None:status['FULL-SFT']={'status':'stopped_budget','reason':'128 steps cannot preserve final evaluation'};finish('stopped_budget');return
    steps,estimates=selected;plan={'status':'frozen_before_any_formal_optimizer_update','steps':steps,'RL_enabled':rl_enabled,'lambda':pre['lambda'] if rl_enabled else None,'beta':.01,
                                 'initialization':'verified P3 COV step256','training_quotas_seconds':estimates,'remaining_seconds':remaining,'reserved_evaluation_seconds':540*(1+int(rl_enabled)),
                                 'safety_exit_seconds':240,'throughput_safety_factor':1.4,'calibration':'first4 fixed training minibatches; zero optimizer updates','diagnostic_actual_seconds':sum(r['charged_seconds'] for r in records)}
    save(root/'training_plan.json',plan);save(root/'training_plan_lock.json',{'sha256':sha((root/'training_plan.json').read_bytes()),'formal_optimizer_updates':0,
         'gate_and_calibration_hashes':{n:sha((root/n).read_bytes()) for n in ('gate.json','preflight/summary.json','preflight/initial_identity.json')}})
    save(code/'reports/medevidence_p4_training_plan.json',plan)
    print(json.dumps({'formal_training_plan':plan}),flush=True)
    launch('FULL-SFT',0,estimates['FULL-SFT'],reserve=plan['reserved_evaluation_seconds']+240)
    if rl_enabled:launch('FULL-GRL',1,estimates['FULL-GRL'],reserve=plan['reserved_evaluation_seconds']+240)
    collect(training=True)
    needed=['FULL-SFT','eval_COV','eval_FULL-SFT']+(['FULL-GRL','eval_FULL-GRL'] if rl_enabled else [])
    complete=all(status.get(k,{}).get('status')=='completed' for k in needed)
    finish('completed' if complete else 'failed' if any(status.get(k,{}).get('status')=='failed' for k in needed) else 'stopped_budget')


def report():
    from scripts.report_medevidence_p3 import full_metrics
    from src.medevidence_p4 import paired_intervals
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';public=base/'code/reports';cfg=read(base/'code/configs/medevidence_p4.json');status=read(root/'stage_status.json');ledger=read(root/'gpu_ledger.json')
    diag={name:read(root/name/'summary.json') for name in ('D1_COV','D1_COV-A','D2','D3','preflight') if (root/name/'summary.json').exists()}
    for name in ('D4','gate'):diag[name]=read(root/(name+'.json')) if (root/(name+'.json')).exists() else {'status':'not_completed'}
    if 'preflight' in diag:diag['preflight'].pop('initial_identity',None)
    save(public/'medevidence_p4_diagnostics.json',diag)
    training={};initial={}
    for name in ('FULL-SFT','FULL-GRL'):
        if (root/name/'summary.json').exists():
            training[name]=read(root/name/'summary.json');training[name].pop('patient_exposures',None)
            rows=lines(root/name/'train.jsonl') if (root/name/'train.jsonl').exists() else []
            from src.medevidence_p2 import distribution
            training[name]['statistics']={k:distribution([r[k] for r in rows if r[k] is not None]) for k in ('L_NLL','gradient_norm_pre_clip','update_seconds','J_GRPO','KL','mean_reward','total_objective')}
            if (root/name/'initial_identity.json').exists():initial[name]=read(root/name/'initial_identity.json')
        else:training[name]={'status':status.get(name,{}).get('status','not_started'),'steps':0,'SFT_exposures':0,'rollouts':0}
    plan=read(root/'training_plan.json') if (root/'training_plan.json').exists() else None
    save(public/'medevidence_p4_training.json',{'branches':training,'plan':plan,'initialization_exact_equal':initial['FULL-SFT']==initial['FULL-GRL'] if len(initial)==2 else None,
        'initialization_matches_preflight':all(v==read(root/'preflight/initial_identity.json') for v in initial.values()) if initial else None,'A_coefficient':0,'seed':17})
    records={};folds=read(root/'protocol/folds.json')
    for name in ('COV','FULL-SFT','FULL-GRL'):
        dest=root/('eval_'+name)
        if (dest/'summary.json').exists() and read(dest/'summary.json')['status']=='completed':records[name]=lines(dest/'predictions.jsonl')
    absolute={name:{**full_metrics(rs,folds),**extended_summary(rs)} for name,rs in records.items()};comparisons={}
    for a,b in [('FULL-SFT','FULL-GRL'),('COV','FULL-SFT'),('COV','FULL-GRL')]:
        if a in records and b in records:comparisons[b+' minus '+a]={'intervals':paired_intervals(records[a],records[b],folds),'paired_success_counts':paired_counts(records[a],records[b])}
    engineering={'status':'not_applicable_RL_not_completed','passed':False}
    if all(name in absolute for name in ('COV','FULL-SFT','FULL-GRL')):
        r,s,c=[absolute[name] for name in ('FULL-GRL','FULL-SFT','COV')]
        conditions={'single_at_least6':r['single_strict_success_count']>=6,'single_net_at_least3':r['single_strict_success_count']-s['single_strict_success_count']>=3,
                    'negative_at_most11_and_COV':r['negative']['valid_nonempty']<=min(11,c['negative']['valid_nonempty']),
                    'region_recall_up_multi_matches_not_down':r['region_recall']>s['region_recall'] and r['strata']['multi']['matched_regions']>=s['strata']['multi']['matched_regions'],
                    'AP_drop_at_most_0_02':r['A_raw_ap']>=c['A_raw_ap']-.02,'invalid_truncated_no_increase':sum(r['positive'][k]+r['negative'][k] for k in ('invalid','truncated'))<=sum(s['positive'][k]+s['negative'][k] for k in ('invalid','truncated'))}
        engineering={'status':'evaluated','passed':all(conditions.values()),'conditions':conditions,'meaning':'R&D confirmation gate only'}
    evaluation={'status':'completed' if 'COV' in records and 'FULL-SFT' in records else 'incomplete','same_protocol_reassessed':absolute,'paired_comparisons':comparisons,
        'main_comparison_status':'completed' if 'FULL-GRL' in records and 'FULL-SFT' in records else 'not_available_RL_not_completed',
        'historical_P3_reference':{'source_commit':cfg['reference_commit'],'generation_cap':80,'results_not_new_reassessment':True},'formal_generation_cap':cfg['max_new_tokens'],
        'engineering_gate':engineering,'patient_bootstrap':'2000 repeats,seed42,original folds retained and thresholds refit each replicate; no training-seed or independent generalization uncertainty',
        'success_false_box_curve':{'status':'not_completed','reason':'no prevalidated frozen localization confidence; optional curve not invented'}}
    save(public/'medevidence_p4_evaluation.json',evaluation)
    budget={'status':ledger['status'],'gpu_hours':ledger['gpu_hours'],'GPU_limit_hours':3,'records':[{k:v for k,v in r.items() if k in ('stage','status','charged_seconds','exit_code')} for r in ledger['records']]}
    save(public/'medevidence_p4_budget.json',budget)
    document='# MedEvidence P4 actual results and decision\n\n'
    document+=f"Overall stage: {ledger['status']}. FULL-GRL: {training['FULL-GRL']['status']}. Test pixels read: 0.\n\n"
    document+='Natural RL startup gate (all three required):\n\n```json\n'+json.dumps(diag['gate'],indent=2,ensure_ascii=False)+'\n```\n\n'
    document+='| Model, P4 shared greedy policy | Single strict | Matched GT regions | Negative nonempty | Positive empty | Multi all covered | AP | Classification FP |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for name,v in absolute.items():document+=f"| {name} | {v['single_strict_success_count']}/25 | {v['matched_regions']}/51 | {v['negative']['valid_nonempty']}/219 | {v['positive']['valid_empty']}/37 | {v['multi_all_covered']}/12 | {v['A_raw_ap']:.6f} | {v['classification_raw_0_5']['FP']}/219 |\n"
    document+='\nTraining actuals:\n\n'+json.dumps({name:{k:v.get(k) for k in ('status','steps','SFT_exposures','rollouts','runtime_seconds')} for name,v in training.items()},indent=2)+'\n\n'
    document+=f"Cumulative GPU occupancy: {ledger['gpu_hours']:.6f}/3.00 hours; includes loading, diagnostics, sampling, calibration, evaluation, failure attempts and saves.\n\n"
    if training['FULL-GRL']['status']=='not_started_gate_failed':document+='FULL-GRL was not started because a prespecified natural candidate gate failed. This run cannot answer whether geometry RL outperforms SFT; no threshold, prompt, sample count or post-SFT RL retry was changed.\n\n'
    document+='Paired differences, 95% intervals, valid bootstrap counts and new/lost/common successes are in the evaluation JSON. Historical 80-token P3 results are historical references, never relabelled as P4 reassessment. Full SFT changes both coverage and additional training; an improvement cannot be attributed solely to coverage. RL, if executed, adds geometry GRPO and fixed KL together, with greater compute. No independent validation, equal-compute superiority, evidence faithfulness or clinical utility is established.\n\n'
    document+='Plan versus actual versus incomplete items: planned405 original patients, fixed shared schedule and COV initialization; actual exposure histograms and step freeze in training JSON. Optional intermediate curves, forced-nonempty diagnosis and localization confidence curve were not executed. No sealed test, new dataset, dep, A training or automatic next round. Protocol deviations: none unless a failed/stopped stage explicitly records its reason; aborted stages retain actual counts and are not promoted to main results.\n\n'
    document+='Public delivery: local commit prepared; not pushed because no P4-specific public push authorization has been received. Patient identifiers, GT, images, per-patient predictions, checkpoints and raw logs remain private.\n'
    (public/'medevidence_p4_decision.md').write_text(document)
    save(public/'medevidence_p4_status.json',{'overall':ledger['status'],'stages':status,'test_pixels_read':0,'public_push':'not_executed_pending_current_authorization'})
    print(json.dumps({'report_status':ledger['status'],'GPU_hours':ledger['gpu_hours'],'new_completed_models':[k for k in records if k!='COV']}),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'controller':controller,'report':report}[os.environ['P4_ACTION']]()
