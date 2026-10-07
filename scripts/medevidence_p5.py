"""Bounded P5 diagnostic: existing rollout audit and paired coordinate fit probes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def save(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def prepare():
    import numpy as np
    import pandas as pd
    from transformers import AutoProcessor
    from src.data import smart_size
    from src.medevidence_p5 import target, canonical, token_parts, decompose
    base=Path(os.environ['PILOT_ROOT']);code=base/'code';out=base/'outputs';old=base.parent/'medevidence_p4/outputs'
    p=out/'protocol';p.mkdir()
    cfg=read(code/'configs/medevidence_p5.json');ledger=read(old/'gpu_ledger.json')
    assert ledger['status']=='completed' and not ledger['active']
    assert abs(ledger['charged_seconds']-cfg['prior_GPU_seconds'])<1e-6
    frame=pd.read_csv(old/'protocol/manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
    rows=frame.set_index('image_id').to_dict('index')
    for r in rows.values():r['boxes']=json.loads(r['boxes'])
    selected=read(old/'protocol/selected_patients.json');rng=np.random.default_rng(cfg['selection_seed'])
    fit={};cal={}
    for name,pool in selected.items():
        ids=[str(x) for x in rng.permutation(sorted(pool))];n=cfg['fit_counts'][name];q=cfg['calibration_counts'][name]
        fit[name]=ids[:n];cal[name]=ids[n:n+q]
    fit_ids=sum(fit.values(),[]);cal_ids=sum(cal.values(),[]);ids=fit_ids+cal_ids
    assert len(ids)==len(set(ids))==96 and len({rows[k]['case_id'] for k in ids})==96
    assert all(rows[k]['split']=='train' for k in ids)
    queues={name:[] for name in fit};rngs={name:np.random.default_rng(cfg['seed']) for name in fit}
    def take(name):
        if not queues[name]:queues[name]=[str(k) for k in rngs[name].permutation(fit[name])]
        return queues[name].pop(0)
    plan=[[take(x) for x in ('single','multi','negative','negative')] for _ in range(cfg['fit_steps'])]
    from collections import Counter
    assert set(Counter(k for batch in plan for k in batch).values())=={32}
    model=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    tokenizer=AutoProcessor.from_pretrained(model,local_files_only=True,use_fast=False).tokenizer
    contracts={};sizes={};maxlen={fmt:0 for fmt in cfg['formats']};maxerr=0.
    for k in ids:
        r=rows[k];h,w=smart_size(r['height'],r['width']);sizes[k]=[w,h];contracts[k]={}
        for fmt in cfg['formats']:
            text=target(r['boxes'],r['width'],r['height'],fmt,(w,h));tokens,parts=token_parts(tokenizer,text)
            assert tokens[-1]==tokenizer.eos_token_id and len(tokens)+16<=cfg['max_new_tokens']
            contracts[k][fmt]={'text':text,'ids':tokens,'parts':parts};maxlen[fmt]=max(maxlen[fmt],len(tokens))
            b,e=canonical(text,fmt,(w,h));assert not e and len(b)==len(r['boxes'])
            from src.medevidence import normalized
            for a,z in zip(b,normalized(r['boxes'],r['width'],r['height'])):
                maxerr=max(maxerr,max(abs(x-y) for x,y in zip(a,z)))
    frame[frame.image_id.isin(ids)].to_csv(p/'manifest.csv',index=False)
    for name,val in [('selection.json',{'fit':fit,'calibration':cal}),('schedule.json',plan),('contracts.json',contracts),('sizes.json',sizes),
                     ('references.json',read(old/'protocol/references.json')),('identity.json',read(old/'protocol/identity.json'))]:save(p/name,val)
    audit={name:decompose(lines(old/name/'predictions.jsonl')) for name in ('D3','D2')}
    audit['selection_scope']='D3 training mechanism audit primary; D2 old development descriptive only; neither selects coordinate format'
    save(out/'reward_audit.json',audit);save(code/'reports/medevidence_p5_reward_audit.json',audit)
    import runpy
    runpy.run_path(str(code/'tests/test_medevidence_p5.py'))['main']()
    report={'status':'passed','fit_patients':32,'calibration_patients':64,'overlap':0,'new_image_reads':0,'test_pixels_read':0,
            'historically_unseen_calibration':False,'max_target_lengths':maxlen,'max_coordinate_rounding_difference_0_1000':maxerr,
            'fit_steps':256,'exposures_per_branch':1024,'exposures_per_fit_patient':32,'unit_tests':'passed',
            'prior_GPU_seconds':ledger['charged_seconds'],'source_commit':read(out/'authorization.json')['source_commit']}
    save(p/'CPU_checks.json',report);save(code/'reports/medevidence_p5_preflight.json',report)
    print(json.dumps(report),flush=True)


def worker():
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from src.data import QUESTIONS
    from src.v2 import DevelopmentData
    from src.model import Model
    from src.experiment import seed_all
    from src.medevidence import normalized
    from src.medevidence_p2 import enrich
    from src.medevidence_p2_run import RNG,set_RNG,BudgetStop,evaluating
    from src.medevidence_p4 import extended_summary,reward,best_single
    from src.medevidence_p4_run import parameters,current_digest,sampling,bind_classification_labels
    from src.medevidence_p5 import canonical,prompt,decompose,diagnostic_gate
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';p=root/'protocol';cfg=read(base/'code/configs/medevidence_p5.json')
    fmt=os.environ['JOB_STAGE'];dest=root/('fit_'+fmt);dest.mkdir()
    if fmt not in cfg['formats'] or not read(root/'authorization.json')['gpu_authorized']:raise PermissionError('Unapproved stage')
    class Context:
        def __init__(self):
            self.cfg={**cfg,'RL':{'sampling_rng_seed':cfg['sampling_seed']}};self.sampling_state=None;self.stopped=False
        def tick(self):
            if self.stopped or time.time()>float(os.environ['JOB_DEADLINE'])-60:raise BudgetStop('P5 stage budget reserve')
    c=Context();signal.signal(signal.SIGUSR1,lambda *_:setattr(c,'stopped',True));started=time.time();completed=0;training=[]
    frame=pd.read_csv(p/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str});rows=frame.set_index('image_id').to_dict('index')
    for r in rows.values():r['boxes']=json.loads(r['boxes'])
    data=DevelopmentData(os.environ['DATA_ROOT'],frame,dest/'data_access.jsonl');data.install_guard()
    sets=read(p/'selection.json');schedule=read(p/'schedule.json');contracts=read(p/'contracts.json');sizes=read(p/'sizes.json')
    references=read(p/'references.json');identity=read(p/'identity.json');cache={}
    model_path=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    for name,meta in identity['base']['files'].items():
        st=(model_path/name).stat();assert (st.st_size,st.st_mtime_ns)==(meta['bytes'],meta['mtime_ns'])
    seed_all(cfg['seed']);m=Model(model_path,references['COV']['checkpoint_path'],True,576,1024)
    assert current_digest(m)==references['COV']['identity']['adapter_digest']
    assert sum(x.numel() for x in parameters(m))==5046272
    bind_classification_labels(m,['no','yes'])
    opt=torch.optim.AdamW(parameters(m),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=.01)
    def encode(k):
        c.tick()
        if k not in cache:
            im=data.image(k).resize(tuple(sizes[k]),Image.Resampling.BICUBIC)
            QUESTIONS['L']=prompt(fmt,sizes[k]);inp,grid=m.prompt(im,'L')
            assert [grid[1]*28,grid[0]*28]==sizes[k]
            cache[k]=(inp,m.encode(inp))
        return cache[k]
    def score(k):
        inp,feat=encode(k);v=m.score(inp,feat,answer=contracts[k][fmt]['text'],include_eos=True)
        assert v['ids'].tolist()==contracts[k][fmt]['ids'];return v
    def record(k,text,tokens):
        b,error=canonical(text,fmt,sizes[k]);trunc=len(tokens)>=cfg['max_new_tokens'] and tokens[-1]!=m.processor.tokenizer.eos_token_id
        canonical_text=json.dumps(b) if not error else 'invalid'
        return enrich({'image_id':k,'case_id':rows[k]['case_id'],'raw_loc_text':text,'loc_text':canonical_text,
                       'format_error':error,'truncated':trunc,'displayed_size':sizes[k],'generated_length':len(tokens)},rows[k])
    def evaluate(name,ids):
        records=[];parts={x:[] for x in ('existence','coordinate','eos','syntax')}
        with evaluating(m),torch.no_grad(),(dest/(name+'.jsonl')).open('x') as f:
            for k in ids:
                inp,feat=encode(k);text,tokens=m.generate(inp,feat,max_tokens=cfg['max_new_tokens'],sample=False)
                r=record(k,text,tokens);values=-score(k)['tokens']
                for part in parts:
                    idx=[i for i,x in enumerate(contracts[k][fmt]['parts']) if x==part]
                    if idx:parts[part].append(float(values[idx].mean()))
                records.append(r);f.write(json.dumps(r)+'\n');f.flush()
        v=extended_summary(records)
        return {'patients':len(records),'positive_strict':sum(r['strict'] for r in records if r['y']),
                'negative_nonempty':sum(r['state']=='valid_nonempty' for r in records if not r['y']),
                'invalid':sum(r['state']=='invalid' for r in records),'truncated':sum(r['truncated'] for r in records),
                'metrics':v,'token_part_mean_NLL':{x:float(np.mean(vv)) if vv else None for x,vv in parts.items()}}
    fit=sum(sets['fit'].values(),[]);cal=sum(sets['calibration'].values(),[])
    result={'format':fmt,'initial_adapter_digest':current_digest(m),'test_pixels_read':0}
    try:
        result['initial_fit']=evaluate('initial_fit',fit);result['initial_calibration']=evaluate('initial_calibration',cal)
        rng=RNG();before=current_digest(m);gradrows=[]
        # Only eight fixed fit patients; no optimizer updates or accumulation into .grad.
        for k in sum([sets['fit']['single'][:2],sets['fit']['multi'][:2],sets['fit']['negative'][:4]],[]):
            v=score(k);parts=contracts[k][fmt]['parts'];entry={'stratum':'single' if len(rows[k]['boxes'])==1 else 'multi' if rows[k]['boxes'] else 'negative','parts':{}}
            present=list(dict.fromkeys(parts))
            for j,part in enumerate(present):
                idx=[i for i,x in enumerate(parts) if x==part];loss=-v['tokens'][idx].sum()/len(parts)
                grads=torch.autograd.grad(loss,parameters(m),retain_graph=j<len(present)-1,allow_unused=True)
                norm=math_norm(grads,torch)
                entry['parts'][part]={'gradient_norm':norm,'original_mean_loss_contribution':float(loss.detach()),'token_count':len(idx)}
            gradrows.append(entry)
        assert before==current_digest(m) and not opt.state
        assert all(x.grad is None for x in parameters(m));set_RNG(rng)
        save(dest/'gradient_parts.json',gradrows)
        aggregates={}
        for stratum in ('single','multi','negative'):
            subset=[r for r in gradrows if r['stratum']==stratum];aggregates[stratum]={}
            for part in ('existence','coordinate','eos','syntax'):
                values=[r['parts'][part] for r in subset if part in r['parts']]
                aggregates[stratum][part]={'patients':len(values),**{key:float(np.mean([r[key] for r in values])) if values else None
                                          for key in ('gradient_norm','original_mean_loss_contribution','token_count')}}
        result['gradient_diagnostic']={'patients':8,'optimizer_updates':0,'adapter_unchanged':True,'RNG_restored':True,'aggregate_by_stratum':aggregates,
                                       'norms_not_additive':'Component vectors may cancel; these norms alone do not establish dominance.'}
        seed_all(cfg['seed']);opt.zero_grad(set_to_none=True)
        with (dest/'training.jsonl').open('x') as f:
            for batch in schedule:
                c.tick();start=time.time();opt.zero_grad(set_to_none=True);losses=[]
                for k in batch:
                    loss=-score(k)['mean'];assert torch.isfinite(loss);(loss/4).backward();losses.append(float(loss.detach()))
                assert all(x.grad is None for x in m.visual.parameters())
                norm=torch.nn.utils.clip_grad_norm_(parameters(m),cfg['gradient_clip']);assert torch.isfinite(norm)
                opt.step();completed+=1;r={'step':completed,'L_NLL':float(np.mean(losses)),'gradient_norm_pre_clip':float(norm),'seconds':time.time()-start}
                training.append(r);f.write(json.dumps(r)+'\n');f.flush()
                if completed%32==0:print(json.dumps(r),flush=True)
        cp=dest/'step_0256';cp.mkdir();m.model.save_pretrained(cp,safe_serialization=True);m.processor.tokenizer.save_pretrained(cp)
        torch.save(opt.state_dict(),cp/'optimizer.pt');torch.save(RNG(),cp/'rng.pt')
        save(cp/'identity.json',{'adapter_digest':current_digest(m),'steps':completed,'format':fmt})
        result['final_fit']=evaluate('final_fit',fit);result['final_calibration']=evaluate('final_calibration',cal)
        groups=[];settings={**cfg['sampling'],'rng_seed':cfg['sampling_seed']}
        with evaluating(m),(dest/'natural_calibration.jsonl').open('x') as f:
            for k in cal:
                inp,feat=encode(k);samples=sampling(c,m,inp,feat,8,settings);gt=normalized(rows[k]['boxes'],rows[k]['width'],rows[k]['height'])
                converted=[]
                for sample in samples:
                    r=record(k,sample['loc_text'],sample['tokens']);converted.append({'loc_text':r['loc_text'],'truncated':r['truncated'],
                        'reward':reward(r['loc_text'],gt,r['truncated']),'invalid':r['state']=='invalid'})
                group={'gt':gt,'image_id':k,'case_id':rows[k]['case_id'],'outputs':converted};groups.append(group);f.write(json.dumps(group)+'\n');f.flush()
        result['natural_calibration']={'patients':64,'rollouts':512,'single_best8_success':sum(best_single(g['outputs'],g['gt'])['strict_success'] for g in groups if len(g['gt'])==1),
                'invalid_or_truncated':sum(r['invalid'] or r['truncated'] for g in groups for r in g['outputs']),'decomposition':decompose(groups)}
        result['gate']=diagnostic_gate(result);result['status']='completed'
    except Exception as exc:
        result['status']='stopped_budget' if isinstance(exc,BudgetStop) else 'failed';result['error_type']=type(exc).__name__;result['reason']=str(exc)
        save(dest/'failure.json',{'error_type':type(exc).__name__,'reason':str(exc)})
    result.update(steps=completed,SFT_exposures=completed*4,training_rollouts=0,diagnostic_rollouts=result.get('natural_calibration',{}).get('rollouts',0),
                  runtime_seconds=time.time()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
                  train_accesses=dict(data.access_counts),mean_update_seconds=float(np.mean([r['seconds'] for r in training])) if training else None)
    save(dest/'summary.json',result);print(json.dumps({k:result[k] for k in ('format','status','steps','runtime_seconds')}),flush=True)
    if result['status']!='completed':sys.exit(1)


def math_norm(grads,torch):
    return float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads if g is not None)))


def controller():
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';code=base/'code';cfg=read(code/'configs/medevidence_p5.json')
    if (out/'gpu_ledger.json').exists():raise FileExistsError('P5 already attempted')
    assert read(out/'protocol/CPU_checks.json')['status']=='passed'
    jobs={};records=[]
    def persist(status):
        now=time.time();charged=sum(r['charged_seconds'] for r in records)+sum(now-j['started'] for j in jobs.values())
        save(out/'gpu_ledger.json',{'status':status,'prior_GPU_seconds':cfg['prior_GPU_seconds'],'new_charged_seconds':charged,
            'combined_GPU_hours':(charged+cfg['prior_GPU_seconds'])/3600,'limit_seconds':cfg['combined_GPU_limit_seconds'],
            'records':records,'active':[{k:v for k,v in j.items() if k not in ('proc','log')} for j in jobs.values()]})
    memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
    gpus=[g for g in cfg['allowed_gpus'] if memory[g]>=cfg['minimum_free_memory_mib']]
    if len(gpus)<2:save(out/'blocked.json',{'status':'blocked','reason':'fewer than two GPUs with sufficient memory'});return
    assert cfg['prior_GPU_seconds']+2*cfg['diagnostic_worker_quota_seconds']+1200<cfg['combined_GPU_limit_seconds']
    for fmt,gpu in zip(cfg['formats'],gpus):
        started=time.time();env=os.environ.copy();env.update(JOB_STAGE=fmt,JOB_DEADLINE=str(started+cfg['diagnostic_worker_quota_seconds']),CUDA_VISIBLE_DEVICES=str(gpu),P5_ACTION='worker',DATASET_NAME='rsna')
        log=(base/'logs'/('fit_'+fmt+'.log')).open('x')
        proc=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
        proc.stdin.write(b"from scripts.medevidence_p5 import worker;worker()\n");proc.stdin.close()
        jobs[fmt]={'proc':proc,'log':log,'started':started,'deadline':started+cfg['diagnostic_worker_quota_seconds'],'pid':proc.pid,'gpu':gpu,'signaled':False};persist('running')
    while jobs:
        for fmt,j in list(jobs.items()):
            proc=j['proc'];now=time.time()
            if proc.poll() is None:
                if now>=j['deadline']-30 and not j['signaled']:proc.send_signal(signal.SIGUSR1);j['signaled']=True
                if now>=j['deadline']:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                else:continue
            j['log'].close();summary=out/('fit_'+fmt)/'summary.json'
            status=read(summary)['status'] if summary.exists() else 'failed'
            if proc.returncode and status=='completed':status='failed'
            records.append({'stage':'fit_'+fmt,'status':status,'exit_code':proc.returncode,'charged_seconds':time.time()-j['started']})
            del jobs[fmt]
        persist('running' if jobs else 'diagnostics_completed')
        if jobs:time.sleep(2)
    report()


def report():
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';public=base/'code/reports';models={}
    for fmt in ('normalized','absolute'):
        p=out/('fit_'+fmt)/'summary.json'
        if p.exists():models[fmt]=read(p)
    eligible=[fmt for fmt,v in models.items() if v['status']=='completed' and v['gate']['passed']]
    selected=max(eligible,key=lambda fmt:(models[fmt]['final_calibration']['metrics']['single_IoU_unconditional']['mean'],fmt=='normalized')) if eligible else None
    decision={'diagnostics_status':'completed' if len(models)==2 and all(v['status']=='completed' for v in models.values()) else 'incomplete',
              'eligible_formats':eligible,'selected_format':selected,'main_status':'ready_for_native_preflight' if selected else 'not_started_gate_failed',
              'selection_uses_development':False,'test_pixels_read':0}
    save(out/'decision.json',decision);save(public/'medevidence_p5_diagnostics.json',models)
    save(public/'medevidence_p5_decision.json',decision);save(public/'medevidence_p5_budget.json',read(out/'gpu_ledger.json'))
    print(json.dumps(decision),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'controller':controller,'worker':worker,'report':report}[os.environ['P5_ACTION']]()
