"""Bounded v2 workers. Only P1/SFT-N/SFT-B have an optimizer; every other stage is frozen."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.data import sha
from src.model import Model
from src.experiment import seed_all,optimizer_snapshot
from src.regions import replace_features,pixel_blur
from src.objectives import ced_reward,advantages,correctness
from src.v2 import (LABELS,DevelopmentData,require_safe_config,require_stage,save,binary_stats,
                    summarize_scores,paired_bootstrap,reward_group,exposure_summary)


class BudgetStop(RuntimeError):pass


class Context:
    def __init__(self):
        self.root=Path(os.environ['OUTPUT_ROOT']);self.old=Path(os.environ['V1_OUTPUT_ROOT']);self.stage=os.environ['V2_STAGE']
        self.cfgpath=Path(os.environ['RUN_CONFIG']);self.cfg=json.loads(self.cfgpath.read_text());require_safe_config(self.cfg)
        if os.environ.get('DATASET_NAME')!='rsna':raise ValueError('RSNA label namespace required')
        self.filter=os.environ.get('V2_MODEL_FILTER','');self.started=time.time();self.deadline=float(os.environ['V2_DEADLINE']);self.dest=self.root/self.stage;self.dest.mkdir(exist_ok=True)
        self.protocol=self.root/'protocol';lock=json.loads((self.protocol/'protocol_lock.json').read_text())
        if sha(self.cfgpath.read_bytes())!=lock['config_sha256']:raise ValueError('v2 config lock changed')
        for n,digest in lock['hashes'].items():
            if sha((self.protocol/n).read_bytes())!=digest:raise ValueError('v2 protocol changed: '+n)
        code=Path(__file__).resolve().parents[1]
        for n,digest in lock['source_hashes'].items():
            if sha((code/n).read_bytes())!=digest:raise ValueError('v2 execution source changed: '+n)
        self.frame=pd.read_csv(self.protocol/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
        self.data=DevelopmentData(os.environ['DATA_ROOT'],self.frame,self.dest/('data_access'+('_'+self.filter if self.filter else '')+'.jsonl'));self.data.install_guard()
        self.rows=self.frame.set_index('image_id').to_dict('index')
        self.regions=json.loads((self.protocol/'regions.json').read_text());self.subsets=json.loads((self.protocol/'subsets.json').read_text())
        self.folds=json.loads((self.protocol/'folds.json').read_text());self.identity=json.loads((self.protocol/'identity.json').read_text())
        self.modelpath=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
        for name,identity in self.identity['base']['files'].items():
            stat=(self.modelpath/name).stat()
            if stat.st_size!=identity['bytes'] or stat.st_mtime_ns!=identity['mtime_ns']:raise ValueError('Base model file identity changed')
        self.statuses=json.loads((self.root/'stage_status.json').read_text());require_stage(self.stage,self.statuses)
        self.tick()

    def tick(self):
        if time.time()>=self.deadline-20:raise BudgetStop('Cumulative six-hour budget exhausted; checkpoint and stop')

    def model(self,name,train=False):
        self.tick();seed_all(self.cfg['seed'])
        if train:
            require_stage(self.stage,self.statuses,training=True)
            if name not in ['P1','SFT-N','SFT-B']:raise PermissionError('Unexpected training model')
            adapter=None
        elif name=='B0':adapter=None
        elif name=='B1':adapter=self.old/'B1'/self.identity['historical_B1']['checkpoint']
        else:
            method,step=name.split('@');adapter=self.root/method/f'step_{int(step):04d}'
        if adapter is not None and not (adapter/'adapter_model.safetensors').is_file():raise FileNotFoundError('Required adapter unavailable')
        if name=='B1':
            identity=self.identity['historical_B1']
            if sha((adapter/'adapter_model.safetensors').read_bytes())!=identity['sha256'] or sha((adapter/'adapter_config.json').read_bytes())!=identity['config_sha256']:raise ValueError('Historical B1 adapter changed')
        model=Model(self.modelpath,adapter=adapter,trainable=train,min_tokens=self.cfg['min_visual_tokens'],max_tokens=self.cfg['max_visual_tokens'])
        ids={x:model.processor.tokenizer.encode(x,add_special_tokens=False) for x in LABELS}
        if ids!=self.identity['label_ids'] or model.processor.tokenizer.eos_token_id!=self.identity['eos_id']:raise ValueError('Token identity changed')
        if train:
            params={n:p for n,p in model.model.named_parameters() if p.requires_grad}
            if sum(p.numel() for p in params.values())!=5046272 or any('lora_' not in n or '.visual.' in n for n in params):raise ValueError('LoRA scope changed')
        return model

    def finish(self,result):
        result.update(stage=self.stage,runtime_seconds=time.time()-self.started,test_images_read=0,posttraining_enabled=False)
        result.setdefault('data_access_counts',dict(self.data.access_counts))
        if self.filter=='aggregate':
            parts=[json.loads(p.read_text()) for p in self.dest.glob('summary_*.json')]
            result['data_access_counts']={split:sum(p.get('data_access_counts',{}).get(split,0) for p in parts) for split in ['train','validation']}
            result['gpu_worker_runtime_seconds_sum']=sum(p['runtime_seconds'] for p in parts)
        suffix='_'+self.filter if self.filter and self.filter!='aggregate' else ''
        save(self.dest/('summary'+suffix+'.json'),result);return result


def unload(model):
    del model;gc.collect();torch.cuda.empty_cache()


def parameter_identity(model):
    digest=hashlib.sha256();versions={}
    for n,p in model.model.named_parameters():
        versions[n]=p._version
        if 'lora_' in n:digest.update(n.encode());digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return {'adapter_parameter_digest':digest.hexdigest(),'parameter_versions':versions,'requires_grad_count':sum(p.requires_grad for p in model.model.parameters())}


@contextmanager
def frozen(model):
    if any(p.requires_grad for p in model.model.parameters()):raise PermissionError('Inference model must have all gradients disabled')
    before=parameter_identity(model)
    original_backward=torch.Tensor.backward;original_step=torch.optim.AdamW.step
    def forbidden(*a,**k):raise PermissionError('Backward/optimizer updates forbidden in frozen diagnostics')
    torch.Tensor.backward=forbidden;torch.optim.AdamW.step=forbidden
    try:
        with torch.no_grad():yield
    finally:
        torch.Tensor.backward=original_backward;torch.optim.AdamW.step=original_step
        after=parameter_identity(model)
        model.v2_identity_check={'before':before,'after':after,'unchanged':before==after}
        if before!=after:raise RuntimeError('Frozen model parameters changed')


def record_scores(ctx,model,image_id,image=None):
    ctx.tick();row=ctx.rows[image_id]
    image=ctx.data.image(image_id) if image is None else image
    inputs,grid=model.prompt(image,'A');features=model.encode(inputs);scores,details=model.classes(inputs,features,labels=LABELS)
    values=scores.detach().float().cpu().numpy();p=float(scores.float().softmax(-1)[1])
    return {'image_id':image_id,'case_id':row['case_id'],'y':LABELS.index(row['pathology']),
            'score_no':float(values[0]),'score_yes':float(values[1]),'margin':float(values[1]-values[0]),'p':p,
            'token_ids':[r['ids'].detach().cpu().tolist() for r in details],'token_lengths':[r['length'] for r in details],
            'eos_included':False,'threshold_0_5_prediction':int(p>=.5),'historical_argmax_prediction':int(np.argmax(values))}


def determinism_check(ctx,model):
    image_id=next(k for k in sorted(ctx.regions) if ctx.regions[k]['eligible'] and ctx.rows[k]['split']=='train')
    image=ctx.data.image(image_id);inputs,grid=model.prompt(image);features=model.encode(inputs);r=ctx.regions[image_id]
    if list(grid)!=r['grid']:raise ValueError('Frozen regions/grid mismatch')
    max_delta=0.;scopes={}
    for name,item in [('original',None),('evidence',r['evidence']),('control_0',r['controls'][0])]:
        feat=features if item is None else replace_features(features,item['tokens'],item['source'])
        first=model.classes(inputs,feat,labels=LABELS)[0]
        second=model.classes(inputs,feat,labels=LABELS)[0]
        delta=float((first-second).abs().max());scopes[name]=delta;max_delta=max(max_delta,delta)
    tolerance=ctx.cfg['P4']['numerical_score_tolerance']
    if max_delta>tolerance:raise RuntimeError('Repeated isolated-answer scores exceed frozen tolerance')
    return {'passed':True,'max_abs_logprob_delta':max_delta,'tolerance':tolerance,'scopes':scopes,'conditions':'fixed model,image,question,isolated answer,regions,replacement sources'}


def old_records(ctx,name):
    if name=='B1':
        path=ctx.old/'B1'/(ctx.identity['historical_B1']['checkpoint']+'_validation.json')
        rows=json.loads(path.read_text())['predictions']
    else:
        rows=[json.loads(s) for s in (ctx.old/'audits/B0/predictions.jsonl').read_text().splitlines()]
    result={}
    for r in rows:
        k=r['image_id'];row=ctx.rows.get(k)
        if row is None or row['split']!='validation' or row['case_id']!=r['case_id'] or r['y']!=LABELS.index(row['pathology']):raise ValueError('Historical score/data identity mismatch')
        v=r.get('original',r);values=v['mean']
        if v['length']!=[1,1]:raise ValueError('Historical label token lengths changed')
        result[k]={'image_id':k,'case_id':r['case_id'],'y':r['y'],'score_no':values[0],'score_yes':values[1],'margin':values[1]-values[0],
                   'p':r['p'],'token_ids':[ctx.identity['label_ids'][x] for x in LABELS],'token_lengths':v['length'],'eos_included':False,
                   'threshold_0_5_prediction':int(r['p']>=.5),'historical_argmax_prediction':int(np.argmax(values)),'origin':'verified_v1_scores'}
    return result


def p0(ctx):
    all_scores={};summary={'models':{},'actual_samples':256,'steps':0,'status':'completed','scope':'exploratory_development'}
    val=ctx.frame[ctx.frame.split=='validation'].sort_values('case_id').image_id.tolist()
    if ctx.filter=='aggregate':
        for name in ['B0','B1']:
            part=json.loads((ctx.dest/('summary_'+name+'.json')).read_text());summary['models'].update(part['models'])
            all_scores[name]=json.loads((ctx.dest/name/'scores.json').read_text())
        summary['bootstrap']=paired_bootstrap(all_scores,ctx.folds,ctx.cfg['bootstrap_repeats'],ctx.cfg['bootstrap_seed'])
        return ctx.finish(summary)
    for name in ([ctx.filter] if ctx.filter else ['B0','B1']):
        model=ctx.model(name);cache=old_records(ctx,name);original_reused=len(cache);dest=ctx.dest/name;dest.mkdir(exist_ok=True)
        with frozen(model):
            repeat=determinism_check(ctx,model);checks=[];started=time.time()
            for k in sorted(cache)[:ctx.cfg['P0']['throughput_cases']]:
                new=record_scores(ctx,model,k);old=cache[k]
                delta=max(abs(new[f]-old[f]) for f in ['score_no','score_yes','p'])
                checks.append(delta)
            elapsed=time.time()-started
            save(dest/'reuse_check.json',{'n':len(checks),'absolute_deltas':checks,'max_delta':max(checks),'tolerance':ctx.cfg['P4']['numerical_score_tolerance']})
            if max(checks)>ctx.cfg['P4']['numerical_score_tolerance']:raise RuntimeError('v1/v2 score semantic mismatch; preserve both and stop')
            rate=elapsed/len(checks)
            estimate=rate*(len(val)-len(cache)+256)
            if estimate>ctx.deadline-time.time()-30:raise BudgetStop('Measured P0 throughput exceeds remaining cumulative budget')
            save(dest/'throughput.json',{'cases':len(checks),'seconds':elapsed,'seconds_per_classification':rate,'estimated_remaining_P0_seconds':rate*(len(val)-len(cache)+256),'budget_remaining_seconds':ctx.deadline-time.time()})
            for k in val:
                if k not in cache:cache[k]=record_scores(ctx,model,k)
            records=[cache[k] for k in val];save(dest/'scores.json',records);all_scores[name]=records
            diagnostic=[]
            ids=ctx.subsets['P0'];permutations=ctx.subsets['mismatch']
            for i,k in enumerate(ids):
                image=ctx.data.image(k)
                views=[('black',Image.new('RGB',image.size,0),None)]+[(f'mismatch_{j+1}',ctx.data.image(perm[i]),perm[i]) for j,perm in enumerate(permutations)]
                diagnostic.append({**cache[k],'condition':'original','donor_image_id':k})
                for condition,im,donor in views:diagnostic.append({**record_scores(ctx,model,k,im),'condition':condition,'donor_image_id':donor})
            save(dest/'image_dependency.json',diagnostic)
        save(dest/'frozen_identity.json',model.v2_identity_check)
        result=summarize_scores(records,ctx.folds);result.update(reused_v1_cases=original_reused,score_reuse_verification={'n':len(checks),'max_delta':max(checks)},determinism=repeat)
        result['image_dependency']={}
        for condition in ['original','black','mismatch_1','mismatch_2','mismatch_3']:
            rows=[r for r in diagnostic if r['condition']==condition];met=binary_stats([r['y'] for r in rows],[r['p'] for r in rows]);met.pop('ap')
            met['mean_margin']=float(np.mean([r['margin'] for r in rows]));met['label_specific_margin']={LABELS[y]:float(np.mean([r['margin'] for r in rows if r['y']==y])) for y in [0,1]}
            result['image_dependency'][condition]=met
        result['image_dependency_warning']='Balanced 32+32 subset; AP omitted; black images are OOD pressure tests, not lesion causality.'
        summary['models'][name]=result
        del model;gc.collect();torch.cuda.empty_cache()
    if not ctx.filter:
        ctx.tick();summary['bootstrap']=paired_bootstrap(all_scores,ctx.folds,ctx.cfg['bootstrap_repeats'],ctx.cfg['bootstrap_seed'])
    return ctx.finish(summary)


@torch.no_grad()
def training_evaluation(ctx,model,ids):
    before=model.model.training;model.set_training(False);records=[];loss=[]
    for k in ids:
        ctx.tick();row=ctx.rows[k];image=ctx.data.image(k);inputs,_=model.prompt(image);features=model.encode(inputs)
        supervised=model.score(inputs,features,answer=row['pathology'],include_eos=True)
        label_loss=float(-supervised['tokens'][:-1].mean());eos_loss=float(-supervised['tokens'][-1]);total=float(-supervised['mean'])
        scores,details=model.classes(inputs,features,labels=LABELS);v=scores.float().cpu().tolist();p=float(scores.float().softmax(-1)[1]);y=LABELS.index(row['pathology'])
        records.append({'image_id':k,'case_id':row['case_id'],'y':y,'score_no':v[0],'score_yes':v[1],'margin':v[1]-v[0],'p':p,
                        'token_ids':[d['ids'].cpu().tolist() for d in details],'token_lengths':[d['length'] for d in details],
                        'threshold_0_5_prediction':int(p>=.5),'historical_argmax_prediction':int(np.argmax(v)),'eos_included':False})
        loss.append({'y':y,'label_loss':label_loss,'eos_loss':eos_loss,'supervised_loss':total})
    model.set_training(before)
    aggregate={'raw_0_5':binary_stats([r['y'] for r in records],[r['p'] for r in records]),'supervised_loss':float(np.mean([r['supervised_loss'] for r in loss])),
               'historical_argmax':binary_stats([r['y'] for r in records],[r['p'] for r in records],[r['historical_argmax_prediction'] for r in records]),
               'eos_loss':float(np.mean([r['eos_loss'] for r in loss])),
               'label_loss':{LABELS[y]:float(np.mean([r['label_loss'] for r in loss if r['y']==y])) for y in [0,1]},
               'margin':{LABELS[y]:float(np.mean([r['margin'] for r in records if r['y']==y])) for y in [0,1]},
               'exact_score_ties':sum(r['score_no']==r['score_yes'] for r in records)}
    return aggregate,records


def train(ctx):
    require_stage(ctx.stage,ctx.statuses,training=True)
    if (ctx.dest/'train.jsonl').exists():raise FileExistsError('Training attempt exists; new attempts require authorization')
    plan=json.loads((ctx.protocol/(ctx.stage+'_schedule.json')).read_text());model=ctx.model(ctx.stage,train=True)
    identity=parameter_identity(model);save(ctx.dest/'initial_identity.json',identity)
    if ctx.stage!='P1':
        p1_identity=json.loads((ctx.root/'P1/initial_identity.json').read_text())
        if identity['adapter_parameter_digest']!=p1_identity['adapter_parameter_digest']:raise RuntimeError('Fresh LoRA initialization differs from P1 B0 initialization')
    opt_cfg=dict(ctx.cfg['optimizer']);opt_cfg['betas']=tuple(opt_cfg['betas']);opt=torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad],**opt_cfg)
    save(ctx.dest/'optimizer_effective.json',[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups])
    checkpoints=ctx.cfg['P1' if ctx.stage=='P1' else 'P2']['checkpoints'];steps=max(checkpoints)
    ids=ctx.subsets['P1'] if ctx.stage=='P1' else ctx.frame[ctx.frame.split=='validation'].sort_values('case_id').image_id.tolist()
    evaluations={};consecutive=0;passed=False;done=0
    def evaluate(step):
        nonlocal consecutive,passed
        result,records=training_evaluation(ctx,model,ids);save(ctx.dest/f'step_{step:04d}_scores.json',records)
        if ctx.stage=='P1':
            met=result['historical_argmax'];meets=met['TP']>=15 and met['TN']>=15
            consecutive=consecutive+1 if meets else 0;passed=consecutive>=2
            result.update(meets_15_of_16_each=meets,consecutive_passes=consecutive,scientific_passed=passed)
        else:result['score_audit']=summarize_scores(records,ctx.folds)
        evaluations[str(step)]=result;save(ctx.dest/'evaluations.json',evaluations)
    try:
        if ctx.stage=='P1':evaluate(0)
        with (ctx.dest/'train.jsonl').open('x') as log:
            for step in range(1,steps+1):
                ctx.tick();opt.zero_grad(set_to_none=True);terms=[]
                for item in plan[(step-1)*4:step*4]:
                    if item['task']!='A':raise RuntimeError('Classification schedule contains localization')
                    k=item['image_id'];row=ctx.rows[k];inputs,_=model.prompt(ctx.data.image(k),'A');features=model.encode(inputs)
                    scored=model.score(inputs,features,answer=row['pathology'],include_eos=True);loss=-scored['mean'];(loss/4).backward()
                    terms.append({'label':row['pathology'],'total':float(loss.detach()),'label_loss':float(-scored['tokens'][:-1].mean().detach()),'eos':float(-scored['tokens'][-1].detach())})
                grad=torch.nn.utils.clip_grad_norm_([p for p in model.model.parameters() if p.requires_grad],ctx.cfg['gradient_clip'])
                if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
                opt.step();done=step
                record={'step':step,'supervised_loss':float(np.mean([t['total'] for t in terms])),'eos_loss':float(np.mean([t['eos'] for t in terms])),
                        'label_loss':{label:float(np.mean([t['label_loss'] for t in terms if t['label']==label])) if any(t['label']==label for t in terms) else None for label in LABELS},
                        'gradient_norm':float(grad),'label_exposures':{label:sum(t['label']==label for t in terms) for label in LABELS},'elapsed_s':time.time()-ctx.started}
                log.write(json.dumps(record)+'\n');log.flush()
                if step in checkpoints:
                    optimizer_snapshot(model,opt,ctx.dest/f'step_{step:04d}',step);evaluate(step)
                    print(json.dumps({'stage':ctx.stage,'step':step,'evaluation':evaluations[str(step)]}),flush=True)
                    if ctx.stage=='P1' and passed:break
                    per_step=(time.time()-ctx.started)/step
                    estimate=per_step*(steps-step)
                    save(ctx.dest/'throughput.json',{'measured_steps':step,'wall_seconds_per_step_including_evaluation':per_step,'estimated_remaining_stage_seconds':estimate,'budget_remaining_seconds':ctx.deadline-time.time()})
                    if estimate>ctx.deadline-time.time()-30:raise BudgetStop('Measured training throughput exceeds remaining budget')
    except BudgetStop:
        optimizer_snapshot(model,opt,ctx.dest/f'budget_stop_{done:04d}',done)
        ctx.finish({'status':'stopped_budget','reason':'cumulative budget limit','actual_samples':len({r['image_id'] for r in plan[:done*4]}),'steps':done,'evaluations':evaluations,'scientific_passed':False})
        raise
    result={'status':'completed' if ctx.stage!='P1' or passed else 'failed','reason':'P1 gate passed' if passed else 'P1 15/16-per-class consecutive-checkpoint criterion not met' if ctx.stage=='P1' else 'fixed final step reached',
            'actual_samples':len({r['image_id'] for r in plan[:done*4]}),'steps':done,'scientific_passed':passed if ctx.stage=='P1' else None,
            'evaluations':evaluations,'schedule_exposures':exposure_summary(plan[:done*4],ctx.frame,ctx.regions),'trainable_parameters':5046272,
            'base_initialization':'fresh B0 LoRA seed 17; no P1/B1 continuation','initial_adapter_digest':identity['adapter_parameter_digest'],
            'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3}
    return ctx.finish(result)


def p2_summary(ctx):
    all_scores={};models={};eligible=[]
    for name in ['SFT-N','SFT-B']:
        if ctx.statuses[name]['status']!='completed':raise FileNotFoundError('Both P2 trials must complete')
        for step in [128,256,512]:
            records=json.loads((ctx.root/name/f'step_{step:04d}_scores.json').read_text());key=name+'@'+str(step)
            all_scores[key]=records;models[key]=summarize_scores(records,ctx.folds)
        metric=models[name+'@512']['crossfit'];raw=models[name+'@512']['raw_0_5']
        if metric and metric['specificity']>=.85 and metric['sensitivity']>=.5:eligible.append((name+'@512',raw['ap'],raw['auroc']))
    selected=[]
    if eligible:
        best=max((r[1],r[2]) for r in eligible);selected=[r[0] for r in eligible if (r[1],r[2])==best]
    save(ctx.root/'working_candidates.json',{'models':selected,'rule':'step512 CF specificity>=.85 sensitivity>=.50; then AP, AUROC; retain exact ties','scope':'development selection, not independent validation'})
    intervals=paired_bootstrap(all_scores,ctx.folds,ctx.cfg['bootstrap_repeats'],ctx.cfg['bootstrap_seed'])
    return ctx.finish({'status':'completed','actual_samples':256,'steps':0,'models':models,'selected':selected,'bootstrap':intervals,'primary_comparison':'SFT-N@512 versus SFT-B@512; intermediate checkpoints are descriptive only'})


def working_models(ctx):
    p=ctx.root/'working_candidates.json'
    return json.loads(p.read_text())['models'] if p.exists() else []


def mean_interval(values,repeats,seed):
    a=np.asarray(values,float)
    if len(a)==0:return {'n':0,'mean':None,'ci95':None,'valid_replicates':0,'requested_replicates':repeats}
    rng=np.random.default_rng(seed);samples=np.mean(a[rng.integers(0,len(a),(repeats,len(a)))],axis=1)
    return {'n':len(a),'mean':float(a.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'valid_replicates':repeats,'requested_replicates':repeats}


def region_case(ctx,model,k):
    row=ctx.rows[k];region=ctx.regions[k];image=ctx.data.image(k);inputs,grid=model.prompt(image);features=model.encode(inputs)
    if list(grid)!=region['grid']:raise ValueError('Region grid mismatch')
    original=model.classes(inputs,features,labels=LABELS)[0].float().cpu().numpy();y=LABELS.index(row['pathology']);p=float(torch.tensor(original).softmax(-1)[1]);pred=int(p>=.5)
    result={'image_id':k,'case_id':row['case_id'],'split':row['split'],'y':y,'prediction':pred,'correct':pred==y,
            'original':original.tolist(),'clinical_review':'pending','wrong_evidence_token_iou':region['wrong_evidence_token_iou']}
    for kind in ['feature','pixel']:
        values={}
        for name,item in [('evidence',region['evidence']),*[(f'control_{i}',c) for i,c in enumerate(region['controls'])],('wrong',region['wrong'])]:
            ctx.tick()
            if kind=='feature':inp=inputs;feat=replace_features(features,item['tokens'],item['source'])
            else:inp,_=model.prompt(pixel_blur(image,item['box'],ctx.cfg['P3']['pixel_blur_radius']));feat=model.encode(inp)
            values[name]=model.classes(inp,feat,labels=LABELS)[0].float().cpu().numpy()
        de=original-values['evidence'];dn=np.mean([original-values[f'control_{i}'] for i in range(3)],axis=0);dw=original-values['wrong']
        support={}
        for label,index in [('ground_truth',y),('predicted_label',pred)]:
            support[label]={'D_E':float(de[index]),'D_N':float(dn[index]),'M':float(de[index]-dn[index]),'D_wrong':float(dw[index]),'D_E_minus_wrong':float(de[index]-dw[index])}
        support['yes_no_margin_drop']={'D_E':float(de[1]-de[0]),'D_N':float(dn[1]-dn[0]),'M':float((de[1]-de[0])-(dn[1]-dn[0])),
                                      'D_wrong':float(dw[1]-dw[0]),'D_E_minus_wrong':float((de[1]-de[0])-(dw[1]-dw[0]))}
        result[kind]={'support':support,'intervention_label_scores':{name:value.tolist() for name,value in values.items()}}
    return result


def p3(ctx):
    names=['B0','B1']+working_models(ctx);all_records={};ids=[k for k in sorted(ctx.regions) if ctx.regions[k]['eligible']]
    if not ids:return ctx.finish({'status':'blocked','reason':'No cases pass frozen geometric checks','actual_samples':0,'steps':0,'clinical_review':'pending'})
    run_names=[] if ctx.filter=='aggregate' else [ctx.filter] if ctx.filter else names
    for name in run_names:
        model=ctx.model(name);records=[]
        with frozen(model):
            for k in ids:
                records.append(region_case(ctx,model,k));save(ctx.dest/(name+'_records.json'),records)
        save(ctx.dest/(name+'_frozen_identity.json'),model.v2_identity_check);all_records[name]=records
        del model;gc.collect();torch.cuda.empty_cache()
    if ctx.filter and ctx.filter!='aggregate':
        return ctx.finish({'status':'completed','actual_samples':len(ids),'steps':0,'model':ctx.filter,'clinical_review':'pending'})
    if ctx.filter=='aggregate':
        all_records={name:json.loads((ctx.dest/(name+'_records.json')).read_text()) for name in names}
    summary={'status':'completed','actual_samples':len(ids),'steps':0,'clinical_review':'pending','analysis_scope':'geometry-only exploratory; no semantic approval',
             'primary_metric':'ground_truth D_E_minus_wrong, separately feature and pixel','models':{},'paired_differences':{}}
    repeats=ctx.cfg['bootstrap_repeats'];seed=ctx.cfg['bootstrap_seed'];fields=['D_E','D_N','M','D_wrong','D_E_minus_wrong']
    for split in ['train','validation']:
        common={r['image_id'] for r in all_records[names[0]] if r['split']==split}
        for rows in all_records.values():common &= {r['image_id'] for r in rows if r['correct']}
        for name,rows in all_records.items():
            selected=[r for r in rows if r['split']==split];report={'n':len(selected),'common_correct_n':len(common),'statistics':{},'common_correct_supplement':{}}
            for kind in ['feature','pixel']:
                for support in ['ground_truth','predicted_label','yes_no_margin_drop']:
                    for field in fields:
                        key='/'.join([kind,support,field]);report['statistics'][key]=mean_interval([r[kind]['support'][support][field] for r in selected],repeats,seed)
                        report['common_correct_supplement'][key]=mean_interval([r[kind]['support'][support][field] for r in selected if r['image_id'] in common],repeats,seed)
            report['wrong_evidence_token_iou']={'mean':float(np.mean([r['wrong_evidence_token_iou'] for r in selected])) if selected else None,'meaning':'fixed half-box-width shift, potentially overlapping pathology; not normal lung'}
            summary['models'][name+'/'+split]=report
        for i,first in enumerate(names):
            a={r['image_id']:r for r in all_records[first] if r['split']==split}
            for second in names[i+1:]:
                b={r['image_id']:r for r in all_records[second] if r['split']==split};pair={}
                if set(a)!=set(b):raise ValueError('Region comparison must be paired')
                for kind in ['feature','pixel']:
                    for field in fields:pair[kind+'/'+field]=mean_interval([b[k][kind]['support']['ground_truth'][field]-a[k][kind]['support']['ground_truth'][field] for k in sorted(a)],repeats,seed)
                summary['paired_differences'][second+' minus '+first+'/'+split]=pair
    return ctx.finish(summary)


def ced_label_cache(ctx,model,inputs,features,region,target):
    original=model.classes(inputs,features,labels=LABELS)[0]
    evidence=model.classes(inputs,replace_features(features,region['evidence']['tokens'],region['evidence']['source']),labels=LABELS)[0]
    controls=torch.stack([model.classes(inputs,replace_features(features,c['tokens'],c['source']),labels=LABELS)[0] for c in region['controls']])
    cache={}
    for index,label in enumerate(LABELS):
        de=original[index]-evidence[index];dn=original[index]-controls[:,index]
        reward,diagnostics=ced_reward(float(label==target),de,dn,eps=ctx.cfg['P4']['ced_epsilon'])
        cache[label]={'reward':float(reward),'D_E':float(de),'D_N':dn.cpu().tolist(),'raw_M':float(de-dn.mean()),**diagnostics}
    return cache


def summarize_groups(groups,variant):
    subsets={'all':groups}
    for stratum in sorted({r['stratum'] for r in groups}):subsets[stratum]=[r for r in groups if r['stratum']==stratum]
    result={}
    for key,rows in subsets.items():
        summaries=[r[variant] for r in rows];usable=[r for r in summaries if r['usable']];noncollapse=[r for r in usable if r['delta_A']>=1e-4]
        near=sum(r['delta_A']<1e-4 for r in usable);origins={}
        for r in summaries:origins[r['difference_origin']]=origins.get(r['difference_origin'],0)+1
        result[key]={'groups':len(rows),'patients':len({r['case_id'] for r in rows}),'answers':sum(len(r['answers']) for r in rows),
                     'legal_answers':sum(r['legal_answers'] for r in rows),'correct_answers':sum(r['correct_answers'] for r in rows),
                     'all_same_answer_groups':sum(r['different_normalized_answers']==1 for r in rows),
                     'all_same_answer_fraction':sum(r['different_normalized_answers']==1 for r in rows)/len(rows) if rows else None,
                     'zero_advantage_groups':sum(r['zero_advantage'] for r in summaries),'zero_advantage_fraction':sum(r['zero_advantage'] for r in summaries)/len(rows) if rows else None,
                     'near_zero_std_groups':sum(r['near_zero_std'] for r in summaries),'ordering_reversals':sum(r['ordering_reversed'] for r in summaries),
                     'usable_groups':len(usable),'delta_A_lt_1e4_groups':near,'delta_A_lt_1e4_fraction':near/len(usable) if usable else None,
                     'difference_origins':origins,'remaining_usable_differences':len(noncollapse),
                     'stop_criterion_met':bool(usable and near/len(usable)>.95 and all(r['difference_origin'] in ['normalization_epsilon_or_float','ordering_reversal'] for r in noncollapse)),
                     'status':'exploratory' if usable else 'no_usable_groups'}
    return result


def annotate_advantage_origin(result):
    c=np.asarray(result['correctness_rewards'],float);r=np.asarray(result['ced_rewards'],float)
    if not np.isin(c,[0,1]).all():origin='illegal_format'
    elif result['ordering_reversed']:origin='ordering_reversal'
    elif result['near_zero_std']:origin='constant_or_near_zero_reward'
    elif result['all_legal_mixed']:
        exact_c=(c-c.mean())/c.std();exact_r=(r-r.mean())/r.std();delta=float(np.max(np.abs(exact_c-exact_r)))
        result['without_epsilon_float64_delta']=delta
        origin='normalization_epsilon_or_float' if delta<1e-9 else 'unexplained'
    else:origin='no_mixed_correctness'
    result['difference_origin']=origin
    return result


def p4(ctx):
    if ctx.statuses['P0']['status']!='completed':raise PermissionError('P0 scoring/determinism must pass before P4')
    names=['B1']+working_models(ctx);models={};p4cfg=ctx.cfg['P4'];selections=ctx.subsets['P4']
    if ctx.filter=='aggregate':
        for name in names:models.update(json.loads((ctx.dest/('summary_'+name+'.json')).read_text())['models'])
        return ctx.finish({'status':'completed','actual_samples':len(selections),'steps':0,'models':models,'working_candidates':working_models(ctx),'clinical_review':'pending','RL_run':False})
    for name in ([ctx.filter] if ctx.filter else names):
        model=ctx.model(name);groups=[];case_diagnostics=[];destination=ctx.dest/name;destination.mkdir(exist_ok=True)
        with frozen(model):
            deterministic=determinism_check(ctx,model)
            with (destination/'samples.jsonl').open('x') as output:
                for selection in selections:
                    k=selection['image_id'];row=ctx.rows[k];ctx.tick();image=ctx.data.image(k);inputs,grid=model.prompt(image);features=model.encode(inputs)
                    region=ctx.regions.get(k,{});eligible=bool(region.get('eligible',False));target=row['pathology']
                    cache=ced_label_cache(ctx,model,inputs,features,region,target) if eligible else {}
                    case_diagnostics.append({'image_id':k,'case_id':row['case_id'],'stratum':selection['stratum'],'eligible_geometry':eligible,'clinical_review':'pending','labels':cache})
                    for group in range(p4cfg['groups']):
                        correct_rewards=[];legacy=[];revised=[];answers=[]
                        for slot in range(p4cfg['group_size']):
                            ctx.tick();sample_seed=selection['sampling_seeds'][group*p4cfg['group_size']+slot];torch.manual_seed(sample_seed);torch.cuda.manual_seed_all(sample_seed)
                            answer,tokens=model.generate(inputs,features,sample=True,sampling_config=p4cfg['generation']);normalized=answer.strip();c=correctness(answer,target)
                            if c<0:old_reward=new_reward=-.1
                            elif eligible:old_reward=new_reward=cache[normalized]['reward']
                            else:old_reward=0.;new_reward=c
                            correct_rewards.append(c);legacy.append(old_reward);revised.append(new_reward)
                            eos=model.last_generation_settings['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
                            record={'image_id':k,'case_id':row['case_id'],'model':name,'stratum':selection['stratum'],'group':group,'slot':slot,'seed':sample_seed,
                                    'text':answer,'tokens':tokens,'normalized_label':normalized if normalized in LABELS else None,'invalid_format':normalized not in LABELS,
                                    'truncated':bool(len(tokens)>=p4cfg['generation']['max_new_tokens'] and (not tokens or tokens[-1] not in eos)),
                                    'correctness_reward':c,'legacy_B4_reward':old_reward,'revised_CED_reward':new_reward}
                            output.write(json.dumps(record)+'\n');output.flush();answers.append(normalized)
                        variants={}
                        for label,rewards in [('correctness_only',correct_rewards),('legacy_B4',legacy),('revised_CED',revised)]:
                            detail=reward_group(correct_rewards,rewards,p4cfg['advantage_epsilon'],p4cfg['near_zero_std'])
                            # Check the numeric audit against the actual historical normalization implementation.
                            actual=advantages(torch.tensor(rewards)).cpu().numpy()
                            if not np.allclose(actual,detail['ced_advantage'],atol=1e-6,rtol=1e-6):raise RuntimeError('Advantage implementation mismatch')
                            variants[label]=annotate_advantage_origin(detail)
                        groups.append({'image_id':k,'case_id':row['case_id'],'stratum':selection['stratum'],'group':group,'answers':answers,
                                       'legal_answers':sum(c>=0 for c in correct_rewards),'different_labels':len(set(answers)&set(LABELS)),
                                       'different_normalized_answers':len(set(answers)),'correct_answers':sum(c==1 for c in correct_rewards),**variants})
                        save(destination/'groups.json',groups)
                    elapsed=time.time()-ctx.started
                    done=len(groups);total=len(selections)*p4cfg['groups']
                    estimate=elapsed/done*(total-done)
                    save(destination/'throughput.json',{'groups_measured':done,'elapsed_seconds':elapsed,'estimated_remaining_model_sampling_seconds':estimate,'budget_remaining_seconds':ctx.deadline-time.time()})
                    if estimate>ctx.deadline-time.time()-30:raise BudgetStop('Measured sampling throughput exceeds remaining budget')
            effective={k:model.last_generation_settings[k] for k in ['temperature','top_p','top_k','max_new_tokens','do_sample','eos_token_id','pad_token_id','use_cache']}
            if any(effective[k]!=v for k,v in p4cfg['generation'].items()):raise RuntimeError('Effective generation settings differ from frozen config')
            save(destination/'effective_generation.json',effective);save(destination/'ced_diagnostics.json',case_diagnostics)
        save(destination/'frozen_identity.json',model.v2_identity_check)
        diagnostics=[v for r in case_diagnostics for v in r['labels'].values()]
        variants={label:summarize_groups(groups,label) for label in ['correctness_only','legacy_B4','revised_CED']}
        relevant=variants['revised_CED'].get('geometry_only_positive',{})
        models[name]={'n_cases':len(selections),'groups':len(groups),'answers':sum(len(r['answers']) for r in groups),'semantic_reviewed_cases':0,
                      'analysis_scope':'geometric-only CED subanalysis; ineligible fallback analyzed separately','determinism':deterministic,'effective_generation':effective,
                      'frozen_parameters_unchanged':model.v2_identity_check['unchanged'],'variants':variants,
                      'ced_summary':{'case_label_pairs':len(diagnostics),'control_std_mean':float(np.mean([r['std'] for r in diagnostics])) if diagnostics else None,
                                     'near_zero_control_std':sum(r['std']<p4cfg['near_zero_std'] for r in diagnostics),'gate_mean':float(np.mean([r['gate'] for r in diagnostics])) if diagnostics else None,
                                     'saturated_pairs':sum(r['saturated'] for r in diagnostics),'saturation_rate':sum(r['saturated'] for r in diagnostics)/len(diagnostics) if diagnostics else None},
                      'branch_recommendation':('stop_current_isolated_yes_no_CED_GRPO_branch_exploratory' if relevant.get('stop_criterion_met') else 'no_usable_groups' if not relevant.get('usable_groups') else 'inspect_differences_before_any_posttraining'),
                      'decision_denominator':'Report all qualifying groups and geometric-evidence qualifying groups separately; fallback equality alone cannot establish evidence-specific collapse.'}
        del model;gc.collect();torch.cuda.empty_cache()
    return ctx.finish({'status':'completed','actual_samples':len(selections),'steps':0,'models':models,'working_candidates':working_models(ctx),'clinical_review':'pending','RL_run':False})


def main():
    ctx=Context()
    if ctx.stage=='P0':result=p0(ctx)
    elif ctx.stage in ['P1','SFT-N','SFT-B']:result=train(ctx)
    elif ctx.stage=='P2-summary':result=p2_summary(ctx)
    elif ctx.stage=='P3':result=p3(ctx)
    elif ctx.stage=='P4':result=p4(ctx)
    elif ctx.stage=='R0':result=ctx.finish({'status':'blocked','reason':ctx.cfg['R0']['reason'],'actual_samples':0,'steps':0})
    else:raise ValueError(ctx.stage)
    print(json.dumps({'stage':ctx.stage,'status':result['status'],'steps':result.get('steps',0),'elapsed_s':result['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
