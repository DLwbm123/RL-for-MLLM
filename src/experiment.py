"""Pilot execution. Private samples stay on the data disk; the test split is sealed."""
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import torch

from src.data import LABELS, transform_box, write_json, sha
from src.evaluation import binary_metrics, cluster_bootstrap, box_iou, select_checkpoint,fit_calibration
from src.model import Model
from src.objectives import correctness, ced_reward, advantages, grpo_loss, evidence_loss, stability_loss
from src.regions import replace_features, pixel_blur


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def context():
    root=Path(os.environ['OUTPUT_ROOT']);data=Path(os.environ['DATA_ROOT']);model=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    config=json.loads(Path(os.environ.get('RUN_CONFIG','configs/pilot.json')).read_text())
    if config.get('dataset','busbra') != os.environ.get('DATASET_NAME','busbra'):raise ValueError('Dataset/config mismatch')
    if config['mode']!='pilot':raise ValueError('Only the authorized pilot can execute through this entry')
    frame=pd.read_csv(root/'protocol/manifest.csv',keep_default_na=False,dtype={'case_id':str,'duplicate_group':str})
    regions=json.loads((root/'protocol/regions.json').read_text())
    lock=json.loads((root/'protocol/protocol_lock.json').read_text())
    for name,digest in lock['hashes'].items():
        if sha((root/'protocol'/name).read_bytes())!=digest:raise ValueError('Protocol lock mismatch '+name)
    if lock.get('config_sha256')!=sha(Path(os.environ.get('RUN_CONFIG','configs/pilot.json')).read_bytes()):
        raise ValueError('Config must be frozen before execution')
    return root,data,model,config,frame,regions


def fixed_cases(frame,split,limit,seed=42):
    subset=frame.loc[frame.split==split].sort_values('image_id').drop_duplicates('case_id')
    if len(subset)<=limit:return subset
    from sklearn.model_selection import train_test_split
    selected,_=train_test_split(subset,train_size=limit,stratify=subset.pathology,random_state=seed)
    return selected.sort_values('case_id')


def sample_image(data,row):
    return Image.open(data/row['image_path']).convert('RGB')


def get_model(path,config,adapter=None,trainable=False):
    seed_all(config['seed'])
    return Model(path,adapter,trainable,config['min_visual_tokens'],config['max_visual_tokens'])


def as_values(scores):
    mean,details=scores
    return {'mean':mean.detach().float().cpu().tolist(),
            'sum':[float(x['sum'].detach()) for x in details],
            'length':[x['length'] for x in details],
            'probabilities':mean.detach().float().softmax(-1).cpu().tolist()}


def intervention_scores(model,image,inputs,features,region,kind):
    if kind=='feature':
        feat=replace_features(features,region['tokens'],region['source'])
        return model.classes(inputs,feat)
    modified=pixel_blur(image,region['box'])
    inp,_=model.prompt(modified)
    feat=model.encode(inp)
    return model.classes(inp,feat)


@torch.no_grad()
def original_validation(model,frame,data,localize=False):
    before=model.model.training;model.set_training(False)
    records=[]
    for row in frame.loc[frame.split=='validation'].to_dict('records'):
        image=sample_image(data,row);inputs,_=model.prompt(image);features=model.encode(inputs)
        scores=as_values(model.classes(inputs,features));y=LABELS.index(row['pathology'])
        record={'image_id':row['image_id'],'case_id':row['case_id'],'y':y,'p':scores['probabilities'][1],**scores}
        if localize and json.loads(row['bbox_xyxy']) is not None:
            loc,_=model.prompt(image,'B');answer,_=model.generate(loc,features,max_tokens=64,sample=False)
            record.update(bbox_prediction=answer,bbox_iou=box_iou(answer,transform_box(json.loads(row['bbox_xyxy']),row['width'],row['height'])))
        records.append(record)
    model.set_training(before)
    metrics=binary_metrics([r['y'] for r in records],[r['p'] for r in records])
    return metrics,records


def audit(tag='B0'):
    started=time.time()
    root,data,path,config,frame,regions=context()
    adapter=None if tag=='B0' else root/tag/json.loads((root/tag/'selected.json').read_text())['checkpoint']
    model=get_model(path,config,adapter)
    selected=fixed_cases(frame,'validation',config['audit_cases'])
    destination=root/'audits'/tag;destination.mkdir(parents=True,exist_ok=True)
    selected[['image_id','case_id','pathology']].to_csv(destination/'selection.csv',index=False)
    records=[]
    with (destination/'predictions.jsonl').open('w') as f,torch.no_grad():
        for i,row in enumerate(selected.to_dict('records')):
            image=sample_image(data,row);inputs,grid=model.prompt(image);features=model.encode(inputs)
            original=as_values(model.classes(inputs,features));y=LABELS.index(row['pathology'])
            rec={'image_id':row['image_id'],'case_id':row['case_id'],'y':y,'p':original['probabilities'][1],
                 'original':original,'correct':int(np.argmax(original['mean']))==y,'mask_area_ratio':row['mask_area_ratio'],
                 'eligible':bool(regions.get(row['image_id'],{}).get('eligible',False))}
            if rec['eligible']:
                region=regions[row['image_id']]
                if list(grid)!=region['grid']:raise ValueError('Region grid does not match processor')
                rec['evidence_tokens']=len(region['evidence']['tokens']);rec['matching']=region['matching']
                for kind in ['feature','pixel']:
                    evidence=as_values(intervention_scores(model,image,inputs,features,region['evidence'],kind))
                    controls=[as_values(intervention_scores(model,image,inputs,features,x,kind)) for x in region['controls']]
                    wrong=as_values(intervention_scores(model,image,inputs,features,region['wrong'],kind))
                    de=original['mean'][y]-evidence['mean'][y]
                    dn=[original['mean'][y]-x['mean'][y] for x in controls]
                    dw=original['mean'][y]-wrong['mean'][y]
                    rec[kind]={'evidence':evidence,'controls':controls,'wrong':wrong,'D_E':de,'D_N':float(np.mean(dn)),
                               'M':de-float(np.mean(dn)),'D_wrong':dw,'evidence_minus_wrong':de-dw,
                               'control_flip_rate':float(np.mean([np.argmax(x['mean'])!=np.argmax(original['mean']) for x in controls]))}
            # Blank/wrong-image diagnostics fixed by selection order, never by correctness.
            if i<8:
                equal,_=model.prompt(image,'A_equal_tokens')
                if [len(model.processor.tokenizer.encode(x,add_special_tokens=False)) for x in ['A','B']]!=[1,1]:raise ValueError('Equal-token diagnostic is not equal length')
                rec['equal_token_format_diagnostic']=as_values(model.classes(equal,features,labels=['A','B']))
                blank=Image.new('RGB',image.size,0);bi,_=model.prompt(blank);bf=model.encode(bi)
                rec['blank']=as_values(model.classes(bi,bf))
                other=selected.iloc[(i+1)%len(selected)].to_dict()
                assert other['case_id']!=row['case_id']
                oi=sample_image(data,other);pi,_=model.prompt(oi);pf=model.encode(pi)
                rec['wrong_image']=as_values(model.classes(pi,pf));rec['wrong_image_case_id']=other['case_id']
            # Independent localization call; masks/boxes never enter class prediction.
            if json.loads(row['bbox_xyxy']) is not None:
                loc,_=model.prompt(image,'B');answer,_=model.generate(loc,features,max_tokens=config['max_bbox_tokens'],sample=False)
                rec['bbox_prediction']=answer;rec['bbox_iou']=box_iou(answer,transform_box(json.loads(row['bbox_xyxy']),row['width'],row['height']))
            f.write(json.dumps(rec)+'\n');f.flush();records.append(rec)
            print(json.dumps({'stage':'audit','tag':tag,'done':i+1,'total':len(selected),'elapsed_s':time.time()-started}),flush=True)
    localized=[r for r in records if 'bbox_iou' in r]
    summary={'tag':tag,'n_cases':len(records),'n_eligible':sum(r['eligible'] for r in records),
             'classification':binary_metrics([r['y'] for r in records],[r['p'] for r in records]),
             'n_localization':len(localized),'bbox_mean_iou':float(np.mean([r['bbox_iou'] for r in localized])) if localized else None,
             'bbox_iou_ge_05':float(np.mean([r['bbox_iou']>=.5 for r in localized])) if localized else None,
             'joint_correct_localization':float(np.mean([r['correct'] and r['bbox_iou']>=.5 for r in localized])) if localized else None,
             'elapsed_s':time.time()-started,'gpu_hours':(time.time()-started)/3600,
             'forward_count':model.forward_count,'vision_count':model.vision_count,
             'score_count':model.score_count,'generated_tokens':model.generated_tokens,'generation_calls':model.generation_calls,
             'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3}
    for kind in ['feature','pixel']:
        eligible=[{'case_id':r['case_id'],'class':r['y'],'correct':r['correct'],'area':r['mask_area_ratio'],'tokens':r['evidence_tokens'],
                   **{k:r[kind][k] for k in ['D_E','D_N','M','D_wrong','evidence_minus_wrong','control_flip_rate']}} for r in records if r['eligible']]
        if not eligible:
            summary[kind]={'n':0,'status':'no_eligible_regions'};continue
        df=pd.DataFrame(eligible);stats={'n':len(df)}
        for k in ['D_E','D_N','M','D_wrong','evidence_minus_wrong','control_flip_rate']:
            stats[k]={'mean':float(df[k].mean()),**cluster_bootstrap(df,lambda d: d[k].mean(),config['bootstrap_repeats'])}
        stats['Pr_M_positive']=float((df.M>0).mean())
        stats['correct_subset']={'n':int(df.correct.sum()),'M':float(df.loc[df.correct,'M'].mean()) if df.correct.any() else None}
        stats['by_class']={str(c):{'n':len(g),'M':float(g.M.mean()),'D_E':float(g.D_E.mean())} for c,g in df.groupby('class')}
        for c in ['area','tokens']:
            rho=spearmanr(df[c],df.M).statistic if df[c].nunique()>1 and df.M.nunique()>1 else np.nan
            stats['M_'+c+'_spearman']=float(rho) if np.isfinite(rho) else None
        df['size_bin']=pd.qcut(df.area,min(4,df.area.nunique()),duplicates='drop').astype(str)
        stats['by_size']={str(c):{'n':len(g),'M':float(g.M.mean())} for c,g in df.groupby('size_bin')}
        summary[kind]=stats
    write_json(destination/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


def schedule(frame,steps,batch,seed,limit=None,task_pattern=None):
    task_pattern=['A','A','A','B'] if task_pattern is None else task_pattern
    if not task_pattern or not set(task_pattern)<= {'A','B'}:raise ValueError('Invalid task pattern')
    train=frame.loc[frame.split=='train'].copy()
    if limit is not None:
        cases=fixed_cases(frame,'train',limit,seed).case_id
        train=train[train.case_id.isin(cases)]
    groups={k:v.to_dict('records') for k,v in train.groupby('case_id')};rng=np.random.default_rng(seed)
    cases=sorted(groups);plan=[];order=[]
    for i in range(steps*batch):
        if not order:order=list(rng.permutation(cases))
        case=order.pop();rows=groups[case];task=task_pattern[i%len(task_pattern)]
        if task=='B':
            rows=[r for r in rows if json.loads(r['bbox_xyxy']) is not None and (os.environ.get('DATASET_NAME')=='rsna' or r['mask_components']==1)]
            if not rows:
                alternatives=[c for c in cases if any(json.loads(r['bbox_xyxy']) is not None and (os.environ.get('DATASET_NAME')=='rsna' or r['mask_components']==1) for r in groups[c])]
                case=alternatives[int(rng.integers(len(alternatives)))];rows=[r for r in groups[case] if json.loads(r['bbox_xyxy']) is not None and (os.environ.get('DATASET_NAME')=='rsna' or r['mask_components']==1)]
        row=rows[int(rng.integers(len(rows)))].copy()
        row['task']=task;plan.append(row)
    return plan


def optimizer_snapshot(model,opt,path,step):
    model.save(path)
    torch.save({'optimizer':opt.state_dict(),'step':step,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},Path(path)/'training_state.pt')


@torch.no_grad()
def fit_diagnostic(model,rows,data):
    before=model.model.training;model.set_training(False);losses=[];records=[]
    for row in rows:
        image=sample_image(data,row);inputs,_=model.prompt(image);features=model.encode(inputs)
        losses.append(float(-model.score(inputs,features,answer=row['pathology'],include_eos=True)['mean']))
        scores=as_values(model.classes(inputs,features));y=LABELS.index(row['pathology'])
        records.append({'image_id':row['image_id'],'case_id':row['case_id'],'y':y,'p':scores['probabilities'][1]})
    model.set_training(before)
    return {'classification_sft_loss':float(np.mean(losses)),'training_metrics':binary_metrics([r['y'] for r in records],[r['p'] for r in records]),'predictions':records}


def train(method='B1',resume=None):
    if os.environ.get('V2_STAGE'):
        raise RuntimeError('Legacy training entry is disabled for diagnostic v2')
    started=time.time()
    root,data,path,cfg,frame,regions=context();dest=root/method;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'train.jsonl').exists() and not resume:
        raise FileExistsError('Existing training attempt: resume a checkpoint or use a new output root; never overwrite provenance')
    if method not in ['shortfit','B1','B2','B3','B4','B5','B6']:raise ValueError(method)
    post=method.startswith('B') and method not in ['B1']
    if post:
        gate=json.loads((root/cfg['post_gate_file']).read_text())
        if gate.get('status')!='passed':raise RuntimeError('Posttraining scientific gate has not passed')
    adapter=root/'B1'/json.loads((root/'B1/selected.json').read_text())['checkpoint'] if post else None
    if resume:adapter=dest/resume
    model=get_model(path,cfg,adapter,trainable=True)
    batch=cfg['gradient_accumulation'];n_cases=frame.loc[frame.split=='train','case_id'].nunique()
    steps=cfg['post_steps'] if post else (cfg['shortfit_steps'] if method=='shortfit' else math.ceil(n_cases*cfg['sft_epochs']/batch))
    plan=schedule(frame,steps,batch,cfg['seed'],cfg['shortfit_cases'] if method=='shortfit' else None,task_pattern=cfg['task_pattern'])
    write_json(dest/'sample_schedule.json',[{'image_id':r['image_id'],'case_id':r['case_id'],'task':r['task']} for r in plan])
    opt=torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad],lr=cfg['post_lr'] if post else cfg['sft_lr'])
    start=0
    if resume:
        saved=torch.load(dest/resume/'training_state.pt',map_location='cpu',weights_only=False)
        opt.load_state_dict(saved['optimizer']);start=saved['step'];random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    candidates=json.loads((dest/'candidates.json').read_text()) if resume and (dest/'candidates.json').exists() else []
    trainable=sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    fit_rows=fixed_cases(frame,'train',cfg['shortfit_cases'],cfg['seed']).to_dict('records') if method=='shortfit' else []
    initial_fit=(json.loads((dest/'initial_fit.json').read_text()) if resume else fit_diagnostic(model,fit_rows,data)) if fit_rows else None
    if initial_fit is not None and not resume:write_json(dest/'initial_fit.json',initial_fit)
    with (dest/(f'train_resume_{time.time_ns()}.jsonl' if resume else 'train.jsonl')).open('w') as log:
        for step in range(start,steps):
            opt.zero_grad(set_to_none=True);stats={'method':method,'step':step+1,'supervised_loss':0.,'evidence_loss':0.,'stability_loss':0.,
                'rl_loss':0.,'rl_zero_variance':None,'rl_unique_answers':None,'aux_used':False}
            for j,row in enumerate(plan[step*batch:(step+1)*batch]):
                image=sample_image(data,row);inputs,grid=model.prompt(image,row['task']);features=model.encode(inputs)
                answer=row['pathology'] if row['task']=='A' else json.dumps(transform_box(json.loads(row['bbox_xyxy']),row['width'],row['height']))
                supervised=-model.score(inputs,features,answer=answer,include_eos=True)['mean']
                (supervised*(cfg['alpha'] if post else 1.)/batch).backward()
                stats['supervised_loss']+=float(supervised.detach())/batch
                if j!=0 or not post:continue
                region=regions.get(row['image_id'],{});eligible=bool(region.get('eligible',False))
                if eligible and list(grid)!=region['grid']:raise ValueError('Training geometry mismatch')
                if method in ['B3','B4','B6']:
                    samples=[];old=[];rewards=[];cache={};reward_diagnostics=[]
                    with torch.no_grad():
                        for _ in range(cfg['group_size']):
                            answer,ids=model.generate(inputs,features,cfg['max_answer_tokens'],sample=True,sampling_config={'temperature':cfg['sampling_temperature'],'top_p':cfg['sampling_top_p'],'top_k':cfg.get('sampling_top_k',0),'max_new_tokens':cfg['max_answer_tokens']})
                            samples.append((answer,ids));old.append(model.score(inputs,features,ids=ids)['tokens'].detach())
                            c=correctness(answer,row['pathology']);reward=c
                            if method=='B4':
                                key=answer.strip()
                                if c<0:reward=-.1
                                elif not eligible:reward=0.
                                elif key in cache:reward=cache[key]
                                else:
                                    original=model.score(inputs,features,answer=key)['mean']
                                    evidence=model.score(inputs,replace_features(features,region['evidence']['tokens'],region['evidence']['source']),answer=key)['mean']
                                    negatives=torch.stack([model.score(inputs,replace_features(features,r['tokens'],r['source']),answer=key)['mean'] for r in region['controls']])
                                    reward,diag=ced_reward(c,original-evidence,original-negatives);reward=float(reward);cache[key]=reward;reward_diagnostics.append(diag)
                            rewards.append(reward)
                    adv=advantages(torch.tensor(rewards,device='cuda'));stats['rl_zero_variance']=bool((adv==0).all());stats['rl_unique_answers']=len({a.strip() for a,_ in samples})
                    stats['reward_std']=float(np.std(rewards));stats['rewards']=rewards;stats['ced']=reward_diagnostics
                    if not stats['rl_zero_variance']:
                        for k,(_,ids) in enumerate(samples):
                            now=model.score(inputs,features,ids=ids)['tokens'];loss=grpo_loss(now,old[k],adv[k])/cfg['group_size']
                            loss.backward();stats['rl_loss']+=float(loss.detach())
                if method in ['B5','B6'] and eligible:
                    original,_=model.classes(inputs,features)
                    evidence,_=model.classes(inputs,replace_features(features,region['evidence']['tokens'],region['evidence']['source']))
                    controls=torch.stack([model.classes(inputs,replace_features(features,r['tokens'],r['source']))[0] for r in region['controls']])
                    aux,_=evidence_loss(original,evidence,controls,LABELS.index(row['pathology']),delta_rel=cfg['delta_rel'],delta_abs=cfg['delta_abs'])
                    stability=stability_loss(original,controls)
                    (cfg['beta']*aux+cfg['gamma']*stability).backward()
                    stats.update(evidence_loss=float(aux.detach()),stability_loss=float(stability.detach()),aux_used=True)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.model.parameters() if p.requires_grad],cfg['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            opt.step();stats.update(gradient_norm=float(norm),elapsed_s=time.time()-started,
                peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,forward_count=model.forward_count,vision_count=model.vision_count)
            log.write(json.dumps(stats)+'\n');log.flush();print(json.dumps(stats),flush=True)
            checkpoint_step=(step+1 in {max(1,steps//2),steps}) if method=='B1' else (step+1==steps)
            if (step+1)%25==0 and not checkpoint_step:
                optimizer_snapshot(model,opt,dest/f'step_{step+1:04d}',step+1)
            if checkpoint_step:
                name=f'step_{step+1:04d}';optimizer_snapshot(model,opt,dest/name,step+1)
                if method!='shortfit':
                    metrics,predictions=original_validation(model,frame,data)
                    write_json(dest/(name+'_validation.json'),{'metrics':metrics,'predictions':predictions})
                    candidates.append({'step':step+1,'checkpoint':name,'auroc':metrics['auroc']})
                    write_json(dest/'candidates.json',candidates)
    if candidates:
        chosen=select_checkpoint(candidates);write_json(dest/'selected.json',chosen)
        validation=json.loads((dest/(chosen['checkpoint']+'_validation.json')).read_text())
        predictions=validation['predictions'];scores=np.asarray([r['mean'] for r in predictions]);y=np.asarray([r['y'] for r in predictions])
        calibration=fit_calibration(scores,y,'validation')
        from scipy.special import softmax
        calibration['validation_fit_metrics']=binary_metrics(y,softmax(scores/calibration['temperature'],axis=-1)[:,1])
        calibration['uncalibrated_validation_metrics']=validation['metrics']
        calibration['evaluation_warning']='Fitted on this validation set; these are calibration-fit diagnostics, not independent test estimates.'
        write_json(dest/'calibration.json',calibration)
    summary={'method':method,'steps':steps,'case_exposures':len(plan),'trainable_parameters':trainable,
             'elapsed_s':time.time()-started,'gpu_hours':(time.time()-started)/3600,'forward_count':model.forward_count,
             'vision_count':model.vision_count,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,
             'score_count':model.score_count,'generated_tokens':model.generated_tokens,'generation_calls':model.generation_calls,
             'selected':select_checkpoint(candidates) if candidates else None,'reference_KL':0.,'case_aggregation':False}
    if method=='shortfit':
        summary['fit_diagnostic_only']=True
        final_fit=fit_diagnostic(model,fit_rows,data)
        summary['initial_fit']=initial_fit;summary['final_fit']=final_fit
        summary['fit_gate_passed']=bool(final_fit['classification_sft_loss'] < initial_fit['classification_sft_loss']*.9)
        summary['elapsed_s']=time.time()-started;summary['gpu_hours']=summary['elapsed_s']/3600
        summary['forward_count']=model.forward_count;summary['vision_count']=model.vision_count
    write_json(dest/'summary.json',summary)
    return summary


def stage_main():
    action=os.environ['ACTION']
    if action=='audit':audit(os.environ.get('TAG','B0'))
    elif action=='train':train(os.environ.get('METHOD','B1'),os.environ.get('RESUME_CHECKPOINT'))
    else:raise ValueError(action)


if __name__=='__main__':stage_main()
