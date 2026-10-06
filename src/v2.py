"""CPU protocol, sealed development access, and diagnostic statistics for RSNA v2."""
import json
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from src.data import sha, write_json

LABELS = ['no', 'yes']
TRAIN_STAGES = {'P1', 'SFT-N', 'SFT-B'}
STAGES = ['P0', 'P1', 'SFT-N', 'SFT-B', 'P2-summary', 'P3', 'P4', 'R0']


def save(path, value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    write_json(temporary,value);os.replace(temporary,path)


def require_safe_config(cfg):
    if cfg.get('diagnostic_only') is not True or cfg.get('posttraining_enabled') is not False:
        raise ValueError('v2 requires diagnostic_only=true and posttraining_enabled=false')
    if cfg['task_pattern']!=['A'] or cfg['label_order']!=LABELS:
        raise ValueError('v2 classification-only task/label mismatch')
    if cfg['classification_score_eos'] or not cfg['sft_include_eos']:
        raise ValueError('Historical scoring/EOS contract must remain unchanged')
    if cfg.get('max_concurrent_gpus',1)>2 or cfg.get('single_gpu_per_experiment',True) is not True:raise ValueError('Parallelism must preserve single-GPU experiments')
    if cfg['gpu_hours_limit']>6 or cfg['gpu_hours_limit']<=0:
        raise ValueError('Initial GPU budget must be within 6 hours')


def require_stage(stage, statuses, training=False):
    if stage not in STAGES:raise ValueError('Unknown v2 stage')
    if training and stage not in TRAIN_STAGES:raise PermissionError('Inference stage cannot enter training')
    if stage in ['SFT-N','SFT-B']:
        p=statuses.get('P1',{})
        if p.get('status')!='completed' or p.get('scientific_passed') is not True:
            raise PermissionError('P1 has not passed; P2 is blocked')


def select_balanced(frame, positive, negative, seed):
    selected=[]
    for label,n in [('yes',positive),('no',negative)]:
        rows=frame[frame.pathology==label].sort_values('case_id')
        if len(rows)<n:raise ValueError('Insufficient prespecified stratum')
        selected.extend(rows.sample(n=n,random_state=seed).image_id.tolist())
    return selected


def classification_schedule(frame, steps, seed, balanced):
    frame=frame.sort_values('case_id')
    if frame.case_id.duplicated().any():raise ValueError('Expected one predetermined image per patient')
    rng=np.random.default_rng(seed);queues={};by_label={k:frame[frame.pathology==k].image_id.tolist() for k in LABELS}
    all_ids=frame.image_id.tolist();plan=[]
    def take(key, pool):
        if not pool:raise ValueError('Empty class pool')
        if not queues.get(key):queues[key]=list(rng.permutation(pool))
        return str(queues[key].pop())
    for step in range(steps):
        ids=[take(k,by_label[k]) for k in ['no','yes','no','yes']] if balanced else [take('all',all_ids) for _ in range(4)]
        plan.extend({'image_id':k,'task':'A','step':step+1} for k in ids)
    return plan


def exposure_summary(plan, frame, regions):
    lookup=frame.set_index('image_id').to_dict('index');counts=Counter(r['image_id'] for r in plan)
    labels=Counter(lookup[r['image_id']]['pathology'] for r in plan)
    return {'exposures':len(plan),'label_exposures':dict(labels),'task_exposures':dict(Counter(r['task'] for r in plan)),
            'unique_patients':len({lookup[k]['case_id'] for k in counts}),'repeat_exposures':len(plan)-len(counts),
            'unique_by_label':{s:sum(lookup[k]['pathology']==s for k in counts) for s in LABELS},
            'evidence_eligible_exposures':sum(regions.get(r['image_id'],{}).get('eligible',False) for r in plan),
            'min_exposures_per_seen_patient':min(counts.values()),'max_exposures_per_seen_patient':max(counts.values())}


def folds_for(frame,seed):
    rows=frame.sort_values('case_id').reset_index(drop=True);y=(rows.pathology=='yes').astype(int)
    result={}
    for fold,(_,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(rows,y)):
        for i in test:result[rows.iloc[i].image_id]=fold
    return result


def mismatch_permutations(ids, frame, count, seed):
    patients=frame.set_index('image_id').case_id.to_dict();rng=np.random.default_rng(seed);result=[]
    for _ in range(count):
        for attempt in range(10000):
            permutation=list(rng.permutation(ids))
            if all(patients[a]!=patients[b] for a,b in zip(ids,permutation)) and permutation not in result:
                result.append(permutation);break
        else:raise ValueError('Unable to build patient-disjoint mismatch permutation')
    return result


class DevelopmentData:
    """Whitelist IDs and resolved paths, and reject reads of any other dataset file."""
    def __init__(self,root,frame,log=None):
        self.root=Path(root).resolve();self.frame=frame.copy();self.log=Path(log) if log else None
        if not set(frame.split)<= {'train','validation'}:raise PermissionError('Sealed/nondevelopment rows in loader')
        if frame.image_id.duplicated().any() or frame.case_id.duplicated().any():raise ValueError('Duplicate development identities')
        self.rows=frame.set_index('image_id').to_dict('index');self.allowed=set()
        for row in self.rows.values():
            for key in ['image_path','mask_path']:
                p=(self.root/row[key]).resolve()
                if not p.is_relative_to(self.root) or not row[key]:raise PermissionError('Image path escapes authorized dataset')
                self.allowed.add(p)
        self.access_counts=Counter()

    def audit_open(self,event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(self.root) and p not in self.allowed:
            raise PermissionError('Sealed or unauthorized dataset path')

    def install_guard(self):sys.addaudithook(self.audit_open)

    def image(self,image_id):
        if image_id not in self.rows:raise PermissionError('Image outside frozen development whitelist')
        row=self.rows[image_id];p=(self.root/row['image_path']).resolve()
        if p not in self.allowed:raise PermissionError('Unauthorized resolved image')
        with Image.open(p) as image:result=image.convert('RGB')
        self.access_counts[row['split']]+=1
        if self.log:
            self.log.parent.mkdir(parents=True,exist_ok=True)
            with self.log.open('a') as f:f.write(json.dumps({'image_id':image_id,'case_id':row['case_id'],'split':row['split']})+'\n')
        return result


def binary_stats(y,p,pred=None,weights=None):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=.5 if pred is None else np.asarray(pred,bool)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    tn,fp,fn,tp=[float(w[mask].sum()) for mask in [(y==0)&~pred,(y==0)&pred,(y==1)&~pred,(y==1)&pred]]
    sensitivity=tp/(tp+fn) if tp+fn else None;specificity=tn/(tn+fp) if tn+fp else None
    fpos=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    fneg=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    both=bool(tp+fn and tn+fp)
    return {'n':int(w.sum()),'auroc':float(roc_auc_score(y,p,sample_weight=w)) if both else None,
            'ap':float(average_precision_score(y,p,sample_weight=w)) if tp+fn else None,
            'balanced_accuracy':(sensitivity+specificity)/2 if both else None,'macro_f1':(fpos+fneg)/2 if both else None,
            'sensitivity':sensitivity,'specificity':specificity,'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp)}


def specificity_threshold(y,p,weights=None,target=.9):
    """Lowest feasible member of frozen candidate rule; ties move together via nextafter."""
    y=np.asarray(y);p=np.asarray(p,float);w=np.ones(len(y)) if weights is None else np.asarray(weights)
    active=(y==0)&(w>0)
    if not active.any():return None
    order=np.argsort(p[active],kind='stable');values=p[active][order];weights=w[active][order]
    boundary=np.searchsorted(np.cumsum(weights),target*weights.sum(),side='left')
    return float(np.nextafter(values[min(boundary,len(values)-1)],np.inf))


def crossfit(y,p,fold,weights=None,target=.9):
    y=np.asarray(y);p=np.asarray(p);fold=np.asarray(fold);w=np.ones(len(y)) if weights is None else np.asarray(weights)
    pred=np.zeros(len(y),bool);thresholds=[]
    for f in range(5):
        train=fold!=f;test=fold==f;t=specificity_threshold(y[train],p[train],w[train],target)
        if t is None:return None,None
        pred[test]=p[test]>=t;thresholds.append(t)
    return pred,thresholds


def summarize_scores(records,fold_map):
    y=np.array([r['y'] for r in records]);p=np.array([r['p'] for r in records]);fold=np.array([fold_map[r['image_id']] for r in records])
    pred,t=crossfit(y,p,fold)
    summary={'raw_0_5':binary_stats(y,p),'crossfit':binary_stats(y,p,pred) if pred is not None else None,'fold_thresholds':t,
             'auprc_definition':'sklearn average_precision_score (AP)','softmax_interpretation':'candidate-label normalized support; not calibrated disease probability',
             'tie_policy':'p>=threshold predicts yes; historical score argmax ties predict no','score_distributions':{}}
    for label in [0,1]:
        subset=[r for r in records if r['y']==label]
        summary['score_distributions'][LABELS[label]]={'n':len(subset),**{key:{'mean':float(np.mean([r[key] for r in subset])),'quantiles_0_25_50_75_100':np.quantile([r[key] for r in subset],[0,.25,.5,.75,1]).tolist()} for key in ['score_no','score_yes','margin','p']}}
    summary['exact_score_ties']=sum(r['score_no']==r['score_yes'] for r in records)
    return summary


def paired_bootstrap(records_by_model,fold_map,repeats=2000,seed=42):
    names=list(records_by_model);reference=sorted(records_by_model[names[0]],key=lambda r:r['case_id'])
    ids=[r['image_id'] for r in reference];patients=[r['case_id'] for r in reference]
    if len(set(patients))!=len(patients):raise ValueError('One fixed image per patient required')
    y=np.array([r['y'] for r in reference]);fold=np.array([fold_map[k] for k in ids]);scores={}
    for name in names:
        rows={r['image_id']:r for r in records_by_model[name]}
        if set(rows)!=set(ids) or any(rows[k]['y']!=y[i] for i,k in enumerate(ids)):raise ValueError('Unpaired scores')
        scores[name]=np.array([rows[k]['p'] for k in ids])
    fields=['auroc','ap','balanced_accuracy','macro_f1','sensitivity','specificity'];samples={};rng=np.random.default_rng(seed)
    for name in names:
        for mode in ['raw_0_5','crossfit','crossfit_minus_raw']:
            for key in fields:samples[(name,mode,key)]=[]
    for i,first in enumerate(names):
        for second in names[i+1:]:
            for mode in ['raw_0_5','crossfit']:
                for key in fields:samples[(second+' minus '+first,mode,key)]=[]
    for _ in range(repeats):
        w=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y));values={}
        # Multiplicity weights keep every duplicate in its original fold.
        for name,p in scores.items():
            pred,_=crossfit(y,p,fold,w)
            for mode,prediction in [('raw_0_5',p>=.5),('crossfit',pred)]:
                metric=binary_stats(y,p,prediction,w) if prediction is not None else {k:None for k in fields}
                for key in fields:
                    value=metric[key];values[(name,mode,key)]=value
                    if value is not None:samples.setdefault((name,mode,key),[]).append(value)
        for name in names:
            for key in fields:
                raw=values[(name,'raw_0_5',key)];cf=values[(name,'crossfit',key)]
                if raw is not None and cf is not None:samples[(name,'crossfit_minus_raw',key)].append(cf-raw)
        for i,first in enumerate(names):
            for second in names[i+1:]:
                for mode in ['raw_0_5','crossfit']:
                    for key in fields:
                        a=values[(first,mode,key)];b=values[(second,mode,key)]
                        if a is not None and b is not None:samples.setdefault((second+' minus '+first,mode,key),[]).append(b-a)
    return {'method':str(repeats)+' patient-paired bootstrap; weights preserve original folds; thresholds refit in each replicate; exploratory development only',
            'intervals':{' / '.join(key):{'ci95':np.quantile(values,[.025,.975]).tolist() if values else None,'valid_replicates':len(values),'requested_replicates':repeats} for key,values in samples.items()}}


def reward_group(correct,ced,epsilon=1e-8,near_zero=1e-8):
    correct=np.asarray(correct,dtype=np.float32);ced=np.asarray(ced,dtype=np.float32)
    def adv(x):
        import torch
        t=torch.as_tensor(x,dtype=torch.float32);std=float(t.std(unbiased=False))
        return (torch.zeros_like(t) if std<near_zero else (t-t.mean())/(std+epsilon)).numpy(),std
    a,sa=adv(correct);b,sb=adv(ced);legal=np.isin(correct,[0,1]);mixed=bool(legal.all() and len(set(correct))==2)
    gap=float(ced[correct==1].min()-ced[correct==0].max()) if mixed else None
    return {'correctness_rewards':correct.tolist(),'ced_rewards':ced.tolist(),'correctness_advantage':a.tolist(),'ced_advantage':b.tolist(),
            'correctness_std':sa,'ced_std':sb,'delta_A':float(np.max(np.abs(b-a))),
            'zero_advantage':bool(np.all(b==0)),'near_zero_std':sb<near_zero,'all_legal_mixed':mixed,
            'correct_reward_gap':gap,'ordering_reversed':bool(gap is not None and gap<0),
            'usable':bool(mixed and sa>=near_zero and sb>=near_zero and gap>0)}
