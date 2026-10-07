"""Fixed coverage selection, crop pairs and patient-cluster summaries."""
from collections import Counter
import numpy as np
from src.medevidence import crop_box, map_box
from src.medevidence_p2 import loc_schedule, summarize, distribution
from src.v2 import binary_stats, crossfit


def select_patients(rows, old_fit, seed=17):
    sort=lambda ks:sorted(ks,key=lambda k:rows[k]['case_id'])
    positive=sort(k for k,r in rows.items() if r['split']=='train' and r['pathology']=='yes' and len(r['boxes'])==1)
    retained=[k for k in old_fit if rows[k]['pathology']=='no']
    if len(positive)!=71 or len(retained)!=16 or len(set(retained))!=16:raise ValueError('Original71/oldFit16 identity mismatch')
    if not {k for k in old_fit if rows[k]['pathology']=='yes'}<=set(positive):raise ValueError('Old Fit positive outside original single-box pool')
    pool=sort(k for k,r in rows.items() if r['split']=='train' and r['pathology']=='no' and k not in retained)
    if len(pool)<240:raise ValueError('Insufficient frozen negatives')
    negatives=retained+[str(k) for k in np.random.default_rng(seed).permutation(pool)[:240]]
    if any(rows[k]['split']!='train' or rows[k]['pathology']!='no' for k in negatives):raise ValueError('Retained negative identity')
    rng=np.random.default_rng(seed);order=[str(k) for k in rng.permutation(positive)]+[str(k) for k in rng.permutation(negatives)]
    plan=loc_schedule(order,rows,256);counts=Counter(k for step in plan for k in step)
    assert len(counts)==327 and sum(counts.values())==1024
    assert Counter(counts[k] for k in positive)=={8:15,7:56}
    assert all(counts[k]==2 for k in negatives)
    return {'positive':positive,'negative':negatives,'order':order},plan


def local_pairs(rows, legacy_views, seed=42):
    positives=sorted((k for k,r in rows.items() if r['split']=='validation' and r['boxes'] and k in legacy_views['crops']),key=lambda k:rows[k]['case_id'])
    negatives=sorted((k for k,r in rows.items() if r['split']=='validation' and not r['boxes']),key=lambda k:rows[k]['case_id'])
    donors=list(np.random.default_rng(seed).permutation(negatives))[:len(positives)]
    result=[];excluded=[]
    for k,n in zip(positives,donors):
        r,d=rows[k],rows[n];views=[]
        try:
            assert len(legacy_views['crops'][k])==len(r['boxes'])
            for scale in (2.,2.5):
                for j,gt in enumerate(r['boxes']):
                    b=crop_box(gt,r['width'],r['height'],scale)
                    if scale==2. and b!=legacy_views['crops'][k][j]['box']:raise ValueError('Legacy main crop geometry mismatch')
                    nb=map_box(b,(r['width'],r['height']),(d['width'],d['height']))
                    delta=(np.asarray(nb)/[d['width'],d['height'],d['width'],d['height']]-np.asarray(b)/[r['width'],r['height'],r['width'],r['height']]).tolist()
                    views.append({'scale':scale,'box_index':j,'positive_box':b,'negative_box':nb,'normalized_geometry_difference':delta})
            result.append({'positive':k,'negative':str(n),'views':views})
        except (ValueError,AssertionError) as error:excluded.append({'positive':k,'negative':str(n),'reason':str(error),'replacement':False})
    assert len({x['negative'] for x in result})==len(result)
    return result,excluded


def metrics(records,folds=None,weights=None):
    w=np.ones(len(records)) if weights is None else np.asarray(weights)
    y=np.array([r['y'] for r in records]);p=np.array([r['p'] for r in records])
    ratio=lambda values,mask:float(np.dot(w,np.asarray(values)*mask)/np.dot(w,mask)) if np.dot(w,mask) else None
    pos=y==1;neg=y==0;single=np.array([r['gt']==1 for r in records]);multi=np.array([r['gt']>1 for r in records])
    pred=np.array([r['pred'] or 0 for r in records]);matches=np.array([r['matches'] for r in records]);gt=np.array([r['gt'] for r in records])
    result={'single_strict_success_rate':ratio([r['strict'] for r in records],single),
            'negative_nonempty_rate':ratio([r['state']=='valid_nonempty' for r in records],neg),
            'negative_invalid_rate':ratio([r['state']=='invalid' for r in records],neg),
            'negative_strict_empty_rate':ratio([r['state']=='valid_empty' for r in records],neg),
            'positive_empty_rate':ratio([r['state']=='valid_empty' for r in records],pos),
            'region_recall':float(np.dot(w,matches)/np.dot(w,gt)) if np.dot(w,gt) else None,
            'region_precision_known_predictions':float(np.dot(w,matches)/np.dot(w,pred)) if np.dot(w,pred) else None,
            'multi_all_covered_rate':ratio([r['all_covered'] for r in records],multi),
            'invalid_rate':ratio([r['state']=='invalid' for r in records],np.ones(len(y),bool))}
    raw=binary_stats(y,p,weights=w)
    result.update({'A_raw_'+k:raw[k] for k in ('ap','auroc','balanced_accuracy','macro_f1','sensitivity','specificity')})
    if folds is not None:
        prediction,_=crossfit(y,p,[folds[r['image_id']] for r in records],w)
        stats=binary_stats(y,p,prediction,w) if prediction is not None else {}
        result.update({'A_crossfit_'+k:stats.get(k) for k in ('ap','auroc','balanced_accuracy','macro_f1','sensitivity','specificity')})
    return result


def paired_intervals(a,b,folds,repeats=2000,seed=42):
    a=sorted(a,key=lambda r:r['case_id']);b=sorted(b,key=lambda r:r['case_id'])
    assert [(r['case_id'],r['image_id'],r['y']) for r in a]==[(r['case_id'],r['image_id'],r['y']) for r in b]
    left,right=metrics(a,folds),metrics(b,folds);samples={k:[] for k in left};rng=np.random.default_rng(seed)
    for _ in range(repeats):
        w=np.bincount(rng.integers(0,len(a),len(a)),minlength=len(a));ma,mb=metrics(a,folds,w),metrics(b,folds,w)
        for k in samples:
            if ma[k] is not None and mb[k] is not None:samples[k].append(mb[k]-ma[k])
    return {k:{'difference':right[k]-left[k] if right[k] is not None and left[k] is not None else None,
               'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,'valid_replicates':len(v),'requested_replicates':repeats} for k,v in samples.items()}


def local_metrics(records):
    result={}
    for scale in (2.,2.5):
        selected=[r for r in records if r['scale']==scale];patients=[]
        for cluster in sorted({r['cluster'] for r in selected}):
            pos=[r['p'] for r in selected if r['cluster']==cluster and r['y']==1];neg=[r['p'] for r in selected if r['cluster']==cluster and r['y']==0]
            if not pos or len(pos)!=len(neg):raise ValueError('Unpaired patient crops')
            patients.append({'cluster':cluster,'positive_support':float(np.mean(pos)),'negative_support':float(np.mean(neg)),
                'positive_detected':float(np.mean(pos))>=.5,'negative_false_positive':float(np.mean(neg))>=.5,
                'pair_correct':float(np.mean(pos))>=.5 and float(np.mean(neg))<.5,
                'within_patient_crop_correctness':float(np.mean([p>=.5 for p in pos]+[p<.5 for p in neg])),
                'all_views_correct':all(p>=.5 for p in pos) and all(p<.5 for p in neg)})
        result[str(scale)]={'pairs':len(patients),'independent_patients':2*len(patients),'crops':len(selected),
            **{k:float(np.mean([p[k] for p in patients])) if patients else None for k in ('positive_detected','negative_false_positive','pair_correct','within_patient_crop_correctness','all_views_correct')},
            'positive_support':distribution([p['positive_support'] for p in patients]),'negative_support':distribution([p['negative_support'] for p in patients])}
    return result
