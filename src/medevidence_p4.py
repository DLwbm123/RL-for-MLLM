"""Prespecified full-annotation schedule, detached geometry reward and PPO terms."""
from collections import Counter
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.medevidence import parse_boxes, matching
from src.medevidence_p2 import iou, distribution, summarize


def schedule(rows, negatives, steps=256, seed=17):
    pools={'single':sorted(k for k,r in rows.items() if r['split']=='train' and len(r['boxes'])==1),
           'multi':sorted(k for k,r in rows.items() if r['split']=='train' and len(r['boxes'])>1),
           'negative':sorted(negatives)}
    if [len(pools[k]) for k in pools]!=[71,78,256]:raise ValueError('Expected original71 single,78 multi,P3 same256 negative patients')
    if any(rows[k]['split']!='train' or rows[k]['boxes'] or rows[k]['pathology']!='no' for k in negatives):raise ValueError('P3 negative identity')
    rng={k:np.random.default_rng(seed) for k in pools};queues={k:[] for k in pools}
    def take(k):
        if not queues[k]:queues[k]=list(rng[k].permutation(pools[k]))
        return str(queues[k].pop(0))
    plan=[[take(k) for k in ('single','multi','negative','negative')] for _ in range(steps)]
    return pools,plan


def giou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    enclosure=(max(a[2],b[2])-min(a[0],b[0]))*(max(a[3],b[3])-min(a[1],b[1]))
    if union<=0 or enclosure<=0:raise ValueError('GIoU requires legal nonzero-area boxes')
    return intersection/union-(enclosure-union)/enclosure


def reward(text,gt,truncated=False):
    boxes,error=parse_boxes(text)
    if error or truncated:return -1.
    if not gt:return 1. if not boxes else -1.
    if not boxes:return -1.
    values=np.asarray([[giou(b,g) for g in gt] for b in boxes],dtype=np.float64)
    ii,jj=linear_sum_assignment(-values);matched=len(ii)
    result=(values[ii,jj].sum()-(len(gt)-matched)-(len(boxes)-matched))/len(gt)
    if not np.isfinite(result):raise FloatingPointError('Nonfinite geometry reward')
    return float(np.clip(result,-1.,1.))


def advantages(values):
    import torch
    r=torch.as_tensor(values,dtype=torch.float32).detach()
    if not torch.isfinite(r).all():raise FloatingPointError('Nonfinite reward')
    std=r.std(unbiased=False)
    return torch.zeros_like(r) if std<=1e-6 else (r-r.mean())/std


def policy_terms(current,old,reference,advantage,mask,epsilon=.2):
    """Mask excludes prompt/padding; sampled EOS included. ref-current k3 direction."""
    import torch
    current=current.float();old=old.detach().float();reference=reference.detach().float();mask=mask.bool()
    if not mask.any() or not all(torch.isfinite(x[mask]).all() for x in (current,old,reference)):raise FloatingPointError('Empty/nonfinite completion logprobs')
    ratio=torch.exp(current[mask]-old[mask]);d=reference[mask]-current[mask]
    kl=torch.exp(d)-d-1
    adv=torch.as_tensor(advantage,dtype=torch.float32,device=current.device).detach()
    objective=torch.minimum(ratio*adv,ratio.clamp(1-epsilon,1+epsilon)*adv).mean()
    if not torch.isfinite(ratio).all() or not torch.isfinite(kl).all() or not torch.isfinite(objective):raise FloatingPointError('Nonfinite ratio/KL/objective')
    return objective,kl.mean()


def best_single(outputs,gt):
    if len(gt)!=1:raise ValueError('Best-of diagnostic only defined on single GT')
    quality=[];success=[]
    for r in outputs:
        boxes,error=parse_boxes(r['loc_text']);valid=not error and not r['truncated'] and len(boxes)==1
        value=iou(boxes[0],gt[0]) if valid else 0.
        quality.append(value);success.append(valid and value>=.5)
    return {'best_iou':max(quality,default=0.),'strict_success':any(success)}


def shuffled_control(groups,ground_truth,repeats=1000,seed=42):
    if len(groups)!=25 or len(ground_truth)!=25:raise ValueError('D4 fixed25 single patients')
    real=np.mean([best_single(g,t)['best_iou'] for g,t in zip(groups,ground_truth)])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        for attempt in range(10000):
            permutation=rng.permutation(len(groups))
            if np.all(permutation!=np.arange(len(groups))):break
        else:raise RuntimeError('No derangement')
        values.append(float(np.mean([best_single(g,ground_truth[j])['best_iou'] for g,j in zip(groups,permutation)])))
    return {'real_mean_best_single_IoU':float(real),'shuffled_mean':float(np.mean(values)),
            'real_minus_shuffled_mean':float(real-np.mean(values)),'shuffled_distribution':distribution(values),'repeats':repeats,'seed':seed,'self_pairings':0}


def gate(d2,d3,d4):
    values={'D2_best_of8_success':d2['positive_best_of8_success'],
            'D3_positive_distinguishable_fraction':d3['positive_distinguishable_fraction'],
            'D4_real_minus_shuffle_IoU':d4['real_minus_shuffled_mean']}
    passed=[values['D2_best_of8_success']>=8,values['D3_positive_distinguishable_fraction']>=.30,values['D4_real_minus_shuffle_IoU']>=.05]
    return {'passed':all(passed),'values':values,'component_passed':passed,'thresholds':[8,.30,.05],'scope':'R&D resource allocation only; no statistical or clinical interpretation'}


def extended_summary(records):
    result=summarize(records);negative=[r for r in records if not r['y']]
    result.update(negative_false_positive_boxes=sum(r['pred'] or 0 for r in negative),
                  negative_false_positive_boxes_per_image=sum(r['pred'] or 0 for r in negative)/len(negative) if negative else None,
                  known_predicted_regions_all_patients=sum(r['pred'] or 0 for r in records),
                  matched_regions=sum(r['matches'] for r in records),
                  parse_errors=dict(Counter(r['parse_error'] for r in records if r['parse_error'])),
                  unknown_prediction_count_patients=sum(r['pred'] is None for r in records))
    result['strata']={name:{'patients':len(subset),'GT_regions':sum(r['gt'] for r in subset),'matched_regions':sum(r['matches'] for r in subset),
                           'strict_success':sum(r['strict'] for r in subset),'empty':sum(r['state']=='valid_empty' for r in subset),'invalid':sum(r['state']=='invalid' for r in subset)}
                       for name,subset in ((name,[r for r in records if (r['gt']==1 if name=='single' else r['gt']>1)]) for name in ('single','multi'))}
    return result


def paired_counts(a,b):
    a=sorted(a,key=lambda r:r['case_id']);b=sorted(b,key=lambda r:r['case_id'])
    assert [(r['case_id'],r['image_id']) for r in a]==[(r['case_id'],r['image_id']) for r in b]
    result={}
    for name,choose in [('single',lambda r:r['gt']==1),('multi',lambda r:r['gt']>1),('negative',lambda r:not r['y'])]:
        pairs=[(x['strict'],y['strict']) for x,y in zip(a,b) if choose(x)]
        result[name]={'patients':len(pairs),'new_success':sum(not x and y for x,y in pairs),'lost_success':sum(x and not y for x,y in pairs),
                      'common_success':sum(x and y for x,y in pairs),'common_failure':sum(not x and not y for x,y in pairs)}
    return result


def paired_intervals(a,b,folds,repeats=2000,seed=42):
    from src.medevidence_p3 import metrics
    a=sorted(a,key=lambda r:r['case_id']);b=sorted(b,key=lambda r:r['case_id'])
    assert [(r['case_id'],r['image_id'],r['y']) for r in a]==[(r['case_id'],r['image_id'],r['y']) for r in b]
    def stats(rs,w):
        out=metrics(rs,folds,w);single=np.asarray([r['gt']==1 for r in rs]);multi=np.asarray([r['gt']>1 for r in rs]);negative=np.asarray([not r['y'] for r in rs])
        mean=lambda vals,mask:float(np.dot(w,np.asarray(vals)*mask)/np.dot(w,mask)) if np.dot(w,mask) else None
        out.update(single_mean_IoU=mean([r['single_iou'] or 0 for r in rs],single),
                   multi_strict_success_rate=mean([r['strict'] for r in rs],multi),
                   multi_matched_regions_per_patient=mean([r['matches'] for r in rs],multi),
                   negative_false_positive_boxes_per_image=mean([r['pred'] or 0 for r in rs],negative))
        return out
    left,right=stats(a,np.ones(len(a))),stats(b,np.ones(len(a)));samples={k:[] for k in left};rng=np.random.default_rng(seed)
    for _ in range(repeats):
        w=np.bincount(rng.integers(0,len(a),len(a)),minlength=len(a));x,y=stats(a,w),stats(b,w)
        for k in samples:
            if x[k] is not None and y[k] is not None:samples[k].append(y[k]-x[k])
    return {k:{'difference':right[k]-left[k] if right[k] is not None and left[k] is not None else None,'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,
               'valid_replicates':len(v),'requested_replicates':repeats} for k,v in samples.items()}
