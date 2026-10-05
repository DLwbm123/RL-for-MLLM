"""Image-level metrics with Case-clustered intervals; no unverified patient diagnosis."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,f1_score,confusion_matrix,cohen_kappa_score


def binary_metrics(y,p,threshold=.5):
    y=np.asarray(y);p=np.asarray(p);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'auroc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'auprc':float(average_precision_score(y,p)) if y.sum() else None,
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'sensitivity':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None,
            'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}


def cluster_bootstrap(frame,statistic,repeats=2000,seed=42):
    groups=[r for _,r in frame.groupby('case_id',sort=True)]
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        sampled=pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))],ignore_index=True)
        value=statistic(sampled)
        if value is not None and np.isfinite(value):values.append(float(value))
    return {'ci95':np.quantile(values,[.025,.975]).tolist() if values else None,'valid_replicates':len(values),'requested_replicates':repeats}


def parse_box(text):
    try:
        b=np.asarray(json.loads(text),dtype=float)
        if b.shape!=(4,) or not np.isfinite(b).all() or (b<0).any() or (b>1000).any() or b[2]<=b[0] or b[3]<=b[1]:return None
        return b
    except (ValueError,TypeError):return None


def box_iou(pred,target):
    p=parse_box(pred)
    if p is None:return 0.
    t=np.asarray(target,dtype=float)
    area=lambda b:max(0,b[2]-b[0])*max(0,b[3]-b[1])
    inter=area([max(p[0],t[0]),max(p[1],t[1]),min(p[2],t[2]),min(p[3],t[3])])
    return float(inter/(area(p)+area(t)-inter))


def fit_calibration(scores,y,split):
    if split!='validation':raise ValueError('Calibration restricted to validation')
    scores=np.asarray(scores); y=np.asarray(y)
    def loss(logt):
        p=softmax(scores/np.exp(logt),axis=1)
        return -np.log(p[np.arange(len(y)),y].clip(1e-12)).mean()
    result=minimize_scalar(loss,bounds=(-3,3),method='bounded')
    return {'temperature':float(np.exp(result.x)),'threshold':.5,'fit_split':'validation','success':bool(result.success)}


def select_checkpoint(candidates):
    """Highest validation AUROC; exact ties choose earliest step."""
    return sorted(candidates,key=lambda r:(-r['auroc'],r['step']))[0]


def paired_auroc_difference(first,second,repeats=2000):
    keys=['image_id','case_id','y']
    merged=first[keys+['p']].merge(second[keys+['p']],on=keys,suffixes=('_a','_b'),validate='one_to_one')
    if len(merged)!=len(first) or len(merged)!=len(second):raise ValueError('Paired comparison requires identical samples and labels')
    def diff(d):
        if d.y.nunique()!=2:return None
        return roc_auc_score(d.y,d.p_b)-roc_auc_score(d.y,d.p_a)
    return {'difference_b_minus_a':float(diff(merged)),**cluster_bootstrap(merged,diff,repeats)}


def birads_metrics(y,pred):
    labels=[2,3,4,5];y=np.asarray(y);pred=np.asarray(pred)
    if not np.isin(y,labels).all():raise ValueError('Unknown gold BI-RADS')
    invalid=~np.isin(pred,labels);pred=np.where(invalid,0,pred)
    return {'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
            'confusion_matrix_labels':[0]+labels,'confusion_matrix':confusion_matrix(y,pred,labels=[0]+labels).tolist(),
            'invalid_predictions':int(invalid.sum()),
            'quadratic_weighted_kappa':float(cohen_kappa_score(y,pred,weights='quadratic')) if not invalid.any() else None}
