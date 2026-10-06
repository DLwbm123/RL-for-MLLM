"""CPU-only attribution of existing binary rewards; no model or training entry."""
from contextlib import contextmanager
from collections import Counter
import math
import numpy as np
import torch


def require_cpu_config(cfg):
    for key in ['training_enabled','posttraining_enabled','resume_P1','run_P2','run_RL','gpu_authorized','push_authorized']:
        if cfg.get(key) is not False:raise PermissionError('v3 forbids '+key)
    if cfg['gpu_hours_limit']!=0 or cfg['label_order']!=['no','yes']:raise PermissionError('CPU-only/label contract changed')
    if cfg['reward_versions']!=['R-C','R-H','R-G','R-A']:raise ValueError('Unplanned reward version')


@contextmanager
def cpu_replay_guard(cfg):
    require_cpu_config(cfg)
    def forbidden(*args,**kwargs):raise PermissionError('v3 CPU replay forbids GPU, autograd and optimizer construction')
    originals=[(torch.Tensor,'backward',torch.Tensor.backward),(torch.autograd,'backward',torch.autograd.backward),
               (torch.autograd,'grad',torch.autograd.grad),(torch.optim.Optimizer,'__init__',torch.optim.Optimizer.__init__),
               (torch.cuda,'_lazy_init',torch.cuda._lazy_init)]
    for obj,key,_ in originals:setattr(obj,key,forbidden)
    try:
        with torch.no_grad():yield
    finally:
        for obj,key,old in originals:setattr(obj,key,old)


def finite(values):
    if not np.isfinite(np.asarray(values,dtype=float)).all():raise ValueError('Nonfinite input; stop replay')


def components(de,dn,c,cfg,dtype=torch.float32):
    finite([de,c,*dn])
    if len(dn)!=3 or c not in [0,1]:raise ValueError('Expected three controls and binary correctness')
    e=torch.tensor(de,dtype=dtype);n=torch.tensor(dn,dtype=dtype)
    mean=n.mean();std=n.std(unbiased=False);margin=e-mean;z=margin/(std+cfg['ced_epsilon'])
    m=torch.tanh(z);gate=.5*(1+torch.tanh(m/cfg['gate_temperature']))
    gated=c*gate;tie=cfg['tie_weight']*m;reward=gated+tie
    return {'D_E':float(e),'D_N':n.tolist(),'mean_D_N':float(mean),'std_D_N':float(std),'M':float(margin),
            'z':float(z),'m':float(m),'g':float(gate),'correctness':c,'correctness_gate_term':float(gated),
            'tie_term':float(tie),'final_reward':float(reward),'control_std_zero':float(std)==0,
            'control_std_small_nonzero':0<float(std)<cfg['small_control_std'],
            'gate_saturated':bool(gate<cfg['gate_saturation_low'] or gate>cfg['gate_saturation_high'])}


def reward_versions(c,m,gate,cfg):
    finite([c,m,gate])
    if c not in [0,1] or not -1<=m<=1 or not 0<=gate<=1:raise ValueError('Invalid reward components')
    m=torch.tensor(m,dtype=torch.float32);g=torch.tensor(gate,dtype=torch.float32)
    return {'R-C':float(c),'R-H':float(c*g+cfg['tie_weight']*m),'R-G':float(c*g),'R-A':float(c+cfg['tie_weight']*m)}


def order(gap):return 'positive' if gap>0 else 'reversed' if gap<0 else 'tie'


def advantage(values,cfg,dtype=torch.float32,epsilon=None):
    finite(values);t=torch.tensor(values,dtype=dtype)
    if not len(t):raise ValueError('Empty group')
    std=t.std(unbiased=False);epsilon=cfg['advantage_epsilon'] if epsilon is None else epsilon
    # Same actual near-zero branch even in the epsilon-free algebraic reference.
    out=torch.zeros_like(t) if std<cfg['near_zero_std'] else (t-t.mean())/(std+epsilon)
    return out.numpy(),float(std)


def errors(actual,reference,floor=1e-12):
    actual=np.asarray(actual,float);reference=np.asarray(reference,float);finite(actual);finite(reference)
    diff=np.abs(actual-reference)
    return {'max_absolute':float(diff.max()) if diff.size else 0.,
            'max_relative_with_floor':float((diff/np.maximum(np.abs(reference),floor)).max()) if diff.size else 0.,
            'relative_denominator_floor':floor,'reference_zero_nonzero_difference':int(((reference==0)&(diff>0)).sum())}


def group_audit(correct,rewards,correct_reward,wrong_reward,cfg):
    finite([*correct,*rewards,correct_reward,wrong_reward])
    if len(correct)!=8 or len(rewards)!=8:raise ValueError('Expected eight candidates')
    c=np.asarray(correct,float);r=np.asarray(rewards,float);legal=bool(np.isin(c,[0,1]).all())
    mixed=bool(legal and len(set(c))==2);b=float(correct_reward-wrong_reward);sign=int(np.sign(b));k=int((c==1).sum())
    if legal and not np.allclose(r,wrong_reward+b*c,rtol=0,atol=cfg['reward_replay_atol']):raise ValueError('Candidate rewards are not fixed per binary label')
    out={'correct_candidates':k,'legal_binary':legal,'mixed':mixed,'reward_gap':b,'sign_b':sign,'order':order(b),'normalizations':{}}
    for label,dtype,eps in [('historical_float32',torch.float32,cfg['advantage_epsilon']),
                            ('float64_with_epsilon',torch.float64,cfg['advantage_epsilon']),
                            ('float64_without_epsilon',torch.float64,0.)]:
        ac,sc=advantage(c,cfg,dtype,eps);ar,sr=advantage(r,cfg,dtype,eps)
        near=sr<cfg['near_zero_std'] or sc<cfg['near_zero_std']
        delta=float(np.max(np.abs(ar-ac)));residual=float(np.max(np.abs(ar-sign*ac)))
        theory_valid=mixed and b!=0 and not near
        scale=(abs(b)*(sc+eps))/(abs(b)*sc+eps) if theory_valid else None
        expected=sign*scale*ac if theory_valid else None
        scale_error=float(np.max(np.abs((scale-1)*ac))) if theory_valid else None
        unexplained=float(np.max(np.abs(ar-expected))) if theory_valid else None
        tolerance=cfg['advantage_replay_atol'] if label=='historical_float32' else cfg['float64_algebra_atol']
        result={'correctness_advantage':ac.tolist(),'reward_advantage':ar.tolist(),'correctness_std':sc,'reward_std':sr,
                'near_zero_branch':near,'zero_advantage':bool(np.all(ar==0)),'Delta_A':delta,'sign_residual':residual,
                'exactly_constant_input':bool(np.all(r==r[0])),
                'constant_input_roundoff_advantage':bool(np.all(r==r[0]) and np.any(ar!=0)),
                'algebra_reference_applicable':theory_valid,'epsilon_scale_factor':scale,'epsilon_explainable_sign_error':scale_error,
                'unexplained_after_scale':unexplained,'unexplained_exceeds_tolerance':bool(unexplained is not None and unexplained>tolerance)}
        if theory_valid and sign<0:
            predicted=2*max(math.sqrt((8-k)/k),math.sqrt(k/(8-k)))
            result['ideal_reversal_Delta_A']=predicted;result['reversal_formula_error']=abs(delta-predicted)
        else:result['ideal_reversal_Delta_A']=None;result['reversal_formula_error']=None
        out['normalizations'][label]=result
    out['float32_vs_float64_advantage_error']=errors(out['normalizations']['historical_float32']['reward_advantage'],out['normalizations']['float64_with_epsilon']['reward_advantage'],cfg['relative_error_denominator_floor'])
    return out


def distribution(values):
    a=np.asarray(values,float);finite(a)
    if not len(a):return {'n':0,'min':None,'q25':None,'median':None,'q75':None,'q95':None,'max':None,'mean':None}
    return {'n':len(a),**dict(zip(['min','q25','median','q75','q95','max'],np.quantile(a,[0,.25,.5,.75,.95,1]).tolist())),'mean':float(a.mean())}


def aggregate_groups(rows,variant):
    selected=[r['versions'][variant] for r in rows];mixed=[v for v in selected if v['mixed']]
    valid=[v for v in mixed if v['normalizations']['historical_float32']['algebra_reference_applicable']]
    positive=[v for v in valid if v['sign_b']>0];reverse=[v for v in valid if v['sign_b']<0]
    return {'groups':len(rows),'patients':len({r['case_id'] for r in rows}),'mixed_groups':len(mixed),'algebra_eligible_mixed_groups':len(valid),
            'positive_mixed_groups':len(positive),'reversed_mixed_groups':len(reverse),
            'mixed_tie_or_near_zero_groups':len(mixed)-len(valid),
            'zero_advantage_groups':sum(v['normalizations']['historical_float32']['zero_advantage'] for v in selected),
            'constant_input_roundoff_groups':sum(v['normalizations']['historical_float32']['constant_input_roundoff_advantage'] for v in selected),
            'all_identical_answer_groups':sum(not v['mixed'] for v in selected),
            'positive_Delta_A':distribution([v['normalizations']['historical_float32']['Delta_A'] for v in positive]),
            'reversed_Delta_A':distribution([v['normalizations']['historical_float32']['Delta_A'] for v in reverse]),
            'float32_sign_residual':distribution([v['normalizations']['historical_float32']['sign_residual'] for v in valid]),
            'float64_no_epsilon_sign_residual':distribution([v['normalizations']['float64_without_epsilon']['sign_residual'] for v in valid]),
            'epsilon_explainable_error':distribution([v['normalizations']['float64_with_epsilon']['epsilon_explainable_sign_error'] for v in valid]),
            'unexplained_groups':sum(any(n['unexplained_exceeds_tolerance'] for n in v['normalizations'].values()) for v in selected),
            'reversal_k_counts':dict(sorted(Counter(v['correct_candidates'] for v in reverse).items()))}


def cluster_intervals(cases,groups,cfg):
    """Resample 32 patients, retaining every patient's four original groups."""
    ids=[r['case_id'] for r in cases];lookup={k:[g for g in groups if g['case_id']==k] for k in ids}
    if any(len(v)!=4 for v in lookup.values()):raise ValueError('Cluster must retain four groups')
    rng=np.random.default_rng(cfg['bootstrap_seed']);samples={k:[] for k in ['cache_reversed_fraction','observed_reversal_patient_fraction','reversed_fraction_of_mixed','positive_near_equivalence_fraction']}
    reversed_ids={g['case_id'] for g in groups if g['versions']['R-H']['mixed'] and g['versions']['R-H']['sign_b']<0}
    for _ in range(cfg['bootstrap_repeats']):
        indexes=rng.integers(0,len(cases),len(cases));chosen=[cases[i] for i in indexes]
        values=[g['versions']['R-H'] for row in chosen for g in lookup[row['case_id']]]
        mixed=[v for v in values if v['mixed']];positive=[v for v in mixed if v['sign_b']>0 and v['normalizations']['historical_float32']['algebra_reference_applicable']]
        samples['cache_reversed_fraction'].append(sum(r['order']=='reversed' for r in chosen)/len(chosen))
        samples['observed_reversal_patient_fraction'].append(sum(r['case_id'] in reversed_ids for r in chosen)/len(chosen))
        if mixed:samples['reversed_fraction_of_mixed'].append(sum(v['sign_b']<0 for v in mixed)/len(mixed))
        if positive:samples['positive_near_equivalence_fraction'].append(sum(v['normalizations']['historical_float32']['Delta_A']<cfg['historical_delta_A_threshold'] for v in positive)/len(positive))
    return {'unit':'patient cluster; all four original groups retained','clusters':len(cases),'seed':cfg['bootstrap_seed'],
            'intervals':{key:{'ci95':np.quantile(values,[.025,.975]).tolist() if values else None,'valid_replicates':len(values),
                              'invalid_replicates':cfg['bootstrap_repeats']-len(values),'requested_replicates':cfg['bootstrap_repeats']} for key,values in samples.items()}}
