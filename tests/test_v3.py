"""Synthetic boundary checks only; these examples are not patient measurements."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.v3 import components,reward_versions,group_audit,cpu_replay_guard,require_cpu_config,cluster_intervals

CFG=json.loads(Path('configs/rsna_v3.json').read_text())


def test_synthetic_gate_tie_reversal_and_protected_boundaries():
    yes=components(-2,[0,0,0],1,CFG);no=components(2,[0,0,0],0,CFG)
    assert yes['control_std_zero'] and yes['m']==-1 and no['m']==1
    correct=reward_versions(1,yes['m'],yes['g'],CFG);wrong=reward_versions(0,no['m'],no['g'],CFG)
    assert correct['R-H']<wrong['R-H']
    assert .1*(no['m']-yes['m'])>yes['g']
    assert correct['R-G']>wrong['R-G'] and correct['R-A']-wrong['R-A']==pytest.approx(.8,abs=1e-6)
    # Saturated gate=0 is a defensive boundary, not claimed for the actual bounded float32 gate.
    assert reward_versions(1,-1,0,CFG)['R-G']==reward_versions(0,1,1,CFG)['R-G']==0
    small=components(1,[0,1e-6,2e-6],1,CFG)
    assert not small['control_std_zero'] and small['control_std_small_nonzero'] and small['gate_saturated']


@pytest.mark.parametrize('k',range(1,8))
def test_synthetic_positive_equivalence_and_each_reversed_k(k):
    c=[1]*k+[0]*(8-k)
    for b in [2.,-.5]:
        r=np.array(c)*b+.3;a=group_audit(c,r.tolist(),.3+b,.3,CFG)
        exact=a['normalizations']['float64_without_epsilon']
        assert exact['sign_residual']<1e-12
        assert not any(x['unexplained_exceeds_tolerance'] for x in a['normalizations'].values())
        if b<0:assert exact['reversal_formula_error']<1e-12


def test_synthetic_tie_same_answers_near_zero_and_nonfinite():
    c=[0,1]*4
    tied=group_audit(c,[.25]*8,.25,.25,CFG)
    assert tied['sign_b']==0 and tied['normalizations']['historical_float32']['zero_advantage']
    near=group_audit(c,[x*1e-10 for x in c],1e-10,0,CFG)
    assert near['normalizations']['historical_float32']['near_zero_branch']
    assert not near['normalizations']['float64_without_epsilon']['algebra_reference_applicable']
    same=group_audit([1]*8,[.8]*8,.8,.1,CFG)
    assert not same['mixed'] and same['normalizations']['float64_without_epsilon']['zero_advantage']
    with pytest.raises(ValueError):components(float('nan'),[0,1,2],1,CFG)
    with pytest.raises(ValueError):group_audit(c,[float('inf')]*8,1,0,CFG)
    with pytest.raises(ValueError):group_audit(c,[0,1,0,2,0,1,0,1],1,0,CFG)


def test_synthetic_constant_float32_roundoff_is_reported_not_repaired():
    from src.objectives import advantages
    tied=group_audit([0,1]*4,[.2]*8,.2,.2,CFG)
    actual=tied['normalizations']['historical_float32']
    assert actual['exactly_constant_input'] and not actual['algebra_reference_applicable']
    assert np.array_equal(actual['reward_advantage'],advantages(torch.tensor([.2]*8)).numpy())
    assert actual['constant_input_roundoff_advantage']==bool(np.any(advantages(torch.tensor([.2]*8)).numpy()!=0))
    assert tied['normalizations']['float64_without_epsilon']['zero_advantage']


def test_synthetic_epsilon_error_matches_scale():
    c=[0,1]*4;r=np.array(c)*1e-6
    result=group_audit(c,r.tolist(),1e-6,0,CFG)['normalizations']['float64_with_epsilon']
    assert result['epsilon_explainable_sign_error']>1e-4
    assert abs(result['sign_residual']-result['epsilon_explainable_sign_error'])<1e-12
    assert result['unexplained_after_scale']<1e-12


def test_cpu_no_training_gpu_or_optional_authorization():
    with cpu_replay_guard(CFG):
        with pytest.raises(PermissionError):torch.tensor(1.).backward()
        with pytest.raises(PermissionError):torch.optim.AdamW([torch.tensor(1.)])
        with pytest.raises(PermissionError):torch.empty(1,device='cuda')
    for key in ['training_enabled','posttraining_enabled','resume_P1','run_P2','run_RL','gpu_authorized','push_authorized']:
        bad=dict(CFG);bad[key]=True
        with pytest.raises(PermissionError):require_cpu_config(bad)


def test_synthetic_patient_clusters_and_empty_mixed_replicates():
    cases=[{'case_id':'synthetic_a','order':'positive'},{'case_id':'synthetic_b','order':'positive'}]
    groups=[{'case_id':row['case_id'],'versions':{'R-H':group_audit([1]*8,[1.]*8,1.,0.,CFG)}} for row in cases for _ in range(4)]
    cfg=dict(CFG,bootstrap_repeats=20)
    result=cluster_intervals(cases,groups,cfg)
    assert result['clusters']==2 and result['intervals']['reversed_fraction_of_mixed']['valid_replicates']==0
    assert result['intervals']['reversed_fraction_of_mixed']['ci95'] is None
    with pytest.raises(ValueError):cluster_intervals(cases,groups[:-1],cfg)
