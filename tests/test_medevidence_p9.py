"""Synthetic CPU checks for correctness heads, prior weighting and reject decisions."""
import tempfile
import os
from pathlib import Path
import numpy as np
import torch
from src.medevidence_p9 import fit_heads,logits,calibrate,calibrated_probabilities,decisions,weighted_cutoff,report
from src.v2 import save


def main():
    rng=np.random.default_rng(17);x=rng.normal(size=(80,4));y=np.stack([x[:,0]>0,x[:,1]>0],-1).astype(float)
    model,checks=fit_heads(x,y,{'head_C':.01,'head_max_iter':1000,'feature_std_floor':.1})
    assert all(v['final_full_loss']<v['initial_full_loss'] for v in checks)
    z=logits(model,x);has=np.ones(len(x),bool);weights=np.ones(len(x))
    shift=calibrate(z,y,weights,has,.01);p=calibrated_probabilities(z,shift,has)
    assert p.shape==y.shape and np.isfinite(p).all()
    actions,confidence=decisions(np.array([[.8,.1],[.1,.9],[.2,.3],[.9,.1]]),np.array([1,1,1,0],bool))
    assert actions.tolist()==[0,8,9,9]
    assert calibrated_probabilities(np.zeros((2,2)),np.zeros(2),np.array([0,1],bool))[0,0]==0
    assert weighted_cutoff(np.array([.1,.2,.8,.9]),np.ones(4),.75)==.2
    assert weighted_cutoff(np.array([.1,.9]),np.array([9.,1.]),.75)==.1
    # Exercise real report export so missing imports cannot recur after successful computation.
    old=os.environ.get('PILOT_ROOT')
    try:
        with tempfile.TemporaryDirectory() as folder:
            b=Path(folder);(b/'outputs').mkdir();(b/'code/reports').mkdir(parents=True);os.environ['PILOT_ROOT']=folder
            save(b/'outputs/FINAL.json',{'status':'stopped_oof_target_support','detector_steps':2400})
            save(b/'outputs/gpu_ledger.json',{'new_charged_seconds':1,'combined_GPU_hours':1.2})
            report();assert (b/'code/reports/medevidence_p9_decision.md').is_file()
            save(b/'outputs/evaluation.json',{'decision':'STOP_CURRENT_CORRECTNESS_BASELINE','metrics':{'synthetic':{
                'coverage':0.,'answered_risk':None,'positive_supported_success_rate':0.,'negative_wrong_yes_rate':0.}}})
            report();assert 'STOP_CURRENT_CORRECTNESS_BASELINE' in (b/'code/reports/medevidence_p9_decision.md').read_text()
    finally:
        if old is None:os.environ.pop('PILOT_ROOT',None)
        else:os.environ['PILOT_ROOT']=old
    assert not torch.cuda.is_initialized();print('P9 correctness/calibration/rejection/report CPU checks passed')


if __name__=='__main__':main()
