"""Reuse P7 supervision and data split without a new detector fit."""
import os
from pathlib import Path
import shutil
from src.medevidence_p7 import read
from src.v2 import save
from scripts.medevidence_p7 import controller as previous_controller


def prepare():
    import runpy
    runpy.run_path('tests/test_medevidence_p7.py',run_name='__main__')
    runpy.run_path('tests/test_medevidence_p8.py',run_name='__main__')
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';old=base.parent/'medevidence_p7';cfg=read(base/'code/configs/medevidence_p8.json')
    ledger=read(old/'outputs/gpu_ledger.json');final=read(old/'outputs/FINAL.json')
    assert not ledger['active'] and ledger['exit_code']==0 and final['status']=='stopped_candidate_gate' and final['detector_steps']==800
    assert abs(ledger['prior_GPU_seconds']+ledger['new_charged_seconds']-cfg['prior_GPU_seconds'])<1e-8
    assert (old/'outputs/detector_final.pt').is_file()
    previous=read(old/'code/configs/medevidence_p7.json')
    for key in ['detector_min_size','detector_max_size','candidate_limit','candidate_score_threshold','candidate_nms_iou','matching_iou',
                'policy_warmup_steps','policy_continuation_steps','policy_batch_size','policy_lr','policy_hidden_size','policy_kl_beta','reward']:
        assert cfg[key]==previous[key],key
    shutil.copytree(old/'outputs/protocol',out/'protocol')
    receipt=read(out/'protocol/CPU_checks.json')
    receipt.update(prior_GPU_seconds=cfg['prior_GPU_seconds'],source_commit=read(out/'authorization.json')['source_commit'],
        detector_reused_steps=800,detector_new_steps=0,matched_coverage_checks='passed',phase_GPU_limit_seconds=cfg['phase_GPU_limit_seconds'])
    save(out/'protocol/CPU_checks.json',receipt);save(base/'code/reports/medevidence_p8_preparation.json',receipt)
    print('P8 preparation passed',flush=True)


def controller():previous_controller('medevidence_p8.json','src.medevidence_p8')
