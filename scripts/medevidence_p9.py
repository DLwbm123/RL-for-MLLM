"""Freeze patient folds and reuse the bounded neutral-process controller."""
import os
from pathlib import Path
import runpy
import shutil
import numpy as np
import pandas as pd
from src.medevidence_p7 import read
from src.v2 import save
from scripts.medevidence_p7 import controller as previous_controller


def prepare():
    runpy.run_path('tests/test_medevidence_p7.py',run_name='__main__')
    runpy.run_path('tests/test_medevidence_p8.py',run_name='__main__')
    runpy.run_path('tests/test_medevidence_p9.py',run_name='__main__')
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';old=base.parent/'medevidence_p8/outputs';cfg=read(base/'code/configs/medevidence_p9.json')
    ledger=read(old/'gpu_ledger.json');assert ledger['status']=='completed' and ledger['exit_code']==0 and not ledger['active']
    assert abs(ledger['prior_GPU_seconds']+ledger['new_charged_seconds']-cfg['prior_GPU_seconds'])<1e-8
    shutil.copytree(old/'protocol',out/'protocol');sets=read(out/'protocol/sets.json')
    frame=pd.read_csv(out/'protocol/manifest.csv',keep_default_na=False).set_index('image_id');rng=np.random.default_rng(17)
    folds=[[] for _ in range(cfg['oof_folds'])]
    for label in ['yes','no']:
        ids=[k for k in sets['train'] if frame.loc[k,'pathology']==label]
        for i,chunk in enumerate(np.array_split(rng.permutation(ids),len(folds))):folds[i].extend(map(str,chunk))
    folds=[sorted(v) for v in folds]
    assert sorted(sum(folds,[]))==sorted(sets['train']) and len(set(sum(folds,[])))==960
    assert all(len(v)==320 and sum(frame.loc[k,'pathology']=='yes' for k in v)==39 for v in folds)
    save(out/'protocol/folds.json',folds)
    receipt=read(out/'protocol/CPU_checks.json');receipt.update(prior_GPU_seconds=cfg['prior_GPU_seconds'],
        source_commit=read(out/'authorization.json')['source_commit'],phase_GPU_limit_seconds=cfg['phase_GPU_limit_seconds'],
        folds=3,heldout_patients_per_fold=320,heldout_positive_per_fold=39,detector_patients_per_fold=640,
        fold_patient_overlap=0,correctness_cpu_checks='passed')
    receipt.pop('detector_new_steps',None);receipt.pop('detector_reused_steps',None)
    save(out/'protocol/CPU_checks.json',receipt);save(base/'code/reports/medevidence_p9_preparation.json',receipt)
    print('P9 preparation passed',flush=True)


def controller():previous_controller('medevidence_p9.json','src.medevidence_p9')
