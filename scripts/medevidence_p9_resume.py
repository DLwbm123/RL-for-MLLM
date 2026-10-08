"""Prepare an isolated, separately charged continuation without retraining detectors."""
import os
from pathlib import Path
import runpy
import shutil
import numpy as np
import torch
from src.medevidence_p7 import read
from src.medevidence_p9_resume import feature_identity,file_receipt
from src.v2 import save
from scripts.medevidence_p7 import controller as previous_controller


def prepare():
    for name in ['p7','p8','p9','p9_resume']:runpy.run_path('tests/test_medevidence_'+name+'.py',run_name='__main__')
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';old=base.parent/'medevidence_p9';cfg=read(base/'code/configs/medevidence_p9_resume.json')
    ledger=read(old/'outputs/gpu_ledger.json');final=read(old/'outputs/FINAL.json')
    assert ledger['status']=='stopped_budget' and ledger['exit_code']==0 and not ledger['active']
    assert final['stage']=='frozen_visual_features' and final['detector_steps']==2400
    assert abs(ledger['prior_GPU_seconds']+ledger['new_charged_seconds']-cfg['prior_GPU_seconds'])<1e-8
    scientific=lambda c:{k:v for k,v in c.items() if k not in {'version','prior_GPU_seconds','phase_GPU_limit_seconds','feature_cache_chunk_size'}}
    assert scientific(cfg)==scientific(read(old/'code/configs/medevidence_p9.json')),'Scientific configuration changed'
    assert cfg['phase_GPU_limit_seconds']==2700 and cfg['feature_cache_chunk_size']==32
    assert not any((old/'outputs'/n).exists() for n in ['head_training.json','evaluation.json'])
    checkpoint_sizes=[(old/f'outputs/fold_{i}/detector_final.pt').stat().st_size for i in range(3)];assert all(s>0 for s in checkpoint_sizes)
    shutil.copytree(old/'outputs/protocol',out/'protocol');sets=read(out/'protocol/sets.json');candidates=read(old/'outputs/candidates.json')
    assert set(candidates)==set(sets['train'])|set(sets['calibration']) and len(candidates)==1024
    for v in candidates.values():
        assert len(v['boxes'])==len(v['scores'])<=8 and np.isfinite(v['boxes']).all() and np.isfinite(v['scores']).all()
    dev_path=base.parent/'medevidence_p8/outputs/dev_features.pt';dev=torch.load(dev_path,map_location='cpu',weights_only=True)
    assert set(dev)==set(sets['dev']) and all(v['x'].shape==(10,7176) for v in dev.values());del dev
    save(out/'protocol/reused_development_sources.json',{n:file_receipt(dev_path.parent/n) for n in ['dev_features.pt','candidates.json']})
    identity=feature_identity(base,os.environ['MODEL_ROOT']);save(out/'protocol/feature_identity.json',identity)
    audit=read(old/'outputs/oof_audit.json');assert audit['top1_supported_positive']==51 and audit['positive_without_supported_candidate']==53
    save(out/'oof_audit.json',audit)
    resume={'status':'prepared','resume_from':'P9 frozen visual features','original_execution_source_commit':read(old/'outputs/authorization.json')['source_commit'],
        'engineering_source_commit':read(out/'authorization.json')['source_commit'],'historical_P9_GPU_seconds':ledger['new_charged_seconds'],
        'prior_combined_GPU_seconds':cfg['prior_GPU_seconds'],'continuation_limit_seconds':2700,'reused_detector_steps':2400,
        'new_detector_steps_planned':0,'reused_training_candidate_patients':960,'reused_calibration_candidate_patients':64,
        'reused_development_cached_patients':256,'chunk_size':32,'scientific_configuration_unchanged':True,'checks':'provenance metadata, patient membership and cache readability; no file hashes'}
    save(out/'resume_receipt.json',resume)
    receipt=read(out/'protocol/CPU_checks.json');receipt.update(source_commit=resume['engineering_source_commit'],prior_GPU_seconds=cfg['prior_GPU_seconds'],
        phase_GPU_limit_seconds=2700,resume_stage='frozen_visual_features',feature_chunk_checks='passed',detector_retraining=False,development_cache_readable=True)
    save(out/'protocol/CPU_checks.json',receipt);save(base/'code/reports/medevidence_p9_resume_preparation.json',receipt)
    assert not torch.cuda.is_initialized();print('P9 continuation preparation passed',flush=True)


def controller():previous_controller('medevidence_p9_resume.json','src.medevidence_p9_resume')
