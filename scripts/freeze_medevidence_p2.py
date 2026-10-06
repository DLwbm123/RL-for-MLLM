"""Lock prepared patients, targets, source and config once CPU checks pass."""
import hashlib
import json
import os
from pathlib import Path
from src.v2 import save


def main():
    base=Path(os.environ['PILOT_ROOT']);code=base/'code';p=base/'outputs/protocol';dest=p/'lock.json'
    resource_revision=os.environ.get('RESOURCE_REVISION_ONLY')=='1'
    previous=json.loads(dest.read_text()) if dest.exists() else None
    if previous and not resource_revision:raise FileExistsError('Already frozen; no re-selection or source substitution')
    if resource_revision:
        if previous is None or (base/'outputs/gpu_ledger.json').exists():raise PermissionError('Resource revision must precede every GPU measurement')
        old_cfg=json.loads((p/'lock_cpu_initial_config.json').read_text());new_cfg=json.loads((code/'configs/medevidence_p2.json').read_text())
        resource_keys={'max_concurrent_gpus','allowed_gpus','single_gpu_per_job','budget_policy'}
        assert {k:v for k,v in old_cfg.items() if k not in resource_keys}=={k:v for k,v in new_cfg.items() if k not in resource_keys}
        assert (p/'lock_cpu_initial.json').exists()
    checks=json.loads((p/'CPU_checks.json').read_text())
    if checks['status']!='passed' or not checks['overlay_engineering_review'].startswith('passed'):raise PermissionError('CPU/alignment checks incomplete')
    names=['manifest.csv','sets.json','schedule.json','token_contracts.json','selected_images.json','alignment_checks.json',
           'CPU_checks.json','reference_checkpoints.json','reference_identity.json']
    sources=['src/medevidence_p2.py','src/medevidence_p2_run.py','src/model.py','src/data.py','src/experiment.py',
             'src/v2.py','src/medevidence.py','src/medevidence_run.py','src/objectives.py','src/regions.py','src/evaluation.py',
             'scripts/prepare_medevidence_p2.py','scripts/medevidence_p2_pipeline.py','scripts/report_medevidence_p2.py',
             'scripts/launch_medevidence_p2.py','scripts/freeze_medevidence_p2.py','tests/test_medevidence_p2.py']
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    lock={'config_sha256':digest(code/'configs/medevidence_p2.json'),'protocol_hashes':{n:digest(p/n) for n in names},
          'source_hashes':{n:digest(code/n) for n in sources},'freeze_stage':'before_first_GPU_model_measurement',
          'test_pixels_read':0,'GPU_authorization':'required separately, never inherited','public_release_authorization':'required separately'}
    if resource_revision:
        assert lock['protocol_hashes']==previous['protocol_hashes']
        lock['resource_revision']={'user_instruction':'我授权你使用所有 gpu','previous_config_sha256':previous['config_sha256'],
            'changed_source_files':[n for n,h in lock['source_hashes'].items() if h!=previous['source_hashes'][n]],
            'scientific_config_and_private_protocol_unchanged':True,'previous_lock_preserved_private':'lock_cpu_initial.json',
            'revision_stage':'before_first_GPU_model_measurement','max_concurrent_gpus':3,'gpu_hours_limit':2}
    save(dest,lock)
    summary=code/'protocol/medevidence_p2/freeze_summary.json';public=json.loads(summary.read_text());public['lock']=lock;save(summary,public)
    print(json.dumps({'status':'ready_to_launch','private_protocol_files_locked':len(names),'source_files_locked':len(sources),'patients':64,'training_exposures':1024,'GPU_launched':False}))


if __name__=='__main__':main()
