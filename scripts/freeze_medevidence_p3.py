"""One freeze after CPU checks and before native model measurement."""
import json
import os
from pathlib import Path
from src.data import sha
from src.v2 import save


def main():
    base=Path(os.environ['PILOT_ROOT']);code=base/'code';p=base/'outputs/protocol';dest=p/'lock.json'
    if dest.exists() or (base/'outputs/gpu_ledger.json').exists():raise FileExistsError('Already frozen/started')
    checks=json.loads((p/'CPU_checks.json').read_text());assert checks['status']=='passed'
    old=json.loads((Path(os.environ['REFERENCE_P2_ROOT'])/'protocol/lock.json').read_text())
    source_names=list(old['source_hashes'])+['src/medevidence_p3.py','src/medevidence_p3_run.py','scripts/prepare_medevidence_p3.py',
        'scripts/medevidence_p3_pipeline.py','scripts/launch_medevidence_p3.py','scripts/freeze_medevidence_p3.py','scripts/report_medevidence_p3.py','tests/test_medevidence_p3.py','configs/medevidence_p3.json']
    protocol_names=['manifest.csv','selected_patients.json','schedule.json','local_pairs.json','token_contracts.json','selected_images.json','reference.json','identity.json','folds.json','CPU_checks.json']
    lock={'code_hashes':{n:sha((code/n).read_bytes()) for n in source_names},'protocol_hashes':{n:sha((p/n).read_bytes()) for n in protocol_names},
          'phase':'before_first_GPU_measurement','initialization':'pilot1 original M1 final','reference_commit':'21e7914370081f504d8efd4f6795a8f24785732b',
          'GPU_limit_hours':3,'max_concurrent_gpus':3,'single_gpu_per_job':True,'test_pixels_read':0}
    save(dest,lock);save(code/'protocol/medevidence_p3/freeze_summary.json',lock)
    print(json.dumps({'status':'ready_to_launch','protocol_files':len(protocol_names),'code_files':len(source_names),'optimizer_updates':0}))


if __name__=='__main__':main()
