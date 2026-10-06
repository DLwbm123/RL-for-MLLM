"""Neutral-argv CPU entry and read-only optional-stage readiness checks."""
import json
import os
from pathlib import Path
import subprocess
from runtime import settings


def main():
    operation=os.environ.get('V3_OPERATION','collect')
    if operation in ['G1','G2']:
        amendment=Path('configs/rsna_v3_execution_amendment.json')
        authorized=amendment.exists() and json.loads(amendment.read_text()).get('gpu_authorized',False)
        status=json.loads(Path('reports/rsna_v3_stage_status.json').read_text())['V3-'+operation]
        print(json.dumps({'stage':'V3-'+operation,**status,'gpu_authorized':authorized,
                          'launched':False,'entry_kind':'readiness_check_only'}));return
    entries={'prepare':'from scripts.replay_rsna_v3 import prepare;prepare()',
             'replay':'from scripts.replay_rsna_v3 import replay;replay()',
             'review':'from scripts.prepare_rsna_v3_review import main;main()',
             'numerics':'from scripts.summarize_rsna_v3_replay import main;main()'}
    if operation not in [*entries,'collect']:raise ValueError('Unsupported operation; no training entry')
    runtime=settings();payload={k:runtime[k] for k in ['REMOTE_ROOT','REMOTE_PYTHON']}
    script='deployment='+repr(payload)+'\noperation='+repr(operation)+'\nentry='+repr(entries.get(operation))+'\n'+'''
import os,subprocess,json,time
from pathlib import Path
b=Path(deployment['REMOTE_ROOT']);r=b/'runs/rsna_diagnostic_v3'
if operation=='collect':
 out={}
 for name in ['stage_status.json','budget.json','cpu_tests.json','protocol/protocol_lock.json','reward_replay/summary.json']:
  p=r/'outputs'/name
  if p.exists():out[name]=p.read_text()
 print(json.dumps(out))
else:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'code'),DATASET_NAME='rsna',DATA_ROOT=str(b/'data/RSNA_Pneumonia'),MODEL_ROOT=str(b/'models'),V1_OUTPUT_ROOT=str(b/'runs/rsna_pilot_v1/outputs'),V2_OUTPUT_ROOT=str(b/'runs/rsna_diagnostic_v2/outputs'),OUTPUT_ROOT=str(r/'outputs'),RUN_CONFIG=str(r/'code/configs/rsna_v3.json'),V3_ORIGIN_HEAD='76816e51c98f6ec2e41260d3784bc6e11b596f5a',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',REVIEW_SEED='42')
 result=subprocess.run([deployment['REMOTE_PYTHON'],'-u','-'],input=entry,text=True,env=env,cwd=r/'code',capture_output=True)
 (r/'logs'/(operation+'_'+str(time.time_ns())+'.log')).write_text(result.stdout+'\\n'+result.stderr)
 print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)
'''
    result=subprocess.run(['ssh',runtime['EXEC_HOST'],'python3 -u -'],input=script,text=True,capture_output=True)
    if operation=='collect' and result.returncode==0:
        out=Path('private/rsna_v3');out.mkdir(exist_ok=True)
        for name,content in json.loads(result.stdout).items():(out/name.replace('/','_')).write_text(content)
        print(json.dumps({'private_status_downloaded':True}))
    else:print(result.stdout);print(result.stderr)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
