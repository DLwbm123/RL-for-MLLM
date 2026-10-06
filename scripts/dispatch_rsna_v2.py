"""Neutral-argv v2 deployment. Upload source only; never uploads data/reports or calls git."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from runtime import settings


def main():
    cfg=settings();operation=os.environ.get('V2_OPERATION','tests')
    if operation not in ['sync','tests','prepare','launch','collect']:raise ValueError('Invalid operation')
    buffer=io.BytesIO()
    if operation=='sync':
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for folder in ['src','scripts','configs','tests']:
                for p in Path(folder).rglob('*'):
                    if p.is_file() and p.suffix in ['.py','.json']:archive.add(p,arcname=str(p))
    authorization=None
    if operation=='launch':authorization=json.loads(Path('private/rsna_v2_gpu_authorization.json').read_text())
    payload={'deployment':{k:cfg[k] for k in ['REMOTE_ROOT','REMOTE_PYTHON','GPU_INDEX']},'operation':operation,
             'source':base64.b64encode(buffer.getvalue()).decode(),'authorization':authorization,
             'origin_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    script='settings='+repr(payload)+'''
import os,io,tarfile,base64,json,subprocess,time
from pathlib import Path
cfg=settings['deployment'];base=Path(cfg['REMOTE_ROOT']);r=base/'runs/rsna_diagnostic_v2';op=settings['operation']
if op=='sync':
 if (r/'outputs/protocol/protocol_lock.json').exists():raise FileExistsError('Frozen protocol: source cannot be overwritten')
 print(subprocess.check_output(['findmnt','-T',str(base),'-o','TARGET,SOURCE,FSTYPE,OPTIONS'],text=True).strip())
 stats=os.statvfs(base);free=stats.f_bavail*stats.f_frsize
 if free<20*1024**3:raise RuntimeError('Less than 20 GiB storage margin')
 probe=base/('.v2_probe_'+str(os.getpid()));probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
 for d in ['code','outputs','logs','tmp']:(r/d).mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(base64.b64decode(settings['source'])),mode='r:gz') as archive:archive.extractall(r/'code',filter='data')
 print(json.dumps({'source_deployed':True,'free_bytes':free}));raise SystemExit(0)
env=os.environ.copy();env.update(DATASET_NAME='rsna',DATA_ROOT=str(base/'data/RSNA_Pneumonia'),MODEL_ROOT=str(base/'models'),V1_OUTPUT_ROOT=str(base/'runs/rsna_pilot_v1/outputs'),OUTPUT_ROOT=str(r/'outputs'),RUN_CONFIG=str(r/'code/configs/rsna_v2.json'),PYTHONPATH=str(r/'code'),TMPDIR=str(r/'tmp'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',FORBIDDEN_ARGV_TERMS='wangbomin,rsna,busbra,evidence,qwen,sft,grpo,mllm',V2_ORIGIN_HEAD=settings['origin_head'])
if op in ['tests','prepare']:
 env['CUDA_VISIBLE_DEVICES']=''
 if op=='tests':
  env.pop('DATASET_NAME',None)
  code="import pytest;raise SystemExit(pytest.main(['-q','tests']))"
 else:code="from scripts.prepare_rsna_v2 import main;main()"
 result=subprocess.run([cfg['REMOTE_PYTHON'],'-u','-'],input=code,text=True,env=env,cwd=r/'code',capture_output=True)
 (r/'logs'/(op+'_'+str(time.time_ns())+'.log')).write_text(result.stdout+'\\n'+result.stderr)
 print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)
if op=='launch':
 auth=settings['authorization']
 if auth.get('authorized') is not True:raise PermissionError('Explicit v2 GPU authorization is missing')
 if (r/'outputs/launch_receipt.json').exists() or (r/'logs/supervisor.pid').exists():raise FileExistsError('v2 already launched; no automatic retry')
 if not (r/'outputs/preflight.json').exists() or json.loads((r/'outputs/preflight.json').read_text())['status']!='passed':raise RuntimeError('Preflight not passed')
 free=int(subprocess.check_output(['nvidia-smi','--id='+cfg['GPU_INDEX'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 if free<30000:raise RuntimeError('Insufficient single-GPU memory margin')
 authfile=r/'outputs/gpu_authorization.json';authfile.write_text(json.dumps(auth,indent=2))
 indices=auth.get('gpu_indices',[cfg['GPU_INDEX']])
 if not 1<=len(indices)<=auth.get('max_concurrent_gpus',1):raise PermissionError('GPU count exceeds authorization')
 for index in indices:
  memory=int(subprocess.check_output(['nvidia-smi','--id='+str(index),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if memory<30000:raise RuntimeError('Insufficient selected GPU memory')
 env.update(CUDA_VISIBLE_DEVICES=str(indices[0]),V2_GPU_INDICES=','.join(map(str,indices)),V2_GPU_AUTH_FILE=str(authfile))
 code="from scripts.rsna_v2_pipeline import main;main()"
 with (r/'logs/supervisor.log').open('x') as log:
  p=subprocess.Popen([cfg['REMOTE_PYTHON'],'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=r/'code',start_new_session=True)
  p.stdin.write(code.encode());p.stdin.close()
 (r/'logs/supervisor.pid').write_text(str(p.pid));print(json.dumps({'pid':p.pid,'gpus':indices,'free_mib':free,'root':str(r)}));raise SystemExit(0)
if op=='collect':
 result={}
 for p in (r/'outputs/reports').glob('*'):
  if p.suffix in ['.json','.md']:result[p.name]=p.read_text()
 for n in ['preflight.json','stage_status.json','budget.json','protocol/protocol_lock.json']:
  p=r/'outputs'/n
  if p.exists():result['private/'+n.replace('/','_')]=p.read_text()
 print(json.dumps(result))
'''
    result=subprocess.run(['ssh',cfg['EXEC_HOST'],'python3 -u -'],input=script,text=True,capture_output=True)
    if operation=='collect' and result.returncode==0:
        collected=json.loads(result.stdout)
        for n,content in collected.items():
            dest=Path('private/rsna_v2')/n if n.startswith('private/') else Path('reports')/n
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(content)
        print(json.dumps({'downloaded_reports':list(collected)}))
    else:print(result.stdout);print(result.stderr)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
