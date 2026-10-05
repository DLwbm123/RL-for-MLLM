"""Start the bounded RSNA smoke + diagnostic chain; no polling or automatic retries."""
import subprocess
from runtime import settings

if __name__=='__main__':
    cfg=settings()
    script='cfg='+repr({k:cfg[k] for k in ['REMOTE_ROOT','REMOTE_PYTHON','GPU_INDEX']})+'''
import json,os,subprocess
from pathlib import Path
base=Path(cfg['REMOTE_ROOT']);r=base/'runs/rsna_pilot_v1';out=r/'outputs'
if (out/'launch_status.json').exists():raise FileExistsError('Run already launched; inspect instead of overwriting')
free=int(subprocess.check_output(['nvidia-smi','--id='+cfg['GPU_INDEX'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<30000:raise RuntimeError('Insufficient free GPU memory')
env=os.environ.copy();env.update(DATASET_NAME='rsna',DATA_ROOT=str(base/'data/RSNA_Pneumonia'),OUTPUT_ROOT=str(out),MODEL_ROOT=str(base/'models'),CACHE_ROOT=str(base/'cache'),RUN_CONFIG=str(r/'code/configs/rsna_pilot.json'),PYTHONPATH=str(r/'code'),CUDA_VISIBLE_DEVICES=cfg['GPU_INDEX'],OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TMPDIR=str(r/'tmp'),WANDB_MODE='disabled')
code="""
import subprocess,sys,os,json,time
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT'])
def state(value): (r/'launch_status.json').write_text(json.dumps(value,indent=2))
try:
 for stage,module in [('smoke','scripts.model_smoke'),('pilot','scripts.pipeline')]:
  state({'status':'running','stage':stage,'started':time.time()})
  with (r.parent/'logs'/(stage+'.log')).open('w') as log:
   p=subprocess.run([sys.executable,'-u','-'],input='import runpy;runpy.run_module('+repr(module)+',run_name="__main__")',text=True,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError(stage+' failed; see stage log')
 state({'status':'chain_exited','pipeline_status':json.loads((r/'pipeline_status.json').read_text())})
except Exception as e:
 state({'status':'failed','error':str(e)});raise
"""
with (r/'logs/supervisor.log').open('w') as log:
 p=subprocess.Popen([cfg['REMOTE_PYTHON'],'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=r/'code',start_new_session=True)
 p.stdin.write(code.encode());p.stdin.close()
(r/'logs/supervisor.pid').write_text(str(p.pid))
print(json.dumps({'pid':p.pid,'gpu':cfg['GPU_INDEX'],'free_mib_at_launch':free,'root':str(r),'log':str(r/'logs/supervisor.log')}))
'''
    raise SystemExit(subprocess.run(['ssh',cfg['EXEC_HOST'],'python3 -u -'],input=script,text=True).returncode)
