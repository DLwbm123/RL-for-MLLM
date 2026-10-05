"""Launch one job using neutral argv; paths and experiment settings travel in stdin/env."""
import json
import os
import subprocess
import sys
from runtime import settings as runtime_settings

if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'pilot'
    if action not in ['pilot','smoke']:raise SystemExit('Choose smoke or pilot')
    deployment=runtime_settings()
    settings={k:deployment[k] for k in ['REMOTE_ROOT','REMOTE_PYTHON','GPU_INDEX']}
    settings['forbidden_argv_terms']=deployment.get('FORBIDDEN_ARGV_TERMS','busbra,evidence,qwen,sft,grpo,mllm')
    settings['action']=action
    script='settings='+repr(settings)+'''\nimport os,subprocess,json
from pathlib import Path
r=Path(settings['REMOTE_ROOT'])
if settings['action']=='pilot' and (r/'outputs/pipeline_status.json').exists():
 raise FileExistsError('A pilot already has a status record; inspect it and resume stages explicitly instead of overwriting')
(r/'tmp').mkdir(exist_ok=True)
env=os.environ.copy();env.update(DATA_ROOT=str(r/'data'),MODEL_ROOT=str(r/'models'),CACHE_ROOT=str(r/'cache'),OUTPUT_ROOT=str(r/'outputs'),PYTHONPATH=str(r/'code'),RUN_CONFIG=str(r/'code/configs/pilot.json'),CUDA_VISIBLE_DEVICES=settings['GPU_INDEX'],OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TMPDIR=str(r/'tmp'),WANDB_MODE='disabled')
env['FORBIDDEN_ARGV_TERMS']=settings['forbidden_argv_terms']
free=int(subprocess.check_output(['nvidia-smi','--id='+settings['GPU_INDEX'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<30000:raise RuntimeError('Insufficient GPU memory')
module='scripts.model_smoke' if settings['action']=='smoke' else 'scripts.pipeline'
code='import runpy; runpy.run_module('+repr(module)+',run_name="__main__")'
log=r/'logs'/(settings['action']+'.log')
p=subprocess.Popen([settings['REMOTE_PYTHON'],'-u','-'],stdin=subprocess.PIPE,stdout=open(log,'w'),stderr=subprocess.STDOUT,env=env,cwd=r/'code',start_new_session=True)
p.stdin.write(code.encode());p.stdin.close()
(r/'logs'/(settings['action']+'.pid')).write_text(str(p.pid))
print(json.dumps({'pid':p.pid,'log':str(log),'action':settings['action']}))
'''
    result=subprocess.run(['ssh',deployment['EXEC_HOST'],'python3 -u -'],input=script,text=True)
    raise SystemExit(result.returncode)
