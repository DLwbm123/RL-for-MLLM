"""Detached neutral controller, own authorization, no repeat or budget reset."""
import json
from pathlib import Path
import subprocess


def main():
    runtime=json.loads(Path('private/runtime.json').read_text());auth=json.loads(Path('private/medevidence_p3/authorization.json').read_text())
    if not auth['gpu_authorized'] or auth['gpu_hours_limit']!=3 or auth['max_concurrent_gpus']!=3:raise PermissionError('Own pilot3 resource receipt required')
    payload={'base':runtime['REMOTE_ROOT']+'/runs/medevidence_p3','python':runtime['REMOTE_PYTHON'],'data':runtime['DATA_ROOT']+'/RSNA_Pneumonia','models':runtime['MODEL_ROOT'],'auth':auth}
    entry='''import json,os,subprocess,time
from pathlib import Path
r=json.loads(PAYLOAD);base=Path(r['base']);root=base/'outputs';code=base/'code'
if (root/'launch.json').exists() or (root/'gpu_ledger.json').exists():raise FileExistsError('Existing attempt; audit first')
if not (root/'protocol/lock.json').exists():raise PermissionError('CPU checks and source/config freeze required')
(root/'authorization.json').write_text(json.dumps(r['auth'],indent=2)+'\\n')
env=os.environ.copy();env.update(PILOT_ROOT=str(base),DATA_ROOT=r['data'],MODEL_ROOT=r['models'],DATASET_NAME='rsna',TMPDIR=str(base/'tmp'))
log=(base/'logs/controller.log').open('x');p=subprocess.Popen([r['python'],'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
p.stdin.write(("import sys;sys.path.insert(0,"+repr(str(code))+");from scripts.medevidence_p3_pipeline import main;main()\\n").encode());p.stdin.close();log.close()
receipt={'status':'launched','controller_pid':p.pid,'started':time.time(),'run':'medevidence_p3'}
(root/'launch.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    result=subprocess.run(['ssh',runtime['EXEC_HOST'],runtime['REMOTE_PYTHON'],'-'],input=entry,text=True,capture_output=True,check=True);print(result.stdout,end='')


if __name__=='__main__':main()
