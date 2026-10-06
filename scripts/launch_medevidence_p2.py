"""Launch only with this pilot's own receipt; remote argv is neutral stdin."""
import json
import subprocess
from pathlib import Path


def main():
    runtime=json.loads(Path('private/runtime.json').read_text())
    auth=json.loads(Path('private/medevidence_p2/authorization.json').read_text())
    if not auth.get('gpu_authorized') or auth['gpu_hours_limit']!=2 or not 1<=auth['max_concurrent_gpus']<=8:raise PermissionError('Pilot-2 needs separate user GPU authorization')
    payload={'base':runtime['REMOTE_ROOT']+'/runs/medevidence_p2','python':runtime['REMOTE_PYTHON'],
             'data':runtime['DATA_ROOT']+'/RSNA_Pneumonia','models':runtime['MODEL_ROOT'],'authorization':auth}
    entry='''import json,os,subprocess,time
from pathlib import Path
r=json.loads(PAYLOAD);base=Path(r['base']);root=base/'outputs';code=base/'code'
if (root/'gpu_ledger.json').exists() or (root/'launch.json').exists():raise FileExistsError('No repeated pilot attempt')
if not (root/'protocol/lock.json').exists():raise PermissionError('CPU checks and freeze required')
(root/'authorization.json').write_text(json.dumps(r['authorization'],indent=2)+'\\n')
env=os.environ.copy();env.update(PILOT_ROOT=str(base),DATA_ROOT=r['data'],MODEL_ROOT=r['models'],DATASET_NAME='rsna',TMPDIR=str(base/'tmp'))
log=(base/'logs/controller.log').open('x')
p=subprocess.Popen([r['python'],'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
p.stdin.write(("import sys;sys.path.insert(0,"+repr(str(code))+");from scripts.medevidence_p2_pipeline import main;main()\\n").encode());p.stdin.close();log.close()
receipt={'status':'launched','controller_pid':p.pid,'started':time.time(),'run':'medevidence_p2'}
(root/'launch.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    result=subprocess.run(['ssh',runtime['EXEC_HOST'],runtime['REMOTE_PYTHON'],'-'],input=entry,text=True,check=True,capture_output=True)
    print(result.stdout,end='')


if __name__=='__main__':main()
