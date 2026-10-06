"""Start the frozen chain using private runtime settings and neutral stdin commands."""
import json
import os
from pathlib import Path
import subprocess


def main():
    runtime=json.loads(Path('private/runtime.json').read_text())
    authorization=json.loads(Path('private/medevidence_p1/authorization.json').read_text())
    if not authorization.get('authorized') or authorization.get('gpu_hours_limit')!=6 or authorization.get('max_concurrent_gpus')!=3:
        raise PermissionError('This pilot requires its own explicit GPU authorization')
    action=os.environ.get('PIPELINE_ACTION','formal')
    if action not in ('smoke','formal'):raise ValueError('Unknown pipeline action')
    payload={'base':runtime['REMOTE_ROOT']+'/runs/medevidence_p1','python':runtime['REMOTE_PYTHON'],
             'data':runtime['DATA_ROOT'],'models':runtime['MODEL_ROOT'],'action':action}
    entry='''import json,os,subprocess,time
from pathlib import Path
r=json.loads(PAYLOAD)
base=Path(r['base']);code=base/'code';root=base/'outputs'
if r['action']=='formal' and (root/'budget_plan.json').exists():raise FileExistsError('Formal attempt already exists; no automatic repeat')
env=os.environ.copy();env.update(PILOT_ROOT=str(base),PIPELINE_ACTION=r['action'],DATA_ROOT=r['data'],MODEL_ROOT=r['models'],DATASET_NAME='rsna',TMPDIR=str(base/'tmp'))
log=(base/'logs'/('controller_'+r['action']+'.log')).open('x')
p=subprocess.Popen([r['python'],'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code,start_new_session=True)
p.stdin.write(("import sys;sys.path.insert(0,"+repr(str(code))+ ");from scripts.medevidence_pipeline import main;main()\\n").encode());p.stdin.close();log.close()
receipt={'controller_pid':p.pid,'started':time.time(),'action':r['action'],'status':'launched','output_root':str(root)}
(root/(r['action']+'_launch.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    result=subprocess.run(['ssh',runtime['EXEC_HOST'],runtime['REMOTE_PYTHON'],'-'],input=entry,text=True,check=True,capture_output=True)
    print(result.stdout,end='')


if __name__=='__main__':main()
