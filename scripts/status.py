"""One read-only snapshot, without scheduling or recurring monitoring."""
import subprocess
from runtime import settings

if __name__=='__main__':
    cfg=settings()
    script='root='+repr(cfg['REMOTE_ROOT'])+'''\nfrom pathlib import Path
import json,subprocess
r=Path(root);p=r/'outputs/pipeline_status.json'
print(p.read_text() if p.exists() else 'No pilot status yet')
pidfile=r/'logs/pilot.pid'
if pidfile.exists():
 pid=pidfile.read_text().strip()
 print(subprocess.check_output(['ps','-p',pid,'-o','pid,ppid,etime,args='],text=True) if Path('/proc/'+pid).exists() else 'Pilot process has exited')
'''
    raise SystemExit(subprocess.run(['ssh',cfg['EXEC_HOST'],'python3 -u -'],input=script,text=True).returncode)
