"""Copy only source/config/tests to an explicitly configured execution root."""
import os
from pathlib import Path
import subprocess
import tarfile
import shlex
from runtime import settings

if __name__ == '__main__':
    cfg=settings();root=Path(cfg['REMOTE_ROOT']);base=Path(cfg['REMOTE_BASE_ROOT'])
    if not root.is_absolute() or root.parent != base or base==Path('/'):
        raise ValueError('This deployment is restricted to the authorized data disk')
    process=subprocess.Popen(['ssh',cfg['EXEC_HOST'],'tar -xf - -C '+shlex.quote(str(base))],stdin=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdin,mode='w|') as archive:
        for folder in ['src','scripts','configs','tests']:
            for file in Path(folder).rglob('*'):
                if file.is_file() and '__pycache__' not in file.parts:
                    archive.add(file,arcname=root.name+'/code/'+str(file))
    process.stdin.close()
    if process.wait():raise SystemExit('source transfer failed')
