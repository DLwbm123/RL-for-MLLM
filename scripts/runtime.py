"""Deployment settings live in a private file or environment, never in source."""
import json
import os
from pathlib import Path


def settings():
    file=Path('private/runtime.json')
    result=json.loads(file.read_text()) if file.exists() else {}
    result.update({k:v for k,v in os.environ.items() if k in ['EXEC_HOST','REMOTE_ROOT','REMOTE_BASE_ROOT','REMOTE_PYTHON','GPU_INDEX','FORBIDDEN_ARGV_TERMS']})
    return result
