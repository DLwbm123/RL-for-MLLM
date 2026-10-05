import json
import os
from pathlib import Path
import time
from huggingface_hub import snapshot_download
from src.model import REVISION

if __name__=='__main__':
    destination=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    for attempt in range(4):
        try:
            result=snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct',revision=REVISION,local_dir=destination,max_workers=2)
            print(json.dumps({'status':'complete','path':result,'revision':REVISION}),flush=True)
            break
        except Exception as error:
            print(json.dumps({'attempt':attempt+1,'error':repr(error)}),flush=True)
            if attempt==3:raise
            time.sleep(5*(attempt+1))
