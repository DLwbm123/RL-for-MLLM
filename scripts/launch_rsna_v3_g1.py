"""Single-use, neutral-argv launcher for the independently frozen G1 manifest."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def watch(workers, deadline):
    """Stop this launch's process groups at the deadline or first worker failure."""
    reason = None
    while True:
        now = time.time()
        for process, record in workers:
            code = process.poll()
            if code is not None and record.get('finished') is None:
                record.update(finished=now, exit_code=code)
        if all(r.get('finished') is not None for _, r in workers):
            return reason
        if any(r.get('exit_code', 0) not in (None, 0) for _, r in workers):
            reason = 'failed_worker'
        elif now >= deadline:
            reason = 'partial_budget'
        if reason:
            for process, record in workers:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process, record in workers:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                if record.get('finished') is None:
                    record.update(finished=time.time(), exit_code=process.returncode)
            return reason
        time.sleep(.1)


def main():
    root = Path(os.environ['OUTPUT_ROOT']).resolve()
    dest = root / 'G1'
    manifest_path = dest / 'protocol/manifest.json'
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != os.environ['V3_G1_MANIFEST_SHA256']:
        raise PermissionError('Manifest differs from externally frozen digest')
    manifest = json.loads(manifest_path.read_text())
    amendment = json.loads((root / 'protocol/execution_amendment.json').read_text())
    if (not amendment['gpu_authorized'] or not manifest['selected']
            or manifest.get('training_enabled') is not False):
        raise PermissionError('Missing authorization, qualified patients or no-training guard')
    gpus = amendment['gpu_indices']
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise PermissionError('Two distinct authorized GPU indices required')
    budget = json.loads((root / 'budget.json').read_text())
    if budget['limit_GPU_seconds'] != 3600:
        raise PermissionError('Frozen cumulative GPU budget must remain 3600 seconds')
    remaining = budget['limit_GPU_seconds'] - budget['consumed_GPU_seconds']
    if remaining <= 60:
        raise PermissionError('Insufficient remaining GPU budget')
    charged = [bool(manifest['selected'][i::2]) for i in range(2)]
    for index, gpu in enumerate(gpus):
        if charged[index]:
            free = int(subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free < 30000:
                raise RuntimeError('Insufficient available GPU memory with peak margin')
    receipt_path = dest / 'launch_receipt.json'
    receipt = {'status': 'running', 'pid': os.getpid(), 'started': time.time(),
               'manifest_sha256': digest, 'workers': []}
    # Exclusive creation is the no-retry/concurrent-launch guard.
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream)
    workers, logs = [], []
    # Reserve shutdown time; each charged worker gets an equal bounded share.
    deadline = time.time() + (remaining - 20) / sum(charged)
    try:
        for index, gpu in enumerate(gpus):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu) if charged[index] else '',
                       V3_G1_SHARD=str(index), V3_G1_SHARDS='2',
                       V3_G1_DEADLINE=str(deadline), V3_G1_MANIFEST=str(manifest_path))
            log = (dest / f'shard_{index}.log').open('x'); logs.append(log)
            record = {'shard': index, 'gpu': gpu if charged[index] else None,
                      'started': time.time(), 'finished': None}
            process = subprocess.Popen([sys.executable, '-u', '-'], stdin=subprocess.PIPE,
                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            workers.append((process, record)); receipt['workers'].append(record)
            record['pid'] = process.pid
            save(receipt_path, receipt)
            process.stdin.write(b'from scripts.run_rsna_v3_g1 import run\nrun()\n')
            process.stdin.close()
        reason = watch(workers, deadline)
        results = [json.loads((dest / f'shard_{i}/result.json').read_text())
                   if (dest / f'shard_{i}/result.json').exists() else {'status': 'failed'}
                   for i in range(2)]
        receipt['status'] = ('partial_budget' if reason == 'partial_budget'
            else 'failed' if reason == 'failed_worker' or any(r['exit_code'] != 0 for _, r in workers)
                or any(r['status'] == 'failed' for r in results)
            else 'partial_budget' if any(r['status'] == 'partial_budget' for r in results)
            else 'completed')
        receipt['stop_reason'] = reason
    except BaseException:
        watch(workers, time.time())
        receipt['status'] = 'failed'
        raise
    finally:
        for log in logs:
            log.close()
        receipt['finished'] = time.time()
        for _, record in workers:
            record['wall_seconds'] = record['finished'] - record['started']
            record['gpu_seconds'] = record['wall_seconds'] if record['gpu'] is not None else 0
        consumed = sum(r['gpu_seconds'] for _, r in workers)
        budget['consumed_GPU_seconds'] += consumed
        budget.setdefault('attempts', []).append({'stage': 'G1', 'gpu_seconds': consumed,
            'status': receipt['status'], 'workers': receipt['workers']})
        save(root / 'budget.json', budget); save(receipt_path, receipt)
        statuses = json.loads((root / 'stage_status.json').read_text())
        statuses['V3-G1'].update(status=receipt['status'], gpu_seconds=consumed)
        save(root / 'stage_status.json', statuses)
        print(json.dumps({'status': receipt['status'], 'gpu_seconds': consumed}), flush=True)


if __name__ == '__main__':
    main()
