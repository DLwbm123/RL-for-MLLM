"""Download the fixed official release; never disable TLS verification."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
import urllib.request
import zipfile

RECORD = 'https://zenodo.org/api/records/8231412'
EXPECTED = '1f8b2be6476d58fc97bfb5e5a1ea9bab'


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected=EXPECTED):
    actual = digest(path, 'md5')
    if actual != expected:
        raise ValueError(f'MD5 mismatch: {actual} != {expected}')
    return {'md5': actual, 'sha256': digest(path, 'sha256')}


def extract(path, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(path) as z:
        for item in z.infolist():
            dest = (target / item.filename).resolve()
            if not dest.is_relative_to(target) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Unsafe archive member: ' + item.filename)
        bad = z.testzip()
        if bad:
            raise ValueError('Corrupt archive member: ' + bad)
        z.extractall(target)


def download(url, path, size, retries=4):
    path = Path(path)
    if path.exists() and path.stat().st_size == size:
        return
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(retries):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset == size:
                part.replace(path)
                return
            if offset > size:
                raise ValueError('Partial download exceeds official size')
            req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req, timeout=45) as r:
                resume = offset and r.status == 206
                if resume and not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid resume range')
                with part.open('ab' if resume else 'wb') as f:
                    shutil.copyfileobj(r, f, 1024 * 1024)
            if part.stat().st_size != size:
                raise ValueError('Incomplete download')
            part.replace(path)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    root = Path(os.environ['DATA_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    record = json.load(urllib.request.urlopen(RECORD, timeout=45))
    (root / 'zenodo_record.json').write_text(json.dumps(record, indent=2))
    item = next(f for f in record['files'] if f['key'] == 'BUSBRA.zip')
    if record['metadata'].get('version') != '1.0' or item['checksum'] != 'md5:' + EXPECTED:
        raise ValueError('Official release changed; review required')
    if shutil.disk_usage(root).free < 1024 ** 3:
        raise OSError('Need at least 1 GiB free')
    path = root / item['key']
    download(item['links']['self'], path, item['size'])
    checks = verify(path)
    (root / 'archive_integrity.json').write_text(json.dumps(checks, indent=2))
    target = root / 'raw'
    marker = root / 'extraction_complete.json'
    if not marker.exists():
        extract(path, target)
        marker.write_text(json.dumps({'archive': checks, 'target': 'raw'}))
    print(json.dumps({'status': 'complete', 'bytes': path.stat().st_size, **checks}), flush=True)


if __name__ == '__main__':
    main()
