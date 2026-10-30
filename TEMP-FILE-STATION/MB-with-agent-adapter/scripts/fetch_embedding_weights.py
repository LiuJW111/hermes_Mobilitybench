"""Fetch the public evaluation weight, preferring the faster official mirror."""
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
EXPECTED_SHA256 = '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
EXPECTED_SIZE = 1191586416
SOURCES = {
    'huggingface': f'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/{REVISION}/model.safetensors?download=true',
    'modelscope': 'https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B/resolve/master/model.safetensors',
}
target = ROOT/'embedding-model/model.safetensors'
partial = target.with_suffix('.safetensors.partial')
if target.exists():
    with target.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == EXPECTED_SHA256
    print('Existing weight verified', flush=True)
    raise SystemExit(0)

speeds = {}
for name, url in SOURCES.items():
    try:
        start = time.monotonic()
        count = 0
        with requests.get(url, headers={'Range': 'bytes=0-8388607'}, stream=True, timeout=(12, 15)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1024*1024):
                count += len(chunk)
                if count >= 8388608 or time.monotonic()-start >= 20:
                    break
        speeds[name] = count/max(time.monotonic()-start, 0.001)
        print(f'{name}: {speeds[name]/1024/1024:.2f} MiB/s', flush=True)
    except requests.RequestException as exc:
        print(f'{name}: {type(exc).__name__}', flush=True)
if not speeds:
    raise RuntimeError('Both official download sources unavailable')

source = max(speeds, key=speeds.get)
print(f'Downloading from {source}', flush=True)
count = 0
digest = hashlib.sha256()
last_progress = time.monotonic()
with requests.get(SOURCES[source], stream=True, timeout=(20, 60)) as response:
    response.raise_for_status()
    if response.status_code != 200:
        raise RuntimeError(f'Expected full file, got HTTP {response.status_code}')
    with partial.open('wb') as f:
        for chunk in response.iter_content(4*1024*1024):
            f.write(chunk)
            digest.update(chunk)
            count += len(chunk)
            if time.monotonic()-last_progress >= 15:
                print(f'{count/EXPECTED_SIZE:.1%}: {count/1024/1024:.0f} MiB', flush=True)
                last_progress = time.monotonic()
assert count == EXPECTED_SIZE, (count, EXPECTED_SIZE)
assert digest.hexdigest() == EXPECTED_SHA256, 'Mirror weight does not match pinned upstream weight'
partial.replace(target)
(ROOT/'docs/audit/embedding-weight.json').write_text(json.dumps({
    'model': 'Qwen/Qwen3-Embedding-0.6B', 'hf_revision': REVISION,
    'source': SOURCES[source], 'sha256': EXPECTED_SHA256, 'size': count,
    'verified': True, 'probe_mib_per_second': {k: round(v/1024/1024, 2) for k,v in speeds.items()},
}, indent=2), encoding='utf-8')
print('Weight downloaded and SHA-256 verified', flush=True)
