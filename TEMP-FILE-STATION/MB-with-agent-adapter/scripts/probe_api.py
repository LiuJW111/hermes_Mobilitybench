"""Small credential/endpoint probe. Does not print credentials."""
import argparse
import json
from pathlib import Path

import httpx
from dotenv import dotenv_values

p = argparse.ArgumentParser()
p.add_argument('--env-file', type=Path, required=True)
args = p.parse_args()
cfg = dotenv_values(args.env_file)
payload = {'model': cfg['LLM_MODEL'], 'messages': [{'role': 'user', 'content': 'Reply OK.'}], 'max_tokens': 64}
if cfg['LLM_MODEL'].lower().startswith('qwen'):
    payload['enable_thinking'] = False  # tiny transport probe, not benchmark configuration
elif cfg['LLM_MODEL'].lower().startswith('glm'):
    payload['thinking'] = {'type': 'disabled'}
response = httpx.post(
    cfg['LLM_BASE_URL'].rstrip('/') + '/chat/completions',
    headers={'Authorization': 'Bearer ' + cfg['LLM_API_KEY']},
    json=payload,
    timeout=60,
)
data = response.json()
print(json.dumps({'status': response.status_code, 'model': cfg['LLM_MODEL'],
                  'usage': data.get('usage'), 'error': data.get('error'),
                  'answer': (data.get('choices') or [{}])[0].get('message', {}).get('content')}, ensure_ascii=False))
response.raise_for_status()
