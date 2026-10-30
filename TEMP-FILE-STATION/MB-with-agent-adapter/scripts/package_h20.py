"""Build a code-only Linux transfer archive, with per-file hashes and exclusions."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    '.git', '.venv', 'venv', 'env', '__pycache__', 'node_modules', '.cache',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', 'data', 'results', 'logs',
    'models', 'embedding-model', 'references', 'dist', 'build', '.plans',
}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.log', '.safetensors', '.gguf', '.pt', '.pth', '.ckpt'}


def allowed(path):
    if any(part in SKIP_DIRS or part.endswith('.egg-info') for part in path.parts):
        return False
    if path.name.startswith('.env') and not path.name.endswith('.example'):
        return False
    return path.suffix.lower() not in SKIP_SUFFIXES and path.name not in {'.coverage', '.DS_Store'}


def source_files(root, selected=None):
    entries = selected or sorted(p.name for p in root.iterdir())
    for entry in entries:
        start = root / entry
        if not start.exists():
            raise FileNotFoundError(start)
        if start.is_symlink() or not allowed(start.relative_to(root)):
            continue
        if start.is_file():
            yield start
            continue
        for current, dirs, files in os.walk(start):
            dirs[:] = sorted(d for d in dirs if not (Path(current)/d).is_symlink()
                             and allowed((Path(current)/d).relative_to(root)))
            for name in sorted(files):
                path = Path(current)/name
                if not path.is_symlink() and allowed(path.relative_to(root)):
                    yield path


def local_secrets():
    secrets = []
    for env in ROOT.glob('.env*'):
        if not env.is_file() or env.name.endswith('.example'):
            continue
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            name, sep, value = line.partition('=')
            if sep and not name.lstrip().startswith('#') and any(
                    marker in name.upper() for marker in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
                value = value.strip().strip('\"\'')
                if len(value) >= 12 and not value.startswith(('<', '${')):
                    secrets.append(value.encode())
    return secrets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-src', type=Path, default=ROOT.parent/'hermes-agent')
    parser.add_argument('--output', type=Path, default=ROOT/'dist'/f'MB-Hermes-H20-code-{datetime.now():%Y%m%d}.tar.gz')
    args = parser.parse_args()
    hermes = args.hermes_src.resolve()
    if not (hermes/'run_agent.py').is_file():
        parser.error('Hermes source is missing')
    output = args.output.resolve()
    if output.exists():
        parser.error('Archive already exists; choose another output filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    secrets = local_secrets()
    records = []
    benchmark_entries = [
        'src', 'configs', 'mcp_server', 'scripts', 'tests', 'pyproject.toml',
        'uv.lock', 'README.md', '.python-version', '.gitignore', '.env.h20.example',
        'docs/h20-quickstart.md',
    ]
    sources = [('MB-with-agent-adapter', ROOT, benchmark_entries), ('hermes-agent', hermes, None)]
    stamp = int(datetime.now(timezone.utc).timestamp())
    pending = output.with_name(output.name+'.partial')
    if pending.exists():
        parser.error('Partial archive already exists; inspect it before rebuilding')

    def add(archive, name, payload, mode=0o644):
        if any(secret in payload for secret in secrets):
            raise RuntimeError(f'Local credential found in package candidate: {name}')
        info = tarfile.TarInfo(name)
        info.size, info.mtime, info.mode = len(payload), stamp, mode
        archive.addfile(info, io.BytesIO(payload))

    with tarfile.open(pending, 'w:gz', compresslevel=6) as archive:
        for prefix, root, selected in sources:
            for path in source_files(root, selected):
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f'Path escaped source root: {path}')
                raw = path.read_bytes()
                executable = path.suffix == '.sh' or raw.startswith(b'#!')
                payload = raw.replace(b'\r\n', b'\n') if executable else raw
                name = prefix+'/'+path.relative_to(root).as_posix()
                add(archive, name, payload, 0o755 if executable else 0o644)
                records.append({'path': name, 'bytes': len(payload),
                                'sha256': hashlib.sha256(payload).hexdigest(),
                                'normalized_shell_line_endings': payload != raw})
        quickstart = (ROOT/'docs/h20-quickstart.md').read_bytes().replace(b'\r\n', b'\n')
        add(archive, 'README-H20.md', quickstart)
        records.append({'path': 'README-H20.md', 'bytes': len(quickstart),
                        'sha256': hashlib.sha256(quickstart).hexdigest(),
                        'normalized_shell_line_endings': False})
        manifest = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'mobilitybench_upstream_commit': '48d005c90c3f807d265a67cb213633473f1c4f80',
            'hermes_commit': '498bfc7bc12a937621b4215312049b1000726df3',
            'includes_local_adapter_changes': True,
            'prompt': 'Native Reporter JSON fields, original zero-based route index, empty when unavailable, -1 only when explicitly unreachable.',
            'excluded': sorted(SKIP_DIRS) + ['real .env credentials', 'model weights', 'bytecode', 'symlinks'],
            'files': records,
        }
        add(archive, 'MANIFEST-H20.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    with tarfile.open(pending, 'r:gz') as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        for record in records:
            payload = archive.extractfile(record['path']).read()
            assert hashlib.sha256(payload).hexdigest() == record['sha256'], record['path']
    pending.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name+'.sha256').write_text(f'{digest}  {output.name}\n', encoding='ascii')
    print(json.dumps({'archive': str(output), 'files': len(records)+1,
                      'archive_bytes': output.stat().st_size, 'sha256': digest}, indent=2))


if __name__ == '__main__':
    main()
