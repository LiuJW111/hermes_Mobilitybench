"""Run real Hermes through MobilityBench's BatchRunner, then selected metrics.

No reference answer or expected tool list is passed to the agent. Every case
gets an isolated Hermes home. Small defaults keep API verification bounded.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dotenv import load_dotenv


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env.api')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/datasets/sample_10.csv')
    parser.add_argument('--sandbox-dir', type=Path)
    parser.add_argument('--model', help='Optional served model ID override')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--full-evaluation', action='store_true', help='Use all five unchanged native metrics')
    parser.add_argument('--allow-native-fallback', action='store_true', help='Allow the native character-similarity fallback when embedding weights are absent')
    parser.add_argument('--resume', action='store_true', help='Continue the same frozen experiment without rerunning saved cases')
    parser.add_argument('--continue-on-error', action='store_true', help='Save isolated case failures and continue; stop after three consecutive failures')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('--limit must be positive; choose an explicit size for API runs')
    load_dotenv(args.env_file, override=True)
    model = args.model or os.environ.get('LLM_MODEL', '')
    base = os.environ.get('LLM_BASE_URL', '')
    key = os.environ.get('LLM_API_KEY', '')
    if not all((model, base, key)):
        parser.error('Configure LLM_MODEL, LLM_BASE_URL and LLM_API_KEY')
    os.environ.setdefault('HERMES_SRC', str(ROOT.parent / 'hermes-agent'))
    source = Path(os.environ['HERMES_SRC']).resolve()
    os.environ.setdefault('HERMES_PY', str(source / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')))
    sandbox = args.sandbox_dir or Path(os.environ.get('MOBILITY_BENCH_SANDBOX_DIR', str(ROOT / 'data/sandbox')))
    sandbox = sandbox.resolve()
    # Resolve the exact filenames from the upstream tool implementations.
    import re
    expected = sorted(set(re.findall(r'"([\w.-]+\.json)"', '\n'.join(
        p.read_text(encoding='utf-8') for p in (ROOT/'src/mobility_bench/tools/sandbox').glob('*.py')
    ))))
    missing = [name for name in expected if not (sandbox/name).is_file()]
    if missing:
        parser.error(f'Missing sandbox files under {sandbox}: {missing}')
    if not (source/'run_agent.py').is_file() or not Path(os.environ['HERMES_PY']).is_file():
        parser.error('HERMES_SRC or HERMES_PY does not exist')
    os.environ['MOBILITY_BENCH_SANDBOX_DIR'] = str(sandbox)
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.chdir(ROOT)

    import pandas as pd
    from mobility_bench.config.settings import ModelConfig, Settings
    from mobility_bench.dataset.loader import DatasetLoader
    from mobility_bench.runner.batch import BatchRunner
    from mobility_bench.runner.base import RunConfig

    frame = pd.read_csv(args.dataset.resolve()).head(args.limit).copy()
    if 'case_id' not in frame:
        frame.insert(0, 'case_id', [f'case_{i+1}' for i in range(len(frame))])
    settings = Settings.load(ROOT/'configs/evaluation/default.yaml')
    settings.agent.sandbox_data_dir = str(sandbox)
    settings.models = {model: ModelConfig.from_dict(model, {'base_url': base, 'api_key': key, 'roles': {'planner': model}})}
    Settings._instance = settings
    from mobility_bench.agent.frameworks.hermes import HermesFramework
    HermesFramework(settings, settings.models[model])._config()  # validate budget before any API calls
    print(f'Preflight OK: {len(frame)} cases, {len(expected)} sandbox files, model={model}', flush=True)
    if args.preflight:
        return 0
    output = (args.output_dir or ROOT/'data/results'/('hermes_api_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if (output/'metadata.json').exists() and not args.resume:
        parser.error('Output already has a run; use a new directory to avoid mixing experiments')
    if args.resume and not (output/'metadata.json').exists():
        parser.error('--resume requires an existing output directory')
    output.mkdir(parents=True, exist_ok=True)
    dataset = output/'dataset.csv'
    runtime_files = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'configs').rglob('*.yaml')) + sorted((ROOT/'src').rglob('*.md'))
    runtime_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files}
    if args.resume:
        previous = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
        if previous['model'] != model or previous['base_url'] != base or previous['runtime_sha256'] != runtime_hashes:
            parser.error('Model, endpoint or runtime changed; use a new experiment directory')
        if previous['dataset_sha256'] != hashlib.sha256(dataset.read_bytes()).hexdigest():
            parser.error('Frozen dataset changed')
        if previous['cases'] != len(frame):
            parser.error('Case count changed')
    else:
        frame.to_csv(dataset, index=False)
    from mobility_bench.agent.frameworks.harness import harness_config
    from dataclasses import asdict
    metadata = {
        'model': model, 'base_url': base, 'agent': 'hermes', 'python': platform.python_version(),
        'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'cases': len(frame),
        'sandbox': str(sandbox), 'toolsets': ['sandbox'], 'harness': asdict(harness_config('hermes')),
        'metrics_requested': ['tool','planning','efficiency','instruction','answer'] if args.full_evaluation else (['tool','planning','efficiency'] if args.evaluate else []),
        'metrics_omitted': {} if args.full_evaluation else {'instruction': 'requires verified embedding weights', 'answer': 'requires verified embedding weights'},
        'evaluation_note': 'Unchanged Hermes adapter and native MobilityBench evaluator; frozen sample baseline',
        'runtime_sha256': runtime_hashes, 'started_at': datetime.now().isoformat(),
    }
    if not args.resume:
        write_json(output/'metadata.json', metadata)
    cases = DatasetLoader(settings).load(str(dataset))
    runner = BatchRunner([model], settings=settings, output_dir=output/'results', framework='hermes')
    config = RunConfig(model_name=model, output_dir=output/'results', framework='hermes')
    from mobility_bench.dataset.schema import RunResult
    raw_file = output/'results'/model/'raw_results.json'
    results = [RunResult(**r) for r in json.loads(raw_file.read_text(encoding='utf-8'))] if args.resume and raw_file.exists() else []
    completed_ids = {r.case_id for r in results}
    consecutive_errors = 0
    for case in cases:
        if case.case_id in completed_ids:
            continue
        os.environ['HERMES_TRACE_DIR'] = str(output/'traces'/case.case_id)
        print(f'START {case.case_id}', flush=True)
        result = runner.run_single(case, config)
        results.append(result)
        write_json(raw_file, [r.to_dict() for r in results])
        total_tokens = sum(sum(u.get('total_tokens', 0) for u in r.token_usage.values() if isinstance(u, dict)) for r in results)
        write_json(output/'progress.json', {'completed': len(results), 'expected': len(cases), 'failed': sum(bool(r.error) for r in results), 'total_tokens': total_tokens, 'last_case': case.case_id, 'updated_at': datetime.now().isoformat()})
        print(f'{len(results)}/{len(cases)} {case.case_id}: {"FAILED" if result.error else "OK"}; tools={len(result.worker_output)}; tokens_so_far={total_tokens}; seconds={result.execution_time:.1f}', flush=True)
        consecutive_errors = consecutive_errors + 1 if result.error else 0
        if result.error:
            print(result.error, flush=True)
            if not args.continue_on_error or consecutive_errors >= 3:
                break  # Do not repeatedly spend API quota on a broken setup/provider.
    if (args.evaluate or args.full_evaluation) and len(results) == len(cases):
        if args.full_evaluation and not args.allow_native_fallback and not (ROOT/'embedding-model/model.safetensors').is_file():
            write_json(output/'completion.json', {'success': False, 'expected': len(cases), 'completed': len(results), 'failed': sum(bool(r.error) for r in results), 'evaluation_pending': 'Embedding weights missing; resume after downloading'})
            raise RuntimeError('Native full evaluation needs embedding-model/model.safetensors; inference is saved')
        from mobility_bench.evaluation.runner import EvaluationRunner
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        write_json(output/'evaluation_backend.json', {'native_fallback_allowed': args.allow_native_fallback,
                   'embedding_weight_present': (ROOT/'embedding-model/model.safetensors').is_file(),
                   'note': 'Evaluator source is unchanged; missing local weights use its built-in simple similarity fallback.'})
        evaluator = EvaluationRunner(output/'results', output/'evaluation', dataset, settings)
        metrics = {}
        metric_names = ['tool', 'planning', 'efficiency', 'instruction', 'answer'] if args.full_evaluation else ['tool', 'planning', 'efficiency']
        for name in metric_names:
            print(f'EVALUATE {name}', flush=True)
            metrics[name] = evaluator.evaluate_metric(name)
        evaluator.save_results(metrics, evaluator.aggregate_results(metrics))
    success = len(results) == len(cases) and all(not r.error for r in results)
    write_json(output/'completion.json', {'success': success, 'expected': len(cases), 'completed': len(results),
                                         'failed': sum(bool(r.error) for r in results)})
    print(f'Results: {output}', flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
