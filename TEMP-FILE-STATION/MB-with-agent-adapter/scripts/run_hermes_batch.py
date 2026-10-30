"""Parallel Hermes baseline with one shared sandbox and a live terminal dashboard."""
import argparse
import atexit
import hashlib
import json
import logging
import multiprocessing
import os
import platform
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from run_hermes_api import write_json


def settings_for(model, sandbox):
    from mobility_bench.config.settings import ModelConfig, Settings
    settings = Settings.load(ROOT/'configs/evaluation/default.yaml')
    settings.agent.sandbox_data_dir = str(sandbox)
    settings.models = {model: ModelConfig.from_dict(model, {
        'base_url': os.environ['LLM_BASE_URL'], 'api_key': os.environ['LLM_API_KEY'], 'roles': {'planner': model},
    })}
    Settings._instance = settings
    return settings


def run_case(case, model, sandbox, output, url):
    """Each pool process owns its settings, trace path, and one isolated Hermes."""
    logging.disable(logging.CRITICAL)
    os.environ['MOBILITY_BENCH_HERMES_MCP_URL'] = url
    os.environ['HERMES_TRACE_DIR'] = str(Path(output)/'traces'/case.case_id)
    settings = settings_for(model, sandbox)
    from mobility_bench.runner.batch import BatchRunner
    from mobility_bench.runner.base import RunConfig
    runner = BatchRunner([model], settings=settings, output_dir=Path(output)/'results', framework='hermes')
    result = runner.run_single(case, RunConfig(model_name=model, output_dir=Path(output)/'results', framework='hermes'))
    return result.to_dict()


def clean(text, limit=240):
    key = os.environ.get('LLM_API_KEY', '')
    if key:
        text = str(text).replace(key, '[REDACTED]')
    return re.sub(r'\s+', ' ', str(text)).strip()[:limit]


def fmt_duration(seconds):
    seconds = int(seconds or 0)
    if seconds >= 3600:
        return f'{seconds//3600}h{(seconds%3600)//60:02d}m'
    if seconds >= 60:
        return f'{seconds//60}m{seconds%60:02d}s'
    return f'{seconds}s'


def is_provider_error(message):
    # Bare 401/403/429 can be bus route numbers or coordinates in task text.
    return bool(re.search(r'(?:error code|status code|HTTP)\s*[:=]?\s*(?:401|403|429)\b|insufficient_quota|rate_limit_exceeded|free quota exhausted|quota exceeded|余额不足|免费额度.*(?:耗尽|用完)|访问量过大', message, re.I))


def describe_result(row, output):
    tools = [t for w in row.get('worker_output', []) for t in w.get('tool_calls', [])]
    failed_tools = 0
    for worker in row.get('worker_output', []):
        for value in worker.get('tool_results', {}).values():
            try:
                value = json.loads(value)
                if isinstance(value, str):
                    value = json.loads(value)
                if 'result' in value:
                    value = json.loads(value['result'])
                failed_tools += value.get('status', 0) != 0
            except (ValueError, TypeError, AttributeError):
                pass
    tokens = sum(u.get('total_tokens', 0) for u in row.get('token_usage', {}).values() if isinstance(u, dict))
    message = row.get('reporter_output') or row.get('error') or ''
    if row.get('error'):
        drivers = sorted((output/'traces'/row['case_id']).glob('*/driver.json'), key=lambda p: p.stat().st_mtime)
        if drivers:
            driver = json.loads(drivers[-1].read_text(encoding='utf-8'))
            stderr = driver.get('stderr', '')
            if 'RuntimeError:' in stderr:
                message = stderr.rsplit('RuntimeError:', 1)[-1]
            elif stderr:
                message = stderr[-2000:]
    return {'tools': len(tools), 'failed_tool_returns': failed_tools, 'tokens': tokens, 'message': clean(message, 1000)}


def start_sandbox(output, sandbox, console):
    import httpx
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    log = (output/'sandbox.log').open('a', encoding='utf-8')
    proc = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT/'mcp_server/shared_sandbox.py'),
        '--port', str(port), '--sandbox-dir', str(sandbox)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def stop():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()
    atexit.register(stop)
    url = f'http://127.0.0.1:{port}'
    start = time.monotonic()
    with console.status('正在加载共享地图沙箱（仅一份，首次约需 20–60 秒）…'):
        with httpx.Client(trust_env=False, timeout=2) as client:
            while time.monotonic()-start < 180:
                if proc.poll() is not None:
                    raise RuntimeError(f'Shared sandbox exited; see {output / "sandbox.log"}')
                try:
                    health = client.get(url+'/health').json()
                    if health.get('ready'):
                        return url+'/mcp', proc, stop, health
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(0.5)
    stop()
    raise TimeoutError('Shared sandbox startup timed out')


def save_selected(output, rows, current_rows, model):
    write_json(output/'results'/model/'raw_results.json', current_rows)
    write_json(output/'selected_results.json', rows)


def evaluate(output, settings, console, rows):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from mobility_bench.evaluation.runner import EvaluationRunner
    # Keep every original attempt under results/. Native evaluation receives
    # one selected outcome per case, with source model preserved in each row.
    groups = {name: [r for r in rows if r['model_name'] == name] for name in sorted({r['model_name'] for r in rows})}
    if len(groups) > 1:
        groups['hermes_mixed'] = rows
    for name, group_rows in groups.items():
        write_json(output/'evaluation_input'/name/'raw_results.json', group_rows)
    evaluator = EvaluationRunner(output/'evaluation_input', output/'evaluation', output/'dataset.csv', settings)
    metrics = {}
    for name in ['tool', 'planning', 'efficiency', 'instruction', 'answer']:
        console.print(f'正在执行原生评分：{name}', style='cyan')
        metrics[name] = evaluator.evaluate_metric(name)
    summary = evaluator.aggregate_results(metrics)
    evaluator.save_results(metrics, summary)
    embedding_present = (ROOT/'embedding-model/model.safetensors').is_file()
    write_json(output/'evaluation_backend.json', {
        'embedding_weight_present': embedding_present,
        'similarity': ('Qwen3-Embedding-0.6B (local weights)'
                       if embedding_present else
                       'native character-similarity fallback (weights missing)'),
        'evaluator_modified': False,
        'score_placeholder': ('native MetricResult.score / average_score are fixed at 0.0; '
                              'use the sub-scores in evaluation/metrics_summary.*'),
    })
    # The native MetricResult.score is a fixed 0.0 placeholder in every
    # metric; the real numbers live in details/sub_scores. Build a compact,
    # complete view (one row per metric/sub-dimension) with coverage counts so
    # no metric disappears when a table is wide or a terminal is narrow.
    metric_rows = []
    for metric_name, per_model in metrics.items():
        for model_name, payload in per_model.items():
            results = payload.get('results', [])
            total = len(results)
            values_by_key = {}
            for result in results:
                if result.error is not None or not result.details:
                    continue
                for key, value in result.details.items():
                    if key in ('task_scenario', 'intent_family', 'source_file'):
                        continue
                    if isinstance(value, bool):
                        value = 1.0 if value else 0.0
                    if isinstance(value, (int, float)):
                        values_by_key.setdefault(key, []).append(float(value))
            if not values_by_key:
                metric_rows.append({'model': model_name, 'metric': metric_name,
                                    'sub': '(no sub-scores)', 'mean': 0.0, 'valid': 0, 'total': total})
                continue
            for key, values in values_by_key.items():
                metric_rows.append({'model': model_name, 'metric': metric_name, 'sub': key,
                                    'mean': round(sum(values)/len(values), 4),
                                    'valid': len(values), 'total': total})

    table = Table(title='MobilityBench 原生指标')
    for col, justify in [('模型', 'left'), ('指标', 'left'), ('子项', 'left'),
                         ('均值', 'right'), ('有效题数', 'right')]:
        table.add_column(col, justify=justify)
    for row in metric_rows:
        table.add_row(row['model'], row['metric'], row['sub'],
                      f"{row['mean']:.4f}", f"{row['valid']}/{row['total']}")
    console.print(table)
    console.print('说明：原生 Metric 只定义子项，顶层 score / average_score 固定为 0，'
                  '不代表任务失败；请以子项数值为准。', style='dim')

    evaluation_dir = output/'evaluation'
    with (evaluation_dir/'metrics_summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        handle.write('model,metric,sub_dimension,mean,valid_cases,total_cases\n')
        for row in metric_rows:
            handle.write(f"{row['model']},{row['metric']},{row['sub']},{row['mean']},"
                         f"{row['valid']},{row['total']}\n")
    md_lines = [
        '# MobilityBench 原生指标汇总', '',
        '| 模型 | 指标 | 子项 | 均值 | 有效题数 | 总题数 |',
        '|---|---|---|---:|---:|---:|',
    ]
    for row in metric_rows:
        md_lines.append(f"| {row['model']} | {row['metric']} | {row['sub']} | "
                        f"{row['mean']:.4f} | {row['valid']} | {row['total']} |")
    md_lines += ['', '> 原生 Metric 的 `score` / `average_score` 是固定占位值 0.0；',
                 '> 上面的子项均值才是实际结果。`valid_cases` 小于总题数表示该子项在部分题目上无定义。']
    (evaluation_dir/'metrics_summary.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT/'.env.api')
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/datasets/sample_10.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2, 4, 8, 16, 32, 48, 64, 96], default=4)
    parser.add_argument('--limit', type=int, default=115)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--continue-with-model', help='Explicitly continue saved cases using another model; record per-case provenance')
    parser.add_argument('--retry-provider-errors', action='store_true', help='Requeue only saved provider failures, keeping task failures and successful cases')
    args = parser.parse_args()
    load_dotenv(args.env_file, override=True)
    for name in ['LLM_MODEL', 'LLM_BASE_URL', 'LLM_API_KEY']:
        if not os.environ.get(name):
            parser.error(f'Missing {name} in env file')
    os.environ.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    os.environ.setdefault('HERMES_SRC', str(ROOT.parent/'hermes-agent'))
    os.environ.setdefault('HERMES_PY', str(Path(os.environ['HERMES_SRC'])/'venv'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')))
    output = args.output_dir.resolve()
    sandbox = Path(os.environ.get('MOBILITY_BENCH_SANDBOX_DIR', ROOT/'data/sandbox')).resolve()
    model = args.continue_with_model or os.environ['LLM_MODEL']
    if args.continue_with_model and not args.resume:
        parser.error('--continue-with-model requires --resume')
    os.chdir(ROOT)
    settings = settings_for(model, sandbox)
    from mobility_bench.dataset.loader import DatasetLoader
    import pandas as pd
    frame = pd.read_csv(args.dataset).head(args.limit).copy()
    if 'case_id' not in frame:
        frame.insert(0, 'case_id', [f'case_{i+1}' for i in range(len(frame))])
    files = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'src').rglob('*.md')) + sorted((ROOT/'configs').rglob('*.yaml')) + sorted((ROOT/'mcp_server').glob('*.py'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    meta_file = output/'metadata.json'
    if meta_file.exists():
        if not (args.resume or args.evaluate_only):
            parser.error('Output exists. Use --resume or choose a new output directory.')
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        if meta['runtime_sha256'] != hashes or meta['base_url'] != os.environ['LLM_BASE_URL']:
            parser.error('Runtime or endpoint changed; cannot mix experiments')
        if meta.get('active_model', meta['model']) != model and not args.continue_with_model:
            parser.error('Model changed; use --continue-with-model to record an explicit continuation')
        if hashlib.sha256((output/'dataset.csv').read_bytes()).hexdigest() != meta['dataset_sha256']:
            parser.error('Frozen dataset changed')
    else:
        if args.resume or args.evaluate_only:
            parser.error('No experiment exists to resume/evaluate')
        output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output/'dataset.csv', index=False)
        from dataclasses import asdict
        from mobility_bench.agent.frameworks.harness import harness_config
        write_json(meta_file, {'model': model, 'base_url': os.environ['LLM_BASE_URL'], 'agent': 'hermes',
            'cases': len(frame), 'workers': args.workers, 'transport': 'shared loopback MCP HTTP',
            'runtime_sha256': hashes, 'dataset_sha256': hashlib.sha256((output/'dataset.csv').read_bytes()).hexdigest(),
            'harness': asdict(harness_config('hermes')), 'python': platform.python_version(),
            'started_at': datetime.now().isoformat(), 'similarity_backend': 'native simple fallback; no weights downloaded',
            'note': 'Original prompt, agent loop, result mapping, sandbox handlers and native scoring rules retained.'})
    cases = DatasetLoader(settings).load(str(output/'dataset.csv'))
    raw_file = output/'results'/model/'raw_results.json'
    current_rows = json.loads(raw_file.read_text(encoding='utf-8')) if raw_file.exists() else []
    selected_file = output/'selected_results.json'
    original_file = output/'results'/json.loads(meta_file.read_text(encoding='utf-8'))['model']/'raw_results.json'
    if selected_file.exists():
        rows = json.loads(selected_file.read_text(encoding='utf-8'))
    elif original_file.exists():
        rows = json.loads(original_file.read_text(encoding='utf-8'))
    else:
        rows = []
    rows_by_id = {r['case_id']: r for r in rows}
    current_by_id = {r['case_id']: r for r in current_rows}
    retry_ids = []
    if args.retry_provider_errors:
        for row in rows:
            if row.get('error') and is_provider_error(describe_result(row, output)['message']):
                retry_ids.append(row['case_id'])
                rows_by_id.pop(row['case_id'])
        rows = [rows_by_id[c.case_id] for c in cases if c.case_id in rows_by_id]
    if args.continue_with_model:
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        meta.setdefault('continuations', []).append({'at': datetime.now().isoformat(), 'model': model,
            'kept_cases': len(rows), 'retry_provider_cases': retry_ids, 'pending_cases': len(cases)-len(rows),
            'reason': 'User requested continuing the existing sample10 after original model free quota was exhausted.'})
        meta['active_model'] = model
        meta['mixed_models'] = True
        write_json(meta_file, meta)
    save_selected(output, rows, current_rows, model)
    console = Console(record=True)
    console.print(Panel(f'Hermes + {model} | {len(cases)} 题 | {args.workers} 并发\n结果目录：{output}\n完成/异常表示程序执行状态，答案得分在全部结束后按原生评估器计算。', title='MobilityBench sample10'))
    if args.evaluate_only:
        if len(rows) != len(cases):
            parser.error('Inference incomplete; resume the run first')
        evaluate(output, settings, console, rows)
        console.save_text(str(output/'terminal-summary.txt'))
        return 0
    pending = [c for c in cases if c.case_id not in rows_by_id]
    if args.resume:
        console.print(f'保留 {len(rows)} 题；补跑供应商错误 {len(retry_ids)} 题；本次待跑 {len(pending)} 题；当前模型 {model}', style='cyan')
    events = (output/'events.jsonl').open('a', encoding='utf-8')
    log = (output/'run.log').open('a', encoding='utf-8')

    def emit(event, **values):
        data = {'time': datetime.now().isoformat(), 'event': event, **values}
        events.write(json.dumps(data, ensure_ascii=False)+'\n')
        events.flush()
        log.write(f"{data['time']} {event} {clean(values, 1800)}\n")
        log.flush()

    progress = Progress(TextColumn('[bold cyan]{task.description}'), BarColumn(), TextColumn('{task.completed}/{task.total}'), TimeElapsedColumn(), TimeRemainingColumn())
    task_id = progress.add_task('总进度', total=len(cases), completed=len(rows))
    active = {}
    recent = []
    last_checkpoint = 0

    def status():
        total_tokens = sum(sum(u.get('total_tokens', 0) for u in r.get('token_usage', {}).values() if isinstance(u, dict)) for r in rows)
        return {'completed': len(rows), 'expected': len(cases), 'failed': sum(bool(r.get('error')) for r in rows), 'active_model': model,
            'selected_by_model': {name: sum(r['model_name'] == name for r in rows) for name in sorted({r['model_name'] for r in rows})},
            'running': [v['case'].case_id for v in active.values()], 'workers': args.workers, 'total_tokens': total_tokens,
            'updated_at': datetime.now().isoformat()}

    def display():
        st = status()
        table = Table(expand=True, title='运行中的题目')
        for col in ['题号', '场景', '耗时', '题目']:
            table.add_column(col)
        for info in active.values():
            table.add_row(info['case'].case_id, clean(info['case'].task_scenario, 24),
                          fmt_duration(time.monotonic()-info['start']), clean(info['case'].query, 110))
        recent_table = Table(expand=True, title='最近完成的题目')
        for col in ['状态', '题号', '场景', '耗时', '工具', 'Token', '报告 / 失败原因']:
            recent_table.add_column(col)
        for item in recent[-8:]:
            recent_table.add_row(Text('异常' if item['error'] else '完成', style='red' if item['error'] else 'green'),
                                 item['case_id'], clean(item.get('scenario', ''), 20), fmt_duration(item['seconds']),
                                 str(item['tools']), f"{item['tokens']:,}", Text(clean(item['message'], 260)))
        return Group(
            progress,
            Text(f"已结束 {len(rows)}/{len(cases)} | 成功 {st['completed']-st['failed']} | 异常 {st['failed']} | 运行中 {len(active)} | 已导出 Token {st['total_tokens']:,}"),
            table, recent_table,
        )

    url, service, stop_service, health = start_sandbox(output, sandbox, console)
    emit('sandbox_ready', **health)
    write_json(output/'processes.json', {'runner_pid': os.getpid(), 'sandbox_pid': service.pid, 'url': url})
    remaining = iter(pending)
    stopped_reason = ''
    provider_failures = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            def submit():
                try:
                    case = next(remaining)
                except StopIteration:
                    return
                future = pool.submit(run_case, case, model, sandbox, output, url)
                active[future] = {'case': case, 'start': time.monotonic()}
                emit('start', case_id=case.case_id, query=case.query)

            for _ in range(args.workers):
                submit()
            with Live(display(), console=console, refresh_per_second=2) as live:
                while active:
                    if service.poll() is not None:
                        stopped_reason = 'Shared sandbox exited unexpectedly'
                    if (output/'STOP').exists():
                        stopped_reason = 'STOP file requested: drain active cases and keep checkpoints'
                    done, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
                    for future in done:
                        info = active.pop(future)
                        try:
                            row = future.result()
                        except Exception as exc:
                            from mobility_bench.dataset.schema import RunResult
                            row = RunResult(case_id=info['case'].case_id, model_name=model, error=clean(exc, 1200), execution_time=time.monotonic()-info['start']).to_dict()
                        rows_by_id[row['case_id']] = row
                        current_by_id[row['case_id']] = row
                        rows[:] = [rows_by_id[c.case_id] for c in cases if c.case_id in rows_by_id]
                        current_rows[:] = [current_by_id[c.case_id] for c in cases if c.case_id in current_by_id]
                        save_selected(output, rows, current_rows, model)
                        detail = describe_result(row, output)
                        item = {'case_id': row['case_id'], 'model': row['model_name'], 'error': bool(row.get('error')), 'seconds': row.get('execution_time', 0),
                                'scenario': info['case'].task_scenario, 'query': info['case'].query, **detail}
                        recent.append(item)
                        emit('failed' if row.get('error') else 'completed', **item)
                        # Print the finished case in full above the live dashboard so the
                        # scrollback shows what the agent actually answered.
                        headline = (f"[green]✓ {row['case_id']}[/green]" if not row.get('error')
                                    else f"[red]✗ {row['case_id']}[/red]")
                        console.print(
                            f"{headline} {info['case'].task_scenario} | {fmt_duration(item['seconds'])} | "
                            f"tools={item['tools']} | tokens={item['tokens']:,}",
                            soft_wrap=True,
                        )
                        console.print(f"  问题: {clean(info['case'].query, 300)}", style='dim', soft_wrap=True)
                        if row.get('error'):
                            console.print(f"  失败: {clean(row.get('error'), 800)}", style='red', soft_wrap=True)
                        else:
                            console.print(f"  报告: {clean(row.get('reporter_output'), 1200)}", soft_wrap=True)
                        progress.update(task_id, completed=len(rows))
                        if row.get('error') and is_provider_error(detail['message']):
                            provider_failures += 1
                        elif not row.get('error'):
                            provider_failures = 0
                        if provider_failures >= 3:
                            stopped_reason = 'Repeated provider errors; stopped submitting new cases'
                        if not stopped_reason:
                            submit()
                    if time.monotonic()-last_checkpoint > 1:
                        write_json(output/'progress.json', status())
                        last_checkpoint = time.monotonic()
                    live.update(display())
        write_json(output/'progress.json', status())
    finally:
        stop_service()
        events.close()
        log.close()
        console.save_text(str(output/'terminal-summary.txt'))
    complete = len(rows) == len(cases)
    if complete:
        evaluate(output, settings, console, rows)
    write_json(output/'completion.json', {**status(), 'inference_complete': complete, 'evaluation_complete': complete, 'stop_reason': stopped_reason})
    console.print(f'已保存结果：{output}', style='bold green')
    if stopped_reason:
        console.print(stopped_reason, style='yellow')
    console.save_text(str(output/'terminal-summary.txt'))
    return 0 if complete else 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
