"""Independently check exported results against the saved Hermes sessions."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mobility_bench.agent.frameworks._cli_common import build_prompt
from mobility_bench.dataset.loader import DatasetLoader


def audit(root):
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    rows = json.loads((root/'results'/meta['model']/'raw_results.json').read_text(encoding='utf-8'))
    dataset = {case.case_id: case for case in DatasetLoader().load(str(root/'dataset.csv'))}
    details = []
    for row in rows:
        case = row['case_id']
        drivers = list((root/'traces'/case).glob('*/driver.json'))
        assert len(drivers) == 1, (case, 'trace count')
        driver = json.loads(drivers[0].read_text(encoding='utf-8'))
        session = next((drivers[0].parent/'sessions').glob('session_*.json'))
        transcript = json.loads(session.read_text(encoding='utf-8'))
        messages = transcript['messages']
        actual = [t for m in messages if m['role']=='assistant' for t in m.get('tool_calls', [])]
        exported = [t for w in row['worker_output'] for t in w['tool_calls']]
        assert len(actual) == len(exported), (case, 'tool count')
        for a, b in zip(actual, exported):
            assert a['id'] == b['id']
            assert a['function']['name'] == 'mcp_sandbox_' + b['name']
            assert json.loads(a['function']['arguments']) == b['args']
        tool_messages = {m['tool_call_id']: m['content'] for m in messages if m['role']=='tool'}
        for worker in row['worker_output']:
            for call_id, value in worker['tool_results'].items():
                assert json.loads(value) == tool_messages[call_id], (case, 'tool return mapping')
        final = [m for m in messages if m['role']=='assistant' and not m.get('tool_calls')][-1]['content']
        assert final.strip() == row['reporter_output'].strip(), (case, 'final answer')
        marker = 'MOBILITYBENCH_HERMES_RESULT='
        payload = json.loads([line[len(marker):] for line in driver['stdout'].splitlines() if line.startswith(marker)][-1])
        for name in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            assert payload[name] == row['token_usage']['reporter'][name], (case, 'token export')
        first_user = next(m['content'] for m in messages if m['role']=='user')
        sample = dataset[case]
        assert first_user == build_prompt({'query': sample.query, 'context': sample.context}), (case, 'input isolation')
        names = [t['function']['name'] for t in transcript['tools']]
        assert names and all(name.startswith('mcp_sandbox_') for name in names), (case, 'tool scope')
        statuses = []
        for text in tool_messages.values():
            value = json.loads(text)
            if 'result' in value:
                value = json.loads(value['result'])
            statuses.append(value.get('status'))
        details.append({'case_id':case, 'tool_calls':len(actual), 'successful_tool_returns':statuses.count(0),
                        'unsuccessful_tool_returns':len(statuses)-statuses.count(0), 'exit':driver['exit'],
                        'seconds':round(driver['elapsed'], 2), 'tokens':row['token_usage']['reporter'],
                        'all_mapping_checks_pass':True})
    result = {'model':meta['model'], 'cases':len(rows), 'all_mapping_checks_pass':True,
              'note':'Token export matches Hermes driver, not an independent provider billing audit. Task correctness is separate.',
              'details':details}
    (root/'chain_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',type=Path)
    audit(p.parse_args().run_dir)
