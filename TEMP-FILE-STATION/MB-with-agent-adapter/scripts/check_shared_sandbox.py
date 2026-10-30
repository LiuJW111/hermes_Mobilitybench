"""Replay saved public tool calls through shared MCP and compare exact payloads."""
import argparse
import asyncio
import json
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def unwrap(value):
    while isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict) and 'result' in value and 'status' not in value:
        return unwrap(value['result'])
    return value


async def check(url, source):
    rows = json.loads(source.read_text(encoding='utf-8'))
    calls = [(call, unwrap(worker['tool_results'][call['id']])) for row in rows for worker in row['worker_output'] for call in worker['tool_calls']]

    async def one(call, expected):
        async with httpx.AsyncClient(trust_env=False, timeout=60) as http:
            async with streamable_http_client(url, http_client=http) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(call['name'], call['args'])
                    assert unwrap(result.content[0].text) == expected, call['id']
                    return call['name']
    names = await asyncio.gather(*(one(call, expected) for call, expected in calls))
    return {'compared_calls': len(names), 'tools': names, 'all_payloads_identical': True, 'source': str(source)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    url = json.loads((args.run_dir/'processes.json').read_text(encoding='utf-8'))['url']
    report = asyncio.run(check(url, args.source))
    (args.run_dir/'transport-parity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
