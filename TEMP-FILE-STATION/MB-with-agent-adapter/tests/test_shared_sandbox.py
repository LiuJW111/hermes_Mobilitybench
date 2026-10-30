"""Verify concurrent MCP clients retain the original schemas and isolated arguments."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def shared_sandbox(tmp_path):
    sandbox = tmp_path/'sandbox'
    sandbox.mkdir()
    data = {f'公园{i}': {'北京': {'name': f'公园{i}', 'location': f'116.{i},39.9', 'address': f'北京{i}号'}} for i in range(8)}
    (sandbox/'nested_poi_data.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with (tmp_path/'server.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT/'mcp_server/shared_sandbox.py'), '--port', str(port), '--sandbox-dir', str(sandbox)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            with httpx.Client(trust_env=False, timeout=1) as client:
                for _ in range(100):
                    assert proc.poll() is None, (tmp_path/'server.log').read_text(encoding='utf-8')
                    try:
                        if client.get(f'http://127.0.0.1:{port}/health').json().get('ready'):
                            break
                    except (httpx.HTTPError, ValueError):
                        time.sleep(0.1)
                else:
                    pytest.fail('Shared MCP startup timed out')
            yield f'http://127.0.0.1:{port}/mcp'
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def test_eight_concurrent_clients_keep_schema_and_arguments(shared_sandbox):
    sys.path.insert(0, str(ROOT/'mcp_server'))
    from sandbox_tools import TOOL_SPECS

    async def client(i):
        async with httpx.AsyncClient(trust_env=False) as http:
            async with streamable_http_client(shared_sandbox, http_client=http) as (reader, writer, _):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    assert {t.name for t in tools} == set(TOOL_SPECS)
                    for tool in tools:
                        assert tool.inputSchema == TOOL_SPECS[tool.name][1]
                        assert tool.description == TOOL_SPECS[tool.name][0]
                    for _ in range(2):
                        response = await session.call_tool('query_poi', {'keywords': f'公园{i}', 'city': '北京'})
                        result = json.loads(response.content[0].text)
                        assert result['status'] == 0
                        assert f'116.{i},39.9' in json.dumps(result)
                    invalid = await session.call_tool('driving_route', {'origin': '999,999', 'destination': '116.4,39.9'})
                    assert json.loads(invalid.content[0].text)['status'] == 1

    async def run():
        await asyncio.gather(*(client(i) for i in range(8)))
    asyncio.run(run())


def test_shared_transport_is_explicit_and_loopback_only(monkeypatch):
    import yaml
    from mobility_bench.agent.frameworks.hermes import HermesFramework
    from mobility_bench.config.settings import ModelConfig, Settings
    framework = HermesFramework(Settings(), ModelConfig.from_dict('fixture', {'api_key': 'fixture', 'roles': {'planner': 'fixture'}}))
    monkeypatch.delenv('MOBILITY_BENCH_HERMES_MCP_URL', raising=False)
    assert 'command' in yaml.safe_load(framework._config())['mcp_servers']['sandbox']
    monkeypatch.setenv('MOBILITY_BENCH_HERMES_MCP_URL', 'http://127.0.0.1:12345/mcp')
    assert yaml.safe_load(framework._config())['mcp_servers']['sandbox'] == {'url': 'http://127.0.0.1:12345/mcp'}
    monkeypatch.setenv('MOBILITY_BENCH_HERMES_MCP_URL', 'https://example.com/mcp')
    with pytest.raises(ValueError, match='loopback'):
        framework._config()
