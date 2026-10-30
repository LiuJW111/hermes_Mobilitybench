"""Real pinned Hermes + real MCP, with a deterministic local model API fixture."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_shared_sandbox import shared_sandbox

from mobility_bench.agent.frameworks.hermes import HermesFramework, hermes_runtime
from mobility_bench.config.settings import ModelConfig, Settings


@pytest.mark.skipif(not hermes_runtime()[1].is_file(), reason="Bundled Hermes environment not installed")
@pytest.mark.parametrize('transport', ['stdio', 'shared_http'])
def test_bundled_hermes_calls_real_mcp(monkeypatch, tmp_path, shared_sandbox, transport):
    requests = []

    class API(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply({"object": "list", "data": [{"id": "fixture", "object": "model", "owned_by": "local"}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            assert payload['max_tokens'] == 4096
            assert payload['tools']
            assert all(t['function']['name'].startswith('mcp_sandbox_') for t in payload['tools'])
            if any(m.get("role") == "tool" for m in payload["messages"]):
                message = {"role": "assistant", "content": "公园4位于北京，坐标116.4,39.9。"}
                reason = "stop"
            else:
                names = [t["function"]["name"] for t in payload.get("tools", [])]
                assert "mcp_sandbox_query_poi" in names
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_fixture", "type": "function",
                    "function": {"name": "mcp_sandbox_query_poi", "arguments": json.dumps({"keywords": "公园4", "city": "北京"}, ensure_ascii=False)}}]}
                reason = "tool_calls"
            response = {"id": "chatcmpl-fixture", "object": "chat.completion", "created": 1, "model": "fixture",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
            if payload.get("stream"):
                for index, call in enumerate(message.get("tool_calls", [])):
                    call["index"] = index
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                data = ("data: " + json.dumps(response, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.reply(response)

    sandbox = tmp_path / "stdio_sandbox"
    sandbox.mkdir()
    (sandbox / "nested_poi_data.json").write_text(json.dumps({"公园4": {"北京": {
        "name": "公园4", "location": "116.4,39.9", "address": "北京"}}}, ensure_ascii=False), encoding="utf-8")
    if transport == 'shared_http':
        monkeypatch.setenv('MOBILITY_BENCH_HERMES_MCP_URL', shared_sandbox)
    else:
        monkeypatch.delenv('MOBILITY_BENCH_HERMES_MCP_URL', raising=False)
    monkeypatch.setenv("HERMES_TRACE_DIR", str(tmp_path / "traces"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), API)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        settings = Settings()
        settings.agent.sandbox_data_dir = str(sandbox)
        model = ModelConfig.from_dict("fixture", {"base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "api_key": "fixture-key", "roles": {"planner": "fixture"}})
        framework = HermesFramework(settings, model)
        state = framework.run_hermes(framework.prepare_initial_state("查找北京测试公园"))
        assert state["_hermes"]["exit"] == 0, state["_hermes"]
        result = framework.extract_result(state)
        assert "116.4,39.9" in state["_hermes"]["answer"]
        assert state["_hermes"]["usage"]["total_tokens"] == 240
        assert state["_hermes"]["tool_calls"][0]["name"] == "query_poi"
        assert "116.4,39.9" in state["_hermes"]["tool_calls"][0]["result"]
        assert result["planner_training_data"]["intent"] == "地点查询"
        assert len(requests) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
