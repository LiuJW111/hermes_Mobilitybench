"""Hermes Agent CLI as a MobilityBench agent framework.

Hermes Agent (Nous Research) is driven in-process through a tiny driver script
that reuses Hermes' own one-shot entry point (``hermes_cli.oneshot._run_agent``)
— the same code path behind ``hermes -z``. Using the driver instead of the
``hermes -z`` subprocess lets the adapter recover token usage, which ``-z``
prints nowhere: the agent exposes ``session_prompt_tokens`` / ``_completion_`` /
``_total_`` and the driver captures them.

Each run gets an isolated ``HERMES_HOME`` whose ``config.yaml`` points the
``custom`` (OpenAI-compatible) provider at the local vLLM endpoint and mounts
the deterministic sandbox tools through the MCP server in
``mcp_server/sandbox_tools.py``. The tool trajectory lives in the session
transcript ``HERMES_HOME/sessions/session_*.json``.

The runner contract (see ``runner/batch.py::_run_with_graph``) requires:

* ``build_graph()`` -> object with an ``ainvoke(state)`` coroutine
* ``prepare_initial_state(query, context)`` -> state dict
* ``extract_result(state)`` -> dict with ``plan_result``,
  ``planner_training_data``, ``worker_training_data`` and ``token_usage``
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from mobility_bench.agent.frameworks._cli_common import (
    MCP_SERVER,
    PROJECT_ROOT,
    VENV_PY,
    build_prompt,
    standard_result,
)
from mobility_bench.agent.frameworks.base import BaseFramework
from mobility_bench.agent.frameworks.harness import harness_config

# Hermes installs its CLI at <repo>/venv/bin/hermes; override with HERMES_BIN.
def hermes_runtime() -> tuple[Path, Path]:
    """Resolve at run time so environment files can configure the deployment."""
    binary = os.environ.get("HERMES_BIN") or shutil.which(harness_config("hermes").bin)
    source = Path(os.environ.get("HERMES_SRC") or (
        str(PROJECT_ROOT / "vendor/hermes-agent") if (PROJECT_ROOT / "vendor/hermes-agent/run_agent.py").is_file()
        else str(Path(binary).resolve().parents[2]) if binary else "/opt/hermes/hermes-agent"
    )).resolve()
    # Do not resolve() the interpreter path: a venv's bin/python is a symlink to
    # the system interpreter, and resolving it makes Python skip pyvenv.cfg and
    # load the system site-packages (mcp 2.x) instead of the venv's mcp 1.x.
    python = Path(os.environ.get("HERMES_PY") or source / "venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    ))
    return source, python

# Configure the real served context length. Pinned Hermes requires >= 64K;
# a 32K vLLM endpoint must not be represented as a 64K endpoint.

# Host env vars that would otherwise leak an OpenAI-compatible endpoint (this
# box sets OPENAI_BASE_URL to a remote API) into Hermes' provider detection.
_STRIP_ENV = (
    "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_API_BASE",
    "HERMES_INFERENCE_MODEL", "HERMES_INFERENCE_PROVIDER",
)

# In-process driver: reuse Hermes' one-shot entry point, capture token usage
# from the agent instance, and print one JSON object to stdout. Agent chatter
# is redirected to /dev/null so stdout stays a single clean JSON line.
_DRIVER_SOURCE = '''\
import contextlib
import json
import os
import sys

sys.path.insert(0, os.environ["HERMES_SRC"])
os.environ["HERMES_YOLO_MODE"] = "1"
os.environ["HERMES_ACCEPT_HOOKS"] = "1"

import run_agent  # noqa: E402

_orig_chat = run_agent.AIAgent.chat
_orig_init = run_agent.AIAgent.__init__
_orig_conversation = run_agent.AIAgent.run_conversation
_cap = {}


def _init(self, *a, **k):
    # Retain Hermes' native loop; set the experiment's explicit step budget.
    limit = int(os.environ.get("MB_HERMES_MAX_STEPS", "0"))
    if limit:
        k["max_iterations"] = limit
    _orig_init(self, *a, **k)


run_agent.AIAgent.__init__ = _init


def _chat(self, *a, **k):
    result = _orig_chat(self, *a, **k)
    _cap["prompt_tokens"] = getattr(self, "session_prompt_tokens", 0) or 0
    _cap["completion_tokens"] = getattr(self, "session_completion_tokens", 0) or 0
    _cap["total_tokens"] = getattr(self, "session_total_tokens", 0) or 0
    return result


run_agent.AIAgent.chat = _chat


def _conversation(self, *a, **k):
    result = _orig_conversation(self, *a, **k)
    if result.get("failed") or result.get("error") or not result.get("completed", False):
        raise RuntimeError("Hermes conversation failed: " + str(result.get("error") or result.get("final_response")))
    return result


run_agent.AIAgent.run_conversation = _conversation

from hermes_cli.oneshot import _run_agent  # noqa: E402

# CLI entry points run MCP discovery explicitly at startup; importing
# hermes_cli.oneshot alone does not, so load configured MCP servers here.
from tools.mcp_tool import discover_mcp_tools  # noqa: E402
_tools = discover_mcp_tools()
if not any(name.startswith("mcp_sandbox_") for name in _tools):
    import mcp  # noqa: E402
    from tools import mcp_tool as _mcp_tool  # noqa: E402
    raise RuntimeError(
        "Hermes did not discover the sandbox MCP tools; "
        f"python={sys.executable}; mcp={getattr(mcp, '__file__', '?')}; "
        f"http_available={_mcp_tool._MCP_HTTP_AVAILABLE}; "
        f"mcp_available={_mcp_tool._MCP_AVAILABLE}; tools={sorted(_tools)[:20]}"
    )

with open(sys.argv[1], encoding="utf-8") as fh:
    _prompt = fh.read()

_devnull = open(os.devnull, "w")
try:
    with contextlib.redirect_stdout(_devnull):
        _answer = _run_agent(_prompt, toolsets=["sandbox"], use_config_toolsets=False) or ""
finally:
    _devnull.close()

print("MOBILITYBENCH_HERMES_RESULT=" + json.dumps({"answer": _answer, **_cap}, ensure_ascii=False))
'''


def _bare_tool_name(name: str) -> str:
    """Reduce a Hermes MCP tool name to the bare tool name.

    Hermes registers MCP tools as ``mcp_<server>_<tool>`` (e.g.
    ``mcp_sandbox_reverse_geocoding``); the tool metric's ground truth uses the
    bare name.
    """
    if name.startswith("mcp_"):
        parts = name.split("_", 2)
        if len(parts) == 3:
            return parts[2]
    return name


class _HermesGraph:
    """Minimal async graph exposing ``ainvoke`` for the batch runner."""

    def __init__(self, framework: "HermesFramework"):
        self._fw = framework

    async def ainvoke(self, initial_state: dict) -> dict:
        return await asyncio.to_thread(self._fw.run_hermes, initial_state)


class HermesFramework(BaseFramework):
    """Framework that drives the hermes agent as the agent under test."""

    @property
    def name(self) -> str:
        return "hermes"

    def build_graph(self) -> Any:
        return _HermesGraph(self)

    def prepare_initial_state(self, query: str, context: str | None = None, **kwargs) -> dict[str, Any]:
        state = {"query": query, "context": context}
        state.update(kwargs)
        return state

    # ----------------------------------------------------------------- hermes
    def _model_name(self) -> str:
        if self.model_config and self.model_config.roles:
            return self.model_config.roles.get("planner") or self.model_config.name
        return self.model_config.name if self.model_config else "qwen3-4b"

    def _base_url(self) -> str:
        url = (self.model_config.base_url if self.model_config else "") or "http://localhost:8080/v1"
        return url.rstrip("/")

    def _api_key(self) -> str:
        return (self.model_config.api_key if self.model_config else "") or "dummy"

    def _config(self) -> str:
        """Build the isolated HERMES_HOME/config.yaml for one run.

        ``provider: custom`` is Hermes' OpenAI-compatible adapter (the bare
        ``vllm`` alias is not understood by its auxiliary clients). The sandbox
        MCP server is mounted over stdio; Hermes discovers its tools as
        ``mcp_sandbox_<tool>``.
        """
        hc = harness_config("hermes")
        if hc.context_window < 64000:
            raise ValueError("Pinned Hermes requires a real context window >= 64000 tokens")
        shared_url = os.environ.get("MOBILITY_BENCH_HERMES_MCP_URL")
        if shared_url:
            # A shared loopback server keeps the multi-GB, read-only replay
            # dictionaries in one process. Each Hermes session stays isolated.
            from urllib.parse import urlparse
            parsed = urlparse(shared_url)
            if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
                raise ValueError("Shared benchmark MCP server must use loopback HTTP")
            sandbox_config = {"url": shared_url}
        else:
            sandbox_config = {
                "command": str(VENV_PY), "args": [str(MCP_SERVER)],
                "env": {
                    "MOBILITY_BENCH_SANDBOX_DIR": str((PROJECT_ROOT / self.settings.agent.sandbox_data_dir).resolve()),
                    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                },
            }
        return yaml.safe_dump({
            "model": {
                "default": self._model_name(), "provider": "custom",
                "base_url": self._base_url(), "api_key": self._api_key(),
                "context_length": hc.context_window,
                "max_tokens": hc.max_output_tokens,
            },
            "auxiliary": {"compression": {"context_length": hc.context_window}},
            "mcp_servers": {"sandbox": sandbox_config},
        }, allow_unicode=True, sort_keys=False)

    def run_hermes(self, state: dict) -> dict:
        hc = harness_config("hermes")
        prompt = build_prompt(state)
        source, python = hermes_runtime()
        if not (source / "run_agent.py").is_file() or not python.is_file():
            raise FileNotFoundError("Set HERMES_SRC to the Hermes checkout and HERMES_PY to its Python interpreter")

        with tempfile.TemporaryDirectory() as home:
            home_path = Path(home)
            (home_path / "config.yaml").write_text(self._config(), encoding="utf-8")
            driver_path = home_path / "hermes_driver.py"
            driver_path.write_text(_DRIVER_SOURCE, encoding="utf-8")
            prompt_path = home_path / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
            env["HERMES_HOME"] = home
            env["HERMES_SRC"] = str(source)
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["MB_HERMES_MAX_STEPS"] = str(hc.max_steps)
            cmd = [str(python), str(driver_path), str(prompt_path)]
            started = time.time()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=home,
                    timeout=hc.timeout + 60,
                )
                stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                raw = e.stdout or ""
                stdout = raw if isinstance(raw, str) else (raw or b"").decode("utf-8", errors="replace")
                stderr = f"hermes driver timed out after {hc.timeout}s"
                code = -1
            elapsed = time.time() - started

            # Session transcript lives under the temp home, so parse it now.
            tool_calls = self._read_session(home_path)
            artifact = Path(os.environ.get("HERMES_TRACE_DIR", PROJECT_ROOT / "data/hermes_traces")) / uuid.uuid4().hex
            artifact.mkdir(parents=True)
            if (home_path / "sessions").exists():
                shutil.copytree(home_path / "sessions", artifact / "sessions")
            (artifact / "driver.json").write_text(json.dumps({
                "query": state.get("query"), "model": self._model_name(),
                "exit": code, "elapsed": elapsed,
                "stdout": stdout.replace(self._api_key(), "[REDACTED]"),
                "stderr": stderr.replace(self._api_key(), "[REDACTED]"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        answer, usage = self._parse_driver_stdout(stdout)
        state["_hermes"] = {
            "answer": answer,
            "usage": usage,
            "tool_calls": tool_calls,
            "stdout": stdout,
            "stderr": stderr[-4000:],
            "exit": code,
            "elapsed": elapsed,
            "artifact": str(artifact),
        }
        return state

    # ---------------------------------------------------------------- mapping
    @staticmethod
    def _parse_driver_stdout(stdout: str) -> tuple[str, dict]:
        marker = "MOBILITYBENCH_HERMES_RESULT="
        records = [line[len(marker):] for line in stdout.splitlines() if line.startswith(marker)]
        payload = records[-1] if records else stdout.strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return "", {}
        if not isinstance(data, dict):
            return "", {}
        answer = (data.get("answer") or "").strip()
        usage: dict = {}
        prompt_tokens = data.get("prompt_tokens", 0) or 0
        completion_tokens = data.get("completion_tokens", 0) or 0
        if prompt_tokens or completion_tokens:
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": data.get("total_tokens", prompt_tokens + completion_tokens),
            }
        return answer, usage

    def _read_session(self, home_path: Path) -> list[dict]:
        """Parse the newest session transcript for the tool trajectory."""
        sessions = sorted(
            (home_path / "sessions").glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not sessions:
            return []
        try:
            data = json.loads(sessions[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        tool_calls: list[dict] = []
        pending: dict[str, dict] = {}
        for msg in data.get("messages") or []:
            role = msg.get("role")
            if role == "assistant":
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function", {}) or {}
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                    tc = {
                        "id": call.get("id", ""),
                        "name": _bare_tool_name(fn.get("name", "")),
                        "args": args,
                        "result": None,
                    }
                    tool_calls.append(tc)
                    pending[tc["id"]] = tc
            elif role == "tool":
                call_id = msg.get("tool_call_id", "")
                if call_id in pending:
                    pending[call_id]["result"] = msg.get("content")
        return tool_calls

    def extract_result(self, final_state: dict) -> dict[str, Any]:
        hermes = final_state.get("_hermes", {})
        if hermes.get("exit") != 0 or not hermes.get("answer"):
            raise RuntimeError(f"Hermes failed or returned no answer; exit={hermes.get('exit')}; details: {hermes.get('artifact', '')}")
        return standard_result(
            final_message=hermes.get("answer") or "",
            reasoning=[],
            tool_calls=hermes.get("tool_calls") or [],
            usage=hermes.get("usage") or {},
            query=final_state.get("query", ""),
        )
