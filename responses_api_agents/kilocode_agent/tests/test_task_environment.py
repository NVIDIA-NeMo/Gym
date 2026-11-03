# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool shells keep the task's home and XDG directories while Kilo's own state stays private."""

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from responses_api_agents.kilocode_agent.tests.test_sandbox import stage_locally


PLUGIN = Path(__file__).parents[1] / "task_environment.mjs"
TASK_DIRECTORIES = ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the Kilo plugin")
@pytest.mark.parametrize("custom_xdg", [False, True])
async def test_shell_environment_keeps_task_files_after_session_close(tmp_path: Path, custom_xdg: bool) -> None:
    session, workdir, home = tmp_path / "session", tmp_path / "task", tmp_path / "task-home"
    for directory in (session, workdir, home):
        directory.mkdir()
    shutil.copyfile(PLUGIN, session / PLUGIN.name)
    # The probe stands in for Kilo: it loads the plugin, applies its shell.env hook, and runs a tool shell.
    (session / "probe.mjs").write_text(
        """
import { TaskEnvironment } from "./task_environment.mjs";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
const hooks = await TaskEnvironment();
const output = {env: {UNRELATED: "retained"}};
await hooks["shell.env"]({}, output);
mkdirSync(process.env.XDG_DATA_HOME + "/kilo", {recursive: true});
writeFileSync(process.env.XDG_DATA_HOME + "/kilo/cli-state", "private");
execFileSync("/bin/sh", ["-c", 'mkdir -p "$XDG_CONFIG_HOME"; printf task > "$HOME/home-marker"; printf config > "$XDG_CONFIG_HOME/config-marker"'],
  {env: {...process.env, ...output.env}});
console.log(JSON.stringify({cliHome: process.env.HOME, cliConfig: process.env.XDG_CONFIG_HOME, tool: output.env}));
"""
    )
    argv = await stage_locally(session, workdir, [shutil.which("node"), str(session / "probe.mjs")])
    # No task directories leak in from the agent-server host.
    env = {key: value for key, value in os.environ.items() if not key.startswith("XDG_")}
    env["HOME"] = str(home)
    if custom_xdg:
        env["XDG_CONFIG_HOME"] = str(tmp_path / "task-config")
    result = subprocess.run(
        argv,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + (session / "stderr.log").read_text()
    assert (session / "task-environment.ready").read_text() == "ready\n"
    observed = json.loads((session / "stdout.jsonl").read_text())
    assert (observed["cliHome"], observed["cliConfig"]) == (str(session / "home"), str(session / "config"))
    assert (session / "data/kilo/cli-state").read_text() == "private"
    task_config = tmp_path / "task-config" if custom_xdg else home / ".config"
    assert observed["tool"]["UNRELATED"] == "retained"
    assert observed["tool"]["HOME"] == str(home)
    assert observed["tool"]["XDG_CONFIG_HOME"] == str(task_config)
    assert observed["tool"]["XDG_DATA_HOME"] == str(home / ".local/share")
    shutil.rmtree(session)
    assert (home / "home-marker").read_text() == "task"
    assert (task_config / "config-marker").read_text() == "config"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the Kilo plugin")
@pytest.mark.parametrize("failure", ["missing", "malformed", "missing_home", "relative_home", "invalid_xdg"])
def test_invalid_task_environment_cannot_acknowledge_hook_initialization(tmp_path: Path, failure: str) -> None:
    shutil.copyfile(PLUGIN, tmp_path / PLUGIN.name)
    env = {key: f"/task/{key.lower()}" for key in TASK_DIRECTORIES}
    if failure == "missing_home":
        del env["HOME"]
    elif failure == "relative_home":
        env["HOME"] = "relative"
    elif failure == "invalid_xdg":
        env["XDG_CONFIG_HOME"] = None
    if failure != "missing":
        (tmp_path / "task-env.json").write_text("{" if failure == "malformed" else json.dumps(env))
    result = subprocess.run(
        [
            shutil.which("node"),
            "--input-type=module",
            "-e",
            'import { TaskEnvironment } from "./task_environment.mjs"; await TaskEnvironment();',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert not (tmp_path / "task-environment.ready").exists()


async def test_runner_removes_stale_task_environment_receipt_before_launch(tmp_path: Path) -> None:
    session, workdir = tmp_path / "session", tmp_path / "task"
    session.mkdir()
    workdir.mkdir()
    receipt = session / "task-environment.ready"
    receipt.write_text("ready\n")
    mcp_receipt = session / "required-mcp.ready"
    mcp_receipt.write_text('["tools"]')
    command = [sys.executable, "-c", "print('Kilo continued without loading the hook')"]
    result = subprocess.run(
        await stage_locally(session, workdir, command),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Kilo continued" in (session / "stdout.jsonl").read_text()
    assert not receipt.exists()
    assert not mcp_receipt.exists()


@pytest.mark.parametrize("task_environment", ["valid", "missing", "malformed", "plugin_error"])
def test_pinned_cli_reports_whether_required_plugin_initialized(tmp_path: Path, task_environment: str) -> None:
    binary = os.environ.get("GYM_TEST_KILO_BINARY")
    if binary is None:
        pytest.skip("Set GYM_TEST_KILO_BINARY to the pinned Kilo 7.4.15 executable")
    plugin = tmp_path / PLUGIN.name
    shutil.copyfile(PLUGIN, plugin)
    if task_environment == "plugin_error":
        plugin.write_text(
            plugin.read_text().replace(
                "const hooks = {", 'throw new Error("hook initialization failed");\n  const hooks = {'
            )
        )
    if task_environment != "missing":
        directories = {key: str(tmp_path / "task" / key.lower()) for key in TASK_DIRECTORIES}
        (tmp_path / "task-env.json").write_text("{" if task_environment == "malformed" else json.dumps(directories))
    (tmp_path / "kilo.json").write_text(json.dumps({"plugin": [plugin.as_uri()]}))
    (tmp_path / "input.json").write_text("{}")
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "KILO_CONFIG": str(tmp_path / "kilo.json"),
        "KILO_DISABLE_PROJECT_CONFIG": "1",
        "KILO_NO_DAEMON": "1",
        "KILO_DB": ":memory:",
    }
    version = subprocess.run([binary, "--version"], env=env, capture_output=True, text=True, timeout=60)
    assert version.returncode == 0 and version.stdout.strip() == "7.4.15"
    # The real plugin loader, without a model request. Kilo logs a plugin failure and still exits 0.
    result = subprocess.run(
        [binary, "debug", "agent", "build", "--print-logs"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    receipt = tmp_path / "task-environment.ready"
    if task_environment == "valid":
        assert receipt.read_text() == "ready\n"
    else:
        assert not receipt.exists()


@pytest.mark.parametrize("home", ["/task/a\tb", "/task/new\nline", '/task/"quoted"\\back'])
def test_recorded_task_directories_are_valid_json(tmp_path: Path, home: str) -> None:
    from shlex import quote

    from responses_api_agents.kilocode_agent.sandbox import _ACTIVATION_SETUP

    env = {"PATH": os.environ["PATH"], "HOME": home, "XDG_CONFIG_HOME": f"{home}\x01\x1fconfig"}
    setup = f"session={quote(str(tmp_path))}\n{_ACTIVATION_SETUP}"
    result = subprocess.run(["bash", "-c", setup], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    recorded = json.loads((tmp_path / "task-env.json").read_text())
    assert recorded["HOME"] == home and recorded["XDG_CONFIG_HOME"] == f"{home}\x01\x1fconfig"


def _plugin_probe(
    tmp_path: Path, script: str, *, preflight: bool = False, required: tuple[str, ...] = ("task.tools",)
) -> dict:
    """Exercise the shipped plugin through its public hooks in a fresh Node process."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the Kilo plugin")
    shutil.copyfile(PLUGIN, tmp_path / PLUGIN.name)
    (tmp_path / "task-env.json").write_text(
        json.dumps({key: str(tmp_path / "task" / key.lower()) for key in TASK_DIRECTORIES})
    )
    (tmp_path / "input.json").write_text(
        json.dumps({"required_mcp_servers": list(required), "mcp_preflight": preflight})
    )
    (tmp_path / "probe.mjs").write_text(
        'import { TaskEnvironment } from "./task_environment.mjs";\n'
        'import { readFile } from "node:fs/promises";\n'
        'const receipt = async () => { try { return JSON.parse(await readFile("./required-mcp.ready", "utf8")); } '
        "catch { return null; } };\n" + script
    )
    result = subprocess.run([node, "probe.mjs"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("status", ["connected", "failed", "needs_auth", "missing", "request_error", "exception"])
def test_required_mcp_hook_checks_connection_and_hides_remote_credentials(tmp_path: Path, status: str) -> None:
    result = _plugin_probe(
        tmp_path,
        f"const status = {json.dumps(status)};\n"
        """
let calls = 0;
const client = {mcp: {status: async () => {
  calls++;
  if (status === "exception") throw new Error("Authorization: Bearer fixture-secret");
  if (status === "request_error") return {error: {message: "Bearer fixture-secret"}};
  return {data: status === "missing" ? {} : {"task.tools": {status}}};
}}};
const hooks = await TaskEnvironment({client});
let error;
try { await hooks["chat.params"](); } catch (failure) { error = failure.message; }
console.log(JSON.stringify({calls, error, receipt: await receipt()}));
""",
    )
    assert result["calls"] == 1
    if status == "connected":
        assert result["receipt"] == ["task.tools"] and "error" not in result
    else:
        assert result["receipt"] is None
        assert result["error"] == "Kilo could not initialize required Gym MCP servers"
    assert "fixture-secret" not in json.dumps(result)


def test_parallel_tool_definitions_share_one_mcp_preflight(tmp_path: Path) -> None:
    result = _plugin_probe(
        tmp_path,
        """
let calls = 0;
const client = {mcp: {status: async () => {
  calls++;
  await new Promise((resolve) => setTimeout(resolve, 20));
  return {data: {"task.tools": {status: "connected"}}};
}}};
const hooks = await TaskEnvironment({client});
await Promise.all(Array.from({length: 8}, () => hooks["tool.definition"]()));
await hooks["tool.definition"]();
console.log(JSON.stringify({calls, receipt: await receipt()}));
""",
        preflight=True,
    )
    assert result == {"calls": 1, "receipt": ["task.tools"]}


@pytest.mark.parametrize("status", ["connected", "failed"])
def test_activation_checks_required_mcp_once_before_model_requests(tmp_path: Path, status: str) -> None:
    """Kilo never refreshes MCP status after connecting, so later requests reuse the first result."""
    result = _plugin_probe(
        tmp_path,
        f"const status = {json.dumps(status)};\n"
        """
let calls = 0;
const client = {mcp: {status: async () => {
  calls++;
  return {data: {"task.tools": {status}}};
}}};
const hooks = await TaskEnvironment({client});
await hooks["tool.definition"]();
const beforeActivation = calls;
const errors = [];
for (let request = 0; request < 3; request++) {
  try { await hooks["chat.params"](); } catch (failure) { errors.push(failure.message); }
}
console.log(JSON.stringify({beforeActivation, calls, errors, receipt: await receipt()}));
""",
    )
    assert result["beforeActivation"] == 0
    assert result["calls"] == 1
    if status == "connected":
        assert result["receipt"] == ["task.tools"] and result["errors"] == []
    else:
        assert result["receipt"] is None
        assert result["errors"] == ["Kilo could not initialize required Gym MCP servers"] * 3


@pytest.mark.parametrize("preflight", [False, True])
def test_sessions_without_required_mcp_never_query_mcp_or_write_receipt(tmp_path: Path, preflight: bool) -> None:
    result = _plugin_probe(
        tmp_path,
        """
let calls = 0;
const client = {mcp: {status: async () => { calls++; throw new Error("unexpected MCP status query"); }}};
const hooks = await TaskEnvironment({client});
await hooks["tool.definition"]();
await hooks["chat.params"]();
await hooks["chat.params"]();
console.log(JSON.stringify({calls, receipt: await receipt()}));
""",
        preflight=preflight,
        required=(),
    )
    assert result == {"calls": 0, "receipt": None}


@contextmanager
def _mcp_preflight_server(*, unavailable: bool) -> Iterator[tuple[str, list[dict], list[dict]]]:
    """Stateless MCP and a model endpoint that records any accidental inference."""
    mcp_requests: list[dict] = []
    model_requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, status: int, payload: dict | None = None) -> None:
            encoded = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path.startswith("/model"):
                model_requests.append(body)
                self._respond(500, {"error": "Preflight must not call a model"})
                return
            mcp_requests.append({"body": body, "authorization": self.headers.get("Authorization")})
            if unavailable:
                self._respond(503, {"error": "MCP unavailable"})
                return
            if self.headers.get("Authorization") != "Bearer fixture-token":
                self._respond(401, {"error": "Missing task credential"})
                return
            method = body["method"]
            if method == "notifications/initialized":
                self._respond(202)
                return
            if method == "initialize":
                result = {
                    "protocolVersion": body["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "gym-test", "version": "1"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "ping",
                            "description": "Return the fixture task state",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                }
            else:
                self._respond(400, {"error": f"Unexpected MCP method: {method}"})
                return
            self._respond(200, {"jsonrpc": "2.0", "id": body["id"], "result": result})

        def do_GET(self) -> None:
            # A stateless JSON transport does not provide the optional SSE listener.
            self._respond(405)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", mcp_requests, model_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("behavior", ["connected", "unavailable", "missing_tool_hook"])
async def test_pinned_cli_mcp_preflight_uses_grant_without_inference(tmp_path: Path, behavior: str) -> None:
    binary = os.environ.get("GYM_TEST_KILO_BINARY")
    if binary is None:
        pytest.skip("Set GYM_TEST_KILO_BINARY to the pinned Kilo 7.4.15 executable")
    session, workdir = tmp_path / "session", tmp_path / "task"
    session.mkdir()
    workdir.mkdir()
    plugin = session / PLUGIN.name
    shutil.copyfile(PLUGIN, plugin)
    if behavior == "missing_tool_hook":
        plugin.write_text(plugin.read_text().replace('"tool.definition":', '"unused.tool.definition":'))
    env = {
        **os.environ,
        "HOME": str(tmp_path / "task-home"),
        **{key: str(tmp_path / key.lower()) for key in TASK_DIRECTORIES if key != "HOME"},
        "KILO_NO_DAEMON": "1",
    }
    version = subprocess.run([binary, "--version"], env=env, capture_output=True, text=True, timeout=60)
    assert version.returncode == 0 and version.stdout.strip() == "7.4.15"
    with _mcp_preflight_server(unavailable=behavior == "unavailable") as (url, mcp_requests, model_requests):
        config = {
            "model": "nemo/fixture",
            "provider": {
                "nemo": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": f"{url}/model/v1", "apiKey": "EMPTY"},
                    "models": {"fixture": {"name": "fixture"}},
                }
            },
            "mcp": {
                "task.tools": {
                    "type": "remote",
                    "url": f"{url}/mcp",
                    "headers": {"Authorization": "Bearer fixture-token"},
                    "enabled": True,
                    "oauth": False,
                    "timeout": 2000,
                }
            },
            "plugin": [plugin.as_uri()],
        }
        argv = await stage_locally(
            session,
            workdir,
            [binary, "debug", "agent", "build", "--print-logs"],
            config=json.dumps(config),
            required_mcp_servers=("task.tools",),
            mcp_preflight=True,
        )
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=90)
    assert model_requests == [], "MCP preflight must not consume model inference"
    assert (session / "task-environment.ready").read_text() == "ready\n"
    receipt = session / "required-mcp.ready"
    if behavior == "connected":
        assert result.returncode == 0, (session / "stderr.log").read_text()[-6000:]
        assert json.loads(receipt.read_text()) == ["task.tools"]
        methods = {request["body"]["method"] for request in mcp_requests}
        assert {"initialize", "tools/list"} <= methods
        assert all(request["authorization"] == "Bearer fixture-token" for request in mcp_requests)
    else:
        assert not receipt.exists(), "Unavailable MCP or an absent hook must not produce readiness"
