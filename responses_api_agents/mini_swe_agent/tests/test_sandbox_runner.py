# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-sandbox runner: how a run ends, the supervisor's stop, and task MCP servers.

The runner needs only the standard library to start; mini-SWE itself is imported from the sandbox runtime. Tests that
run real mini-SWE 2.4.6 need that runtime's interpreter in ``MINI_SWE_RUNTIME_PYTHON``, a venv built from
``runtime/requirements.in`` with ``-c runtime/requirements.txt --override runtime/overrides.txt``; conftest.py builds
one when ``uv`` is on PATH.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from responses_api_agents.mini_swe_agent import sandbox_runner
from responses_api_agents.mini_swe_agent.trajectory import project


RUNNER = Path(sandbox_runner.__file__)
RUNTIME_PYTHON = os.environ.get("MINI_SWE_RUNTIME_PYTHON")
needs_runtime = pytest.mark.skipif(
    not RUNTIME_PYTHON, reason="set MINI_SWE_RUNTIME_PYTHON to a mini-SWE 2.4.6 runtime"
)
HARNESS_VARIABLES = (
    "MSWEA_GLOBAL_CONFIG_DIR",
    "MSWEA_CONFIGURED",
    "MSWEA_SILENT_STARTUP",
    "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT",
    "LITELLM_LOCAL_MODEL_COST_MAP",
)


@pytest.fixture(autouse=True)
def restore_process_state(monkeypatch):
    """The runner owns its process: it sets mini-SWE's variables and ignores SIGTERM on the way out."""
    for name in HARNESS_VARIABLES:
        # Recorded before removal, so what the runner sets is removed again after the test.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    handler = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, handler)


class FakeAgent:
    def __init__(self, run):
        self._run = run
        self.tasks = []

    def run(self, task):
        self.tasks.append(task)
        return self._run()


def payload(session_dir: Path, **overrides) -> dict:
    return {
        "task": "Fix it.",
        "skills_dir": None,
        "config": {},
        "trajectory_path": str(session_dir / "trajectory.json"),
        "mcp": None,
        "stop_request_path": str(session_dir / "stop.request"),
        **overrides,
    }


def test_a_finished_loop_exits_zero_and_records_how_it_ended(tmp_path, monkeypatch) -> None:
    agent = FakeAgent(lambda: {"exit_status": "LimitsExceeded"})
    monkeypatch.setattr(sandbox_runner, "build_agent", lambda payload, attribute_calls: agent)

    code = sandbox_runner.run(payload(tmp_path, skills_dir="/skills"), tmp_path, tmp_path / "output.json")

    assert code == 0
    output = json.loads((tmp_path / "output.json").read_text())
    assert output["exit_status"] == "LimitsExceeded" and output["error"] is None
    assert output["runtime"]["pid"] == os.getpid()
    assert agent.tasks == ["Fix it.\nTask skills are in /skills. Read the relevant SKILL.md files.\n"]
    # mini-SWE's own configuration stays in the session directory, never in the task's HOME.
    assert os.environ["MSWEA_GLOBAL_CONFIG_DIR"] == str(tmp_path / "config")
    assert os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "1"


def test_an_exception_from_the_loop_exits_one_with_its_traceback(tmp_path, monkeypatch) -> None:
    def fail():
        raise ValueError("bad config")

    monkeypatch.setattr(sandbox_runner, "build_agent", lambda payload, attribute_calls: FakeAgent(fail))

    assert sandbox_runner.run(payload(tmp_path), tmp_path, tmp_path / "output.json") == 1
    error = json.loads((tmp_path / "output.json").read_text())["error"]
    assert (error["type"], error["message"]) == ("ValueError", "bad config")
    assert "Traceback" in error["traceback"]


@pytest.mark.parametrize(("stop_requested", "reason"), [(False, "wall_time"), (True, "cancelled")])
def test_the_supervisors_sigterm_stops_the_loop_and_says_why(tmp_path, monkeypatch, stop_requested, reason) -> None:
    def interrupted():
        if stop_requested:
            (tmp_path / "stop.request").write_text("")
        # Without the runner's handler, this signal would stop pytest itself.
        assert signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, signal.SIG_IGN)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        raise AssertionError("SIGTERM did not stop the loop")

    monkeypatch.setattr(sandbox_runner, "build_agent", lambda payload, attribute_calls: FakeAgent(interrupted))

    assert sandbox_runner.run(payload(tmp_path), tmp_path, tmp_path / "output.json") == 143
    output = json.loads((tmp_path / "output.json").read_text())
    assert output["stop_reason"] == reason and output["error"] is None


def test_terminated_is_not_an_exception_a_harness_can_swallow() -> None:
    # mini-SWE turns any Exception from a step into a model or format error; a stop must not become one.
    assert not issubclass(sandbox_runner.Terminated, Exception)


FAKE_CLI = """
import json, os, sys, time
from pathlib import Path
here = Path(__file__).parent
if sys.argv[1] == "serve":
    (here / "served.json").write_text(json.dumps({"pid": os.getpid(), "servers": json.loads((here / "mcp_servers.json").read_text())}))
    (here / "mcp.sock").write_text("")
    time.sleep(60)
elif sys.argv[1] == "list":
    print((here / "listing.json").read_text())
"""


def stage_fake_mcp(session_dir: Path, listing: dict) -> None:
    (session_dir / "mcp_cli.py").write_text(FAKE_CLI)
    (session_dir / "mcp_servers.json").write_text(json.dumps([{"name": n, "headers": {"A": "t"}} for n in listing]))
    (session_dir / "listing.json").write_text(json.dumps(listing))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it no longer runs.
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split(")")[-1].split()[0] == "Z")


def test_mcp_tools_reach_the_task_and_the_server_stops_with_the_run(tmp_path, monkeypatch) -> None:
    stage_fake_mcp(
        tmp_path,
        {"task_tools": {"tools": [{"name": "get_weather", "inputSchema": {}}]}, "extra": {"error": "ConnectError"}},
    )
    agent = FakeAgent(lambda: {"exit_status": "Submitted"})
    attributed = []

    def build(payload, attribute_calls):
        attributed.append(attribute_calls)
        return agent

    monkeypatch.setattr(sandbox_runner, "build_agent", build)
    mcp = {"servers": ["task_tools", "extra"], "required": ["task_tools"]}

    assert sandbox_runner.run(payload(tmp_path, mcp=mcp), tmp_path, tmp_path / "output.json") == 0

    served = json.loads((tmp_path / "served.json").read_text())
    assert served["servers"][0]["headers"] == {"A": "t"}
    # The grant headers were read by the server and are no longer on disk for the model's shell.
    assert not (tmp_path / "mcp_servers.json").exists()
    assert not alive(served["pid"])
    task = agent.tasks[0]
    assert '"get_weather"' in task and "extra" not in task
    assert f"{sys.executable} -E -s {tmp_path / 'mcp_cli.py'} call SERVER TOOL 'JSON_ARGUMENTS'" in task
    assert attributed == [True]
    output = json.loads((tmp_path / "output.json").read_text())
    assert output["mcp"] == {"servers": ["task_tools"], "unavailable": [{"server": "extra", "error": "ConnectError"}]}


def test_a_required_mcp_server_that_did_not_connect_stops_the_run_before_any_model_call(tmp_path, monkeypatch) -> None:
    stage_fake_mcp(tmp_path, {"task_tools": {"error": "ConnectError: refused"}})
    built = []
    monkeypatch.setattr(sandbox_runner, "build_agent", lambda payload, attribute_calls: built.append(1))
    mcp = {"servers": ["task_tools"], "required": ["task_tools"]}

    assert sandbox_runner.run(payload(tmp_path, mcp=mcp), tmp_path, tmp_path / "output.json") == 3
    assert built == []
    error = json.loads((tmp_path / "output.json").read_text())["error"]
    # The reason each required server did not connect stays in the failure.
    assert error == {
        "type": "mcp_unavailable",
        "message": "Required MCP servers did not connect: task_tools (ConnectError: refused)",
    }


def test_an_mcp_server_process_that_exits_before_it_is_ready_stops_the_run(tmp_path, monkeypatch) -> None:
    # A byte that is not UTF-8 in the server's output still leaves a readable reason.
    (tmp_path / "mcp_cli.py").write_text(
        "import sys; sys.stderr.buffer.write(b'\\xff no servers file\\n'); sys.exit(4)\n"
    )
    monkeypatch.setattr(sandbox_runner, "build_agent", lambda payload, attribute_calls: pytest.fail("ran"))
    mcp = {"servers": ["task_tools"], "required": ["task_tools"]}

    assert sandbox_runner.run(payload(tmp_path, mcp=mcp), tmp_path, tmp_path / "output.json") == 3
    message = json.loads((tmp_path / "output.json").read_text())["error"]["message"]
    assert "exited with 4" in message and "no servers file" in message


def test_main_removes_the_staged_input_before_the_run(tmp_path, monkeypatch) -> None:
    (tmp_path / "input.json").write_text(json.dumps(payload(tmp_path)))
    seen = []
    monkeypatch.setattr(
        sandbox_runner,
        "run",
        lambda payload, directory, output: seen.append(directory.joinpath("input.json").exists()),
    )
    monkeypatch.setattr(sys, "argv", ["sandbox_runner.py", str(tmp_path / "input.json"), str(tmp_path / "out.json")])

    sandbox_runner.main()

    assert seen == [False]


# -- real mini-SWE 2.4.6 ---------------------------------------------------------------------------------------------


class ScriptedModel(ThreadingHTTPServer):
    """A Responses API endpoint that answers with scripted bash calls and records each request."""

    def __init__(self, commands: list[str], *, hang_after: int | None = None):
        self.commands = list(commands)
        self.requests: list[dict] = []
        self.hang_after = hang_after
        super().__init__(("127.0.0.1", 0), _ScriptedHandler)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/ng-rollout/rollout-7/v1"


class _ScriptedHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        server: ScriptedModel = self.server
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        server.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        index = len(server.requests) - 1
        if server.hang_after is not None and index >= server.hang_after:
            time.sleep(60)
        command = server.commands[min(index, len(server.commands) - 1)]
        response = {
            "id": f"resp_{index}",
            "object": "response",
            "created_at": 1,
            "model": body.get("model", "m"),
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": f"fc_{index}",
                    "call_id": f"call_{index}",
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5,
            },
        }
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def scripted_model():
    servers = []

    def start(commands, **kwargs):
        server = ScriptedModel(commands, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()


def stage_run(tmp_path: Path, model: ScriptedModel, *, command_timeout: int = 30) -> tuple[Path, Path, Path]:
    session_dir, workdir, home = tmp_path / "session", tmp_path / "task", tmp_path / "home"
    for directory in (session_dir, workdir, home):
        directory.mkdir()
    shutil.copy(RUNNER, session_dir / "sandbox_runner.py")
    staged = payload(
        session_dir,
        config={
            "agent": {"step_limit": 5, "cost_limit": 0},
            "environment": {"cwd": str(workdir), "timeout": command_timeout},
            "model": {
                "model_class": "litellm_response",
                "model_name": "openai/served-model",
                "cost_tracking": "ignore_errors",
                "model_kwargs": {
                    "api_base": model.base_url,
                    "api_key": "gym",
                    "timeout": 60,
                    "max_retries": 0,
                    "temperature": 0.3,
                    "extra_headers": {"x-session-id": "rollout-7"},
                    "extra_body": {"max_output_tokens": 8},
                },
            },
        },
    )
    (session_dir / "input.json").write_text(json.dumps(staged))
    return session_dir, workdir, home


def launch(session_dir: Path, workdir: Path, home: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            RUNTIME_PYTHON,
            "-E",
            "-s",
            str(session_dir / "sandbox_runner.py"),
            str(session_dir / "input.json"),
            str(session_dir / "output.json"),
        ],
        cwd=workdir,
        env={"PATH": os.environ["PATH"], "HOME": str(home)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


@needs_runtime
def test_real_mini_swe_runs_the_task_through_the_rollout_url_and_submits(tmp_path, scripted_model) -> None:
    model = scripted_model(["echo done > marker.txt && echo $HOME", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"])
    session_dir, workdir, home = stage_run(tmp_path, model)

    process = launch(session_dir, workdir, home)
    log, _ = process.communicate(timeout=120)

    assert process.returncode == 0, log.decode()
    assert (workdir / "marker.txt").read_text() == "done\n"
    trajectory = json.loads((session_dir / "trajectory.json").read_text())
    assert trajectory["info"]["exit_status"] == "Submitted"
    # mini-SWE saves no observation for the submission; the exit message keeps its output for the record.
    assert trajectory["messages"][-1]["extra"]["output"].startswith("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
    output = json.loads((session_dir / "output.json").read_text())
    assert output["exit_status"] == "Submitted" and output["error"] is None
    # Every call carries the rollout path, its owner, mini-SWE's bash tool, and the request's sampling.
    assert len(model.requests) == 2
    assert [request["body"]["max_output_tokens"] for request in model.requests] == [8, 8]
    first = model.requests[0]
    assert first["path"] == "/ng-rollout/rollout-7/v1/responses"
    assert first["headers"]["x-session-id"] == "rollout-7"
    assert [tool["name"] for tool in first["body"]["tools"]] == ["bash"]
    assert first["body"]["tools"][0]["strict"] is False
    assert (first["body"]["temperature"], first["body"]["max_output_tokens"]) == (0.3, 8)
    # The tool shell kept the task's HOME, and mini-SWE wrote nothing there.
    observation = next(m for m in trajectory["messages"] if m.get("type") == "function_call_output")
    assert str(home) in observation["output"] and observation["extra"]["duration_ms"] >= 0
    assert list(home.iterdir()) == []
    assert not (session_dir / "input.json").exists()
    # The trajectory stays where the model's commands can read it, so it keeps no rollout URL, capture key or API key.
    saved = (session_dir / "trajectory.json").read_text()
    assert model.base_url not in saved and "rollout-7" not in saved
    assert {"api_base", "api_key", "extra_headers"}.isdisjoint(trajectory["info"]["config"]["model"]["model_kwargs"])


@needs_runtime
def test_real_mini_swe_saves_its_steps_when_the_supervisor_stops_it(tmp_path, scripted_model) -> None:
    model = scripted_model(["echo step one"], hang_after=1)
    session_dir, workdir, home = stage_run(tmp_path, model)

    process = launch(session_dir, workdir, home)
    deadline = time.monotonic() + 60
    while len(model.requests) < 2 and time.monotonic() < deadline:
        time.sleep(0.1)
    process.send_signal(signal.SIGTERM)
    log, _ = process.communicate(timeout=30)

    assert process.returncode == 143, log.decode()
    trajectory = json.loads((session_dir / "trajectory.json").read_text())
    # The first step was saved; the model call in flight at the stop is absent, and no exit status was recorded.
    assert trajectory["info"]["exit_status"] == ""
    assert [m.get("type") for m in trajectory["messages"] if m.get("type")] == ["function_call_output"]
    assert json.loads((session_dir / "output.json").read_text())["stop_reason"] == "wall_time"
    # mini-SWE counted the interrupted call, so the first call's usage is not reported as the run's.
    assert trajectory["info"]["model_stats"]["api_calls"] == 2
    assert project(trajectory, []).usage is None


@needs_runtime
def test_real_mini_swe_keeps_the_model_call_of_a_step_whose_command_killed_it(tmp_path, scripted_model) -> None:
    # The command kills the runner (its parent) before mini-SWE's end-of-step save.
    model = scripted_model(["kill -9 $PPID"])
    session_dir, workdir, home = stage_run(tmp_path, model)

    process = launch(session_dir, workdir, home)
    log, _ = process.communicate(timeout=120)

    assert process.returncode == -signal.SIGKILL, log.decode()
    assert not (session_dir / "output.json").exists()
    # The runner saved the trajectory before the command ran: the model's call is there, with no exit status.
    trajectory = json.loads((session_dir / "trajectory.json").read_text())
    assert trajectory["info"]["exit_status"] == ""
    projection = project(trajectory, [])
    assert [response.id for response in projection.responses] == ["resp_0"]
    assert [(tool.tool_call_id, tool.output) for tool in projection.tools] == [("call_0", None)]


@needs_runtime
def test_a_background_process_holding_stdout_open_does_not_hang_the_run(tmp_path, scripted_model) -> None:
    model = scripted_model(
        ["sh -c 'echo $$ > bg.pid; exec sleep 300' &", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    )
    session_dir, workdir, home = stage_run(tmp_path, model, command_timeout=3)

    process = launch(session_dir, workdir, home)
    try:
        log, _ = process.communicate(timeout=120)
    finally:
        background = workdir / "bg.pid"
        if background.exists():
            try:
                os.kill(int(background.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass

    assert process.returncode == 0, log.decode()
    trajectory = json.loads((session_dir / "trajectory.json").read_text())
    assert trajectory["info"]["exit_status"] == "Submitted"
    first = next(m for m in trajectory["messages"] if m.get("type") == "function_call_output")
    # mini-SWE kills the command's process group at the command timeout, background job included.
    assert first["extra"]["exception_type"] == "TimeoutExpired"


def test_a_model_failure_records_the_http_status_of_its_answer(tmp_path, monkeypatch) -> None:
    class APIError(Exception):
        # LiteLLM 1.106.0 raises APIError for a 402, 403, 409, 413 or 425 answer, with the status on the exception.
        status_code = 403

    def fail():
        raise APIError("forbidden")

    monkeypatch.setattr(sandbox_runner, "build_agent", lambda payload, attribute_calls: FakeAgent(fail))

    assert sandbox_runner.run(payload(tmp_path), tmp_path, tmp_path / "output.json") == 1
    error = json.loads((tmp_path / "output.json").read_text())["error"]
    assert (error["type"], error["status_code"]) == ("APIError", 403)


def test_an_mcp_cli_that_fails_to_start_is_stopped(tmp_path, monkeypatch) -> None:
    stage_fake_mcp(tmp_path, {"task_tools": {"error": "ConnectError: refused"}})
    started = []
    popen = subprocess.Popen

    def recorded(*args, **kwargs):
        started.append(popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(sandbox_runner.subprocess, "Popen", recorded)

    with pytest.raises(sandbox_runner.MCPUnavailable):
        sandbox_runner.start_mcp(tmp_path / "mcp_cli.py", required=["task_tools"], log=tmp_path / "mcp.log")

    # The run gets no process to stop, so the CLI and its stdio servers must not outlive it.
    [process] = [process for process in started if "serve" in process.args]
    assert process.poll() is not None
