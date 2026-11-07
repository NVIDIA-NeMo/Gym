# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox runner, run as the supervisor runs it: a separate process next to its harness files."""

import importlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from uuid import uuid4

import httpx
import pytest
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError


RUNNER = Path(__file__).resolve().parents[1] / "sandbox_runner.py"
needs_tmux = pytest.mark.skipif(
    shutil.which("tmux") is None or shutil.which("script") is None, reason="needs tmux and script"
)


def _runner_module():
    # Importing the runner configures LiteLLM through the environment; keep that out of this process.
    with mock.patch.dict(os.environ):
        return importlib.import_module("responses_api_agents.terminus_2_agent.sandbox_runner")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (
            httpx.HTTPStatusError("busy", request=httpx.Request("POST", "http://m"), response=httpx.Response(429)),
            ("model_request_error", 429),
        ),
        (httpx.ConnectError("refused"), ("model_transport_error", None)),
        (httpx.ReadTimeout("slow"), ("model_transport_error", None)),
        (ContextLengthExceededError("too long"), ("context_limit", None)),
        (OutputLengthExceededError("truncated", truncated_response="x"), ("output_limit", None)),
        (RuntimeError("tmux exited"), ("harness_error", None)),
    ],
)
def test_errors_are_classified_by_what_ended_the_run(error: BaseException, kind: tuple) -> None:
    assert _runner_module()._classify_error(error) == kind


def test_the_runner_modules_import_without_nemo_gym() -> None:
    """The sandbox runtime has only Harbor, so the runner and the modules beside it must not need nemo_gym."""
    code = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'nemo_gym':\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {str(RUNNER.parent)!r})\n"
        "import sandbox_runner, terminus, llm\n"
        "llm.NemoGymLLM(model_name='m', api_base='http://x/v1', chat_params={})\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, timeout=120)


def test_an_unreadable_input_still_writes_a_result(tmp_path: Path) -> None:
    output = tmp_path / "output.json"

    finished = subprocess.run(
        [sys.executable, "-s", str(RUNNER), str(tmp_path / "missing.json"), str(output)],
        capture_output=True,
        timeout=120,
    )

    assert finished.returncode == 1
    result = json.loads(output.read_text())
    assert (result["stop_reason"], result["error_kind"], result["error_type"]) == (
        "error",
        "harness_error",
        "FileNotFoundError",
    )


class _Model:
    """Scripted Terminus replies: one command, then a confirmed completion; optionally stall before completing.

    Without ``complete``, every reply sends the command again and never claims completion.
    """

    def __init__(self, command: str, *, stall: float = 0, complete: bool = True) -> None:
        self.headers: list[str | None] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.headers.append(self.headers.get("x-session-id"))
                turns = sum(1 for message in body["messages"] if message["role"] == "assistant")
                if turns and stall:
                    time.sleep(stall)
                reply = {"analysis": "a", "plan": "p", "commands": [], "task_complete": True}
                if turns == 0 or not complete:
                    reply = {**reply, "commands": [{"keystrokes": command, "duration": 1}], "task_complete": False}
                data = json.dumps(
                    {
                        "id": f"chatcmpl-{uuid4().hex}",
                        "choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def episode(tmp_path: Path):
    """An input for one run and a private tmux server, stopped afterwards: the runner leaves its terminal up."""
    workdir = tmp_path / "task"
    workdir.mkdir()
    # A socket path must stay short (104 bytes on macOS), shorter than pytest's temporary directories.
    sockets = Path(tempfile.mkdtemp(prefix="terminus-tmux-", dir="/tmp"))
    environment = {**os.environ, "TMUX_TMPDIR": str(sockets), "TASK_MARKER": "from-task"}
    environment.pop("TMUX", None)

    def payload(model: _Model, *, max_turns: int = 5) -> Path:
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps(
                {
                    "instruction": "write the marker",
                    "model": "test-model",
                    "model_base_url": model.url,
                    "chat_params": {"temperature": 0.0},
                    "invocation_id": "terminus2_test",
                    "model_timeout_sec": 60,
                    "command_timeout_sec": 30,
                    "logs_dir": str(tmp_path / "logs"),
                    "stop_request_path": str(tmp_path / "stop.request"),
                    "workdir": str(workdir),
                    "terminus": {
                        "max_turns": max_turns,
                        "parser_name": "json",
                        "temperature": 0.0,
                        "reasoning_effort": None,
                        "collect_rollout_details": False,
                        "enable_summarize": False,
                        "proactive_summarization_threshold": 0,
                        "max_thinking_tokens": None,
                        "model_info": {"max_input_tokens": 32768, "max_output_tokens": 4096},
                        "trajectory_config": {"raw_content": False},
                        "tmux_pane_width": 160,
                        "tmux_pane_height": 40,
                        "store_all_messages": False,
                        "interleaved_thinking": False,
                    },
                }
            )
        )
        return path

    yield workdir, environment, payload
    subprocess.run(["tmux", "kill-server"], env=environment, capture_output=True, timeout=30)
    shutil.rmtree(sockets, ignore_errors=True)


@needs_tmux
def test_a_run_completes_in_tmux_with_the_task_environment(tmp_path: Path, episode) -> None:
    workdir, environment, payload = episode
    model = _Model('echo "$TASK_MARKER:${LITELLM_LOCAL_MODEL_COST_MAP:-unset}" > marker.txt\n')
    try:
        finished = subprocess.run(
            [sys.executable, "-s", str(RUNNER), str(payload(model)), str(tmp_path / "output.json")],
            env=environment,
            capture_output=True,
            timeout=240,
        )
    finally:
        model.close()

    assert finished.returncode == 0, finished.stderr.decode(errors="replace")[-2000:]
    result = json.loads((tmp_path / "output.json").read_text())
    assert (result["stop_reason"], result["finished_naturally"], result["error"]) == ("completed", True, None)
    assert len(result["response_ids"]) == len(model.headers) == 3
    assert result["calls_without_usage"] == 0
    # Every model call names the invocation, so model-call capture can join it.
    assert set(model.headers) == {"terminus2_test"}
    # Tools run in the task directory with the environment the runner started with.
    assert (workdir / "marker.txt").read_text().strip() == "from-task:unset"
    assert [step["source"] for step in result["trajectory"]["steps"]].count("agent") == 3


@needs_tmux
@pytest.mark.parametrize(("stop_requested", "reason"), [(False, "wall_time"), (True, "cancelled")])
def test_sigterm_saves_the_turns_taken_so_far(tmp_path: Path, episode, stop_requested: bool, reason: str) -> None:
    workdir, environment, payload = episode
    model = _Model("touch marker.txt\n", stall=120)
    try:
        process = subprocess.Popen(
            [sys.executable, "-s", str(RUNNER), str(payload(model)), str(tmp_path / "output.json")],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 120
        # The second model call stalls once the first command has run.
        while len(model.headers) < 2 and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.2)
        assert (workdir / "marker.txt").exists()
        if stop_requested:
            (tmp_path / "stop.request").touch()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 0, process.stdout.read().decode(errors="replace")[-2000:]
    finally:
        if process.poll() is None:
            process.kill()
        model.close()

    result = json.loads((tmp_path / "output.json").read_text())
    assert result["stop_reason"] == reason
    assert result["finished_naturally"] is False
    # The pending second request must not turn the first response subtotal into complete usage.
    assert result["calls_without_usage"] == 1
    assert [step["source"] for step in result["trajectory"]["steps"]].count("agent") >= 1


def _run(tmp_path: Path, environment: dict[str, str], input_path: Path) -> dict:
    """Run the runner to its end and return its result."""
    finished = subprocess.run(
        [sys.executable, "-s", str(RUNNER), str(input_path), str(tmp_path / "output.json")],
        env=environment,
        capture_output=True,
        timeout=240,
    )
    assert finished.returncode == 0, finished.stderr.decode(errors="replace")[-2000:]
    return json.loads((tmp_path / "output.json").read_text())


@needs_tmux
def test_a_run_that_uses_every_turn_ends_at_max_turns(tmp_path: Path, episode) -> None:
    workdir, environment, payload = episode
    model = _Model("echo turn >> turns.txt\n", complete=False)
    try:
        result = _run(tmp_path, environment, payload(model, max_turns=2))
    finally:
        model.close()

    assert (result["stop_reason"], result["finished_naturally"], result["error"]) == ("max_turns", False, None)
    assert len(model.headers) == 2


@needs_tmux
def test_a_command_that_moves_the_workdir_away_does_not_end_the_run(tmp_path: Path, episode) -> None:
    workdir, environment, payload = episode
    moved = workdir.with_name("task.bak")
    model = _Model(f"mv {shlex.quote(str(workdir))} {shlex.quote(str(moved))}\n")
    try:
        result = _run(tmp_path, environment, payload(model))
    finally:
        model.close()

    # Terminus's own tmux commands no longer run in the workdir, so the run continues to its completion.
    assert (result["stop_reason"], result["finished_naturally"], result["error"]) == ("completed", True, None)
    assert moved.is_dir() and not workdir.exists()


@needs_tmux
def test_a_background_process_holding_the_terminal_does_not_hold_the_run(tmp_path: Path, episode) -> None:
    workdir, environment, payload = episode
    model = _Model("(sleep 60; echo late > late.txt) & echo started > marker.txt\n")
    try:
        result = _run(tmp_path, environment, payload(model))
    finally:
        model.close()

    assert result["stop_reason"] == "completed"
    assert (workdir / "marker.txt").read_text().strip() == "started"
    # The run ended while the background process was still running.
    assert not (workdir / "late.txt").exists()
