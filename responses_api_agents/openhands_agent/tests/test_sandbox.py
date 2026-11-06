# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime preparation, activation staging and artifact collection in the sandbox."""

import asyncio
import importlib.metadata
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from packaging.requirements import Requirement

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    ROLLOUT_STOP_METADATA_KEY,
    SandboxPythonRuntime,
    SandboxSession,
)
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.openhands_agent import sandbox as adapter
from responses_api_agents.openhands_agent.tests.test_app import agent, seed
from responses_api_agents.openhands_agent.tests.test_sandbox_runner import STUB


def exec_result(stdout="", *, code=0, stderr=""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=code, error_type=None)


class LocalSandbox:
    """A sandbox whose filesystem is this machine's."""

    def __init__(self):
        self.commands: list[str] = []

    async def upload(self, source, target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    async def download(self, source, target):
        shutil.copyfile(source, target)

    async def exec(self, command, *, cwd=None, timeout_s=None, **_):
        self.commands.append(command)
        done = subprocess.run(["bash", "-c", command], cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        return exec_result(done.stdout, code=done.returncode, stderr=done.stderr)


def test_the_runtime_installs_the_pinned_harness_and_checks_its_imports():
    requirements = {Requirement(line).name: Requirement(line) for line in adapter.RUNTIME.requirements}
    assert str(requirements["openhands-sdk"].specifier) == "==1.59.0"
    assert str(requirements["openhands-tools"].specifier) == "==1.59.0"
    # func-timeout publishes only an sdist; a direct URL with its digest is built on the host.
    digest = "74cd3c428ec94f4edfba81f9b2f14904846d5ffccc27c92433b8b5939b5575dd"  # pragma: allowlist secret
    assert requirements["func-timeout"].url.endswith(f"/func_timeout-4.3.5.tar.gz#sha256={digest}")
    for module in ("openhands.sdk", "openhands.tools.terminal", "openhands.tools.preset.default"):
        assert module in adapter.RUNTIME.import_check
    # The interpreter link sits beside python/, so 'pkill -f python' cannot match the runner's command line.
    assert adapter.INTERPRETER == f"{adapter.RUNTIME.directory}/openhands-gym"
    assert "python" not in adapter.INTERPRETER
    assert adapter.RUNTIME.directory.startswith(f"{adapter.RUNTIME_PREFIX}-")


def test_the_agent_server_environment_has_the_openhands_version_the_sandbox_runs():
    # The runtime installs the versions found in this environment, so the lock must pin the adapter's version.
    try:
        installed = {name: importlib.metadata.version(name) for name in ("openhands-sdk", "openhands-tools")}
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("openhands-sdk is not installed in this environment")
    assert installed == {"openhands-sdk": adapter.OPENHANDS_VERSION, "openhands-tools": adapter.OPENHANDS_VERSION}


@pytest.mark.asyncio
async def test_a_healthy_runtime_is_reused_and_linked(monkeypatch):
    install = AsyncMock()
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=True))
    monkeypatch.setattr(SandboxPythonRuntime, "install", install)
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=exec_result()))
    assert await adapter.prepare_runtime(sandbox, timeout=10) == adapter.INTERPRETER
    install.assert_not_awaited()
    link = shlex.quote(adapter.INTERPRETER)
    sandbox.exec.assert_awaited_once_with(
        f"test -L {link} || ln -s python/bin/python3 {link} || test -L {link}", cwd="/", timeout_s=30
    )


@pytest.mark.asyncio
async def test_a_missing_runtime_is_installed_within_the_seed_budget(monkeypatch):
    install = AsyncMock()
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=False))
    monkeypatch.setattr(SandboxPythonRuntime, "install", install)
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=exec_result()))
    assert await adapter.prepare_runtime(sandbox, timeout=10) == adapter.INTERPRETER
    install.assert_awaited_once_with(sandbox, timeout_s=10)


@pytest.mark.asyncio
async def test_a_failed_install_or_link_fails_the_seed(monkeypatch):
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=False))
    monkeypatch.setattr(SandboxPythonRuntime, "install", AsyncMock(side_effect=RuntimeError("no wheel for glibc")))
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=exec_result()))
    with pytest.raises(RuntimeError, match="no wheel for glibc"):
        await adapter.prepare_runtime(sandbox, timeout=10)
    sandbox.exec.assert_not_awaited()

    monkeypatch.setattr(SandboxPythonRuntime, "install", AsyncMock())
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=exec_result(code=1, stderr="read-only")))
    with pytest.raises(RuntimeError, match="interpreter link failed .*read-only"):
        await adapter.prepare_runtime(sandbox, timeout=10)


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
async def test_the_interpreter_link_runs_the_runtime_python_and_tolerates_a_concurrent_seed(monkeypatch, tmp_path):
    base = Path(sys.base_prefix)
    if not (base / "bin/python3").exists():
        pytest.skip("the base interpreter has no bin/python3")
    # A runtime directory whose python/ is this interpreter's installation, as the installer publishes it.
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "python").symlink_to(base)
    monkeypatch.setattr(adapter, "INTERPRETER", str(runtime / "openhands-gym"))
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=True))
    sandbox = LocalSandbox()
    for _ in range(2):
        assert await adapter.prepare_runtime(sandbox, timeout=10) == str(runtime / "openhands-gym")
    assert os.readlink(runtime / "openhands-gym") == "python/bin/python3"
    # Run through the link, the interpreter still finds its own standard library.
    done = subprocess.run(
        [str(runtime / "openhands-gym"), "-I", "-c", "import json, sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert Path(done.stdout.strip()).resolve() == base.resolve()


def local_state(session_dir: Path, *, interpreter: str = "/rt/openhands-gym", workdir: str = "/app"):
    async def output_log():
        return "supervisor log"

    session = SimpleNamespace(
        session_dir=str(session_dir),
        workdir=workdir,
        sandbox=LocalSandbox(),
        read_output_log=output_log,
        cleanup=None,
        expired=False,
        expired_before_launch=False,
    )
    return adapter.OpenHandsSandboxSession(request=seed(), session=session, interpreter=interpreter)


@pytest.mark.asyncio
async def test_collection_keeps_events_and_tolerates_a_missing_result_or_runtime(tmp_path):
    state = local_state(tmp_path)
    (tmp_path / "events.jsonl").write_text('[1.0, {"kind": "MessageEvent"}]\n')
    (tmp_path / "stderr.log").write_text("x" * 10 + "tail")
    (tmp_path / "runtime.json").write_text('{"hostname": "box", "pid": "12"}')
    collected = await state.collect_artifacts()
    assert (collected.events, collected.result, collected.stderr) == (
        '[1.0, {"kind": "MessageEvent"}]\n',
        None,
        "x" * 10 + "tail",
    )
    # A malformed runtime record is a diagnostic gap, not a failed episode.
    assert state.runtime_info is None


@pytest.mark.asyncio
async def test_collection_without_events_fails_with_the_runner_stderr(tmp_path):
    state = local_state(tmp_path)
    (tmp_path / "stderr.log").write_text("Traceback: ImportError: openhands")
    with pytest.raises(RuntimeError, match="recorded no events: Traceback: ImportError: openhands"):
        await state.collect_artifacts()


@pytest.mark.asyncio
async def test_an_unreadable_result_is_logged_and_left_to_the_activation(tmp_path, caplog):
    state = local_state(tmp_path)
    (tmp_path / "events.jsonl").write_text('[1.0, {"kind": "MessageEvent"}]\n')
    with caplog.at_level(logging.WARNING, logger=adapter.LOG.name):
        collected = await state.collect_artifacts()
    assert collected.result is None
    assert "Could not read OpenHands result.json" in caplog.text


class SupervisedSandbox:
    """Keeps uploads by path and answers the supervised launch with ``receipt``, once ``finish`` is set.

    The harness never writes events.jsonl.
    """

    uploads_independent_of_execs = True
    sandbox_id = "sbx"

    def __init__(self, receipt: dict):
        self.receipt = receipt
        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        self.launched, self.finish = asyncio.Event(), asyncio.Event()

    async def upload(self, source, target):
        self.files[target] = Path(source).read_text()

    async def download(self, source, target):
        if source not in self.files:
            raise FileNotFoundError(source)
        Path(target).write_text(self.files[source])

    async def exec(self, command, *, cwd=None, timeout_s=None, **_):
        self.commands.append(command)
        receipt = re.search(r"--receipt (\S+)", command)
        if receipt:
            self.launched.set()
            await self.finish.wait()
            self.files[receipt.group(1)] = json.dumps(self.receipt)
            return exec_result(code=self.receipt["return_code"] or 0)
        if "echo absent" in command:
            return exec_result("absent\n")
        return exec_result()


def supervised_state(sandbox: SupervisedSandbox) -> adapter.OpenHandsSandboxSession:
    session = SandboxSession(sandbox=sandbox, session_dir="/s", workdir="/app", harness="OpenHands")
    return adapter.OpenHandsSandboxSession(request=seed(), session=session, interpreter="/rt/openhands-gym")


BODY = NeMoGymResponseCreateParamsNonStreaming(input="Build it")


@pytest.mark.asyncio
async def test_a_rollout_stop_before_launch_is_graded_as_an_empty_stopped_run():
    # The supervisor finds the stop request and never starts the harness: no events, no return code.
    sandbox = SupervisedSandbox({"cleanup_confirmed": True, "return_code": None, "timed_out": True, "error": None})
    sandbox.finish.set()
    state = supervised_state(sandbox)
    assert await state.session.expire(timeout=1, reason="budget") is True
    response = await agent()._activate(state, BODY, prompt="Build it", system="")
    assert (response.status, response.metadata["stop_reason"], response.output) == ("incomplete", "timeout", [])
    assert json.loads(response.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "budget"}
    # OpenHands never ran, so collection reads nothing from the sandbox.
    assert not [command for command in sandbox.commands if "stderr.log" in command or "events.jsonl" in command]


@pytest.mark.asyncio
async def test_a_rollout_stop_before_the_first_event_is_graded_as_an_empty_stopped_run():
    sandbox = SupervisedSandbox({"cleanup_confirmed": True, "return_code": 143, "timed_out": True, "error": None})
    state = supervised_state(sandbox)
    activation = asyncio.create_task(agent()._activate(state, BODY, prompt="Build it", system=""))
    await asyncio.wait_for(sandbox.launched.wait(), timeout=5)
    assert await state.session.expire(timeout=1, reason="budget") is True
    sandbox.finish.set()
    response = await asyncio.wait_for(activation, timeout=5)
    assert (response.status, response.metadata["stop_reason"], response.output) == ("incomplete", "timeout", [])
    assert json.loads(response.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "budget"}


@pytest.mark.asyncio
async def test_a_deadline_without_events_is_a_failure_of_unknown_cause():
    # Without a rollout stop, a harness that recorded nothing by its deadline got no model reply, and may not have
    # made a model call; the Environment Server's outcome policy retries the episode instead of grading it.
    sandbox = SupervisedSandbox({"cleanup_confirmed": True, "return_code": 143, "timed_out": True, "error": None})
    sandbox.finish.set()
    response = await agent()._activate(supervised_state(sandbox), BODY, prompt="Build it", system="")
    assert (response.status, response.metadata["stop_reason"], response.output) == ("failed", "timeout", [])
    assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {"failure_kind": "unknown"}
    assert ROLLOUT_STOP_METADATA_KEY not in response.metadata


@pytest.mark.asyncio
async def test_a_timed_out_run_whose_events_cannot_be_read_is_still_a_failure(tmp_path):
    state = local_state(tmp_path)
    state.session.cleanup = {"timed_out": True}
    # events.jsonl exists but cannot be read, so the run may have recorded work that would be lost.
    (tmp_path / "events.jsonl").mkdir()
    with pytest.raises(RuntimeError, match="recorded no events"):
        await state.collect_artifacts()


@pytest.fixture
def interpreter(tmp_path: Path) -> Path:
    """A private interpreter with the stand-in SDK installed, run with -I like the runtime's openhands-gym."""
    root = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    (site,) = root.glob("lib/python3*/site-packages")
    modules = {
        "openhands/__init__.py": "",
        "openhands/sdk/__init__.py": STUB,
        "openhands/tools/__init__.py": "",
        "openhands/tools/file_editor.py": "class FileEditorTool:\n    name = 'file_editor'\n",
        "openhands/tools/task_tracker.py": "class TaskTrackerTool:\n    name = 'task_tracker'\n",
        "openhands/tools/terminal.py": "class TerminalTool:\n    name = 'terminal'\n",
        "openhands/tools/preset/__init__.py": "",
        "openhands/tools/preset/default.py": "def get_default_condenser(llm):\n    return llm\n",
    }
    for name, source in modules.items():
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        (site / name).write_text(source)
    # A tool command that leaves a background process holding the runner's stdout and stderr open.
    sdk = site / "openhands/sdk/__init__.py"
    sdk.write_text(
        sdk.read_text().replace(
            'if mode == "finish":',
            'if mode == "finish":\n'
            "            import subprocess\n"
            '            child = subprocess.Popen(["sleep", "60"], start_new_session=True)\n'
            '            open(os.environ["STUB_CHILD"], "w").write(str(child.pid))\n'
            "            import sys\n"
            "            note(stdin=sys.stdin.read())\n",
        )
    )
    # A venv finds its site-packages only from its own bin/, so the link stays there.
    link = root / "bin/openhands-gym"
    link.symlink_to(root / "bin/python")
    return link


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_staged_activation_runs_isolated_and_finishes_with_a_background_process(tmp_path, interpreter):
    session_dir = tmp_path / "session"
    (session_dir / "state").mkdir(parents=True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copyfile(adapter.RUNNER_SOURCE, session_dir / adapter.RUNNER_NAME)
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, session_dir / HARNESS_RUNNER_FILE)
    state = local_state(session_dir, interpreter=str(interpreter), workdir=str(workdir))
    runner_input = {
        "workdir": str(workdir),
        "prompt": "Build it",
        "system_message_suffix": None,
        "llm": {"model": "openai/m"},
        "condenser": False,
        "max_iterations": 3,
        "state_dir": state.path("state"),
        "events_path": state.path("events.jsonl"),
        "result_path": state.path("result.json"),
    }
    command = asyncio.run(state.stage_activation(runner_input)).argv
    params = (session_dir / "params.sh").read_text()
    # The runner has no .py name and the interpreter ignores PYTHONPATH and user site-packages.
    assert f"command=({shlex.quote(str(interpreter))} -I -B {state.path('openhands_runner')} " in params
    assert ".py" not in params
    child_file = tmp_path / "child.pid"
    shadow = tmp_path / "task_pythonpath/openhands"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise ImportError('the task PYTHONPATH shadowed the private runtime')\n")
    env = {
        **os.environ,
        "STUB_MODE": "finish",
        "STUB_LOG": str(tmp_path / "stub.log"),
        "STUB_CHILD": str(child_file),
        # -I must keep this from shadowing the private runtime's packages.
        "PYTHONPATH": str(shadow.parent),
    }
    started = time.monotonic()
    try:
        # Pipes stand in for the provider's exec channel: the background child must not hold them open.
        done = subprocess.run(command, capture_output=True, env=env, timeout=30)
    finally:
        if child_file.exists():
            try:
                os.killpg(int(child_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert done.returncode == 0, (done.stdout, done.stderr, (session_dir / "stderr.log").read_text())
    assert time.monotonic() - started < 20
    assert json.loads((session_dir / "result.json").read_text())["execution_status"] == "finished"
    notes = [json.loads(line) for line in (tmp_path / "stub.log").read_text().splitlines()]
    assert {"stdin": ""} in notes
    runtime = adapter.parse_runtime_info(json.loads((session_dir / "runtime.json").read_text()))
    assert runtime is not None and runtime.pid > 0
    collected = asyncio.run(state.collect_artifacts())
    assert json.loads(collected.result)["status"] == "returned"
