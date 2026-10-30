# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Pool's adapter plumbing with local file transfer, real processes, and a fake CLI."""

import asyncio
import json
import shlex
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.base import SandboxExecResult
from responses_api_agents.pool_agent.configuration import pool_config
from responses_api_agents.pool_agent.sandbox import PoolSession
from responses_api_agents.pool_agent.tests.test_app import PoolAgent, agent, seed


FAKE_POOL = r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--agent-config-file') + 1]).parent
config = json.loads((root / 'agent_config.json').read_text())
mode = os.environ.get('TEST_MODE', 'normal')
trajectory = root / 'state' / 'trajectory-test_gym-root.ndjson'
def emit(kind, payload):
    with trajectory.open('a') as f:
        f.write(json.dumps({'type': kind, kind.replace('.', '_'): payload}) + '\n')
def interrupt(*args):
    Path('signal-received').write_text('TERM')
    emit('session.exit', {'reason': 'exit_tool_called'})
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == 'stubborn' else interrupt)
shell = config['tools']['shell']
subprocess.run([shell['shell'], *shell['shell_args'],
    'printf "%s\\n%s\\n%s\\n" "$HOME" "$XDG_CONFIG_HOME" "$TASK_TEST" > proof.txt'], check=True)
print(json.dumps({'home': os.environ['HOME'], 'config_home': os.environ['XDG_CONFIG_HOME'],
    'provider_keys': sorted(k for k in os.environ if k.startswith(('POOL_', 'POOLSIDE_', 'OPENAI_'))),
    'base_url': config['model']['provider']['openai']['base_url'],
    'cwd': os.getcwd(), 'prompt': (root / 'prompt.txt').read_text()}), flush=True)
if mode not in ('missing', 'stall', 'lost'):
    emit('assistant_message.end', {'assistant_message': 'working'})
Path('ready').touch()
if mode in ('stall', 'lost'):
    from urllib.request import Request, urlopen
    url = config['model']['provider']['openai']['base_url'] + '/chat/completions'
    with urlopen(Request(url, data=b'{"messages":[]}'), timeout=20) as response:
        response.read()
if mode in ('term', 'stubborn', 'stall', 'lost'):
    time.sleep(60)
if mode != 'missing':
    emit('session.exit', {'reason': 'exit_tool_called'})
"""


class LocalSandbox:
    """Only substitute installation; execute every other sandbox command locally."""

    def __init__(self):
        self.events = []
        self.disconnected = False

    async def upload(self, source, destination):
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        self.events.append(f"read {Path(source).name}")
        shutil.copyfile(source, destination)

    async def exec(self, command, *, cwd=None, timeout_s=30):
        argv = shlex.split(command)
        if len(argv) > 1 and argv[1].endswith("/install_pool.sh"):
            assert argv[4] == "true"
            binary = Path(argv[2]) / "bin" / "pool"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text(f"#!{sys.executable}\n" + FAKE_POOL)
            binary.chmod(0o755)
            return SandboxExecResult("", "", 0)
        process = await asyncio.create_subprocess_exec(
            "sh", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)

    async def disconnect(self):
        self.events.append("disconnect")
        self.disconnected = True

    async def stop(self):
        pytest.fail("A borrowed sandbox must not be stopped")


def session(tmp_path, mode="normal"):
    workdir = tmp_path / "task with spaces"
    workdir.mkdir()
    body = seed(workdir=str(workdir))
    body.sandbox_access.env = {
        "TEST_MODE": mode,
        "TASK_TEST": "task-value",
        "HOME": str(tmp_path / "task-home"),
        "XDG_CONFIG_HOME": str(tmp_path / "task-config"),
        "POOL_REMOTE_SINK": "must-not-inherit",
        "POOLSIDE_API_KEY": "must-not-inherit",
        "OPENAI_BASE_URL": "http://must-not-inherit",
        "OPENAI_API_KEY": "must-not-inherit",
    }
    sandbox = LocalSandbox()
    return PoolSession(
        request=body,
        provider_name="local",
        sandbox_id="test",
        session=SandboxSession(
            sandbox=sandbox, session_dir=str(tmp_path / "private session"), workdir=str(workdir), harness="Pool"
        ),
    ), sandbox


async def execute(state, timeout=10):
    config = pool_config(base_url="http://model.invalid/ng-rollout/test/v1", model="test", context_window=1024)
    return await state.session.execute(
        stage_activation=lambda: state.stage(prompt="task input", config=json.dumps(config)),
        collect=state.collect,
        timeout=timeout,
        close_timeout=10,
    )


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize("ending", ["normal", "timeout", "close", "missing", "stubborn"])
async def test_real_stage_run_collect_close(tmp_path, ending):
    mode = "term" if ending in ("timeout", "close") else ending
    state, sandbox = session(tmp_path, mode)
    await state.install(version="v1.0.16", accept_eula=True, timeout=10)
    task = asyncio.create_task(execute(state, timeout=2 if ending in ("timeout", "stubborn") else 20))
    try:
        if ending == "close":
            async with asyncio.timeout(10):
                while not (Path(state.session.workdir) / "ready").exists():
                    await asyncio.sleep(0.02)
            await asyncio.wait_for(state.close(timeout=10), timeout=30)
        artifacts = await asyncio.wait_for(task, timeout=30)
        assert state.session.cleanup_confirmed
        assert artifacts.runner_pid > 0
        details = json.loads(artifacts.stdout)
        assert details["home"] == str(tmp_path / "private session/home")
        assert details["config_home"] == str(tmp_path / "private session/config")
        assert details["provider_keys"] == ["POOLSIDE_API_KEY"]
        assert details["base_url"].startswith("http://127.0.0.1:")
        assert details["cwd"] == state.session.workdir
        assert details["prompt"] == "task input"
        assert artifacts.trajectory_error == ("trajectory_unavailable" if ending == "missing" else None)
        assert sandbox.events.index("read cleanup.json") < sandbox.events.index("read stdout.jsonl")
        if ending in ("timeout", "close"):
            assert (Path(state.session.workdir) / "signal-received").read_text() == "TERM"
            assert artifacts.exit_code == 0
        if ending == "stubborn":
            # The supervisor's grace is shorter than the runner's fallback wait.
            assert artifacts.exit_code is None
            # The bash supervisor reports a signal death as 128 + SIGKILL.
            assert state.session.cleanup["return_code"] == 137
        assert state.session.cleanup["timed_out"] is (ending in ("timeout", "stubborn"))
    finally:
        await asyncio.wait_for(state.close(timeout=10), timeout=30)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert sandbox.disconnected
    assert not Path(state.session.session_dir).exists()
    assert (Path(state.session.workdir) / "proof.txt").read_text().splitlines() == [
        str(tmp_path / "task-home"),
        str(tmp_path / "task-config"),
        "task-value",
    ]
    assert sandbox.events.count("read stdout.jsonl") == 1
    assert sandbox.events.index("read stdout.jsonl") < sandbox.events.index("disconnect")


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap", ["same", "ancestor", "descendant"])
async def test_install_rejects_symlink_workdir_overlap(tmp_path, overlap):
    state, _ = session(tmp_path)
    root = Path(state.session.session_dir)
    root.mkdir()
    target = root if overlap == "same" else root.parent if overlap == "ancestor" else root / "nested"
    target.mkdir(exist_ok=True)
    link = tmp_path / "workdir-link"
    link.symlink_to(target, target_is_directory=True)
    state.session.workdir = str(link)
    with pytest.raises(RuntimeError, match="disjoint"):
        await state.install(version="v1.0.16", accept_eula=True, timeout=10)
    assert link.is_dir()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize("confirmed", [False, "true", 1])
async def test_bad_termination_evidence_never_collects_before_confirmation(tmp_path, confirmed):
    state, sandbox = session(tmp_path)
    await state.install(version="v1.0.16", accept_eula=True, timeout=10)
    download = sandbox.download
    corrupt = True

    async def read(source, destination):
        await download(source, destination)
        if corrupt and Path(source).name == "cleanup.json":
            receipt = json.loads(Path(destination).read_text())
            receipt["cleanup_confirmed"] = confirmed
            Path(destination).write_text(json.dumps(receipt))

    sandbox.download = read
    try:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(execute(state), timeout=30)
        assert state.session.artifacts is None
        assert "read stdout.jsonl" not in sandbox.events
        if confirmed is False:
            await asyncio.wait_for(state.close(timeout=5), timeout=15)
            assert state.session.termination_error is not None
            assert state.session.artifacts is None
        else:
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(state.close(timeout=1), timeout=5)
            assert not sandbox.disconnected
            corrupt = False
            await asyncio.wait_for(state.close(timeout=5), timeout=15)
            assert state.session.artifacts is not None
    finally:
        corrupt = False
        await asyncio.wait_for(state.close(timeout=5), timeout=15)
    assert sandbox.disconnected


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize("mode", ["stall", "lost"])
async def test_activation_uses_proxy_evidence_when_cli_has_no_model_output(tmp_path, mode):
    release = threading.Event()
    paths = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            paths.append(self.path)
            if mode == "stall":
                release.wait(timeout=15)
                return
            payload = b'{"choices":[{"message":{"role":"assistant","content":"done"}}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    state, sandbox = session(tmp_path, mode)
    server = agent(timeout=3, failure_artifacts_dir=tmp_path / "failures")
    try:
        await state.install(version="v1.0.16", accept_eula=True, timeout=10)
        with patch.object(
            PoolAgent,
            "model_endpoint",
            return_value=MagicMock(base_url=f"http://127.0.0.1:{upstream.server_port}/ng-rollout/test/v1"),
        ):
            call = server._activate(state, NeMoGymResponseCreateParamsNonStreaming(input="task"))
            if mode == "stall":
                with pytest.raises(RuntimeError, match="before any model response"):
                    await asyncio.wait_for(call, timeout=20)
                assert state.failure == "RuntimeError"
            else:
                response = await asyncio.wait_for(call, timeout=20)
                assert response.status == "incomplete" and response.output == []
        assert paths == ["/ng-rollout/test/v1/chat/completions"]
        assert state.session.artifacts.completed_model_calls == int(mode == "lost")
        assert state.session.cleanup["timed_out"] is True
        await asyncio.wait_for(server._close_agent_session_state(state), timeout=20)
        assert sandbox.disconnected
        if mode == "stall":
            saved = json.loads(state.failure_artifact_path.read_text())
            assert saved["artifacts"]["completed_model_calls"] == 0
            assert saved["cleanup"]["cleanup_confirmed"] is True
    finally:
        release.set()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
        await asyncio.wait_for(state.close(timeout=5), timeout=15)


def test_runner_executes_fake_cli_without_changing_host_environment(tmp_path, monkeypatch):
    import os
    import signal

    from responses_api_agents.pool_agent import sandbox_runner

    state, _ = session(tmp_path)
    asyncio.run(state.install(version="v1.0.16", accept_eula=True, timeout=10))
    asyncio.run(
        state.stage(
            prompt="task input",
            config=json.dumps(pool_config(base_url="http://model.invalid/v1", model="test", context_window=1024)),
        )
    )
    directory = Path(state.session.session_dir)
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))
    monkeypatch.setenv("POOL_REMOTE_SINK", "host-value")
    previous = signal.getsignal(signal.SIGTERM)
    assert sandbox_runner.run(directory, state.session.workdir) == 0
    assert signal.getsignal(signal.SIGTERM) == previous
    assert os.environ["HOME"] == str(tmp_path / "host-home")
    assert os.environ["POOL_REMOTE_SINK"] == "host-value"
    assert json.loads((directory / "exit.json").read_text()) == 0
    assert json.loads((directory / "model_calls.json").read_text()) == {"completed": 0}
    output = json.loads((directory / "stdout.jsonl").read_text())
    assert output["home"] == str(directory / "home")
    assert output["provider_keys"] == ["POOLSIDE_API_KEY"]
    assert (Path(state.session.workdir) / "proof.txt").read_text().splitlines() == [
        str(tmp_path / "task-home"),
        str(tmp_path / "task-config"),
        "task-value",
    ]
