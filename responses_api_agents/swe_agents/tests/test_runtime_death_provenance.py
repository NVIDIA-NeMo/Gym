# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real subprocess regressions for the host observer and startup overlay."""

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from responses_api_agents.swe_agents.runtime_death_provenance import RuntimeDeathMonitor


OVERLAY = Path(__file__).parents[1] / "runtime_provenance_overlay"


@pytest.fixture
def runtime_fixture(tmp_path):
    for package in ["openhands", "openhands/runtime", "openhands/runtime/impl", "openhands/runtime/impl/local"]:
        directory = tmp_path / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()
    (tmp_path / "openhands/runtime/action_execution_server.py").write_text("""
import os, signal, subprocess, sys, time
time.sleep(0.2)
mode = os.environ['OBSERVER_CASE']
if mode == 'audit_refused':
    import runpy
    def refuse(event, args):
        if event == 'sys.addaudithook':
            raise RuntimeError('hook installation refused')
    sys.addaudithook(refuse)
    runpy.run_path(os.environ['OBSERVER_OVERLAY'] + '/sitecustomize.py')
if mode in {'setup', 'self_kill'}:
    subprocess.run([sys.executable, '-c', 'pass'], check=True)
if mode == 'self_kill':
    os.kill(os.getpid(), signal.SIGKILL)
if mode == 'signal_before_setup':
    os.kill(os.getpid(), signal.SIGKILL)
raise RuntimeError('real startup failure before execution')
""")
    (tmp_path / "openhands/runtime/impl/local/local_runtime.py").write_text("""
import os, subprocess, sys, time
from types import SimpleNamespace

def _create_server():
    env = os.environ.copy()
    if env['OBSERVER_CASE'] == 'missing':
        env.pop('NEMO_GYM_RUNTIME_NONCE')
    if env['OBSERVER_CASE'] == 'stale':
        env['NEMO_GYM_RUNTIME_NONCE'] = 'different-run'
    if env['OBSERVER_CASE'] == 'audit_refused':
        env['PYTHONPATH'] = env['PYTHONPATH'].split(':', 1)[1]
    process = subprocess.Popen([sys.executable, '-m', 'openhands.runtime.action_execution_server'], env=env)
    return SimpleNamespace(process=process), 'unused'

class LocalRuntime:
    async def execute_action(self):
        return None
    def _wait_until_alive(self):
        while self.server_process.poll() is None:
            time.sleep(0.01)
        raise RuntimeError('Server process died')
""")
    controller = tmp_path / "evaluation/benchmarks/swe_bench/run_infer.py"
    controller.parent.mkdir(parents=True)
    controller.write_text("""
import asyncio, os, subprocess, sys
if os.environ['OBSERVER_CASE'] == 'controller_setup':
    subprocess.run([sys.executable, '-c', 'pass'], check=True)
from openhands.runtime.impl.local import local_runtime as local
runtime = local.LocalRuntime()
for attempt in range(2 if os.environ['OBSERVER_CASE'] in {'retry_after_action', 'retry_missing'} else 1):
    if attempt and os.environ['OBSERVER_CASE'] == 'retry_missing':
        os.environ['OBSERVER_CASE'] = 'missing'
    runtime.server_process = local._create_server()[0].process
    if os.environ['OBSERVER_CASE'] == 'retry_after_action' and attempt == 0:
        asyncio.run(runtime.execute_action())
    try:
        runtime._wait_until_alive()
    except RuntimeError:
        pass
""")
    return controller


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "cause"),
    [
        ("pre_execution", "pre_execution"),
        ("forwarding_launcher", "pre_execution"),
        ("setup", "unknown"),
        ("controller_setup", "unknown"),
        ("self_kill", "unknown"),
        ("signal_before_setup", "unknown"),
        ("missing", "unknown"),
        ("stale", "unknown"),
        ("audit_refused", "unknown"),
        ("retry_after_action", "unknown"),
        ("retry_missing", "unknown"),
    ],
)
async def test_observer_tracks_real_processes_and_retries(runtime_fixture, case, cause):
    monitor = RuntimeDeathMonitor()
    await monitor.start()
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=f"{OVERLAY}:{runtime_fixture.parents[3]}",
        NEMO_GYM_RUNTIME_NONCE=monitor.nonce,
        NEMO_GYM_RUNTIME_SOCKET=monitor.path,
        OBSERVER_CASE=case,
        OBSERVER_OVERLAY=str(OVERLAY),
    )
    command = [sys.executable, str(runtime_fixture)]
    if case == "forwarding_launcher":
        launcher = runtime_fixture.parents[3] / "poetry"
        launcher.write_text("import subprocess, sys\nsubprocess.run([sys.executable, *sys.argv[3:]], check=True)\n")
        command = [sys.executable, str(launcher), "run", "python", str(runtime_fixture)]
    process = await asyncio.create_subprocess_exec(
        *command,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    monitor.set_root(process.pid)
    stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
    assert process.returncode == 0, (stdout, stderr)
    observed = await monitor.finish()
    assert observed == cause, (monitor.peers, monitor.expected, monitor.exits, monitor.invalid, stderr)
    assert not Path(monitor.path).exists()


@pytest.mark.asyncio
async def test_missing_observations_never_mask():
    monitor = RuntimeDeathMonitor()
    await monitor.start()
    assert await monitor.finish() == "unknown"


def test_observer_failure_preserves_token_capture_overlay(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_RUNTIME_NONCE", "test-run")
    monkeypatch.setenv("NEMO_GYM_RUNTIME_SOCKET", "/missing-observer.sock")
    monkeypatch.setenv("NEMO_GYM_MODEL_SERVER_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "openhands.runtime.action_execution_server"])
    spec = importlib.util.spec_from_file_location("test_runtime_overlay", OVERLAY / "sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    with (
        patch("socket.socket") as socket_factory,
        patch("os.path.isfile", return_value=True),
        patch("runpy.run_path") as capture_overlay,
    ):
        socket_factory.return_value.connect.side_effect = OSError("observer unavailable")
        with pytest.raises(OSError, match="observer unavailable"):
            spec.loader.exec_module(module)
        capture_overlay.assert_called_once_with("/nemo_gym_capture_overlay/sitecustomize.py")


@pytest.mark.parametrize(
    "entrypoint",
    [
        ["poetry", "run", "python", "evaluation/benchmarks/swe_bench/run_infer.py"],
        ["poetry", "run", "python", "-m", "openhands.runtime.action_execution_server"],
    ],
)
def test_forwarding_launcher_does_not_register_as_runtime(monkeypatch, entrypoint):
    monkeypatch.setenv("NEMO_GYM_RUNTIME_NONCE", "test-run")
    monkeypatch.setattr(sys, "orig_argv", ["python", *entrypoint])
    monkeypatch.delenv("NEMO_GYM_MODEL_SERVER_BASE_URL", raising=False)
    spec = importlib.util.spec_from_file_location("test_runtime_launcher", OVERLAY / "sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    with patch("socket.socket") as socket_factory:
        spec.loader.exec_module(module)
        socket_factory.assert_not_called()
