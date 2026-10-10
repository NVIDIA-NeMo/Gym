# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock
from urllib.request import urlopen

import pytest

from responses_api_agents.nooa_agent import sandbox_supervisor
from responses_api_agents.nooa_agent.sandbox_entrypoint import SandboxResult
from responses_api_agents.nooa_agent.tests.test_sandbox_entrypoint import payload, run_result


@pytest.mark.parametrize(
    "confirmed,error,exit_code", [(True, None, 0), (False, "cleanup failed", 1), (True, "launch failed", 1)]
)
def test_nooa_launcher_uses_main_reaper_without_a_second_deadline(
    monkeypatch, tmp_path: Path, confirmed: bool, error: str | None, exit_code: int
) -> None:
    receipt = {"cleanup_confirmed": confirmed, "return_code": 0, "error": error, "timed_out": False}
    reaper = MagicMock(return_value=receipt)
    monkeypatch.setattr(sandbox_supervisor.process_supervisor, "_supervise", reaper)
    assert sandbox_supervisor.supervise(tmp_path) == exit_code
    assert reaper.call_args.args == (
        [
            sys.executable,
            "-I",
            "-m",
            "responses_api_agents.nooa_agent.sandbox_entrypoint",
            str(tmp_path / "input.json"),
            str(tmp_path / "result.json"),
            str(tmp_path / "runner.stop"),
            str(tmp_path / "completion.json"),
        ],
    )
    assert reaper.call_args.kwargs == {
        "timeout": math.inf,
        "cleanup_timeout": 5,
        "stop_path": tmp_path / "runner.stop",
    }
    assert int((tmp_path / "runner.pid").read_text()) == os.getpid()
    assert json.loads((tmp_path / "cleanup.json").read_text()) == receipt
    assert not (tmp_path / "cleanup.tmp").exists()
    assert not (tmp_path / "completion.json").exists()


def wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Worker did not write {path.name}")
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform != "linux", reason="Main supervisor requires Linux subreaper and /proc")
@pytest.mark.parametrize(
    "ending", ["success-stop", "success-signal", "cancel-stop", "cancel-signal", "fatal", "crash"]
)
def test_worker_services_survive_completion_then_main_reaper_cleans_them(tmp_path: Path, ending: str) -> None:
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "input.json").write_text(payload().model_dump_json())
    result = run_result()
    (tmp_path / "template.json").write_text(
        SandboxResult(
            response=result.episode.response,
            observations=result.episode.observations,
            model_cookies={},
            resource_cookies={},
        ).model_dump_json()
    )
    (tmp_path / "marker.txt").write_text("service survived task completion")
    service = tmp_path / "service.py"
    service.write_text(
        "from http.server import HTTPServer, SimpleHTTPRequestHandler\n"
        "from pathlib import Path\n"
        "server = HTTPServer(('127.0.0.1', 0), SimpleHTTPRequestHandler)\n"
        "Path('port.tmp').write_text(str(server.server_port))\n"
        "Path('port.tmp').replace('port')\n"
        "server.serve_forever()\n"
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import asyncio, os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from responses_api_agents.nooa_agent import sandbox_entrypoint as entrypoint\n"
        "class Client:\n"
        "    async def close(self): pass\n"
        "entrypoint.set_global_aiohttp_client = lambda _: Client()\n"
        "async def execute(payload):\n"
        "    child = subprocess.Popen([sys.executable, 'service.py'], start_new_session=True, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "    Path('service.tmp').write_text(str(child.pid))\n"
        "    Path('service.tmp').replace('service.pid')\n"
        "    result = entrypoint.SandboxResult.model_validate_json(Path('template.json').read_text())\n"
        "    if sys.argv[1] == 'crash': os._exit(7)\n"
        "    if sys.argv[1] == 'fatal':\n"
        "        result.error = entrypoint.RunnerError(kind='fatal', message='fixture failure')\n"
        "    if sys.argv[1].startswith('cancel'):\n"
        "        try: await asyncio.Event().wait()\n"
        "        except asyncio.CancelledError:\n"
        "            result.error = entrypoint.RunnerError(kind='cancelled', message='fixture cancellation')\n"
        "    return result\n"
        "entrypoint.execute = execute\n"
        "asyncio.run(entrypoint._main(Path('input.json'), Path('result.json'), "
        "stop_path=Path('runner.stop'), completion_path=Path('completion.json')))\n"
    )
    # Replace only the model-bearing command with a local scripted worker. The
    # NOOA receipt wrapper, worker lifecycle and main's Linux reaper all run.
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from responses_api_agents.nooa_agent import sandbox_supervisor as supervisor\n"
        "reaper = supervisor.process_supervisor._supervise\n"
        "def supervise(command, **kwargs):\n"
        f"    return reaper([sys.executable, {str(worker)!r}, sys.argv[1]], **kwargs)\n"
        "supervisor.process_supervisor._supervise = supervise\n"
        "raise SystemExit(supervisor.supervise(Path.cwd()))\n"
    )
    process = subprocess.Popen([sys.executable, str(launcher), ending], cwd=tmp_path)
    service_pid = None
    try:
        wait_for_file(tmp_path / "service.pid")
        service_pid = int((tmp_path / "service.pid").read_text())
        assert int((tmp_path / "runner.pid").read_text()) == process.pid
        if ending.startswith("success"):
            wait_for_file(tmp_path / "completion.json")
            assert json.loads((tmp_path / "completion.json").read_text()) == {"task_completed": True}
            assert process.poll() is None
            assert not (tmp_path / "cleanup.json").exists()
            wait_for_file(tmp_path / "port")
            port = int((tmp_path / "port").read_text())
            with urlopen(f"http://127.0.0.1:{port}/marker.txt", timeout=2) as response:
                assert response.read() == b"service survived task completion"
        if ending.endswith("stop"):
            (tmp_path / "runner.stop").touch()
        elif ending.endswith("signal"):
            process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        assert process.returncode == 0
        receipt = json.loads((tmp_path / "cleanup.json").read_text())
        assert receipt["cleanup_confirmed"] is True
        assert receipt["error"] is None
        assert receipt["timed_out"] is False
        if ending == "crash":
            assert receipt["return_code"] == 7
            assert not (tmp_path / "completion.json").exists()
        else:
            saved = SandboxResult.model_validate_json((tmp_path / "result.json").read_text())
            if ending.startswith("cancel"):
                assert saved.error.kind == "cancelled"
            elif ending == "fatal":
                assert saved.error.kind == "fatal"
        with pytest.raises(ProcessLookupError):
            os.kill(service_pid, 0)
    finally:
        (tmp_path / "runner.stop").touch()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)
        if service_pid is not None:
            try:
                os.kill(service_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
