# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import base64
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import SandboxExecResult, SandboxSpec
from responses_api_agents.collab_intel_agent import boxed_exec
from responses_api_agents.collab_intel_agent.boxed_exec import WRAPPER, BoxedExec, SandboxExecError


def _load_cli():
    path = Path(boxed_exec.__file__).with_name("boxed_cli.py")
    spec = importlib.util.spec_from_file_location("boxed_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _tar(files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    return _b64(buf.getvalue())


class FakeBox:
    """A sandbox that records what it is asked and answers the egress probe and the wrapper from `script`."""

    instances: list["FakeBox"] = []
    egress = False
    exec_stdout: str | None = None
    start_error: Exception | None = None
    wrapper_result: dict = {"rc": 0, "stdout": "ok", "stderr": "", "changed": {}, "deleted": []}

    def __init__(self, provider, owns_provider=True):
        self.uploads: list[tuple[str, int]] = []
        self.commands: list[str] = []
        self.stopped = False
        FakeBox.instances.append(self)

    async def start(self, spec):
        if FakeBox.start_error is not None:
            raise FakeBox.start_error
        self.spec = spec
        return self

    async def upload(self, local, remote):
        self.uploads.append((remote, Path(local).stat().st_size))

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
        if "socket" in command:
            out = FakeBox.exec_stdout or ("egress\n" if FakeBox.egress else "blocked\n")
            return SandboxExecResult(stdout=out, stderr="", return_code=0)
        if "wrap.py" in command:
            return SandboxExecResult(stdout="noise\n" + json.dumps(FakeBox.wrapper_result), stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    FakeBox.instances, FakeBox.egress, FakeBox.start_error, FakeBox.exec_stdout = [], False, None, None
    FakeBox.wrapper_result = {"rc": 0, "stdout": "ok", "stderr": "", "changed": {}, "deleted": []}
    monkeypatch.setattr(boxed_exec, "AsyncSandbox", FakeBox)


def _boxes() -> BoxedExec:
    provider = SimpleNamespace(aclose=AsyncMock())
    return BoxedExec(provider, SandboxSpec(image="python:3.12-slim", workdir="/tmp"))


def _request(runtime: bool = False, **extra) -> dict:
    request = {
        "argv": ["@python", "-m", "pytest"],
        "env": {"HOME": "@dir"},
        "timeout": 5,
        "files": _tar({"a.py": "x"}),
    }
    return request | ({"runtime": _tar({"pytest/__init__.py": ""})} if runtime else {}) | extra


def test_the_first_command_of_a_phase_starts_a_sandbox_and_the_next_one_reuses_it():
    boxes = _boxes()
    result = asyncio.run(boxes.run("p", _request(runtime=True)))
    assert result["rc"] == 0 and result["stdout"] == "ok"
    asyncio.run(boxes.run("p", _request()))
    (box,) = FakeBox.instances
    assert box.spec.image == "python:3.12-slim"
    # the runtime and the wrapper go in once, the workspace with every command
    assert [r for r, _ in box.uploads].count("/tmp/_rt.tar") == 1
    assert [r for r, _ in box.uploads].count("/tmp/wrap.py") == 1
    assert sum(r.startswith("/tmp/runs/") for r, _ in box.uploads) == 2
    assert sum("wrap.py" in c for c in box.commands) == 2


def test_a_sandbox_with_network_egress_is_refused_and_stopped():
    FakeBox.egress = True
    with pytest.raises(SandboxExecError, match="open connections"):
        asyncio.run(_boxes().run("p", _request(runtime=True)))
    assert FakeBox.instances[0].stopped
    assert not any("wrap.py" in c for c in FakeBox.instances[0].commands)


def test_egress_is_allowed_only_when_the_operator_says_so(caplog):
    FakeBox.egress = True
    boxes = BoxedExec(
        SimpleNamespace(aclose=AsyncMock()),
        SandboxSpec(image="python:3.12-slim", workdir="/tmp"),
        require_no_egress=False,
    )
    assert asyncio.run(boxes.run("p", _request(runtime=True)))["rc"] == 0
    assert "network egress" in caplog.text


def test_a_sandbox_without_python_is_refused():
    FakeBox.exec_stdout = "sh: python: not found"
    with pytest.raises(SandboxExecError, match="no working python"):
        asyncio.run(_boxes().run("p", _request(runtime=True)))


def test_a_provider_failure_is_a_sandbox_error_that_names_only_the_class():
    FakeBox.start_error = ConnectionError("https://secret.example/key=abc")
    with pytest.raises(SandboxExecError) as error:
        asyncio.run(_boxes().run("p", _request(runtime=True)))
    assert "ConnectionError" in str(error.value) and "secret" not in str(error.value)


def test_a_phase_without_a_runtime_does_not_start_a_sandbox():
    with pytest.raises(SandboxExecError, match="runtime"):
        asyncio.run(_boxes().run("p", _request()))
    assert not FakeBox.instances


def test_a_malformed_request_is_a_sandbox_error():
    no_argv = _request(runtime=True)
    del no_argv["argv"]
    with pytest.raises(SandboxExecError, match="KeyError"):
        asyncio.run(_boxes().run("p", no_argv))
    with pytest.raises(SandboxExecError, match="Error"):
        asyncio.run(_boxes().run("q", _request(runtime=True, files="not base64 !!")))


def test_files_that_grow_past_the_limit_fail_the_run():
    FakeBox.wrapper_result = {"rc": 0, "stdout": "", "stderr": "", "changed": {}, "deleted": [], "too_large": True}
    with pytest.raises(SandboxExecError, match="changed more than"):
        asyncio.run(_boxes().run("p", _request(runtime=True)))


def test_release_stops_the_sandbox_and_close_closes_the_provider():
    boxes = _boxes()
    asyncio.run(boxes.run("p", _request(runtime=True)))
    asyncio.run(boxes.release("p"))
    assert FakeBox.instances[0].stopped
    asyncio.run(boxes.run("q", _request(runtime=True)))
    asyncio.run(boxes.close())
    assert FakeBox.instances[1].stopped
    boxes._provider.aclose.assert_awaited_once()


def _wrapper(tmp_path: Path, argv: list[str], timeout: int = 20) -> dict:
    """The in-sandbox wrapper on trusted test commands: the protocol, not any model's code."""
    tmp_path.mkdir(exist_ok=True)
    tar = tmp_path / "ws.tar"
    tar.write_bytes(base64.b64decode(_tar({"keep.txt": "k", "gone.txt": "g", "edit.txt": "old"})))
    args = {
        "dir": str(tmp_path / "run"),
        "tar": str(tar),
        "argv": argv,
        "env": {"HOME": "@dir"},
        "timeout": timeout,
        "pythonpath": "",
        "max_bytes": 1000,
    }
    done = subprocess.run(
        [sys.executable, "-c", WRAPPER, _b64(json.dumps(args).encode())], capture_output=True, text=True, check=True
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_wrapper_reports_output_and_the_files_a_command_changed(tmp_path):
    code = "import os; open('new.txt','w').write('n'); open('edit.txt','w').write('new'); os.remove('gone.txt'); print('hi')"
    result = _wrapper(tmp_path, ["@python", "-c", code])
    assert (result["rc"], result["stdout"].strip()) == (0, "hi")
    assert {k: base64.b64decode(v) for k, v in result["changed"].items()} == {"new.txt": b"n", "edit.txt": b"new"}
    assert result["deleted"] == ["gone.txt"]
    assert not (tmp_path / "run").exists() and not (tmp_path / "ws.tar").exists()


def test_wrapper_maps_the_workspace_placeholder_and_reports_a_timeout(tmp_path):
    result = _wrapper(tmp_path, ["@python", "-c", "import sys; print(sys.argv[1])", "@dir/x"])
    assert result["stdout"].strip() == str(tmp_path / "run" / "x")
    assert _wrapper(tmp_path / "t", ["@python", "-c", "import time; time.sleep(30)"], timeout=1)["rc"] is None


def test_wrapper_stops_collecting_past_the_byte_limit(tmp_path):
    result = _wrapper(tmp_path, ["@python", "-c", "open('big.txt','w').write('x' * 5000)"])
    assert result["too_large"] and result["changed"] == {}


# the harness side: boxed_cli.py


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "pkg" / "__pycache__").mkdir(parents=True)
    (ws / "pkg" / "__pycache__" / "m.pyc").write_bytes(b"cache")
    (ws / "pkg" / "m.py").write_text("x = 1\n")
    (ws / "test_m.py").write_text("def test_x(): pass\n")
    return ws


def _cli_with_reply(monkeypatch, reply: dict):
    cli = _load_cli()
    sent = []

    def post(payload, timeout_s):
        sent.append(payload)
        return reply

    monkeypatch.setattr(cli, "_post", post)
    monkeypatch.setattr(cli, "_runtime_tar", lambda: b"runtime")
    return cli, sent


def test_a_python_command_is_sent_to_the_sandbox_with_the_workspace_and_comes_back_as_a_completed_process(
    tmp_path, monkeypatch
):
    ws = _workspace(tmp_path)
    reply = {"rc": 1, "stdout": "out", "stderr": "err", "changed": {"new.txt": _b64(b"n")}, "deleted": ["test_m.py"]}
    cli, sent = _cli_with_reply(monkeypatch, reply)
    env = {"PATH": "/host/bin", "HOME": str(ws), "PYTHONDONTWRITEBYTECODE": "1"}
    done = cli.run(
        [sys.executable, "-m", "pytest", "-q", "--rootdir", str(ws)],
        cwd=ws,
        env=env,
        capture_output=True,
        text=True,
        timeout=9,
    )
    (payload,) = sent
    assert payload["argv"] == ["@python", "-m", "pytest", "-q", "--rootdir", "@dir"]
    assert payload["env"] == {"HOME": "@dir", "PYTHONDONTWRITEBYTECODE": "1"}  # the host PATH stays here
    assert payload["timeout"] == 9
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload["files"]))) as tar:
        assert sorted(tar.getnames()) == ["pkg/m.py", "test_m.py"]  # no caches
    assert base64.b64decode(payload["runtime"]) == b"runtime"
    assert (done.returncode, done.stdout, done.stderr) == (1, "out", "err")
    assert (ws / "new.txt").read_text() == "n" and not (ws / "test_m.py").exists()


def test_only_the_first_command_of_the_process_carries_the_runtime(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    cli, sent = _cli_with_reply(monkeypatch, {"rc": 0, "stdout": "", "stderr": ""})
    for _ in range(2):
        cli.run([sys.executable, "-c", "pass"], cwd=ws)
    assert "runtime" in sent[0] and "runtime" not in sent[1]


def test_a_timeout_and_a_failing_check_behave_as_subprocess_run_does(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    cli, _ = _cli_with_reply(monkeypatch, {"rc": None, "stdout": "", "stderr": ""})
    with pytest.raises(subprocess.TimeoutExpired):
        cli.run([sys.executable, "-c", "pass"], cwd=ws, timeout=3)
    cli, _ = _cli_with_reply(monkeypatch, {"rc": 2, "stdout": "", "stderr": ""})
    with pytest.raises(subprocess.CalledProcessError):
        cli.run([sys.executable, "-c", "pass"], cwd=ws, check=True)


def test_files_outside_the_workspace_are_not_written_back(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    reply = {"rc": 0, "stdout": "", "stderr": "", "changed": {"../escape.txt": _b64(b"x")}, "deleted": ["../ws2"]}
    cli, _ = _cli_with_reply(monkeypatch, reply)
    cli.run([sys.executable, "-c", "pass"], cwd=ws)
    assert not (tmp_path / "escape.txt").exists()


def test_other_commands_run_as_before(tmp_path, monkeypatch):
    cli, sent = _cli_with_reply(monkeypatch, {})
    done = cli.run(["echo", "hello"], capture_output=True, text=True)
    assert done.stdout.strip() == "hello" and not sent


def _exits(monkeypatch, cli):
    def stop(code):
        raise SystemExit(code)

    monkeypatch.setattr(cli.os, "_exit", stop)


def test_a_sandbox_failure_ends_the_process_instead_of_returning_a_result(tmp_path, monkeypatch, capsys):
    cli = _load_cli()
    _exits(monkeypatch, cli)
    monkeypatch.setenv("CI_EXEC_URL", "http://127.0.0.1:9/ci_exec/k/run")  # nothing listens
    monkeypatch.setattr(cli, "_runtime_tar", lambda: b"")
    with pytest.raises(SystemExit) as stop:
        cli.run([sys.executable, "-c", "pass"], cwd=_workspace(tmp_path), timeout=1)
    assert stop.value.code == cli.EXIT_CODE and "CI_EXEC_ERROR" in capsys.readouterr().err


def test_a_python_command_without_a_cwd_cannot_be_sent(monkeypatch):
    cli = _load_cli()
    _exits(monkeypatch, cli)
    with pytest.raises(SystemExit):
        cli.run([sys.executable, "-c", "pass"])


def test_starting_this_interpreter_any_other_way_is_refused(monkeypatch):
    cli = _load_cli()
    _exits(monkeypatch, cli)
    with pytest.raises(SystemExit):
        cli.GuardedPopen([sys.executable, "-c", "pass"])
    done = cli.GuardedPopen(["true"])  # other commands still start
    done.wait()
    assert done.returncode == 0


def test_the_runtime_is_pytest_and_its_dependencies_without_caches_or_metadata():
    data = _load_cli()._runtime_tar()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        names = tar.getnames()
    assert {"pytest/__init__.py", "_pytest/compat.py", "pluggy/__init__.py", "iniconfig/__init__.py"} <= set(names)
    assert not [n for n in names if "__pycache__" in n or ".dist-info" in n or n.startswith("..")]
