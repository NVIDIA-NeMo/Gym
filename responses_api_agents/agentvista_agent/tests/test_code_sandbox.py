# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real namespace probes plus host-side untrusted image/transport regressions."""

import asyncio
import base64
import io
import json
import os
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from responses_api_agents.agentvista_agent.runtime import code_sandbox, sandbox_bridge


def test_missing_isolation_runtime_never_uses_the_host(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="host execution is forbidden"):
        sandbox_bridge.sandbox_command({"kernel_python": sys.executable}, tmp_path)


def test_a_python_prefix_that_exposes_task_data_is_rejected(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: '["/", "/"]')
    with pytest.raises(ValueError, match="Unsafe Python runtime"):
        sandbox_bridge.sandbox_command({"kernel_python": sys.executable, "source_root": str(tmp_path)}, tmp_path)


@pytest.mark.skipif(not shutil.which("bwrap"), reason="Bubblewrap is required for real namespace probes")
async def test_real_namespaces_hide_answers_credentials_processes_and_network(tmp_path, monkeypatch):
    private = tmp_path / "private"
    (private / "app").mkdir(parents=True)
    (private / "channel").mkdir()
    answers = tmp_path / "answers.json"
    answers.write_text("HOST_REFERENCE_CANARY")
    monkeypatch.setenv("VERIFIER_API_KEY", "HOST_JUDGE_CANARY")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    probe = f"""
import json,os,pathlib,socket
paths = [{str(answers)!r}, '/proc/{os.getpid()}/environ', '/proc/{os.getpid()}/root{str(answers)}']
values=[]
for name in paths:
 try: values.append(pathlib.Path(name).read_text())
 except OSError: values.append('DENIED')
pathlib.Path('/workspace/link').symlink_to({str(answers)!r})
try: symlink=pathlib.Path('/workspace/link').read_text()
except OSError: symlink='DENIED'
s=socket.socket(); s.settimeout(1)
network=s.connect_ex(('127.0.0.1',{listener.getsockname()[1]}))
print(json.dumps({{'values':values,'symlink':symlink,'env':dict(os.environ),'network':network,
 'pid_ns':os.readlink('/proc/self/ns/pid'),'net_ns':os.readlink('/proc/self/ns/net')}}))
"""
    (private / "app/sandbox_runtime.py").write_text(probe)
    command = sandbox_bridge.sandbox_command(
        {"kernel_python": sys.executable, "source_root": str(tmp_path / "source")}, private
    )
    try:
        async with AsyncSandbox({"local": {}}, SandboxSpec(workdir=str(private))) as sandbox:
            await sandbox.start()
            result = await sandbox.exec(command, timeout_s=20)
        if result.return_code and any(
            s in (result.stderr or "")
            for s in ["No permissions to create new namespace", "Creating new namespace failed"]
        ):
            pytest.skip("Host policy disables unprivileged namespace creation")
        assert result.return_code == 0 and result.error_type is None, result.stderr
        data = json.loads(result.stdout)
        assert data["values"] == ["DENIED"] * 3
        assert data["symlink"] == "DENIED"
        assert "HOST_JUDGE_CANARY" not in json.dumps(data)
        assert "VERIFIER_API_KEY" not in data["env"]
        assert data["network"] != 0
        assert data["pid_ns"] != os.readlink("/proc/self/ns/pid")
        assert data["net_ns"] != os.readlink("/proc/self/ns/net")
        assert answers.read_text() == "HOST_REFERENCE_CANARY"
    finally:
        listener.close()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="Bubblewrap is required for real cleanup probe")
async def test_cancel_kills_a_detached_process_in_the_namespace(tmp_path):
    private = tmp_path / "private"
    (private / "app").mkdir(parents=True)
    (private / "channel").mkdir()
    marker = "agentvista-child-" + os.urandom(8).hex()
    # The detached descendant has an independent process group. Killing only the
    # host exec group would leave it alive without PID-namespace teardown.
    program = (
        "import os,time,pathlib; os.setsid(); pathlib.Path('/channel/started').write_text('ready'); time.sleep(60)"
    )
    (private / "app/sandbox_runtime.py").write_text(
        'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",'
        + repr(program)
        + ","
        + repr(marker)
        + "])\ntime.sleep(60)\n"
    )
    command = sandbox_bridge.sandbox_command(
        {"kernel_python": sys.executable, "source_root": str(tmp_path / "source")}, private
    )
    async with AsyncSandbox({"local": {}}, SandboxSpec(workdir=str(private))) as sandbox:
        await sandbox.start()
        task = asyncio.create_task(sandbox.exec(command, timeout_s=30))
        try:
            for _ in range(100):
                if (private / "channel/started").exists() or task.done():
                    break
                await asyncio.sleep(0.1)
            if task.done():
                result = await task
                if any(
                    s in (result.stderr or "")
                    for s in ["No permissions to create new namespace", "Creating new namespace failed"]
                ):
                    pytest.skip("Host policy disables unprivileged namespace creation")
                pytest.fail(str(result))
            assert (private / "channel/started").exists()
            import subprocess

            assert marker in subprocess.check_output(["ps", "-eo", "args="], text=True)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
    for _ in range(20):
        processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
        if marker not in processes:
            break
        await asyncio.sleep(0.1)
    assert marker not in processes


def test_returned_markdown_cannot_make_the_host_open_an_arbitrary_file(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    original = tmp_path / "private.png"
    Image.new("RGB", (11, 7), "red").save(original)
    image = Image.new("RGB", (3, 2), "blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    proxy = code_sandbox.CodeSandbox.__new__(code_sandbox.CodeSandbox)
    proxy.output = tmp_path / "returned"
    proxy.output.mkdir()
    proxy.timeout = 2
    proxy.failed = False
    import threading

    proxy.lock = threading.Lock()
    proxy.process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(code_sandbox, "write_request", lambda *a: None)
    monkeypatch.setattr(
        code_sandbox,
        "read_reply",
        lambda *a: {
            "result": f"![evil]({original}) ![tool_image_1/../../escape](/workspace/out.png)",
            "images": {"/workspace/out.png": base64.b64encode(buf.getvalue()).decode()},
        },
    )
    result = proxy.call({"code": "ignored"})
    assert "/../" not in result
    assert str(original) not in result
    assert "[unavailable sandbox image]" in result
    files = list(proxy.output.glob("*.png"))
    assert len(files) == 1 and Image.open(files[0]).size == (3, 2)
    assert Image.open(original).size == (11, 7)
    proxy.process = None


def test_a_broken_transport_is_recorded_and_never_reused(tmp_path, monkeypatch):
    pytest.importorskip("PIL.Image")
    import threading

    proxy = code_sandbox.CodeSandbox.__new__(code_sandbox.CodeSandbox)
    proxy.output = tmp_path / "returned"
    proxy.output.mkdir()
    proxy.timeout = 1
    proxy.failed = False
    proxy.lock = threading.Lock()
    proxy.process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(code_sandbox, "write_request", lambda *a: None)
    monkeypatch.setattr(
        code_sandbox, "read_reply", lambda *a: (_ for _ in ()).throw(code_sandbox.SandboxFailure("closed"))
    )
    monkeypatch.setattr(proxy, "close", lambda: None)
    with pytest.raises(code_sandbox.SandboxFailure):
        proxy.call({"code": "x"})
    assert proxy.failed and (tmp_path / "code_sandbox_failed.json").is_file()
    with pytest.raises(code_sandbox.SandboxFailure, match="unavailable"):
        proxy.call({"code": "x"})
    proxy.process = None


def test_failed_startup_marks_the_episode_before_returning_a_tool_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTVISTA_ROOT", str(tmp_path / "source"))
    monkeypatch.setattr(
        code_sandbox.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(OSError("missing helper"))
    )
    with pytest.raises(OSError):
        code_sandbox.CodeSandbox({}, tmp_path)
    assert (tmp_path / "code_sandbox_failed.json").is_file()


@pytest.mark.parametrize(
    "blocked", ["/usr/lib/private-benchmark/answers.json", str(Path(sys.prefix) / "answers.json")]
)
def test_runtime_mounts_cannot_contain_a_protected_dataset(tmp_path, monkeypatch, blocked):
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: json.dumps([sys.prefix, sys.base_prefix]))
    with pytest.raises(ValueError, match="Unsafe Python runtime"):
        sandbox_bridge.sandbox_command(
            {"kernel_python": sys.executable, "source_root": str(tmp_path / "source"), "blocked_paths": [blocked]},
            tmp_path,
        )
