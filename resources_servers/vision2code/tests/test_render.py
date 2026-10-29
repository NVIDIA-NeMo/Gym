# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from nemo_gym.sandbox import SandboxExecResult


module = importlib.import_module("resources_servers.vision2code.render")


@pytest.fixture
def fake_sandbox(monkeypatch):
    instances = []

    class FakeSandbox:
        mode = "ok"
        gate = None
        stop_gate = None

        def __init__(self, provider, spec):
            self.spec = spec
            self.environment = dict(os.environ)
            self.provider = provider
            self.uploads = {}
            self.commands = []
            self.stopped = False
            self.downloaded = False
            self.entered = asyncio.Event()
            self.stop_entered = asyncio.Event()
            instances.append(self)

        async def start(self):
            if self.mode == "create_failure":
                raise RuntimeError("unpack failed")
            if self.mode == "cancel_start":
                self.entered.set()
                await self.gate.wait()

        async def upload(self, local_path, remote_path):
            self.uploads[remote_path] = Path(local_path).read_text()

        async def exec(self, command, **kwargs):
            self.commands.append(command)
            if "mountinfo" in command:
                if self.mode == "isolation_failure":
                    return SandboxExecResult("", "unshare denied", 1)
                if self.mode == "mount_failure":
                    return SandboxExecResult('{"unexpected":["/run/secrets"]}', "", 1)
                return SandboxExecResult('{"mounts":["/","/workspace"],"unexpected":[]}', "", 0)
            if "launcher.py" in command:
                self.entered.set()
                if self.mode == "cancel_exec":
                    await self.gate.wait()
                if self.mode == "timeout":
                    return SandboxExecResult("", "deadline", 125, "timeout")
                if self.mode == "code_error":
                    return SandboxExecResult("", "SyntaxError", 1)
                if self.mode == "runtime_failure":
                    return SandboxExecResult("", "lost init", 125, "sandbox")
                return SandboxExecResult("rendered", "", 0)
            info = {
                "exists": self.mode != "missing_output",
                "regular": self.mode != "symlink",
                "bytes": module.MAX_BYTES + 1 if self.mode == "oversized" else 100,
            }
            return SandboxExecResult(json.dumps(info), "", 0)

        async def download(self, remote_path, local_path):
            assert not self.stopped
            self.downloaded = True
            if self.mode == "download_failure":
                raise OSError("transfer error")
            if self.mode == "invalid":
                Path(local_path).write_text("bad PNG")
            else:
                Image.new("RGB", (40, 50), "white").save(local_path)

        async def stop(self):
            self.stop_entered.set()
            if self.stop_gate is not None:
                await self.stop_gate.wait()
            if self.mode == "stop_failure":
                raise RuntimeError("failed to stop")
            self.stopped = True

    monkeypatch.setattr(module, "AsyncSandbox", FakeSandbox)
    return FakeSandbox, instances


async def attempt(tmp_path, **kwargs):
    return await module.render(
        code="print('hello')",
        width=40,
        height=50,
        image="/local.sqsh",
        provider={"enroot": {}},
        artifact_dir=tmp_path,
        timeout=60,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,status",
    [
        ("ok", "ok"),
        ("code_error", "code_error"),
        ("timeout", "timeout"),
        ("missing_output", "missing_output"),
        ("symlink", "invalid_output"),
        ("oversized", "invalid_output"),
        ("invalid", "invalid_output"),
        ("create_failure", "infrastructure_failure"),
        ("isolation_failure", "infrastructure_failure"),
        ("mount_failure", "infrastructure_failure"),
        ("runtime_failure", "infrastructure_failure"),
        ("download_failure", "infrastructure_failure"),
        ("stop_failure", "infrastructure_failure"),
    ],
)
async def test_lifecycle(fake_sandbox, tmp_path, mode, status):
    cls, instances = fake_sandbox
    cls.mode = mode
    previous = dict(os.environ)
    result = await attempt(tmp_path)
    assert dict(os.environ) == previous
    assert instances[0].environment["ENROOT_MOUNT_HOME"] == "no"
    assert instances[0].provider["enroot"]["create"]["rw"] is False
    assert instances[0].provider["enroot"]["create"]["mount_point"] == "/workspace"
    assert result.status == status
    assert instances[0].stopped == (mode != "stop_failure")
    assert json.loads((tmp_path / "execution.json").read_text())["status"] == status
    if mode != "create_failure":
        assert set(instances[0].uploads) == {"/workspace/code.py", "/workspace/launcher.py"}
        assert instances[0].commands[0].startswith("/usr/bin/env -i ")
        assert "unshare --user --map-current-user --net" in instances[0].commands[0]
    if mode in {"symlink", "oversized", "missing_output", "isolation_failure", "mount_failure"}:
        assert not instances[0].downloaded
    if mode in {"isolation_failure", "mount_failure"}:
        assert not any("launcher.py" in command for command in instances[0].commands)
    if mode == "ok":
        assert (result.width, result.height, result.dimension_mismatch) == (40, 50, False)
        assert (tmp_path / "output.png").exists()
    if mode == "invalid":
        assert not (tmp_path / "output.png").exists()


@pytest.mark.asyncio
async def test_dimensions_are_diagnostics(fake_sandbox, tmp_path):
    result = await module.render(
        code="", width=80, height=90, image="/local.sqsh", provider={"enroot": {}}, artifact_dir=tmp_path, timeout=60
    )
    assert result.status == "ok"
    assert result.dimension_mismatch is True
    assert Image.open(tmp_path / "output.png").size == (40, 50)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["cancel_start", "cancel_exec"])
async def test_cancellation_waits_for_cleanup(fake_sandbox, tmp_path, stage):
    cls, instances = fake_sandbox
    cls.mode = stage
    cls.gate = asyncio.Event()
    cls.stop_gate = asyncio.Event()
    task = asyncio.create_task(attempt(tmp_path))
    while not instances:
        await asyncio.sleep(0)
    await asyncio.wait_for(instances[0].entered.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    cls.gate.set()
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()  # Repeated cancellation cannot orphan the stop task.
    cls.stop_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert instances[0].stopped
    assert json.loads((tmp_path / "execution.json").read_text())["status"] == "cancelled"


@pytest.mark.asyncio
async def test_concurrent_attempts_do_not_share_files(fake_sandbox, tmp_path):
    results = await asyncio.gather(*(attempt(tmp_path / str(i)) for i in range(4)))
    assert all(r.status == "ok" for r in results)
    assert len(fake_sandbox[1]) == 4
    assert all(s.stopped for s in fake_sandbox[1])
    assert len(list(tmp_path.glob("*/output.png"))) == 4


@pytest.mark.parametrize("text", ["print(1)", "```python\nprint(1)\n```", "```\nprint(1)\n```"])
def test_code_normalization(text):
    assert module.normalize_code(text) == "print(1)\n"


@pytest.mark.parametrize("kind", ["symlink", "jpeg", "bytes", "pixels", "corrupt"])
def test_output_limits(tmp_path, kind):
    path = tmp_path / "output.png"
    if kind == "symlink":
        target = tmp_path / "target.png"
        Image.new("RGB", (40, 40)).save(target)
        path.symlink_to(target)
    elif kind == "jpeg":
        Image.new("RGB", (40, 40)).save(path, format="JPEG")
    elif kind == "bytes":
        with path.open("wb") as f:
            f.truncate(module.MAX_BYTES + 1)
    elif kind == "pixels":
        Image.new("1", (5001, 5000)).save(path)
    else:
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises((ValueError, OSError)):
        module.validate_png(path)


@pytest.mark.asyncio
async def test_cancellation_during_stop_is_recorded(fake_sandbox, tmp_path):
    cls, instances = fake_sandbox
    cls.stop_gate = asyncio.Event()
    task = asyncio.create_task(attempt(tmp_path))
    while not instances:
        await asyncio.sleep(0)
    await instances[0].stop_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    cls.stop_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert instances[0].stopped
    assert json.loads((tmp_path / "execution.json").read_text())["status"] == "cancelled"
