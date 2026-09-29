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
import logging
from pathlib import Path

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, SandboxStatus, create_provider
from nemo_gym.sandbox.providers.local import LocalProvider


pytestmark = pytest.mark.sandbox


async def test_exec_runs_in_the_workspace_with_layered_env(tmp_path: Path) -> None:
    provider = LocalProvider(workspace_root=str(tmp_path))
    handle = await provider.create(SandboxSpec(env={"FROM_SPEC": "spec", "OVERRIDDEN": "spec"}))

    result = await provider.exec(
        handle,
        "pwd && echo $FROM_SPEC $OVERRIDDEN $FROM_CALL",
        env={
            "OVERRIDDEN": "call",
            "FROM_CALL": "call",
        },
    )

    workspace, cwd, values = handle.raw["workspace"], *result.stdout.splitlines()
    assert result.return_code == 0 and result.error_type is None
    assert Path(cwd).resolve() == workspace.resolve()
    assert values.split() == ["spec", "call", "call"]
    assert workspace.parent == tmp_path


async def test_command_failure_is_reported_not_raised(tmp_path: Path) -> None:
    provider = LocalProvider(workspace_root=str(tmp_path))
    handle = await provider.create(SandboxSpec())

    result = await provider.exec(handle, "echo out && echo err >&2 && exit 7")

    assert (result.return_code, result.error_type) == (7, None)
    assert result.stdout.strip() == "out" and result.stderr.strip() == "err"


async def test_timeout_kills_the_whole_process_tree(tmp_path: Path) -> None:
    provider = LocalProvider(workspace_root=str(tmp_path))
    handle = await provider.create(SandboxSpec())
    marker = tmp_path / "survived"

    result = await provider.exec(handle, f"(sleep 1 && touch {marker}) & sleep 30", timeout_s=0.2)

    assert result.return_code == 125 and result.error_type == "timeout"
    await asyncio.sleep(2)
    assert not marker.exists(), "backgrounded child outlived the timed-out command"


async def test_timeout_returns_even_when_a_child_escapes_the_process_group(tmp_path: Path) -> None:
    # setsid/nohup/daemonized children survive killpg and hold the pipes open, so the
    # provider must never wait on output again after a timeout.
    provider = LocalProvider(workspace_root=str(tmp_path))
    handle = await provider.create(SandboxSpec())

    escaped = "python3 -c 'import os,time;os.setsid();time.sleep(60)' & sleep 30"
    result = await asyncio.wait_for(provider.exec(handle, escaped, timeout_s=0.5), timeout=15)

    assert result.return_code == 125 and result.error_type == "timeout"


async def test_create_warns_about_container_fields_it_cannot_honor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Pointing a container-shaped config at this provider must not silently drop its limits.
    provider = LocalProvider(workspace_root=str(tmp_path))

    with caplog.at_level(logging.WARNING):
        await provider.create(SandboxSpec(image="repo/img", ttl_s=18000, ready_timeout_s=1200, ports=(8080,)))

    assert "image, ttl_s, ready_timeout_s, ports" in caplog.text
    assert "resource limits" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await provider.create(SandboxSpec())
    assert caplog.text == ""


async def test_exec_rejects_running_as_another_user(tmp_path: Path) -> None:
    provider = LocalProvider(workspace_root=str(tmp_path))
    handle = await provider.create(SandboxSpec())

    with pytest.raises(ValueError, match="cannot run commands as another user"):
        await provider.exec(handle, "id", user="root")


async def test_file_transfer_resolves_relative_paths_against_the_workspace(tmp_path: Path) -> None:
    provider = LocalProvider(workspace_root=str(tmp_path))
    handle = await provider.create(SandboxSpec())
    workspace = handle.raw["workspace"]
    source = tmp_path / "source.txt"
    source.write_text("payload\n")

    await provider.upload_file(handle, source, "nested/in.txt")
    await provider.download_file(handle, "nested/in.txt", tmp_path / "out" / "relative.txt")
    await provider.download_file(handle, str(workspace / "nested/in.txt"), tmp_path / "out" / "absolute.txt")

    assert (workspace / "nested" / "in.txt").read_text() == "payload\n"
    assert (tmp_path / "out" / "relative.txt").read_text() == "payload\n"
    assert (tmp_path / "out" / "absolute.txt").read_text() == "payload\n"


async def test_close_removes_an_allocated_workspace_but_keeps_a_supplied_one(tmp_path: Path) -> None:
    provider = LocalProvider(workspace_root=str(tmp_path))
    supplied = tmp_path / "caller_owned"

    allocated_handle = await provider.create(SandboxSpec(image="ignored:latest"))
    supplied_handle = await provider.create(SandboxSpec(workdir=str(supplied)))
    allocated = allocated_handle.raw["workspace"]

    assert await provider.status(allocated_handle) is SandboxStatus.RUNNING
    await provider.close(allocated_handle)
    await provider.close(supplied_handle)
    await provider.aclose()

    assert not allocated.exists()
    assert await provider.status(allocated_handle) is SandboxStatus.STOPPED
    assert supplied.is_dir(), "a caller-supplied workdir must survive close"


async def test_registered_provider_drives_the_public_sandbox_api(tmp_path: Path) -> None:
    sandbox = AsyncSandbox(create_provider({"local": {"workspace_root": str(tmp_path)}}))
    await sandbox.start(SandboxSpec(image="swebench/ignored", files={"seed.txt": "seeded\n"}))
    workspace = sandbox._handle.raw["workspace"]

    result = await sandbox.exec("cat seed.txt && echo done > out.txt")
    await sandbox.download("out.txt", tmp_path / "out.txt")

    assert result.stdout.strip() == "seeded"
    assert (tmp_path / "out.txt").read_text().strip() == "done"
    assert await sandbox.status() is SandboxStatus.RUNNING
    await sandbox.stop()
    assert not workspace.exists()
