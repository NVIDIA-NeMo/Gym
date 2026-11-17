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

"""Runs the bwrap provider against a real ``bwrap`` binary (Linux with user namespaces only)."""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.bwrap import BwrapProvider


BWRAP = shutil.which("bwrap")


def _user_namespaces_work() -> bool:
    if BWRAP is None or not sys.platform.startswith("linux"):
        return False
    probe = subprocess.run(
        [BWRAP, "--unshare-user", "--unshare-pid", "--ro-bind", "/", "/", "--proc", "/proc", "true"],
        capture_output=True,
        timeout=30,
    )
    return probe.returncode == 0


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(not _user_namespaces_work(), reason="needs Linux, bwrap on PATH and user namespaces"),
]


@pytest.fixture
def provider(tmp_path: Path) -> BwrapProvider:
    return BwrapProvider(bwrap_path=BWRAP, base_dir=str(tmp_path / "base"), probe_on_create=True)


async def test_real_bwrap_isolation(provider: BwrapProvider, tmp_path: Path) -> None:
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir="/workspace", env={"TASK": "t1"}))
    inst = sandbox._handle.raw
    try:
        result = await sandbox.exec("pwd; echo $HOME $TASK; echo hi > out.txt; echo $$")
        assert result.return_code == 0, result.stderr
        lines = result.stdout.split()
        assert lines[:3] == ["/workspace", "/home/sandbox", "t1"]
        assert lines[3] == "2"  # a fresh pid namespace per exec
        assert (inst.workspace / "out.txt").read_text() == "hi\n"
        # The workspace is also reachable at its host path.
        assert (await sandbox.exec(f"cat {inst.workspace}/out.txt")).stdout == "hi\n"
        # /tmp persists across execs of one sandbox but is private to it.
        assert (await sandbox.exec("echo kept > /tmp/state")).return_code == 0
        assert (await sandbox.exec("cat /tmp/state")).stdout == "kept\n"
        # The root and the system binds are read-only; the environment is whitelisted.
        assert (await sandbox.exec("touch /newfile")).return_code != 0
        assert (await sandbox.exec("touch /usr/newfile")).return_code != 0
        env = (await sandbox.exec("env")).stdout
        assert "NVIDIA_VISIBLE_DEVICES=void" in env and "PYTEST_CURRENT_TEST" not in env
        # No network but loopback.
        interfaces = (await sandbox.exec("tail -n +3 /proc/net/dev | cut -d: -f1")).stdout.split()
        assert interfaces == ["lo"]
        # Background processes do not outlive their exec: the pid namespace dies with it.
        await sandbox.exec("(sleep 1; touch /tmp/late) > /dev/null 2>&1 &")
        await asyncio.sleep(2)
        assert not (inst.tmp / "late").exists()
        # Timeouts kill the namespace.
        timed_out = await sandbox.exec("sleep 30", timeout_s=0.5)
        assert timed_out.error_type == "timeout"

        other = await AsyncSandbox(provider, owns_provider=False).start(SandboxSpec(workdir="/workspace"))
        try:
            assert (await other.exec("cat /tmp/state")).return_code != 0
            assert (await other.exec("ls /workspace")).stdout == ""
        finally:
            await other.stop()

        borrowed = await AsyncSandbox.connect(await sandbox.serialize(), provider=provider, owns_provider=False)
        assert (await borrowed.exec("cat out.txt")).stdout == "hi\n"
        await borrowed.disconnect()
    finally:
        await sandbox.stop()
    assert not inst.root.exists()
