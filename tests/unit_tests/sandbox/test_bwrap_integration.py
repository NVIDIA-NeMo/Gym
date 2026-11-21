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
import os
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


_REAL_BWRAP = _user_namespaces_work()
# CI sets this so a runner without bwrap or user namespaces fails these tests instead of silently skipping them.
_REQUIRED = os.environ.get("NEMO_GYM_REQUIRE_BWRAP") == "1"

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(not _REAL_BWRAP and not _REQUIRED, reason="needs Linux, bwrap on PATH and user namespaces"),
]


@pytest.fixture(autouse=True)
def _require_real_bwrap() -> None:
    if not _REAL_BWRAP:
        pytest.fail("NEMO_GYM_REQUIRE_BWRAP=1, but bwrap is not on PATH or user namespaces do not work here")


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
        # The workspace is also reachable at its host path, unless that path lies in sandbox-writable space
        # (tmp_path is usually under /tmp).
        at_host_path = await sandbox.exec(f"cat {inst.workspace}/out.txt")
        if provider._binds_workspace_host_path(inst):
            assert at_host_path.stdout == "hi\n"
        else:
            assert at_host_path.return_code != 0
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
        # Some kernels add their fallback tunnel devices (tunl0, sit0, ip_vti0, ...) to every new namespace; they
        # are down and route nowhere.
        fallback_tunnels = {"tunl0", "gre0", "gretap0", "erspan0", "ip_vti0", "ip6_vti0", "sit0", "ip6tnl0", "ip6gre0"}
        assert [name for name in interfaces if name not in fallback_tunnels] == ["lo"]
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


async def test_kill_commands_cannot_reach_processes_outside_the_exec(provider: BwrapProvider) -> None:
    # The property the provider exists for: an agent's pkill or kill -9 -1 stays inside its exec's pid namespace.
    # The pattern is built in the shell, so the exec's own `sh -c` argv does not match it.
    host = subprocess.Popen(["sleep", "61.7777"])
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir="/workspace"))
    try:
        result = await sandbox.exec(
            'm="61.$((7776 + 1))"; sleep "$m" & inner=$!; '
            'until pgrep -f "sleep $m" > /dev/null; do sleep 0.05; done; '
            'pkill -f "sleep $m"; wait "$inner"; echo "inner $?"; '
            "kill -9 -1; echo survived",
            timeout_s=30,
        )
        # pkill found the exec's own sleep (TERM: 128 + 15) and kill -9 -1 spared the calling shell ...
        assert result.stdout.splitlines() == ["inner 143", "survived"], (result.stdout, result.stderr)
        # ... but neither reached the host's sleep with the same command line, nor this test process.
        assert host.poll() is None
        assert (await sandbox.exec("true")).return_code == 0
    finally:
        host.kill()
        host.wait()
        await sandbox.stop()


async def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.02)


async def test_process_proof_follows_real_execs_from_creator_and_borrower(provider: BwrapProvider) -> None:
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir="/workspace"))
    borrowed = await AsyncSandbox.connect(await sandbox.serialize(), provider=provider, owns_provider=False)
    try:
        assert await sandbox.exec_processes_terminated() is True
        running = asyncio.create_task(sandbox.exec("sleep 2"))
        await _wait_for(lambda: bool(provider._processes.get(sandbox._handle.sandbox_id)))
        assert await sandbox.exec_processes_terminated() is False
        await running
        # An exec that left a background process: its namespace died with it, so the proof holds.
        assert (await borrowed.exec("sleep 60 > /dev/null 2>&1 &")).return_code == 0
        assert await sandbox.exec_processes_terminated() is True
        assert await borrowed.exec_processes_terminated() is True
    finally:
        await borrowed.disconnect()
        await sandbox.stop()


async def test_info_fd_child_pid_is_the_namespaces_pid_1(provider: BwrapProvider) -> None:
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir="/workspace"))
    inst = sandbox._handle.raw
    try:
        running = asyncio.create_task(sandbox.exec("sleep 2"))
        await _wait_for(lambda: any(p.name.endswith(".init") for p in inst.run_dir.iterdir()))
        record = next(p for p in inst.run_dir.iterdir() if p.name.endswith(".init"))
        child = int(record.read_text().split()[0])
        nspid = next(
            line for line in Path(f"/proc/{child}/status").read_text().splitlines() if line.startswith("NSpid:")
        )
        assert nspid.split()[1:] == [str(child), "1"]
        await running
    finally:
        await sandbox.stop()


async def test_a_timeout_kills_background_and_setsid_children(provider: BwrapProvider) -> None:
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir="/workspace"))
    marker = "61.4242"
    try:
        result = await sandbox.exec(f"setsid sleep {marker} & sleep {marker} & sleep 30", timeout_s=0.5)
        assert result.error_type == "timeout"
        await asyncio.sleep(0.5)
        survivors = subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True).stdout
        assert survivors.strip() == ""
    finally:
        await sandbox.stop()


async def test_stop_from_a_borrower_kills_an_in_flight_exec(provider: BwrapProvider) -> None:
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir="/workspace"))
    borrower_provider = BwrapProvider(bwrap_path=BWRAP, base_dir=str(provider.base_dir), probe_on_create=False)
    borrowed = await AsyncSandbox.connect(await sandbox.serialize(), provider=borrower_provider, owns_provider=False)
    running = asyncio.create_task(sandbox.exec("sleep 30"))
    await _wait_for(lambda: bool(provider._processes.get(sandbox._handle.sandbox_id)))
    await borrowed.stop()
    result = await asyncio.wait_for(running, 10)
    assert result.return_code != 0
    assert not sandbox._handle.raw.root.exists()


async def test_symlinks_planted_by_sandboxed_code_are_refused(provider: BwrapProvider, tmp_path: Path) -> None:
    from nemo_gym.sandbox.providers.bwrap import BwrapPathError

    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir="/workspace"))
    try:
        planted = await sandbox.exec("ln -s /etc/hostname link && ln -s /tmp dir")
        assert planted.return_code == 0, planted.stderr
        (tmp_path / "src").write_text("payload")
        for path in ("/workspace/link", "/workspace/dir/x"):
            with pytest.raises(BwrapPathError):
                await sandbox.download(path, tmp_path / "got")
            with pytest.raises(BwrapPathError):
                await sandbox.upload(tmp_path / "src", path)
        assert not (tmp_path / "got").exists()
    finally:
        await sandbox.stop()
