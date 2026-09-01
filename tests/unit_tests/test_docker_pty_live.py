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

"""Live PTY tests against a real local docker daemon.

Skipped unless a reachable docker daemon is available. Uses
``python:3.12-slim`` (multi-arch: runs natively on amd64 and arm64), pulled on
first run.
"""

import shutil
import subprocess

import pytest

from nemo_gym.sandbox.providers.base import SandboxPtyError, SandboxPtySpec, SandboxSpec
from nemo_gym.sandbox.providers.docker import DockerProvider


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipduringci,
    pytest.mark.skipif(not _docker_available(), reason="docker daemon not available"),
]

# Small public image that carries python3 (required by the docker PTY broker).
IMAGE = "python:3.12-slim"


async def _read_until(session, needle: bytes, *, timeout_s: float = 30.0) -> bytes:
    buffer = bytearray()
    while needle not in buffer:
        buffer.extend(await session.read(timeout_s=timeout_s))
    return bytes(buffer)


async def test_pty_cross_process_attach_flow() -> None:
    provider = DockerProvider()
    handle = await provider.create(SandboxSpec(image=IMAGE, ttl_s=600))
    try:
        creator_session = await provider.create_pty(handle, SandboxPtySpec())
        assert creator_session.mode == "pty"
        assert not creator_session.closed

        await creator_session.write(b"echo hello-$((1+1))\n")
        output = await _read_until(creator_session, b"hello-2")
        assert b"hello-2" in output

        # Simulate another process: a fresh provider instance that shares only
        # the docker daemon rebuilds the handle from the descriptor and
        # attaches to the same session by id.
        descriptor = await provider.serialize_handle(handle)
        other_provider = DockerProvider()
        other_handle = await other_provider.connect(descriptor)
        attached = await other_provider.attach_pty(other_handle, creator_session.session_id, takeover=True, since=0)
        try:
            assert attached.session_id == creator_session.session_id
            assert attached.mode == "pty"

            # since=0 replays the retained output, which includes the earlier command.
            replay = await _read_until(attached, b"hello-2")
            assert b"hello-2" in replay

            # The evicted creator's next use fails.
            with pytest.raises(SandboxPtyError):
                await creator_session.read(timeout_s=10.0)
            with pytest.raises(SandboxPtyError):
                await creator_session.write(b"echo nope\n")
            assert creator_session.closed

            # The attached session is live: shell state still works.
            await attached.write(b"echo again-$((2+2))\n")
            assert b"again-4" in await _read_until(attached, b"again-4")

            # run_detached: marker discipline over the polled output log.
            output, exit_code = await attached.run_detached("sleep 1; echo done", poll_interval_s=0.5)
            assert b"done" in output
            assert exit_code == 0
        finally:
            await attached.close()  # attached: releases, does not end the session

        # Released (not ended): a non-takeover attach now succeeds.
        reattached = await other_provider.attach_pty(other_handle, creator_session.session_id, takeover=False)
        await reattached.close()
        await other_provider.aclose()
    finally:
        await provider.close(handle)
        await provider.aclose()


async def test_pty_command_session_exit_and_cleanup() -> None:
    provider = DockerProvider()
    handle = await provider.create(SandboxSpec(image=IMAGE, ttl_s=600))
    try:
        session = await provider.create_pty(handle, SandboxPtySpec(command="echo bye; exit 7", rows=30, cols=100))
        output = await _read_until(session, b"bye")
        assert b"bye" in output
        assert await session.wait_exit(timeout_s=30.0) == 7
        # Drain to EOF.
        while await session.read(timeout_s=10.0):
            pass
        await session.close()

        # The creator's close ends the session: its state dir is gone.
        probe = await provider.exec(handle, f"test -d /tmp/.nemo-gym-pty/{session.session_id}")
        assert probe.return_code != 0

        # And re-attaching raises.
        with pytest.raises(SandboxPtyError):
            await provider.attach_pty(handle, session.session_id)
    finally:
        await provider.close(handle)
        await provider.aclose()


async def test_pty_resize_and_signal() -> None:
    provider = DockerProvider()
    handle = await provider.create(SandboxSpec(image=IMAGE, ttl_s=600))
    try:
        session = await provider.create_pty(handle, SandboxPtySpec(rows=24, cols=80))
        try:
            await session.write(b"stty size\n")
            assert b"24 80" in await _read_until(session, b"24 80")

            await session.resize(40, 120)
            await session.write(b"stty size\n")
            assert b"40 120" in await _read_until(session, b"40 120")

            # SIGKILL the shell's process group (interactive shells ignore
            # SIGTERM): the session process exits with 128 + 9.
            await session.send_signal("SIGKILL")
            assert await session.wait_exit(timeout_s=30.0) == 128 + 9
        finally:
            await session.close()
    finally:
        await provider.close(handle)
        await provider.aclose()
