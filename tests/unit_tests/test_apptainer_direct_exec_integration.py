# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-Apptainer regression for detached daemons, not a fake CLI."""

import asyncio
import contextlib
import os
import shlex
import socket

import pytest

from nemo_gym.sandbox.providers.apptainer.provider import ApptainerProvider
from nemo_gym.sandbox.providers.base import SandboxSpec


@pytest.mark.skipif(
    not os.environ.get("NEMO_GYM_APPTAINER_TEST_IMAGE"), reason="requires a local Python-capable image"
)
@pytest.mark.parametrize("termination", ["close", "cancel"])
async def test_direct_exec_reaps_setsid_daemon(termination):
    provider = ApptainerProvider(
        bin_path=os.environ.get("NEMO_GYM_APPTAINER_TEST_BIN_PATH"),
        create={"use_instance": False, "apply_resource_limits": False, "extra_start_args": ["--no-home"]},
        probe={"command": None},
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handle = await provider.create(SandboxSpec(image=os.environ["NEMO_GYM_APPTAINER_TEST_IMAGE"]))
    # Double-fork and setsid escape the exec process group, but not its PID namespace.
    daemon = (
        "import os,socket,time; "
        "pid=os.fork(); "
        "os._exit(0) if pid else None; os.setsid(); "
        "pid=os.fork(); os._exit(0) if pid else None; "
        "s=socket.socket(); s.bind(('127.0.0.1'," + str(port) + ")); s.listen(); "
        "open('/sandbox/daemon-ready','w').write('ready'); time.sleep(600)"
    )
    task = asyncio.create_task(provider.exec(handle, f"python3 -c {shlex.quote(daemon)}; sleep 600", timeout_s=300))

    def listening():
        with socket.socket() as probe:
            probe.settimeout(0.2)
            return probe.connect_ex(("127.0.0.1", port)) == 0

    try:
        async with asyncio.timeout(240):
            while not (handle.raw.staging_dir / "daemon-ready").exists():
                if task.done():
                    pytest.fail(f"Apptainer exited before daemon readiness: {await task}")
                await asyncio.sleep(0.1)
        assert listening()
        if termination == "close":
            await provider.close(handle)
        else:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(10):
            while listening():
                await asyncio.sleep(0.1)
        await provider.close(handle)
        assert not handle.raw.staging_dir.exists()
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await provider.close(handle)
