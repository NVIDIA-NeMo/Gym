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
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest

from nemo_gym.process_utils import await_cleanup, kill_process_tree


def test_await_cleanup_finishes_after_repeated_cancellation() -> None:
    async def run() -> None:
        release = asyncio.Event()

        async def child() -> str:
            await release.wait()
            return "reaped"

        communication = asyncio.create_task(child())
        cleanup = asyncio.create_task(await_cleanup(communication))
        await asyncio.sleep(0)
        cleanup.cancel()
        await asyncio.sleep(0)
        cleanup.cancel()
        assert not cleanup.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert communication.result() == "reaped"

    asyncio.run(run())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group lifecycle")
@pytest.mark.parametrize("detached", [False, True])
def test_kill_process_tree_stops_stdout_holding_descendants(tmp_path: Path, detached: bool) -> None:
    child_pid = tmp_path / "child.pid"
    launcher = (
        "import subprocess,sys; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], start_new_session={detached!r}); "
        f"Path({str(child_pid)!r}).write_text(str(child.pid)); child.wait()"
    )

    async def run() -> None:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", launcher, stdout=asyncio.subprocess.PIPE, start_new_session=True
        )
        sibling = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)", start_new_session=True
        )
        try:
            async with asyncio.timeout(5):
                while not child_pid.exists():
                    await asyncio.sleep(0.01)
                kill_process_tree(proc)
                await proc.communicate()
            assert proc.returncode == -signal.SIGKILL
            assert sibling.returncode is None
            try:
                assert psutil.Process(int(child_pid.read_text())).status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                pass
        finally:
            # Keep the regression safe to run against an implementation that leaks children.
            if child_pid.exists():
                with suppress(ProcessLookupError):
                    os.kill(int(child_pid.read_text()), signal.SIGKILL)
            for process in (proc, sibling):
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    asyncio.run(run())


@pytest.mark.parametrize("error", [ProcessLookupError, AttributeError, OSError])
def test_kill_process_tree_handles_exited_process_and_unavailable_groups(error) -> None:
    proc = MagicMock(pid=123)
    proc.kill.side_effect = ProcessLookupError
    with (
        patch("nemo_gym.process_utils.psutil.Process", side_effect=psutil.NoSuchProcess(123)),
        patch("nemo_gym.process_utils.os.killpg", create=True, side_effect=error),
    ):
        kill_process_tree(proc)
    assert proc.kill.call_count == (0 if error is ProcessLookupError else 1)
