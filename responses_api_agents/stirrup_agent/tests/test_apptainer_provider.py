# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Startup/teardown resource-safety tests for the Apptainer code-exec provider.

A readiness failure in ``__aenter__`` (which skips ``__aexit__``) must not leak
the subprocess, the stderr log handle, or the temp bind directory.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from responses_api_agents.stirrup_agent.apptainer_provider import ApptainerCodeExecToolProvider


def _make_fake_process(wait_side_effect=None) -> MagicMock:
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = None
    proc.pid = 4242
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    if wait_side_effect is not None:
        proc.wait = AsyncMock(side_effect=wait_side_effect)
    else:
        proc.wait = AsyncMock(return_value=0)
    return proc


async def test_failed_readiness_cleans_up_process_handle_and_tempdir() -> None:
    """A readiness failure must terminate the shell, close the log, and drop the temp dir."""
    provider = ApptainerCodeExecToolProvider(sif_path="/fake.sif")
    fake_proc = _make_fake_process()
    create_mock = AsyncMock(return_value=fake_proc)

    # `echo ready` returns non-zero -> __aenter__ raises before returning.
    with (
        patch("asyncio.create_subprocess_shell", new=create_mock),
        patch.object(ApptainerCodeExecToolProvider, "_exec", new=AsyncMock(return_value=(1, b"", b""))),
    ):
        with pytest.raises(RuntimeError, match="Failed to start Apptainer shell"):
            await provider.__aenter__()

    # The graceful terminate path was taken and awaited (no leaked live shell).
    fake_proc.terminate.assert_called_once()
    fake_proc.wait.assert_awaited()
    assert provider._process is None

    # The stderr log handle handed to the subprocess is closed and its temp
    # bind directory removed.
    stderr_fh = create_mock.call_args.kwargs["stderr"]
    assert stderr_fh.closed
    assert not Path(stderr_fh.name).parent.exists()
    assert provider._stderr_fh is None
    assert provider._temp_dir is None


async def test_failed_readiness_kills_when_terminate_times_out() -> None:
    """If the shell ignores SIGTERM, the cleanup escalates to kill() and reaps it."""
    provider = ApptainerCodeExecToolProvider(sif_path="/fake.sif")
    # First wait (after terminate) times out; second wait (after kill) reaps.
    fake_proc = _make_fake_process(wait_side_effect=[asyncio.TimeoutError(), 0])

    with (
        patch("asyncio.create_subprocess_shell", new=AsyncMock(return_value=fake_proc)),
        patch.object(ApptainerCodeExecToolProvider, "_exec", new=AsyncMock(return_value=(1, b"", b""))),
    ):
        with pytest.raises(RuntimeError, match="Failed to start Apptainer shell"):
            await provider.__aenter__()

    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()
    assert fake_proc.wait.await_count == 2
    assert provider._process is None
    assert provider._temp_dir is None


async def test_normal_exit_closes_stderr_handle_and_tempdir() -> None:
    """On a successful enter/exit the stderr handle and temp dir are released."""
    provider = ApptainerCodeExecToolProvider(sif_path="/fake.sif", capture_git_diff=False)
    fake_proc = _make_fake_process()

    with (
        patch("asyncio.create_subprocess_shell", new=AsyncMock(return_value=fake_proc)),
        patch.object(ApptainerCodeExecToolProvider, "_exec", new=AsyncMock(return_value=(0, b"", b""))),
        patch.object(ApptainerCodeExecToolProvider, "get_code_exec_tool", new=MagicMock(return_value=MagicMock())),
    ):
        await provider.__aenter__()
        stderr_fh = provider._stderr_fh
        temp_dir = provider._temp_dir
        assert stderr_fh is not None and not stderr_fh.closed
        assert temp_dir is not None and temp_dir.exists()

        await provider.__aexit__(None, None, None)

    assert stderr_fh.closed
    assert provider._stderr_fh is None
    assert not temp_dir.exists()
    assert provider._temp_dir is None
