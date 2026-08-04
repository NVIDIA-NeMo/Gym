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
"""Isolation of the direct apptainer sandbox from the host process tree.

Tasks run arbitrary shell commands inside the sandbox. Without a private PID
namespace a pattern-matching kill reaches host processes, and without a new
session a group-directed signal reaches this server's own process group.
"""

import asyncio
import contextlib
import os
import signal
import sys
from unittest.mock import AsyncMock, patch

import pytest

from responses_api_agents.pinchbench.tests.test_app import make_agent


async def _capture_launch(agent, tmp_path, apptainer_cfg):
    captured = {}

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        proc = AsyncMock()
        proc.wait = AsyncMock(return_value=0)
        proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        with contextlib.suppress(Exception):
            await agent._run_in_apptainer_direct("task_x", tmp_path, apptainer_cfg)
    return captured


async def _sleeping_child(**kwargs):
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_direct_exec_isolates_the_pid_namespace_by_default(tmp_path):
    agent = make_agent(sandbox_spec={"image": "/img.sif"})
    captured = await _capture_launch(agent, tmp_path, {"direct_exec": True})

    assert "--pid" in captured["argv"]


@pytest.mark.asyncio
async def test_explicit_direct_exec_args_are_honoured(tmp_path):
    agent = make_agent(sandbox_spec={"image": "/img.sif"})
    captured = await _capture_launch(agent, tmp_path, {"direct_exec": True, "direct_exec_args": ["--cleanenv"]})

    assert "--pid" not in captured["argv"]


@pytest.mark.asyncio
async def test_direct_exec_launches_in_a_new_session(tmp_path):
    agent = make_agent(sandbox_spec={"image": "/img.sif"})
    captured = await _capture_launch(agent, tmp_path, {"direct_exec": True})

    assert captured["kwargs"]["start_new_session"] is True


@pytest.mark.asyncio
async def test_new_session_puts_the_child_in_its_own_process_group():
    proc = await _sleeping_child(start_new_session=True)
    try:
        assert os.getpgid(proc.pid) == proc.pid
        assert os.getpgid(proc.pid) != os.getpgid(0)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        await proc.wait()


@pytest.mark.asyncio
async def test_without_new_session_the_child_shares_our_process_group():
    proc = await _sleeping_child()
    try:
        assert os.getpgid(proc.pid) == os.getpgid(0)
    finally:
        proc.kill()
        await proc.wait()
