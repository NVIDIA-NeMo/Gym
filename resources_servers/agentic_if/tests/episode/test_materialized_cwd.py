# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialized prompt facts must agree with the actual bound task directory."""

import os

import pytest

from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.workspace import verify_materialized_cwd
from resources_servers.agentic_if.tests.support import LocalTransport


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["physical", "symlink", "omitted"])
async def test_cwd_fact_accepts_equivalent_directories_without_changing_identity(tmp_path, spelling):
    task = tmp_path / "task root"
    task.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(task)
    transport = LocalTransport()
    sandbox = BoundSandbox(transport, user=os.getuid(), workdir=str(task), timeout_s=60)
    expected = {"physical": str(task), "symlink": str(alias), "omitted": None}[spelling]
    await verify_materialized_cwd(sandbox, expected)
    assert sandbox.workdir == str(task)
    assert all(user == os.getuid() for _, user in transport.calls)
    assert len(transport.calls) == (0 if expected is None else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", ["different", "missing", "relative", "not-text", "newline"])
async def test_inaccurate_cwd_fails_before_actor_execution(tmp_path, expected):
    task = tmp_path / "task"
    task.mkdir()
    other = tmp_path / "different"
    other.mkdir()
    value = {
        "different": str(other),
        "missing": str(tmp_path / "absent"),
        "relative": "task",
        "not-text": 1,
        "newline": "/task\n/other",
    }[expected]
    sandbox = BoundSandbox(LocalTransport(), user=os.getuid(), workdir=str(task), timeout_s=60)
    with pytest.raises((ValueError, RuntimeError), match="materialized_cwd"):
        await verify_materialized_cwd(sandbox, value)
    assert sandbox.workdir == str(task)
