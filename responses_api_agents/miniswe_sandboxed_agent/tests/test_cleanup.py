# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig, MiniSWEHarness


@pytest.fixture
def harness(tmp_path):
    return MiniSWEHarness(
        sandbox=SimpleNamespace(exec=AsyncMock()),
        context=HarnessContext(session_id="unknown-launch", instruction="task"),
        config=MiniSWEConfig(),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=AsyncMock(),
        model_name="model",
        directory=tmp_path,
    )


@pytest.mark.parametrize(
    "evidence",
    [
        SandboxExecResult("", "missing cleanup receipt", 1),
        SandboxExecResult('{"status":"failed","remaining_pids":[42]}', "", 0),
    ],
)
async def test_unknown_launch_or_negative_cleanup_blocks_disposal(harness, evidence):
    harness.launch_attempted = True
    harness.sandbox.exec.return_value = evidence
    with pytest.raises(RuntimeError, match="cleanup"):
        await harness.close()
    assert not harness.cleanup_confirmed
    with pytest.raises(RuntimeError, match="before confirmed cleanup"):
        await harness.dispose()
    harness.sandbox.exec.return_value = SandboxExecResult('{"status":"stopped","remaining_pids":[]}', "", 0)
    await harness.close()
    assert harness.cleanup_confirmed


async def test_file_cleanup_failure_is_retryable(harness):
    harness.sandbox.exec.return_value = SandboxExecResult("", "read-only filesystem", 1)
    with pytest.raises(RuntimeError, match="session file cleanup failed"):
        await harness.dispose()
    assert not harness.disposed
    harness.sandbox.exec.return_value = SandboxExecResult("", "", 0)
    await harness.dispose()
    assert harness.disposed


async def test_missing_prerequisite_reports_actionable_failure(harness):
    harness.sandbox.exec.return_value = SandboxExecResult("", "Missing prerequisite: setsid", 127)
    with pytest.raises(RuntimeError, match="(?s)127.*Missing prerequisite: setsid"):
        await harness.setup()
