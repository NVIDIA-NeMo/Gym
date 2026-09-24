# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference execution stays independent of task loading and file staging."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.miniswe_sandboxed_agent.app import empty_response
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext
from responses_api_agents.miniswe_sandboxed_agent.oracle import OracleHarness


@pytest.mark.parametrize("user", [None, "agent", 1000, "root"])
@pytest.mark.parametrize("exit_code", [0, 1])
async def test_oracle_preserves_agent_identity_and_exit_status(tmp_path, user, exit_code):
    sandbox = SimpleNamespace(
        upload=AsyncMock(),
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="uid=test\n/task\n", stderr=""),
                SimpleNamespace(return_code=exit_code, stdout="solution", stderr=""),
            ]
        ),
    )
    context = HarnessContext(
        session_id="safe-id",
        task_id="test",
        rollout_id="run",
        instruction="task",
        user=user,
        workdir="/task",
        setup_timeout_sec=30,
        mcp_servers=[],
        skills_dir=None,
    )
    harness = OracleHarness(
        sandbox=sandbox,
        context=context,
        directory=tmp_path / "oracle",
        response=empty_response(NeMoGymResponseCreateParamsNonStreaming(input=[]), "unused"),
    )
    await harness.setup()
    response, outcome, extra = await harness.execute(30)
    sandbox.upload.assert_not_awaited()
    for call in sandbox.exec.await_args_list:
        assert call.kwargs["user"] == user and call.kwargs["cwd"] == "/task"
    assert outcome.reason == ("completed" if exit_code == 0 else "nonzero_exit")
    assert extra["oracle_exit_code"] == exit_code and response.output == []
    assert json.loads((tmp_path / "oracle/identity.json").read_text())["requested_user"] == user
