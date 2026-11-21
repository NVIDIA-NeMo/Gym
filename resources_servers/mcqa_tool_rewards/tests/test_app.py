# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.opencode_multimodal_benchmarks.tests import tool_adapter_checks


class TestAdapter(tool_adapter_checks.AdapterChecks):
    component = "mcqa_tool_rewards"
    parent = "mcqa"
    invalid_fields = {"grading_mode": "invalid", "options": "A"}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_rewards,expected", [(None, 1.0), ({}, 0.95)])
async def test_grading_evidence_binds_final_result(tool_rewards, expected):
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from nemo_gym.core import rollout_context
    from nemo_gym.server_runtime import observe_verifier
    from nemo_gym.trajectory import grading_result_binding
    from resources_servers.mcqa_tool_rewards.app import (
        ToolRewardConfig,
        ToolRewardServer,
        ToolRewardVerifyRequest,
    )
    from resources_servers.opencode_multimodal_benchmarks.tests.test_rewards import tool_response, trajectory

    server = ToolRewardServer(
        config=ToolRewardConfig(name="mcqa", host="localhost", port=1, entrypoint="app.py", tool_rewards=tool_rewards),
        server_client=MagicMock(spec=ServerClient),
    )
    body = ToolRewardVerifyRequest(
        responses_create_params={"input": []},
        response=tool_response("\\boxed{A}"),
        expected_answer="A",
        options=[{"A": "cat"}, {"B": "dog"}],
        ng_trajectory=trajectory("failed"),
    )
    with rollout_context("mcqa-final-grade"):
        result = await observe_verifier(server.verify, enabled=True, verifier_name="mcqa")(body)
    saved = result.model_dump(mode="json")
    assert saved["reward"] == pytest.approx(expected)
    assert saved["response"] == body.response.model_dump(mode="json")
    [invocation] = saved["ng_verifier_invocations"]
    [operation] = invocation["grading_operations"]
    assert operation["outcome"] == "completed"
    assert operation["dependencies"] == []
    assert operation["result_binding_sha256"] == grading_result_binding(saved)
