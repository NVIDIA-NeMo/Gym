# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.safe_child_llm.app import (
    SafeChildLLMConfig,
    SafeChildLLMResourcesServer,
    SafeChildLLMVerifyRequest,
)


async def test_verify_never_claims_an_automatic_safety_score():
    server = SafeChildLLMResourcesServer(
        config=SafeChildLLMConfig(host="0.0.0.0", port=8080, entrypoint="", name="safe_child_llm"),
        server_client=MagicMock(spec=ServerClient),
    )
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="test",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    result = await server.verify(
        SafeChildLLMVerifyRequest(
            responses_create_params={"input": "test"},
            response=response,
            safe_child_id="safe-child-6-12-001",
            annotation_status="pending_human_review",
        )
    )
    assert result.reward == 0.0
    assert result.annotation_status == "pending_human_review"
