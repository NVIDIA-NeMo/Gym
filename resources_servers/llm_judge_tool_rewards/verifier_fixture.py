# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline judge replies exercise the real wrapper without network requests."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import BaseVerifyResponse
from nemo_gym.verification import VerifierFixture


if TYPE_CHECKING:
    from resources_servers.llm_judge_tool_rewards.app import FixtureRequest, ToolRewardServer


def create_server() -> "ToolRewardServer":
    from resources_servers.llm_judge_tool_rewards.app import ToolRewardConfig, ToolRewardServer

    return ToolRewardServer(
        config=ToolRewardConfig(
            name="fixture",
            host="localhost",
            port=1,
            entrypoint="app.py",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            judge_responses_create_params={"input": []},
        ),
        server_client=MagicMock(spec=ServerClient),
    )


async def invoke(server: "ToolRewardServer", body: "FixtureRequest") -> BaseVerifyResponse:
    from resources_servers.llm_judge_tool_rewards.app import ToolRewardVerifyRequest

    reply = body.response.model_copy(deep=True)
    reply.id = "scripted-judge"
    reply.output = [reply.output[-1]]
    reply.output[0].content[0].text = body.scripted_verdict
    http = MagicMock(status=200)
    http.read = AsyncMock(return_value=reply.model_dump_json().encode())
    server.server_client.post = AsyncMock(return_value=http)
    request = ToolRewardVerifyRequest.model_validate(body.model_dump(exclude={"scripted_verdict"}))
    return await server.verify(request)


def fixture(request_model: type[BaseModel]) -> VerifierFixture:
    return VerifierFixture(
        server_factory=create_server,
        request_model=request_model,
        invoke=invoke,
        cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
    )
