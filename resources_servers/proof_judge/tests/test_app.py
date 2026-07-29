# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.proof_judge.app import (
    ProofWithJudgeResourcesServer,
    ProofWithJudgeResourcesServerConfig,
    ProofWithJudgeVerifyRequest,
)


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class TestApp:
    def test_sanity(self) -> None:
        config = ProofWithJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        )
        ProofWithJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_verify_judge_failure_recorded(self) -> None:
        """A judge transport error is recorded as judge_failed, not a wrong answer."""
        config = ProofWithJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        )
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock(side_effect=RuntimeError("judge timeout"))
        server = ProofWithJudgeResourcesServer(config=config, server_client=server_mock)

        response = _make_response("## Solution\nmy proof.\n## Self Evaluation\nlooks fine \\boxed{1}")
        request = ProofWithJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=response,
            problem="Prove it.",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.judge_failed is True
        assert "judge timeout" in result.judge_failure_reason
