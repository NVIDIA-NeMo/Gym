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

    async def test_verify_judge_failure_routed_to_sidecar(self) -> None:
        # A judge call that errors is a distinct outcome, not a wrong answer:
        # reward 0.0, the model's answer carried, and the row flagged for the
        # failures sidecar instead of contaminating accuracy.
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

        request = ProofWithJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("## Solution\nProof here.\n## Self Evaluation\n\\boxed{1}"),
            problem="Prove X.",
        )

        data = (await server.verify(request)).model_dump()
        assert data["reward"] == approx(0.0)
        assert data["_ng_failure_class"] == "judge_failed"
        assert data["_ng_failure_judge_failed"] is True
        assert "judge timeout" in data["_ng_failure_judge_error"]
        assert data["response"] is not None
