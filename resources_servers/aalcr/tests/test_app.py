# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import AsyncMock, MagicMock

from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.aalcr.app import AalcrResourcesServer, AalcrResourcesServerConfig, AALCRVerifyRequest


def _config() -> AalcrResourcesServerConfig:
    return AalcrResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server={
            "type": "responses_api_models",
            "name": "abcd",
        },
        judge_responses_create_params_overrides=dict(),
    )


def _model_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="test_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_id",
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
        AalcrResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))

    async def test_verify_judge_failure_routed_to_sidecar(self) -> None:
        # A failed judge call is a distinct outcome, not a wrong answer: reward 0.0
        # and the row is flagged for the failures sidecar instead of deflating accuracy.
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=RuntimeError("judge timeout"))
        server = AalcrResourcesServer(config=_config(), server_client=server_client)

        req = AALCRVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_model_response("candidate answer"),
            document_category="c",
            document_set_id="s",
            question_id=1,
            question="q",
            answer="a",
            data_source_filenames="f",
            data_source_urls="u",
            input_tokens=1000,
            input_tokens_band="<80k",
        )

        data = (await server.verify(req)).model_dump()
        assert data["reward"] == approx(0.0)
        assert data["_ng_failure_class"] == "judge_failed"
        assert data["_ng_failure_judge_failed"] is True
        assert "judge timeout" in data["_ng_failure_judge_error"]
        assert data["response"] is not None
