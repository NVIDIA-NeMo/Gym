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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.aalcr.app import (
    LEGACY_JUDGE_PROTOCOL,
    V1_0_DATASET_REVISION,
    V1_1_DATASET_REVISION,
    V1_1_JUDGE_PROTOCOL,
    V1_1_SYSTEM_PROMPT,
    AalcrResourcesServer,
    AalcrResourcesServerConfig,
    AALCRVerifyRequest,
    _build_judge_input,
    _parse_judge_verdict,
)


def _config(judge_protocol: str = LEGACY_JUDGE_PROTOCOL) -> AalcrResourcesServerConfig:
    return AalcrResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server={
            "type": "responses_api_models",
            "name": "abcd",
        },
        judge_protocol=judge_protocol,
        judge_responses_create_params_overrides=dict(),
    )


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
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


def _request(judge_protocol: str) -> AALCRVerifyRequest:
    version, revision = {
        LEGACY_JUDGE_PROTOCOL: ("1.0.0", V1_0_DATASET_REVISION),
        V1_1_JUDGE_PROTOCOL: ("1.1", V1_1_DATASET_REVISION),
    }[judge_protocol]
    return AALCRVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_response("candidate"),
        document_category="Company Documents",
        document_set_id="set",
        question_id=28,
        question="What percentage?",
        answer="65.8%",
        data_source_filenames="document.txt",
        data_source_urls="https://example.com/document",
        input_tokens=100,
        input_tokens_band="<80k",
        aa_lcr_version=version,
        aa_lcr_dataset_revision=revision,
        aa_lcr_judge_protocol=judge_protocol,
    )


class TestApp:
    def test_sanity(self) -> None:
        AalcrResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_legacy_prompt_is_preserved(self) -> None:
        messages = _build_judge_input(
            LEGACY_JUDGE_PROTOCOL,
            question="question",
            official_answer="answer",
            candidate_answer="candidate",
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "The question, for reference only: question" in messages[0]["content"]
        assert "START QUESTION" not in messages[0]["content"]
        assert messages[0]["content"].endswith("Reply only with CORRECT or INCORRECT.")

    def test_v1_1_prompt_uses_official_system_and_user_messages(self) -> None:
        messages = _build_judge_input(
            V1_1_JUDGE_PROTOCOL,
            question="question",
            official_answer="answer",
            candidate_answer="candidate",
        )

        assert messages[0] == {"role": "system", "content": V1_1_SYSTEM_PROMPT}
        assert messages[1]["role"] == "user"
        assert "START QUESTION question\n\nEND QUESTION" in messages[1]["content"]
        assert "BEGIN CANDIDATE ANSWER TO ASSESS\n\ncandidate" in messages[1]["content"]
        assert messages[1]["content"].endswith("Reply as JSON, with a verdict of CORRECT or INCORRECT.")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"verdict": "CORRECT"}', "CORRECT"),
            ('{"verdict": "INCORRECT"}', "INCORRECT"),
        ],
    )
    def test_v1_1_parser_accepts_official_json_verdict(self, text: str, expected: str) -> None:
        assert _parse_judge_verdict(text, V1_1_JUDGE_PROTOCOL) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "CORRECT",
            '{"verdict": "MAYBE"}',
            '{"verdict": "CORRECT", "reason": "extra"}',
        ],
    )
    def test_v1_1_parser_rejects_non_protocol_responses(self, text: str) -> None:
        with pytest.raises(JudgeError, match="AA-LCR v1.1 judge"):
            _parse_judge_verdict(text, V1_1_JUDGE_PROTOCOL)

    async def test_v1_1_verify_sends_system_prompt_and_parses_json(self) -> None:
        server = AalcrResourcesServer(
            config=_config(V1_1_JUDGE_PROTOCOL),
            server_client=MagicMock(spec=ServerClient),
        )
        judge = AsyncMock(return_value=_response('{"verdict": "CORRECT"}'))

        with patch("resources_servers.aalcr.app.call_judge", judge):
            result = await server.verify(_request(V1_1_JUDGE_PROTOCOL))

        assert result.reward == 1.0
        assert result.invalid_judge_response is False
        judge_input = judge.await_args.kwargs["json"]["input"]
        assert [message["role"] for message in judge_input] == ["system", "user"]

    async def test_verify_rejects_dataset_and_judge_protocol_mismatch(self) -> None:
        server = AalcrResourcesServer(
            config=_config(V1_1_JUDGE_PROTOCOL),
            server_client=MagicMock(spec=ServerClient),
        )

        with pytest.raises(ValueError, match="dataset and judge protocol mismatch"):
            await server.verify(_request(LEGACY_JUDGE_PROTOCOL))

    async def test_legacy_invalid_verdict_remains_a_zero_reward(self) -> None:
        server = AalcrResourcesServer(
            config=_config(LEGACY_JUDGE_PROTOCOL),
            server_client=MagicMock(spec=ServerClient),
        )
        judge = AsyncMock(return_value=_response("NOT A VERDICT"))

        with patch("resources_servers.aalcr.app.call_judge", judge):
            result = await server.verify(_request(LEGACY_JUDGE_PROTOCOL))

        assert result.reward == 0.0
        assert result.invalid_judge_response is True
