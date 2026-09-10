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

        expected_prompt = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: question
The OFFICIAL ANSWER: answer
CANDIDATE ANSWER TO ASSESS: candidate

Reply only with CORRECT or INCORRECT."""
        assert messages == [{"role": "user", "content": expected_prompt}]

    def test_v1_1_prompt_uses_official_system_and_user_messages(self) -> None:
        messages = _build_judge_input(
            V1_1_JUDGE_PROTOCOL,
            question="question",
            official_answer="answer",
            candidate_answer="candidate",
        )

        expected_system_prompt = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""
        expected_user_prompt = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION question

END QUESTION

The OFFICIAL ANSWER: answer

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

candidate

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""
        assert V1_1_SYSTEM_PROMPT == expected_system_prompt
        assert messages == [
            {"role": "system", "content": expected_system_prompt},
            {"role": "user", "content": expected_user_prompt},
        ]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("CORRECT", "CORRECT"),
            (" INCORRECT\n", "INCORRECT"),
            ("NOT A VERDICT", None),
            ("", None),
            (" \n", None),
        ],
    )
    def test_legacy_parser_preserves_previous_verdict_behavior(self, text: str, expected: str | None) -> None:
        assert _parse_judge_verdict(text, LEGACY_JUDGE_PROTOCOL) == expected

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
            "",
            " \n",
        ],
    )
    def test_v1_1_parser_rejects_non_protocol_responses(self, text: str) -> None:
        with pytest.raises(JudgeError, match="empty judge response|AA-LCR v1.1 judge"):
            _parse_judge_verdict(text, V1_1_JUDGE_PROTOCOL)

    def test_legacy_rows_without_provenance_use_legacy_defaults(self) -> None:
        legacy_row = _request(LEGACY_JUDGE_PROTOCOL).model_dump(
            exclude={"aa_lcr_version", "aa_lcr_dataset_revision", "aa_lcr_judge_protocol"}
        )

        request = AALCRVerifyRequest.model_validate(legacy_row)

        assert request.aa_lcr_version == "1.0.0"
        assert request.aa_lcr_dataset_revision == V1_0_DATASET_REVISION
        assert request.aa_lcr_judge_protocol == LEGACY_JUDGE_PROTOCOL

    @pytest.mark.parametrize(("verdict", "expected_reward"), [("CORRECT", 1.0), ("INCORRECT", 0.0)])
    async def test_v1_1_verify_sends_system_prompt_and_parses_json(self, verdict: str, expected_reward: float) -> None:
        server = AalcrResourcesServer(
            config=_config(V1_1_JUDGE_PROTOCOL),
            server_client=MagicMock(spec=ServerClient),
        )
        judge = AsyncMock(return_value=_response(f'{{"verdict": "{verdict}"}}'))

        with patch("resources_servers.aalcr.app.call_judge", judge):
            result = await server.verify(_request(V1_1_JUDGE_PROTOCOL))

        assert result.reward == expected_reward
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

    @pytest.mark.parametrize(
        ("judge_text", "expected_reward", "expected_invalid"),
        [
            ("CORRECT", 1.0, False),
            ("INCORRECT", 0.0, False),
            ("NOT A VERDICT", 0.0, True),
            ("", 0.0, True),
            (" \n", 0.0, True),
        ],
    )
    async def test_legacy_verdict_preserves_previous_reward(
        self, judge_text: str, expected_reward: float, expected_invalid: bool
    ) -> None:
        server = AalcrResourcesServer(
            config=_config(LEGACY_JUDGE_PROTOCOL),
            server_client=MagicMock(spec=ServerClient),
        )
        judge = AsyncMock(return_value=_response(judge_text))

        with patch("resources_servers.aalcr.app.call_judge", judge):
            result = await server.verify(_request(LEGACY_JUDGE_PROTOCOL))

        assert result.reward == expected_reward
        assert result.invalid_judge_response is expected_invalid
