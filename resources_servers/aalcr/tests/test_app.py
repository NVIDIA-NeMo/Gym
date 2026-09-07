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
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.aalcr.app import (
    AalcrResourcesServer,
    AalcrResourcesServerConfig,
    AALCRVerifyRequest,
    _parse_bare_verdict,
    _parse_json_verdict,
)
from resources_servers.aalcr.versions import DEFAULT_VERSION, VERSIONS, get_version


def _config(dataset_version: str = DEFAULT_VERSION) -> AalcrResourcesServerConfig:
    return AalcrResourcesServerConfig(
        dataset_version=dataset_version,
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


class TestApp:
    def test_sanity(self) -> None:
        AalcrResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))


class FakeHTTPResponse:
    ok = True

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _make_response(text: str) -> dict:
    return NeMoGymResponse(
        id="judge_response",
        created_at=0.0,
        model="test-judge",
        object="response",
        output=[
            {
                "id": "message_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    ).model_dump()


def _server(judge_reply: str, dataset_version: str = DEFAULT_VERSION) -> AalcrResourcesServer:
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(return_value=FakeHTTPResponse(_make_response(judge_reply)))
    return AalcrResourcesServer(config=_config(dataset_version), server_client=server_client)


def _request(candidate_answer: str = "65.8%") -> AALCRVerifyRequest:
    return AALCRVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "BEGIN INPUT DOCUMENTS..."}]},
        response=NeMoGymResponse(
            id="policy_response",
            created_at=0.0,
            model="test-policy",
            object="response",
            output=[
                {
                    "id": "message_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": candidate_answer, "annotations": []}],
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        ),
        document_category="Company_Reports",
        document_set_id="set_1",
        question_id=28,
        question="What was the margin?",
        answer="65.8%",
        data_source_filenames="a.txt",
        data_source_urls="https://example.invalid/a",
        input_tokens=90000,
        input_tokens_band="80k-100k",
    )


class TestJudgePrompts:
    """The prompts must stay byte-faithful to the upstream dataset card at each pinned revision."""

    def test_v1_1_adds_a_system_prompt_with_the_four_grading_rules(self) -> None:
        system_prompt = VERSIONS["1.1"].judge_system_prompt
        assert system_prompt.startswith(
            "Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER."
        )
        assert system_prompt.count("\n- ") == 4
        assert '0.675, "67.5%" and "67.5 percentage points" all match' in system_prompt

    def test_v1_0_has_no_system_prompt(self) -> None:
        assert VERSIONS["1.0"].judge_system_prompt is None

    def test_v1_1_user_prompt_uses_delimited_blocks_and_asks_for_json(self) -> None:
        prompt = VERSIONS["1.1"].judge_user_prompt.format(question="Q?", official_answer="A", candidate_answer="C")
        for marker in (
            "START QUESTION Q?",
            "END QUESTION",
            "The OFFICIAL ANSWER: A",
            "END OFFICIAL ANSWER",
            "BEGIN CANDIDATE ANSWER TO ASSESS",
            "END CANDIDATE ANSWER TO ASSESS",
        ):
            assert marker in prompt
        assert prompt.endswith("Reply as JSON, with a verdict of CORRECT or INCORRECT.")
        # The superseded instruction must not linger; it is what makes a judge emit a bare verdict.
        assert "Reply only with CORRECT or INCORRECT." not in prompt

    def test_v1_0_user_prompt_is_unchanged(self) -> None:
        prompt = VERSIONS["1.0"].judge_user_prompt.format(question="Q?", official_answer="A", candidate_answer="C")
        assert "The question, for reference only: Q?" in prompt
        assert "CANDIDATE ANSWER TO ASSESS: C" in prompt
        assert prompt.endswith("Reply only with CORRECT or INCORRECT.")


class TestVersionTable:
    def test_each_version_pairs_a_revision_with_a_protocol(self) -> None:
        assert set(VERSIONS) == {"1.0", "1.1"}
        assert VERSIONS["1.0"].revision != VERSIONS["1.1"].revision
        assert VERSIONS["1.0"].judge_replies_json is False
        assert VERSIONS["1.1"].judge_replies_json is True

    def test_unknown_version_is_rejected_by_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown AA-LCR version"):
            get_version("1.2")


class TestParseJudgeVerdict:
    @pytest.mark.parametrize(
        "reply, expected",
        [
            ('{"verdict": "CORRECT"}', "CORRECT"),
            ('{"verdict": "INCORRECT"}', "INCORRECT"),
            ('```json\n{"verdict": "CORRECT"}\n```', "CORRECT"),
            ('Here is my assessment:\n{"verdict": "INCORRECT"}\nDone.', "INCORRECT"),
            ('{"verdict": "correct"}', "CORRECT"),
            ('{"Verdict": " CORRECT "}', "CORRECT"),
            ('{"reasoning": "matches after unit conversion", "verdict": "CORRECT"}', "CORRECT"),
        ],
    )
    def test_json_parser_accepts_json_verdicts(self, reply: str, expected: str) -> None:
        assert _parse_json_verdict(reply) == expected

    @pytest.mark.parametrize(
        "reply",
        [
            "CORRECT",  # the superseded v1.0 bare-verdict format
            "INCORRECT",
            "",
            "The answer looks right to me.",
            '{"verdict": "MAYBE"}',
            '{"result": "CORRECT"}',
            '{"verdict": true}',
            '["CORRECT"]',
            '{"verdict": "CORRECT"',  # truncated JSON
        ],
    )
    def test_json_parser_rejects_everything_else(self, reply: str) -> None:
        assert _parse_json_verdict(reply) is None

    @pytest.mark.parametrize("reply, expected", [("CORRECT", "CORRECT"), ("INCORRECT", "INCORRECT")])
    def test_bare_parser_accepts_v1_0_verdicts(self, reply: str, expected: str) -> None:
        assert _parse_bare_verdict(reply) == expected

    @pytest.mark.parametrize("reply", ['{"verdict": "CORRECT"}', "", "Correct", "CORRECT."])
    def test_bare_parser_rejects_everything_else(self, reply: str) -> None:
        assert _parse_bare_verdict(reply) is None


class TestVerify:
    async def test_correct_verdict_scores_one_and_sends_the_system_prompt(self) -> None:
        response = await _server('{"verdict": "CORRECT"}').verify(_request())
        assert response.reward == 1.0
        assert response.invalid_judge_response is False
        assert response.invalid_model_response is False
        assert response.reward_80k_100k == 1.0
        assert response.judge_responses_create_params.instructions == VERSIONS["1.1"].judge_system_prompt

    async def test_incorrect_verdict_scores_zero_and_is_not_flagged(self) -> None:
        response = await _server('{"verdict": "INCORRECT"}').verify(_request())
        assert response.reward == 0.0
        assert response.invalid_judge_response is False

    async def test_unparseable_judge_reply_is_flagged_not_silently_zero(self) -> None:
        """A judge that ignores the JSON instruction must be visible in `invalid_judge_response`.

        `nemo_gym.judge` reserves exceptions for failed judge *calls*; a received-but-unparseable reply is
        scored, so the flag is the only signal that a run was mis-graded rather than merely answered badly.
        A bare `CORRECT` is exactly what a v1.0-shaped judge returns, so this also covers the prompt/parser
        mismatch that would otherwise drive every rollout to zero.
        """
        response = await _server("CORRECT").verify(_request())
        assert response.reward == 0.0
        assert response.invalid_judge_response is True

    async def test_selecting_v1_0_restores_the_whole_v1_0_protocol(self) -> None:
        """One selector moves the data and the protocol together, so v1.0 grades v1.0 the way it did."""
        server = _server("CORRECT", dataset_version="1.0")
        response = await server.verify(_request())
        assert response.reward == 1.0
        assert response.invalid_judge_response is False
        assert response.judge_responses_create_params.instructions is None

    async def test_v1_0_rejects_a_json_reply(self) -> None:
        response = await _server('{"verdict": "CORRECT"}', dataset_version="1.0").verify(_request())
        assert response.reward == 0.0
        assert response.invalid_judge_response is True

    async def test_empty_candidate_answer_short_circuits_the_judge(self) -> None:
        server = _server('{"verdict": "CORRECT"}')
        response = await server.verify(_request(candidate_answer="   "))
        assert response.reward == 0.0
        assert response.invalid_model_response is True
        server.server_client.post.assert_not_called()
