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
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.encoders import jsonable_encoder
from pytest import approx, fixture, mark

from nemo_gym.config_types import AggregateMetricsRequest, ModelServerRef
from nemo_gym.judge import judge_failsafe
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.equivalence_llm_judge.app import (
    VERIFIER_FIXTURE,
    LLMJudgeResourcesServer,
    LLMJudgeResourcesServerConfig,
    LLMJudgeVerifyRequest,
    _extract_question_text,
    _match_verdict_labels,
)


async def test_hle_verified_fixture() -> None:
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
    )
    assert [(result.kind, result.observed_rewards) for result in results] == [
        ("full_reward", (1.0,)),
        ("zero_reward", (0.0,)),
        ("malformed", ()),
    ]


class TestApp:
    def test_checkpoint_recovery_mode_is_stateless(self) -> None:
        assert LLMJudgeResourcesServerConfig.CHECKPOINT_RECOVERY_MODE == "stateless"

    @fixture
    def config(self) -> LLMJudgeResourcesServerConfig:
        judge_prompt_template_fpath = str(
            Path(__file__).resolve().parents[1] / "prompt_templates/equivalence_llm_judge.txt"
        )

        cfg = LLMJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=judge_prompt_template_fpath,
        )
        cfg.judge_equal_label = "[[A=B]]"
        cfg.judge_not_equal_label = "[[A!=B]]"
        return cfg

    def _create_response(self, id: str, output_item: NeMoGymResponseOutputItem) -> str:
        return NeMoGymResponse(
            id=id,
            created_at=123.0,
            model="judge_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump_json()

    def _msg(self, text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id="msg_id",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )

    @mark.parametrize("limit,answer_length,skip", [(100, 101, True), (100, 100, False), (None, 100001, False)])
    async def test_final_answer_limit_preserves_response(self, config, limit, answer_length, skip):
        assert config.max_answer_chars is None
        if limit is not None:
            config.max_answer_chars = limit
        client = MagicMock(spec=ServerClient)
        reply = MagicMock(ok=True)
        reply.read = AsyncMock(return_value=self._create_response("judge", self._msg("[[A=B]]")))
        client.post = AsyncMock(return_value=reply)
        server = LLMJudgeResourcesServer(config=config, server_client=client)
        response = NeMoGymResponse.model_validate_json(self._create_response("policy", self._msg("é" * answer_length)))
        request = LLMJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=response,
            expected_answer="answer",
        )
        before = request.model_dump()
        result = await server.verify(request)
        assert request.model_dump() == before
        assert result.response.model_dump() == before["response"]
        if skip:
            client.post.assert_not_awaited()
            assert result.reward == 0
            assert result.judge_evaluations == []
            assert result.failure_reason == "final_answer_too_long: 101 characters exceeds 100"
        else:
            client.post.assert_awaited_once()
            assert result.reward == 1
            assert result.failure_reason is None

    async def test_answer_limit_excludes_thinking_and_previous_messages(self, config):
        config.max_answer_chars = 100
        client = MagicMock(spec=ServerClient)
        reply = MagicMock(ok=True)
        reply.read = AsyncMock(return_value=self._create_response("judge", self._msg("[[A=B]]")))
        client.post = AsyncMock(return_value=reply)
        server = LLMJudgeResourcesServer(config=config, server_client=client)
        response_data = json.loads(self._create_response("policy", self._msg("short answer")))
        response_data["output"][:0] = [
            self._msg("previous answer " * 100).model_dump(),
            {
                "id": "thinking",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking " * 10000}],
            },
        ]
        request = LLMJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse.model_validate(response_data),
            expected_answer="answer",
        )
        result = await server.verify(request)
        assert result.reward == 1
        client.post.assert_awaited_once()
        prompt = result.judge_evaluations[0].responses_create_params.input[-1].content
        assert "short answer" in prompt
        assert "thinking" not in prompt
        assert "previous answer" not in prompt

    async def test_answer_limit_precedes_regex_and_counts_all_text_blocks(self, config):
        config.max_answer_chars = 100
        config.response_extract_regex = r"Answer: (.*)"
        client = MagicMock(spec=ServerClient)
        client.post = AsyncMock()
        server = LLMJudgeResourcesServer(config=config, server_client=client)
        message = self._msg("x" * 95)
        message.content.append(NeMoGymResponseOutputText(annotations=[], text="Answer: x", type="output_text"))
        request = LLMJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse.model_validate_json(self._create_response("policy", message)),
            expected_answer="x",
        )
        result = await server.verify(request)
        assert result.reward == 0
        assert result.failure_reason == "final_answer_too_long: 105 characters exceeds 100"
        client.post.assert_not_awaited()

    async def test_exhausted_judge_failure_preserves_response_for_recovery(self, config):
        client = MagicMock(spec=ServerClient)
        client.post = AsyncMock(side_effect=RuntimeError("upstream judge unavailable"))
        server = LLMJudgeResourcesServer(config=config, server_client=client)
        request = LLMJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse.model_validate_json(self._create_response("policy", self._msg("answer"))),
            expected_answer="answer",
        )
        result = await judge_failsafe(server.verify)(request)
        failed = json.loads(result.body)
        assert failed["_ng_failure_class"] == "judge_failed"
        assert "upstream judge unavailable" in failed["_ng_failure_judge_error"]
        assert failed["response"] == request.response.model_dump(mode="json")
        assert failed["expected_answer"] == "answer"
        client.post.assert_awaited_once()

        reply = MagicMock(ok=True)
        reply.read = AsyncMock(return_value=self._create_response("judge", self._msg("[[A=B]]")))
        client.post = AsyncMock(return_value=reply)
        recovered = await server.verify(request)
        assert recovered.reward == 1
        assert recovered.response.model_dump() == request.response.model_dump()

    async def test_verify_equal_then_confirm(self, config: LLMJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        # Default: check_twice_swap = False
        rs = LLMJudgeResourcesServer(config=config, server_client=server_mock)

        # First: judge says equal; Second: judge says equal => reward 1
        post_mock = MagicMock()
        post_mock.read = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)

        # Only the first call is used when check_twice_swap is False
        post_mock.read.side_effect = [
            self._create_response("first", self._msg("some text [[A=B]] trailing")),
        ]

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q: 1+1?"}])
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("It is 2.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert res.expected_answer == "2"
        assert len(res.judge_evaluations) == 1

        # Now enable double-check and ensure two evaluations are returned
        config_twice = config.model_copy(deep=True)
        config_twice.check_twice_swap = True
        rs_twice = LLMJudgeResourcesServer(config=config_twice, server_client=server_mock)

        post_mock2 = MagicMock()
        post_mock2.read = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock2)
        post_mock2.read.side_effect = [
            self._create_response("first", self._msg("[[A=B]]")),
            self._create_response("second", self._msg("[[A=B]]")),
        ]
        req2 = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res2 = await rs_twice.verify(req2)
        assert res2.reward == approx(1.0)
        assert len(res2.judge_evaluations) == 2

    async def test_verify_not_equal_first(self, config: LLMJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        rs = LLMJudgeResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_response("f", self._msg("[[A!=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q: 1+1?"}])
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("It is 3.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res = await rs.verify(req)
        assert res.reward == approx(0.0)
        assert len(res.judge_evaluations) == 1

    async def test_unexpected_judge_output_defaults_to_not_equal(self, config: LLMJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        rs = LLMJudgeResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_response("f", self._msg("no label present")))
        server_mock.post = AsyncMock(return_value=post_mock)

        req = LLMJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="m",
                object="response",
                output=[self._msg("text")],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
            expected_answer="x",
        )
        res = await rs.verify(req)
        assert res.reward == approx(0.0)

    @mark.parametrize("case", ["missing", "reasoning_only", "empty", "whitespace", "no_text", "last_empty"])
    async def test_empty_final_answer_skips_judge_and_preserves_response(
        self, config: LLMJudgeResourcesServerConfig, case: str
    ) -> None:
        config.check_twice_swap = True
        config.check_full_generation_on_fail = True
        client = MagicMock(spec=ServerClient)
        client.post = AsyncMock()
        server = LLMJudgeResourcesServer(config=config, server_client=client)
        outputs = {
            "missing": [],
            "reasoning_only": [
                {
                    "id": "thinking",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "The answer is 42"}],
                }
            ],
            "empty": [self._msg("")],
            "whitespace": [self._msg(" \n\t ")],
            "no_text": [self._msg("").model_copy(update={"content": []})],
            "last_empty": [self._msg("Earlier answer"), self._msg("")],
        }
        response = json.loads(self._create_response("policy", self._msg("")))
        response["output"] = outputs[case]
        request = LLMJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse.model_validate(response),
            expected_answer="42",
            template_metadata={"output_regex": r"ANSWER: (.*)"},
        )
        before = request.model_dump()

        result = await server.verify(request)

        client.post.assert_not_awaited()
        assert result.reward == 0
        assert result.failure_reason == "empty_final_answer"
        assert result.judge_evaluations == []
        assert result.response.model_dump() == before["response"]
        assert request.model_dump() == before

    async def test_swap_fails_uses_configured_reward(self, config: LLMJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.check_twice_swap = True
        cfg.reward_if_swap_fails = -1.0
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.read = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)
        # First pass equal, second pass not equal -> use configured -1.0
        post_mock.read.side_effect = [
            self._create_response("first", self._msg("[[A=B]]")),
            self._create_response("second", self._msg("[[A!=B]]")),
        ]

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q?"}])
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("A")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="B",
        )
        res = await rs.verify(req)
        assert res.reward == approx(-1.0)
        assert len(res.judge_evaluations) == 2

    async def test_per_record_regex_extraction(self, config: LLMJudgeResourcesServerConfig) -> None:
        """Test that template_metadata.output_regex extracts answer correctly."""
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.use_per_record_regex = True
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_response("first", self._msg("[[A=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What is 2+2?"}]
        )
        # Model generates answer wrapped in \boxed{}
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("Let me explain: The answer is \\boxed{4} because 2+2=4.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        # Request with per-record regex in template_metadata
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="4",
            template_metadata={"output_regex": r"\\boxed\{(.*?)\}"},
        )

        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert len(res.judge_evaluations) == 1
        # Verify the regex extraction worked by checking judge was called once
        assert server_mock.post.call_count == 1

    @mark.parametrize("pattern", [r"ANSWER:\s*(.+)", r"(?=The)"])
    async def test_full_generation_rescue_on_extraction_failure(
        self, config: LLMJudgeResourcesServerConfig, pattern: str
    ) -> None:
        """When regex-extracted answer fails, retry with full generation for partial credit."""
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.use_per_record_regex = True
        cfg.check_full_generation_on_fail = True
        cfg.reward_if_full_generation_succeeds = 0.5
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.read = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)
        # First call (extracted answer) fails, second call (full generation) succeeds
        post_mock.read.side_effect = [
            self._create_response("first", self._msg("[[A!=B]]")),
            self._create_response("second", self._msg("[[A=B]]")),
        ]

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What is 2+2?"}]
        )
        # Model output contains an answer even if extraction misses or matches empty text.
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("The final answer is clearly 4")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        # Both a missing match and an empty match must retain full-generation rescue.
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="4",
            template_metadata={"output_regex": pattern},
        )

        res = await rs.verify(req)
        assert res.reward == approx(0.5)  # Partial credit
        assert len(res.judge_evaluations) == 2  # Both passes recorded
        assert server_mock.post.call_count == 2

    async def test_extraction_length_threshold_skips_regex(self, config: LLMJudgeResourcesServerConfig) -> None:
        """Long expected answers skip regex extraction and use full generation."""
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.use_per_record_regex = True
        cfg.extraction_length_threshold = 50  # 50 characters
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_response("first", self._msg("[[A=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Explain photosynthesis."}]
        )
        # Long answer that exceeds threshold
        long_answer = "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose."
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg(long_answer)],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        # Even though regex is provided, it should be ignored due to length threshold
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer=long_answer,  # >50 chars, will skip regex
            template_metadata={"output_regex": r"\\boxed\{(.*?)\}"},  # Should be ignored
        )

        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert len(res.judge_evaluations) == 1
        # Verify only one judge call (no second pass due to length threshold)
        assert server_mock.post.call_count == 1

    def test_question_extracted_from_multimodal_user_content(self) -> None:
        """A vision row's user turn is a content list, not a string.

        Shape mirrors a prepared HLE vision row: an ``input_text`` block carrying the
        question alongside an ``input_image`` block carrying a base64 data URI. Without
        list handling the judge receives an empty question and grades on nothing.
        """
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": "Answer the question."},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Which piece delivers mate?"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                            "detail": "high",
                        },
                    ],
                },
            ]
        )

        question = _extract_question_text(params, None)

        assert question == "Which piece delivers mate?"
        # The image must not leak into the judge prompt: base64 payloads are large and
        # would blow the judge's context while telling it nothing.
        assert "base64" not in question

    def test_question_extraction_from_string_user_content_is_unchanged(self) -> None:
        """The text-only path is untouched by multimodal handling: last user turn wins."""
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": "  second question  "},
            ]
        )

        assert _extract_question_text(params, None) == "second question"

    def test_question_extraction_returns_empty_when_no_text_blocks(self) -> None:
        """An image-only user turn has no text to extract, so the empty-string contract holds."""
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "auto",
                        }
                    ],
                }
            ]
        )

        assert _extract_question_text(params, None) == ""


class TestJudgeParsing:
    @fixture
    def server(self) -> LLMJudgeResourcesServer:
        cfg = LLMJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=str(
                Path(__file__).resolve().parents[1] / "prompt_templates/equivalence_llm_judge.txt"
            ),
            judge_equal_label="Judgement: yes",
            judge_not_equal_label="Judgement: no",
        )
        return LLMJudgeResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    async def _judge(self, server, text: str | None, status: str = "completed"):
        output = []
        if text is not None:
            output = [
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ]
        response = NeMoGymResponse(
            id="judge_resp",
            created_at=0.0,
            model="judge",
            object="response",
            output=output,
            status=status,
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        post = MagicMock()
        post.read = AsyncMock(return_value=response.model_dump_json())
        server.server_client.post = AsyncMock(return_value=post)
        return await server._generate_judge_evaluation(question="Q?", expected_answer="B", generated_answer="A")

    @mark.parametrize(
        "text,status,verdict,issues",
        [
            ("Judgement: yes\nConfidence: 95%", "completed", "Judgement: yes", []),
            ("Judgement: no\nJudgement: yes", "completed", "Judgement: yes", ["conflicting_verdicts"]),
            ("Judgement: yes\nJudgement: no", "completed", "Judgement: no", ["conflicting_verdicts"]),
            ("Judgement: yes\nJudgement: yes", "completed", "Judgement: yes", ["repeated_verdict"]),
            ("Unable to assess.", "completed", None, ["no_verdict"]),
            ("Reasoning: the answer is", "incomplete", None, ["truncated_judge_output", "no_verdict"]),
            ("Judgement: yes\nConfid", "incomplete", "Judgement: yes", []),
            (None, "completed", None, ["unparseable_judge_output"]),
            (None, "incomplete", None, ["truncated_judge_output", "unparseable_judge_output"]),
        ],
    )
    async def test_verdict_and_parsing_issues(self, server, text, status, verdict, issues) -> None:
        is_equal, record = await self._judge(server, text, status)
        assert is_equal is (verdict == "Judgement: yes")
        assert record.verdict_label == verdict
        assert record.judgement_parsing_issues == issues

        serialized = jsonable_encoder(record)
        if issues:
            assert serialized["judgement_parsing_issues"] == issues
        else:
            assert "judgement_parsing_issues" not in serialized

        # Old rollouts have the judge response but no parsing diagnostics.
        legacy = deepcopy(serialized)
        legacy.pop("judgement_parsing_issues", None)
        expected_metrics = {"judgement_parsing_issue_rate": float(bool(issues))}
        expected_metrics.update({f"judgement_parsing_issue_rate/{issue}": 1.0 for issue in issues})
        assert server.compute_metrics([[{"judge_evaluations": [legacy]}]]) == expected_metrics

    @mark.parametrize(
        "text,verdict,issues",
        [
            ("The answer is INCORRECT.", "INCORRECT", []),
            ("INCORRECT INCORRECT", "INCORRECT", ["repeated_verdict"]),
            ("Initially INCORRECT, finally CORRECT", "CORRECT", ["conflicting_verdicts"]),
            ("Initially CORRECT, finally INCORRECT", "INCORRECT", ["conflicting_verdicts"]),
        ],
    )
    async def test_overlapping_labels(self, server, text, verdict, issues) -> None:
        server.config.judge_equal_label = "CORRECT"
        server.config.judge_not_equal_label = "INCORRECT"
        is_equal, record = await self._judge(server, text)
        assert is_equal is (verdict == "CORRECT")
        assert record.verdict_label == verdict
        assert record.judgement_parsing_issues == issues

    @mark.parametrize(
        "equal_label,not_equal_label,text,verdict,issues",
        [
            ("OK", "OKAY", "OKAY", "OKAY", []),
            ("OKAY", "OK", "OKAY", "OKAY", []),
            ("AB", "BCD", "ABCD", "AB", []),
            ("AB", "BCD", "ABCD BCD", "BCD", ["conflicting_verdicts"]),
        ],
    )
    async def test_verdict_uses_same_matches_as_diagnostics(
        self, server, equal_label, not_equal_label, text, verdict, issues
    ) -> None:
        server.config.judge_equal_label = equal_label
        server.config.judge_not_equal_label = not_equal_label
        is_equal, record = await self._judge(server, text)
        assert is_equal is (verdict == equal_label)
        assert record.verdict_label == verdict
        assert record.judgement_parsing_issues == issues

    @mark.parametrize(
        "text,equal_label,not_equal_label,expected",
        [
            ("[[A=B]] then [[A!=B]]", "[[A=B]]", "[[A!=B]]", ["[[A=B]]", "[[A!=B]]"]),
            ("A+B A?B", "A+B", "A?B", ["A+B", "A?B"]),
            ("AAB", "A+B", "A?B", []),
            ("anything", "", "", []),
            ("NO", "", "NO", ["NO"]),
            ("YES", "YES", "", ["YES"]),
        ],
    )
    def test_literal_and_empty_labels(self, text, equal_label, not_equal_label, expected) -> None:
        assert _match_verdict_labels(text, equal_label, not_equal_label) == expected

    @staticmethod
    def _rollout(*issues: list[str]) -> dict:
        return {"judge_evaluations": [{"judgement_parsing_issues": value} for value in issues]}

    def test_issue_rates_count_rollouts_and_exclude_unjudged_rows(self, server) -> None:
        tasks = [
            [self._rollout(["no_verdict"], ["no_verdict"]), self._rollout([], [])],
            [self._rollout(["truncated_judge_output", "no_verdict"])],
            [{"reward": 0.0}, {"reward": 0.0, "judge_evaluations": []}],
            [{"judge_evaluations": [{"verdict_label": "JUDGE_ERROR", "response": None}]}],
            [{"judge_evaluations": [{}]}],
        ]
        assert server.compute_metrics(tasks) == {
            "judgement_parsing_issue_rate": approx(2 / 3),
            "judgement_parsing_issue_rate/no_verdict": approx(2 / 3),
            "judgement_parsing_issue_rate/truncated_judge_output": approx(1 / 3),
        }

    @mark.parametrize(
        "tasks,expected",
        [
            ([[{"judge_evaluations": [{"judgement_parsing_issues": []}]}]], {"judgement_parsing_issue_rate": 0.0}),
            ([[{"judge_evaluations": [{}]}]], {}),
            ([[{"judge_evaluations": [{"verdict_label": "JUDGE_ERROR", "response": None}]}]], {}),
            ([[{"reward": 0.0}]], {}),
            ([], {}),
        ],
    )
    def test_clean_or_unjudged_run_metrics(self, server, tasks, expected) -> None:
        assert server.compute_metrics(tasks) == expected

    async def test_legacy_aggregation_preserves_saved_scores(self, server) -> None:
        _, record = await self._judge(server, "Judgement: no\nJudgement: yes")
        legacy = jsonable_encoder(record)
        legacy.pop("judgement_parsing_issues")
        legacy["verdict_label"] = "Judgement: no"
        rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 0.0, "judge_evaluations": [legacy]}]
        original = deepcopy(rows)

        metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))

        assert metrics.agent_metrics["mean/reward"] == 0.0
        assert metrics.key_metrics["judgement_parsing_issue_rate"] == 1.0
        assert metrics.agent_metrics["judgement_parsing_issue_rate/conflicting_verdicts"] == 1.0
        assert rows == original

    @mark.parametrize("include_rate", [False, True])
    def test_key_metrics(self, server, include_rate) -> None:
        metrics = {"mean/reward": 0.31, "std/reward": 0.02}
        expected = {"mean/reward": 0.31}
        if include_rate:
            metrics.update({"judgement_parsing_issue_rate": 0.4, "judgement_parsing_issue_rate/no_verdict": 0.4})
            expected["judgement_parsing_issue_rate"] = 0.4
        assert server.get_key_metrics(metrics) == expected
