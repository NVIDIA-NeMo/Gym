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
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.encoders import jsonable_encoder
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.equivalence_llm_judge.app import (
    LLMJudgeResourcesServer,
    LLMJudgeResourcesServerConfig,
    LLMJudgeVerifyRequest,
    _extract_question_text,
)


class TestApp:
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

    async def test_missing_assistant_text_uses_configured_failure_message(
        self, config: LLMJudgeResourcesServerConfig
    ) -> None:
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.msg_extraction_failure = "[CUSTOM EXTRACTION FAILURE]"
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._create_response("f", self._msg("[[A!=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        req = LLMJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="m",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
            expected_answer="x",
        )

        res = await rs.verify(req)

        judge_prompt = res.judge_evaluations[0].responses_create_params.input[-1].content
        assert cfg.msg_extraction_failure in judge_prompt
        assert res.reward == approx(0.0)

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

    async def test_full_generation_rescue_on_extraction_failure(self, config: LLMJudgeResourcesServerConfig) -> None:
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
        # Model output: correct answer is in full text but regex won't match properly
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

        # Regex pattern that won't match the model output
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="4",
            template_metadata={"output_regex": r"ANSWER:\s*(.+)"},  # Won't match
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


class TestVerdictParsing:
    """The verdict is the label whose last occurrence ends latest in the judge's message.

    Judges reason out loud before committing, so they routinely name the losing verdict
    on the way to the winning one. Reading the first occurrence graded on that aside.
    """

    def _config(self, equal_label: str, not_equal_label: str) -> LLMJudgeResourcesServerConfig:
        cfg = LLMJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=str(
                Path(__file__).resolve().parents[1] / "prompt_templates/equivalence_llm_judge.txt"
            ),
        )
        cfg.judge_equal_label = equal_label
        cfg.judge_not_equal_label = not_equal_label
        return cfg

    async def _judge(self, cfg: LLMJudgeResourcesServerConfig, judge_text: str) -> tuple[bool, str | None]:
        """Run one judge pass over ``judge_text`` and return (is_equal, verdict_label)."""
        server_mock = MagicMock(spec=ServerClient)
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.read = AsyncMock(
            return_value=NeMoGymResponse(
                id="judge_resp",
                created_at=0.0,
                model="judge_model",
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id="msg_id",
                        content=[NeMoGymResponseOutputText(annotations=[], text=judge_text, type="output_text")],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ).model_dump_json()
        )
        server_mock.post = AsyncMock(return_value=post_mock)

        is_equal, record = await rs._generate_judge_evaluation(
            question="Q?", expected_answer="B", generated_answer="A"
        )
        return is_equal, record.verdict_label

    # --- HLE labels: the case this change exists for -----------------------------

    async def test_verdict_after_contrary_reasoning_wins(self) -> None:
        cfg = self._config("Judgement: yes", "Judgement: no")
        text = (
            "Reasoning: If the response had used different units this would be\n"
            "Judgement: no, but the values agree exactly.\n"
            "Judgement: yes\n"
            "Confidence: 95%"
        )
        assert await self._judge(cfg, text) == (True, "Judgement: yes")

    async def test_trailing_not_equal_verdict_wins(self) -> None:
        cfg = self._config("Judgement: yes", "Judgement: no")
        text = (
            "Reasoning: A sloppier grader might say Judgement: yes here, but the\n"
            "extracted answer is off by an order of magnitude.\n"
            "Judgement: no\n"
            "Confidence: 90%"
        )
        assert await self._judge(cfg, text) == (False, "Judgement: no")

    # --- Overlapping labels: lc_judge grades with CORRECT / INCORRECT ------------

    async def test_incorrect_is_not_read_as_correct(self) -> None:
        """Regression guard: "INCORRECT" contains "CORRECT" one character later.

        Ranking by match start would hand every INCORRECT verdict a passing grade.
        """
        cfg = self._config("CORRECT", "INCORRECT")
        assert await self._judge(cfg, "INCORRECT") == (False, "INCORRECT")
        assert await self._judge(cfg, "The candidate answer is INCORRECT.") == (False, "INCORRECT")

    async def test_overlapping_labels_still_honour_the_last_verdict(self) -> None:
        cfg = self._config("CORRECT", "INCORRECT")
        assert await self._judge(cfg, "At first glance INCORRECT, but on reflection: CORRECT") == (True, "CORRECT")
        assert await self._judge(cfg, "Initially CORRECT, however the units are wrong: INCORRECT") == (
            False,
            "INCORRECT",
        )

    # --- Degenerate judge output -------------------------------------------------

    async def test_missing_verdict_is_not_equal_and_unlabelled(self) -> None:
        cfg = self._config("Judgement: yes", "Judgement: no")
        assert await self._judge(cfg, "I could not determine an answer.") == (False, None)

    async def test_single_label_present_is_used_regardless_of_position(self) -> None:
        cfg = self._config("[[A=B]]", "[[A!=B]]")
        assert await self._judge(cfg, "[[A=B]] trailing commentary") == (True, "[[A=B]]")
        assert await self._judge(cfg, "[[A!=B]] trailing commentary") == (False, "[[A!=B]]")


class TestJudgementParsingIssueRate:
    """The per-row flags roll up into aggregate metrics.

    Without this the flags only exist in the rollout JSONL, so a run whose judge
    silently stopped grading looks exactly like a run the model failed.
    """

    def _server(self) -> LLMJudgeResourcesServer:
        cfg = LLMJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=str(
                Path(__file__).resolve().parents[1] / "prompt_templates/equivalence_llm_judge.txt"
            ),
        )
        return LLMJudgeResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    @staticmethod
    def _rollout(*evaluation_issues: list[str]) -> dict:
        """One rollout, one entry per judge pass. Clean passes omit the key, as served."""
        return {
            "reward": 0.0,
            "judge_evaluations": [
                {"judgement_parsing_issues": issues} if issues else {} for issues in evaluation_issues
            ],
        }

    def test_clean_run_reports_a_zero_rate_rather_than_nothing(self) -> None:
        """0.0 and a missing key mean different things: "checked" versus "not measured"."""
        metrics = self._server().compute_metrics([[self._rollout([])], [self._rollout([])]])
        assert metrics == {"judgement_parsing_issue_rate": 0.0}

    def test_rate_is_the_share_of_rollouts_with_any_issue(self) -> None:
        tasks = [
            [self._rollout(["no_verdict"])],
            [self._rollout([])],
            [self._rollout(["truncated_judge_output", "no_verdict"])],
            [self._rollout([])],
        ]
        metrics = self._server().compute_metrics(tasks)
        assert metrics["judgement_parsing_issue_rate"] == 0.5
        assert metrics["judgement_parsing_issue_rate/no_verdict"] == 0.5
        assert metrics["judgement_parsing_issue_rate/truncated_judge_output"] == 0.25

    def test_a_rollout_judged_twice_still_counts_once(self) -> None:
        """Swap and rescue configs make two judge calls per row.

        Counting evaluations rather than rollouts would let a two-pass benchmark
        report a rate above 1.0.
        """
        tasks = [[self._rollout(["no_verdict"], ["no_verdict"])], [self._rollout([], [])]]
        metrics = self._server().compute_metrics(tasks)
        assert metrics["judgement_parsing_issue_rate"] == 0.5
        assert metrics["judgement_parsing_issue_rate/no_verdict"] == 0.5

    def test_rows_that_never_reached_the_judge_are_not_in_the_denominator(self) -> None:
        """A row that errored before grading is not evidence about the judge."""
        tasks = [[self._rollout(["no_verdict"])], [{"reward": 0.0}], [{"reward": 0.0, "judge_evaluations": []}]]
        assert self._server().compute_metrics(tasks)["judgement_parsing_issue_rate"] == 1.0

    def test_a_run_with_no_judged_rows_reports_nothing(self) -> None:
        assert self._server().compute_metrics([[{"reward": 0.0}]]) == {}

    def test_rate_is_promoted_into_key_metrics(self) -> None:
        """The default key set is `mean/*` only, which would bury the rate.

        It is the one number saying whether the accuracy beside it can be trusted, so
        it has to appear where the headline numbers are read, not only in the full
        metrics JSON.
        """
        agent_metrics = {
            "mean/accuracy": 0.31,
            "std/accuracy": 0.02,
            "judgement_parsing_issue_rate": 0.4,
            "judgement_parsing_issue_rate/no_verdict": 0.4,
        }
        key = self._server().get_key_metrics(agent_metrics)
        assert key == {"mean/accuracy": 0.31, "judgement_parsing_issue_rate": 0.4}

    def test_key_metrics_without_a_rate_is_just_the_default_set(self) -> None:
        """Other servers' metrics must not gain a phantom key."""
        assert self._server().get_key_metrics({"mean/accuracy": 0.31, "std/accuracy": 0.02}) == {"mean/accuracy": 0.31}


class TestJudgementParsingIssues:
    """Verdicts that do not parse cleanly are recorded on the row that they graded.

    A judge that stops emitting parseable verdicts is otherwise indistinguishable
    from a model that got every question wrong: both are a column of zeros.
    """

    def _config(self, equal_label: str = "Judgement: yes", not_equal_label: str = "Judgement: no"):
        cfg = LLMJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=str(
                Path(__file__).resolve().parents[1] / "prompt_templates/equivalence_llm_judge.txt"
            ),
        )
        cfg.judge_equal_label = equal_label
        cfg.judge_not_equal_label = not_equal_label
        return cfg

    def _server(self, cfg) -> LLMJudgeResourcesServer:
        return LLMJudgeResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    async def _judge(self, rs, text: str | None, status: str = "completed"):
        """One judge pass. ``text=None`` produces a response with no message output."""
        output = []
        if text is not None:
            output = [
                NeMoGymResponseOutputMessage(
                    id="msg_id",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ]
        pm = MagicMock()
        pm.read = AsyncMock(
            return_value=NeMoGymResponse(
                id="judge_resp",
                created_at=0.0,
                model="judge_model",
                object="response",
                output=output,
                status=status,
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ).model_dump_json()
        )
        rs.server_client.post = AsyncMock(return_value=pm)
        return await rs._generate_judge_evaluation(question="Q?", expected_answer="B", generated_answer="A")

    async def test_clean_verdict_is_not_flagged(self) -> None:
        rs = self._server(self._config())
        is_equal, rec = await self._judge(rs, "Reasoning: they match.\nJudgement: yes\nConfidence: 95%")
        assert is_equal is True
        assert rec.judgement_parsing_issues == []

    async def test_no_verdict_is_flagged(self) -> None:
        rs = self._server(self._config())
        is_equal, rec = await self._judge(rs, "I am unable to assess this answer.")
        # Still scores not-equal, but no longer silently: the row says why.
        assert (is_equal, rec.verdict_label) == (False, None)
        assert rec.judgement_parsing_issues == ["no_verdict"]

    async def test_conflicting_verdicts_are_flagged_but_still_scored(self) -> None:
        rs = self._server(self._config())
        is_equal, rec = await self._judge(rs, "Judgement: no\nOn reflection:\nJudgement: yes")
        # The last-occurrence rule still decides; the flag records that it had to.
        assert (is_equal, rec.verdict_label) == (True, "Judgement: yes")
        assert rec.judgement_parsing_issues == ["conflicting_verdicts"]

    async def test_repeated_verdict_is_flagged(self) -> None:
        rs = self._server(self._config())
        _, rec = await self._judge(rs, "Judgement: yes\n...restating...\nJudgement: yes")
        assert rec.judgement_parsing_issues == ["repeated_verdict"]

    async def test_truncated_judge_output_is_flagged_when_it_cost_the_verdict(self) -> None:
        rs = self._server(self._config())
        _, rec = await self._judge(rs, "Reasoning: the extracted answer is", status="incomplete")
        # Truncation severed the verdict line, so both facts are recorded.
        assert rec.judgement_parsing_issues == ["truncated_judge_output", "no_verdict"]

    async def test_truncation_after_the_verdict_is_not_flagged(self) -> None:
        """The HLE judge prompt puts `Confidence:` after `Judgement:`.

        A judge cut off on that trailing line still committed to a verdict, and the row
        is graded correctly. Flagging it would count good rows as problems and inflate
        judgement_parsing_issue_rate -- exactly the signal the rate exists to give.
        """
        rs = self._server(self._config())
        is_equal, rec = await self._judge(rs, "Reasoning: they match.\nJudgement: yes\nConfid", status="incomplete")
        assert (is_equal, rec.verdict_label) == (True, "Judgement: yes")
        assert rec.judgement_parsing_issues == []

    async def test_unparseable_judge_output_is_flagged(self) -> None:
        rs = self._server(self._config())
        is_equal, rec = await self._judge(rs, None)
        assert (is_equal, rec.verdict_label) == (False, None)
        assert rec.judgement_parsing_issues == ["unparseable_judge_output"]

    async def test_truncation_that_severed_the_whole_message_records_both(self) -> None:
        rs = self._server(self._config())
        _, rec = await self._judge(rs, None, status="incomplete")
        assert rec.judgement_parsing_issues == ["truncated_judge_output", "unparseable_judge_output"]

    async def test_overlapping_labels_do_not_produce_phantom_conflicts(self) -> None:
        """lc_judge grades CORRECT / INCORRECT; "INCORRECT" also contains "CORRECT".

        A naive text.count() would flag every ordinary not-equal verdict as a
        conflict, which would bury the real signal under 100% false positives.
        """
        rs = self._server(self._config(equal_label="CORRECT", not_equal_label="INCORRECT"))
        is_equal, rec = await self._judge(rs, "The candidate answer is INCORRECT.")
        assert (is_equal, rec.verdict_label) == (False, "INCORRECT")
        assert rec.judgement_parsing_issues == []

    async def test_clean_record_omits_the_issues_key_entirely(self) -> None:
        """A clean row carries no issues field at all, rather than an empty list.

        Most rows are clean; an always-present empty list would be a column in the
        rollout JSONL that never carries information.
        """
        rs = self._server(self._config())
        _, clean = await self._judge(rs, "Judgement: yes")
        assert "judgement_parsing_issues" not in jsonable_encoder(clean)
        # The attribute still exists on the object -- only serialisation drops it.
        assert clean.judgement_parsing_issues == []

        _, flagged = await self._judge(rs, "no verdict here")
        assert jsonable_encoder(flagged)["judgement_parsing_issues"] == ["no_verdict"]

    async def test_flags_are_recorded_per_record_and_not_shared_between_rows(self) -> None:
        """Each evaluation owns its own list.

        The flags are the only surviving record of a broken judge, so a list shared
        between records -- the classic mutable-default bug -- would smear one row's
        issues across every later row and make the field useless for slicing.
        """
        rs = self._server(self._config())
        records = [(await self._judge(rs, "no verdict here"))[1] for _ in range(50)]
        assert all(rec.judgement_parsing_issues == ["no_verdict"] for rec in records)

        _, clean = await self._judge(rs, "Judgement: yes")
        assert clean.judgement_parsing_issues == []
