# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.imo_proofbench_judge.app import (
    PASS_THRESHOLD,
    ImoProofBenchJudgeConfig,
    ImoProofBenchJudgeServer,
    ImoProofBenchVerifyRequest,
    _strip_thinking_traces,
    extract_text_from_response,
    math_equal,
    parse_judgement_verdict,
    parse_points,
    search_boxed,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestStripThinkingTraces:
    def test_strips_paired_think(self):
        assert _strip_thinking_traces("<think>r</think>Answer") == "Answer"

    def test_strips_paired_thinking(self):
        assert _strip_thinking_traces("<thinking>r</thinking>Result") == "Result"

    def test_strips_unpaired_closing_think(self):
        # No opening tag: model started reasoning then closed it.
        assert _strip_thinking_traces("reasoning</think>Final") == "Final"

    def test_strips_unpaired_closing_thinking(self):
        assert _strip_thinking_traces("reasoning</thinking>Final") == "Final"

    def test_no_tags_passthrough(self):
        assert _strip_thinking_traces("plain") == "plain"

    def test_multiline(self):
        assert _strip_thinking_traces("<think>\nl1\nl2\n</think>\nA") == "A"

    def test_empty_string(self):
        assert _strip_thinking_traces("") == ""


class TestSearchBoxed:
    """Mirror Skills' `nemo_skills.evaluation.math_grader.search_boxed`."""

    def test_simple(self):
        assert search_boxed("answer is \\boxed{42}") == "42"

    def test_rightmost_wins(self):
        # Skills walks back from the LAST `\\boxed` — final boxed answer wins
        # over any earlier `\\boxed` expressions in the reasoning trace.
        assert search_boxed("first \\boxed{wrong} then \\boxed{right}") == "right"

    def test_nested_braces(self):
        # Brace-nesting bookkeeping — `\\boxed{f(x)=2x+c}` returns the inner
        # expression unchanged, including any nested braces.
        assert search_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_no_boxed(self):
        assert search_boxed("just text, no boxed expression") is None

    def test_unbalanced_braces(self):
        # Open `\\boxed{` without a closing brace returns None.
        assert search_boxed("\\boxed{unclosed") is None

    def test_empty(self):
        assert search_boxed("") is None
        assert search_boxed(None) is None  # type: ignore[arg-type]


class TestParseJudgementVerdict:
    """Mirror Skills' 3-format `is_correct_judgement` priority."""

    def test_format1_yes(self):
        assert parse_judgement_verdict("Judgement: Yes") is True

    def test_format1_no(self):
        assert parse_judgement_verdict("**Judgement**: No") is False

    def test_format1_priority_over_points(self):
        # Format 1 wins over Format 3 when both appear.
        assert parse_judgement_verdict("Judgement: Yes\n<points>1 out of 7</points>") is True

    def test_format2_correct(self):
        assert parse_judgement_verdict("Reasoning. \\boxed{Correct}") is True

    def test_format2_incorrect(self):
        assert parse_judgement_verdict("Analysis. \\boxed{Incorrect}") is False

    def test_format2_priority_over_points(self):
        assert parse_judgement_verdict("\\boxed{Correct} <points>0 out of 7</points>") is True

    def test_format3_pass(self):
        assert parse_judgement_verdict("<points>7 out of 7</points>") is True
        assert parse_judgement_verdict("<points>6 out of 7</points>") is True

    def test_format3_fail(self):
        assert parse_judgement_verdict("<points>1 out of 7</points>") is False
        assert parse_judgement_verdict("<points>0 out of 7</points>") is False

    def test_no_format_match(self):
        assert parse_judgement_verdict("inconclusive verdict") is None

    def test_empty(self):
        assert parse_judgement_verdict("") is None
        assert parse_judgement_verdict(None) is None  # type: ignore[arg-type]


class TestParsePoints:
    def test_seven(self):
        assert parse_points("Some reasoning. <points>7 out of 7</points>") == 7

    def test_six(self):
        assert parse_points("<points>6 out of 7</points>") == 6

    def test_one(self):
        assert parse_points("<points>1 out of 7</points>") == 1

    def test_zero(self):
        assert parse_points("<points>0 out of 7</points>") == 0

    def test_case_insensitive(self):
        assert parse_points("<POINTS>6 OUT OF 7</POINTS>") == 6

    def test_extra_whitespace(self):
        assert parse_points("<points>  6  out of 7  </points>") == 6

    def test_first_block_wins(self):
        # If somehow two blocks: first one wins (matches Skills' re.search).
        assert parse_points("<points>6 out of 7</points> ... <points>0 out of 7</points>") == 6

    def test_no_block(self):
        assert parse_points("nothing relevant") is None

    def test_empty(self):
        assert parse_points("") is None

    def test_none_handled(self):
        assert parse_points(None) is None  # type: ignore[arg-type]

    def test_non_integer(self):
        # Regex requires \d+, so non-integers won't match.
        assert parse_points("<points>three out of 7</points>") is None


class TestMathEqual:
    """Tests for the sympy-based equivalence helper that mirrors Skills'
    ``math_grader.math_equal`` and powers the ``symbolic_correct`` path."""

    def test_literal_match(self):
        assert math_equal("3", "3") is True
        assert math_equal("180", " 180 ") is True

    def test_none_or_empty(self):
        assert math_equal("3", None) is False
        assert math_equal("3", "") is False

    def test_numeric_equivalence_with_latex_expression(self):
        """The case that motivated this whole helper: model emits a numeric
        value that's symbolically equal to a LaTeX expected_answer.
        ``binom(20,10)^2 - binom(20,9)^2 = 5924217936`` — Gym's old literal-
        only prefill missed all 14 PB-Basic-012 hits Skills caught."""
        assert math_equal(r"$\binom{20}{10}^2 - \binom{20}{9}^2$ ", "5924217936") is True

    def test_latex_to_latex_equivalence(self):
        # Skills handles fractions / equivalent LaTeX forms.
        assert math_equal(r"$\frac{1}{2}$", "0.5") is True

    def test_unequal_numbers(self):
        assert math_equal("3", "4") is False

    def test_unrelated_strings(self):
        assert math_equal("hello", "world") is False

    def test_handles_garbage_without_raising(self):
        # math_verify's parser raises on some pathological inputs; the
        # wrapper must swallow and return False (matches Skills' best-
        # effort behaviour — the rollout falls through to the LLM judge).
        assert math_equal("3", r"\boxed{") is False


# ---------------------------------------------------------------------------
# extract_text_from_response
# ---------------------------------------------------------------------------


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


def _empty_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _make_parsed_reasoning_response(
    reasoning_text: str,
    message_text: str | None,
) -> NeMoGymResponse:
    """Build a response shaped like vLLM's `--reasoning-parser` output.

    `output[0]` is a separate `reasoning` item (model thinking, no tags),
    and `output[1]` is the post-think message. When `message_text` is
    None the rollout truncated mid-reasoning and no message was emitted.
    """
    output = [
        NeMoGymResponseReasoningItem(
            id="rsn",
            summary=[NeMoGymSummary(text=reasoning_text, type="summary_text")],
            type="reasoning",
        )
    ]
    if message_text is not None:
        output.append(
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=message_text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        )
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class TestExtractText:
    def test_plain(self):
        assert extract_text_from_response(_make_response("Hello")) == "Hello"

    def test_strips_think(self):
        assert extract_text_from_response(_make_response("<think>x</think>Hi")) == "Hi"

    def test_no_strip(self):
        text = "<think>x</think>Hi"
        assert extract_text_from_response(_make_response(text), strip_thinking=False) == text

    def test_empty_output(self):
        assert extract_text_from_response(_empty_response()) == ""


# ---------------------------------------------------------------------------
# Server.verify
# ---------------------------------------------------------------------------


class TestServer:
    @fixture
    def config(self) -> ImoProofBenchJudgeConfig:
        return ImoProofBenchJudgeConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
                max_output_tokens=4096,
                temperature=0.0,
                top_p=1.0,
            ),
            use_chat_completions_for_judge=True,
        )

    @fixture
    def chat_config(self, config) -> ImoProofBenchJudgeConfig:
        config.use_chat_completions_for_judge = True
        return config

    @fixture
    def responses_config(self, config) -> ImoProofBenchJudgeConfig:
        config.use_chat_completions_for_judge = False
        return config

    def _judge_chat_response(self, body: str) -> dict:
        return NeMoGymChatCompletion(
            id="judge",
            created=0,
            model="judge_model",
            object="chat.completion",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content=body),
                )
            ],
        ).model_dump()

    def _judge_responses_response(self, body: str) -> dict:
        return _make_response(body).model_dump()

    @pytest.mark.asyncio
    async def test_verify_correct_seven(self, chat_config):
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(
            return_value=self._judge_chat_response("Excellent. <points>7 out of 7</points>")
        )
        server_mock.post = AsyncMock(return_value=response_mock)

        # Model output ending in `\boxed{X}` — the verifier mirrors Skills'
        # `eval_type=math` extraction and sends just `X` to the judge.
        model_response = _make_response("<think>reasoning</think>The answer is X. \\boxed{X}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="Prove X.",
            reference_solution="Proof by induction.",
            rubric="7: rigorous; 6: minor gaps; 1: partial; 0: incorrect.",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.judge_correct == approx(1.0)
        assert result.judge_points == 7
        assert result.no_judge_score == approx(0.0)
        # extracted_answer is the contents of the rightmost \boxed{}, not
        # the full message text.
        assert result.extracted_answer == "X"

    @pytest.mark.asyncio
    async def test_verify_correct_six(self, chat_config):
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>6 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("<think>r</think>partial proof attempt \\boxed{partial}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.judge_points == 6

    @pytest.mark.asyncio
    async def test_verify_partial_one(self, chat_config):
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>1 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("</think>weak attempt \\boxed{guess}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.judge_correct == approx(0.0)
        assert result.judge_points == 1

    @pytest.mark.asyncio
    async def test_verify_incorrect_zero(self, chat_config):
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>0 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("</think>nope \\boxed{wrong}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.judge_points == 0

    @pytest.mark.asyncio
    async def test_verify_judgement_yes_format(self, chat_config):
        """Format 1: `Judgement: Yes` — the highest-priority parser path
        (matches Skills' `is_correct_judgement`). Used when the judge emits
        the legacy boolean format alongside or instead of `<points>`.
        """
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(
            return_value=self._judge_chat_response("**Judgement**: Yes\n\n<points>1 out of 7</points>")
        )
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("</think>solution \\boxed{42}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        # Format 1 (Judgement: Yes) wins over Format 3 (1 out of 7) in
        # Skills' priority order, so this rollout counts as correct even
        # though the points block alone would be incorrect.
        assert result.reward == approx(1.0)
        assert result.judge_correct == approx(1.0)
        assert result.judge_points == 1  # legacy field still parsed

    @pytest.mark.asyncio
    async def test_verify_boxed_correct_format(self, chat_config):
        """Format 2: `\\boxed{Correct}` — second-priority parser path
        (matches Skills' `is_correct_judgement`)."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("Reasoning... \\boxed{Correct}"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("</think>my proof \\boxed{42}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.judge_correct == approx(1.0)

    @pytest.mark.asyncio
    async def test_verify_no_boxed_skips_judge(self, chat_config):
        """No `\\boxed{...}` ⇒ Skills' `prefill_judgement` synthesises
        "Reasoning: No answer was provided.\\nJudgement: No" and skips the
        LLM judge. We mirror that synthetic verdict (parseable as
        Judgement: No) so the rollout is graded-incorrect rather than
        unjudged."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        server_mock.post = AsyncMock()  # should NOT be called

        model_response = _make_response("Let me think... not sure")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.extracted_answer == ""
        assert result.judge_correct == approx(0.0)
        assert result.judge_points is None
        # Synthetic "Judgement: No" parses as a verdict, so this is NOT
        # counted toward no_judge_score (matches Skills' accounting).
        assert result.no_judge_score == approx(0.0)
        assert result.judge_output == "Reasoning: No answer was provided.\nJudgement: No"
        server_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_prefill_exact_match_skips_judge(self, chat_config):
        """Skills' `prefill_judgement` shortcut: when the extracted boxed
        answer literally matches `expected_answer` (str.strip equality),
        the LLM judge is skipped and a synthetic "Judgement: Yes" verdict
        is returned. This is the source of Skills' headline pass rate on
        IMO-ProofBench problems with short numerical answers."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        server_mock.post = AsyncMock()  # judge must NOT be called

        model_response = _make_response("</think>after lots of work \\boxed{180}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
            expected_answer=" 180  ",  # whitespace-equivalent match
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.judge_correct == approx(1.0)
        assert result.extracted_answer == "180"
        assert result.judge_output == "Reasoning: The two answers are identical.\nJudgement: Yes"
        assert result.judge_points is None
        assert result.no_judge_score == approx(0.0)
        server_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_prefill_no_match_calls_judge(self, chat_config):
        """When extracted != expected, prefill is skipped and the LLM judge
        runs. Confirms the shortcut only fires on literal match."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>0 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("</think>I think \\boxed{42}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
            expected_answer="180",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.judge_points == 0
        server_mock.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_symbolic_match_skips_judge(self, chat_config):
        """The case the post-hoc analysis surfaced: the model emits a
        numeric value (``5924217936``) that's symbolically equivalent to
        a LaTeX expected answer. Skills catches this via ``math_equal``
        and counts it as ``symbolic_correct``; Gym's literal-only prefill
        used to miss every such case (-5 pp on the 16-seed 120B run).
        After the fix, the symbolic shortcut fires and the LLM judge is
        skipped."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        server_mock.post = AsyncMock()  # judge MUST NOT be called

        model_response = _make_response(r"</think>The answer is \boxed{5924217936}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
            expected_answer=r"$\binom{20}{10}^2 - \binom{20}{9}^2$ ",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.symbolic_correct == approx(1.0)
        assert result.judge_correct == approx(1.0)
        assert result.any_correct == approx(1.0)
        assert result.extracted_answer == "5924217936"
        assert result.judge_output == "Reasoning: The two answers are identical.\nJudgement: Yes"
        server_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_symbolic_mismatch_calls_judge_and_sets_symbolic_zero(self, chat_config):
        """When the boxed answer is symbolically different from expected,
        ``symbolic_correct`` is 0 and the LLM judge runs as before. The
        verifier output exposes both fields independently (mirroring
        Skills' separate ``symbolic_correct`` and ``judge_correct`` columns)."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>0 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response(r"</think>\boxed{42}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
            expected_answer="180",
        )
        result = await server.verify(request)
        assert result.symbolic_correct == approx(0.0)
        assert result.judge_correct == approx(0.0)
        assert result.any_correct == approx(0.0)
        assert result.judge_points == 0
        server_mock.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_judge_yes_with_symbolic_no_yields_any_correct(self, chat_config):
        """If sympy says no but the LLM judge says yes, ``any_correct`` is 1
        (mirrors Skills' ``symbolic_correct OR judge_correct`` aggregation)."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>7 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response(r"</think>\boxed{some_proof_summary_not_sympy_equal}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
            expected_answer="42",
        )
        result = await server.verify(request)
        assert result.symbolic_correct == approx(0.0)
        assert result.judge_correct == approx(1.0)
        assert result.any_correct == approx(1.0)
        assert result.reward == approx(1.0)

    @pytest.mark.asyncio
    async def test_verify_prefill_no_expected_answer_calls_judge(self, chat_config):
        """When `expected_answer` is None (e.g. open-ended proof problem with
        no canonical short answer), the prefill shortcut is bypassed and the
        judge runs as normal."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>1 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("</think>I think \\boxed{42}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
            expected_answer=None,
        )
        result = await server.verify(request)
        assert result.judge_points == 1
        server_mock.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_parsed_reasoning_complete(self, chat_config):
        """vLLM --reasoning-parser path: separate `type=reasoning` item plus
        a non-empty `message` item ⇒ message text used as predicted answer
        even though no `</think>` tag appears anywhere."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>7 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_parsed_reasoning_response(
            reasoning_text="thinking step by step ...",
            message_text="MY_PROOF_ANSWER \\boxed{42}",
        )
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        # extracted_answer is the contents of \boxed{}, not the full message.
        assert result.extracted_answer == "42"
        assert result.reward == approx(1.0)
        assert result.judge_points == 7
        # Confirm the judge prompt's `{predicted_answer}` got the boxed
        # content (the full message text is intentionally NOT sent — that
        # was a Skills-vs-Gym misread; see COMPARISON_RESULTS.md).
        json_payload = server_mock.post.call_args.kwargs["json"]
        assert "42" in json_payload.messages[0]["content"]
        assert "MY_PROOF_ANSWER" not in json_payload.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_verify_parsed_reasoning_truncated(self, chat_config):
        """vLLM --reasoning-parser path: only a `type=reasoning` item, no
        message item ⇒ truncated mid-reasoning ⇒ judge skipped."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        server_mock.post = AsyncMock()

        model_response = _make_parsed_reasoning_response(
            reasoning_text="ran out of tokens partway through ...",
            message_text=None,
        )
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.extracted_answer == ""
        assert result.judge_correct == approx(0.0)
        assert result.judge_points is None
        # Empty extraction now mirrors Skills' synthetic "Judgement: No" —
        # parseable verdict, so no_judge_score=0 not 1.
        assert result.no_judge_score == approx(0.0)
        server_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_parsed_reasoning_empty_message(self, chat_config):
        """vLLM --reasoning-parser path: reasoning item plus a message item
        with empty content ⇒ also no-answer (synthetic Judgement: No)."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        server_mock.post = AsyncMock()

        model_response = _make_parsed_reasoning_response(
            reasoning_text="thinking ...",
            message_text="",
        )
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.extracted_answer == ""
        assert result.judge_correct == approx(0.0)
        assert result.judge_points is None
        assert result.no_judge_score == approx(0.0)
        server_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_judge_unparseable(self, chat_config):
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("Inconclusive — no clear verdict."))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("<think>r</think>some attempt \\boxed{X}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.judge_points is None
        assert result.no_judge_score == approx(1.0)

    @pytest.mark.asyncio
    async def test_verify_responses_path(self, responses_config):
        """Cover the /v1/responses branch, mirroring omniscience."""
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=responses_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_responses_response("<points>7 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("<think>r</think>my proof \\boxed{Q.E.D.}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="P",
            reference_solution="S",
            rubric="R",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.judge_points == 7
        # Confirm /v1/responses URL was used.
        url_path = server_mock.post.call_args.kwargs["url_path"]
        assert url_path == "/v1/responses"

    @pytest.mark.asyncio
    async def test_judge_prompt_contains_all_placeholders(self, chat_config):
        server_mock = MagicMock(spec=ServerClient)
        server = ImoProofBenchJudgeServer(config=chat_config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._judge_chat_response("<points>7 out of 7</points>"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = _make_response("<think>r</think>some proof \\boxed{MY_BOXED_ANSWER}")
        request = ImoProofBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            problem="MY_PROBLEM",
            reference_solution="MY_REFSOL",
            rubric="MY_RUBRIC",
        )
        await server.verify(request)

        json_payload = server_mock.post.call_args.kwargs["json"]
        judge_text = json_payload.messages[0]["content"]
        assert "MY_PROBLEM" in judge_text
        assert "MY_REFSOL" in judge_text
        assert "MY_RUBRIC" in judge_text
        # `{predicted_answer}` is filled with the contents of \boxed{}, not
        # the surrounding proof text.
        assert "MY_BOXED_ANSWER" in judge_text


# ---------------------------------------------------------------------------
# Score function + threshold
# ---------------------------------------------------------------------------


class TestScoreFn:
    def test_threshold_value(self):
        assert PASS_THRESHOLD == 6  # documented Skills behaviour: points >= 6 ⇒ correct

    def test_seven_correct(self):
        # _score_fn now keys correctness off the verifier-provided
        # `judge_correct` field (which already encodes Skills' 3-format
        # priority); `judge_points` only drives the per-score-band buckets.
        s = ImoProofBenchJudgeServer._score_fn({"judge_correct": 1.0, "judge_points": 7})
        assert s["judge_correct"] == 1.0
        assert s["judge_score_7"] == 1.0
        assert s["judge_score_6"] == 0.0

    def test_six_correct(self):
        s = ImoProofBenchJudgeServer._score_fn({"judge_correct": 1.0, "judge_points": 6})
        assert s["judge_correct"] == 1.0
        assert s["judge_score_6"] == 1.0
        assert s["judge_score_7"] == 0.0

    def test_one_incorrect(self):
        s = ImoProofBenchJudgeServer._score_fn({"judge_correct": 0.0, "judge_points": 1})
        assert s["judge_correct"] == 0.0
        assert s["judge_score_1"] == 1.0

    def test_zero_incorrect(self):
        s = ImoProofBenchJudgeServer._score_fn({"judge_correct": 0.0, "judge_points": 0})
        assert s["judge_correct"] == 0.0
        assert s["judge_score_0"] == 1.0

    def test_none_incorrect(self):
        s = ImoProofBenchJudgeServer._score_fn({"judge_correct": 0.0, "judge_points": None})
        assert s["judge_correct"] == 0.0
        assert all(s[k] == 0.0 for k in ("judge_score_0", "judge_score_1", "judge_score_6", "judge_score_7"))

    def test_format1_correct_overrides_low_points(self):
        # Format 1 (Judgement: Yes) wins over Format 3 in Skills' priority
        # order — verifier sets judge_correct=1.0 even when judge_points<6.
        s = ImoProofBenchJudgeServer._score_fn({"judge_correct": 1.0, "judge_points": 1})
        assert s["judge_correct"] == 1.0
        # Score-band buckets still reflect the parsed integer.
        assert s["judge_score_1"] == 1.0

    def test_off_rubric_score_below_threshold(self):
        # Score of 4 is "off-rubric" (rubric only enumerates 0/1/6/7); the
        # verifier still maps it to incorrect via the points >= 6 rule.
        s = ImoProofBenchJudgeServer._score_fn({"judge_correct": 0.0, "judge_points": 4})
        assert s["judge_correct"] == 0.0


# ---------------------------------------------------------------------------
# compute_metrics + get_key_metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    @fixture
    def server(self) -> ImoProofBenchJudgeServer:
        cfg = ImoProofBenchJudgeConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        return ImoProofBenchJudgeServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_compute_metrics_basic(self, server):
        # 2 tasks × 2 rollouts each. Task 0: both correct (points=7,6). Task 1: both wrong (points=1,0).
        tasks = [
            [
                {
                    "judge_correct": 1.0,
                    "judge_points": 7,
                    "extracted_answer": "a",
                    "category": "Algebra",
                    "level": "IMO-easy",
                },
                {
                    "judge_correct": 1.0,
                    "judge_points": 6,
                    "extracted_answer": "b",
                    "category": "Algebra",
                    "level": "IMO-easy",
                },
            ],
            [
                {
                    "judge_correct": 0.0,
                    "judge_points": 1,
                    "extracted_answer": "c",
                    "category": "Geometry",
                    "level": "IMO-hard",
                },
                {
                    "judge_correct": 0.0,
                    "judge_points": 0,
                    "extracted_answer": "d",
                    "category": "Geometry",
                    "level": "IMO-hard",
                },
            ],
        ]
        metrics = server.compute_metrics(tasks)
        # Task 0 contributes 1 each pass@1 (both correct), task 1 contributes 0
        # → average pass@1 = 0.5.
        keys = [k for k in metrics if "pass@1[avg-of-2]/judge_correct" in k]
        assert keys, f"missing pass@1 key in {sorted(metrics.keys())}"
        # Gym reports rates as percentages (0-100), not fractions (0-1).
        assert metrics[keys[0]] == approx(50.0)
        # Per-category breakdown should appear (Algebra/Geometry).
        cat_keys = [k for k in metrics if k.startswith("Algebra/") or k.startswith("Geometry/")]
        assert any("pass@1[avg-of-2]/judge_correct" in k for k in cat_keys)

    def test_compute_metrics_empty(self, server):
        # Empty input shouldn't crash.
        metrics = server.compute_metrics([])
        assert isinstance(metrics, dict)

    def test_compute_metrics_no_optional_field(self, server):
        # If category/level isn't in the rollout dict, subset metrics should
        # silently skip — not raise.
        tasks = [[{"judge_correct": 1.0, "judge_points": 7, "extracted_answer": "x"}]]
        metrics = server.compute_metrics(tasks)
        # No Algebra/IMO-easy keys expected.
        assert not any(k.startswith("Algebra/") for k in metrics)

    def test_get_key_metrics_subset(self, server):
        agent_metrics = {
            "pass@1[avg-of-4]/judge_correct": 0.42,
            "pass@4/judge_correct": 0.8,
            "majority@4/judge_correct": 0.5,
            "pass@1[avg-of-4]/no_answer": 0.1,
            "pass@4/no_answer": 0.05,
            "mean/output_tokens": 12345.0,
            "noise": 0.0,
        }
        key = server.get_key_metrics(agent_metrics)
        assert "pass@1[avg-of-4]/judge_correct" in key
        assert "pass@4/judge_correct" in key
        assert "majority@4/judge_correct" in key
        assert "mean/output_tokens" in key
        # no_answer is excluded from highest_k_metrics calls for pass@k/majority@k.
        assert "pass@4/no_answer" not in key
