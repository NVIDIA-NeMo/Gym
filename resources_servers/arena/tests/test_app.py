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
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import yaml
from pytest import approx, fixture, raises

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.arena.app import (
    ArenaResourcesServer,
    ArenaResourcesServerConfig,
    ArenaVerifyRequest,
)
from resources_servers.arena.arena import (
    _ALL_VERDICT_LABELS,
    _bootstrap,
    _compute_raw_style_feature,
    _extract_style_metadata,
    _extract_thinking_content,
    _extract_verdict,
    _fit_bt_with_offset,
    _score_verdict_as_a,
    _score_verdict_as_b,
    _strip_thinking_blocks,
    _weighted_scores_as_a,
    _weighted_scores_as_b,
)


# Load style constants from the lmarena_260311 config so tests stay in sync with the YAML.
_LMARENA_CFG_PATH = Path(__file__).parent.parent / "configs" / "lmarena_260311.yaml"
with open(_LMARENA_CFG_PATH) as _f:
    _lmarena_cfg = yaml.safe_load(_f)
_ARENA_CFG = _lmarena_cfg["lmarena_260311"]["resources_servers"]["arena"]
_TEST_STYLE_NORM_MEAN: dict[str, list[float]] = _ARENA_CFG["style_norm_mean"]
_TEST_STYLE_NORM_STD: dict[str, list[float]] = _ARENA_CFG["style_norm_std"]
_TEST_STYLE_COEFS: dict[str, list[float]] = _ARENA_CFG["style_coefs"]

# Load style constants from the arena_hard_v2 config for multi-category tests.
_ARENA_HARD_V2_CFG_PATH = Path(__file__).parent.parent / "configs" / "arena_hard_v2.yaml"
with open(_ARENA_HARD_V2_CFG_PATH) as _f:
    _arena_hard_v2_cfg = yaml.safe_load(_f)
_ARENA_HARD_V2_CFG = _arena_hard_v2_cfg["arena_hard_v2"]["resources_servers"]["arena"]
_TEST_MULTI_STYLE_NORM_MEAN: dict[str, list[float]] = _ARENA_HARD_V2_CFG["style_norm_mean"]
_TEST_MULTI_STYLE_NORM_STD: dict[str, list[float]] = _ARENA_HARD_V2_CFG["style_norm_std"]
_TEST_MULTI_STYLE_COEFS: dict[str, list[float]] = _ARENA_HARD_V2_CFG["style_coefs"]


# _score_verdict_as_a/b and _weighted_scores_as_a/b are used by unit tests below.


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_output_message(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg-{text[:20]}",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_response(id: str, output_item: Any) -> dict[str, Any]:
    return NeMoGymResponse(
        id=id,
        created_at=0.0,
        model="test-model",
        object="response",
        output=[output_item],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump()


def _make_model_response(text: str, id: str = "model-resp") -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(_make_response(id, _make_output_message(text)))


def _make_post_mock(response_json: str) -> MagicMock:
    post_mock = MagicMock()
    post_mock.read = AsyncMock(return_value=response_json)
    return post_mock


# ── Unit tests for module-level helpers ───────────────────────────────────────


class TestExtractVerdict:
    def test_a_strong_win(self):
        assert _extract_verdict("My verdict is [[A>>B]]") == "[[A>>B]]"

    def test_a_win(self):
        assert _extract_verdict("Final verdict: [[A>B]]") == "[[A>B]]"

    def test_tie(self):
        assert _extract_verdict("They are equal: [[A=B]]") == "[[A=B]]"

    def test_b_win(self):
        assert _extract_verdict("B is slightly better [[B>A]]") == "[[B>A]]"

    def test_b_strong_win(self):
        assert _extract_verdict("B wins strongly [[B>>A]]") == "[[B>>A]]"

    def test_both_bad(self):
        assert _extract_verdict("Both answers are poor: [[BB]]") == "[[BB]]"

    def test_no_verdict(self):
        assert _extract_verdict("I cannot determine which is better.") is None

    def test_empty_string(self):
        assert _extract_verdict("") is None

    def test_rightmost_wins(self):
        # Judge first mentions [[A>B]] in reasoning, then gives final [[B>A]] verdict.
        assert _extract_verdict("Initially [[A>B]] but after reconsideration [[B>A]]") == "[[B>A]]"

    def test_rightmost_wins_strong_after_weak(self):
        # Strong verdict [[A>>B]] appears after a weak one [[A>B]] — strong wins by position.
        assert _extract_verdict("Leaning [[A>B]] but on reflection [[A>>B]]") == "[[A>>B]]"

    def test_rightmost_wins_weak_after_strong(self):
        # Weak verdict [[A>B]] appears after strong [[A>>B]] — weak wins by position.
        assert _extract_verdict("Strong at first [[A>>B]] then reconsidered [[A>B]]") == "[[A>B]]"

    def test_all_labels_covered(self):
        # Smoke-test: every verdict label is recognized.
        for label in _ALL_VERDICT_LABELS:
            assert _extract_verdict(f"verdict is {label}") == label


class TestScoreVerdictAsA:
    def test_a_strong_win(self):
        assert _score_verdict_as_a("[[A>>B]]") == approx(1.0)

    def test_a_win(self):
        assert _score_verdict_as_a("[[A>B]]") == approx(1.0)

    def test_tie(self):
        assert _score_verdict_as_a("[[A=B]]") == approx(0.5)

    def test_b_win(self):
        assert _score_verdict_as_a("[[B>A]]") == approx(0.0)

    def test_b_strong_win(self):
        assert _score_verdict_as_a("[[B>>A]]") == approx(0.0)

    def test_both_bad(self):
        assert _score_verdict_as_a("[[BB]]") == approx(0.0)

    def test_none(self):
        assert _score_verdict_as_a(None) == approx(0.0)


class TestScoreVerdictAsB:
    def test_b_strong_win(self):
        assert _score_verdict_as_b("[[B>>A]]") == approx(1.0)

    def test_b_win(self):
        assert _score_verdict_as_b("[[B>A]]") == approx(1.0)

    def test_tie(self):
        assert _score_verdict_as_b("[[A=B]]") == approx(0.5)

    def test_a_win(self):
        assert _score_verdict_as_b("[[A>B]]") == approx(0.0)

    def test_a_strong_win(self):
        assert _score_verdict_as_b("[[A>>B]]") == approx(0.0)

    def test_both_bad(self):
        assert _score_verdict_as_b("[[BB]]") == approx(0.0)

    def test_none(self):
        assert _score_verdict_as_b(None) == approx(0.0)


class TestWeightedScores:
    def test_strong_win_repeated_weight_times(self):
        assert _weighted_scores_as_a("[[A>>B]]", weight=3) == [1.0, 1.0, 1.0]

    def test_weak_win_not_repeated(self):
        assert _weighted_scores_as_a("[[A>B]]", weight=3) == [1.0]

    def test_tie_not_repeated(self):
        assert _weighted_scores_as_a("[[A=B]]", weight=3) == [0.5]

    def test_strong_loss_repeated_weight_times(self):
        assert _weighted_scores_as_a("[[B>>A]]", weight=3) == [0.0, 0.0, 0.0]

    def test_weak_loss_not_repeated(self):
        assert _weighted_scores_as_a("[[B>A]]", weight=3) == [0.0]

    def test_b_perspective_strong_win_repeated(self):
        assert _weighted_scores_as_b("[[B>>A]]", weight=3) == [1.0, 1.0, 1.0]

    def test_b_perspective_strong_loss_repeated(self):
        assert _weighted_scores_as_b("[[A>>B]]", weight=3) == [0.0, 0.0, 0.0]

    def test_weight_one_behaves_like_unweighted(self):
        for verdict in ["[[A>>B]]", "[[B>>A]]", "[[A>B]]", "[[A=B]]", "[[B>A]]"]:
            assert len(_weighted_scores_as_a(verdict, weight=1)) == 1
            assert len(_weighted_scores_as_b(verdict, weight=1)) == 1

    def test_none_verdict_not_repeated(self):
        assert _weighted_scores_as_a(None, weight=3) == [0.0]
        assert _weighted_scores_as_b(None, weight=3) == [0.0]


class TestStyleFeatures:
    def test_extract_style_metadata_plain_text(self):
        meta = _extract_style_metadata("hello world")
        assert meta["token_len"] > 0
        assert meta["header_count"] == 0
        assert meta["list_count"] == 0
        assert meta["bold_count"] == 0

    def test_extract_style_metadata_markdown(self):
        text = "## Header\n- item one\n- item two\n**bold text**"
        meta = _extract_style_metadata(text)
        assert meta["header_count"] == 1
        assert meta["list_count"] == 2
        assert meta["bold_count"] == 1

    def test_extract_style_metadata_strips_code_blocks(self):
        # Code blocks should be stripped before counting markdown elements.
        text = "```\n## fake header\n- fake list\n```\n## Real header"
        meta = _extract_style_metadata(text)
        assert meta["header_count"] == 1  # only the real one outside the fence
        assert meta["list_count"] == 0

    def test_compute_raw_style_feature_shape(self):
        feat = _compute_raw_style_feature("short answer", "a much longer baseline answer with lots of words")
        assert feat.shape == (4,)

    def test_compute_raw_style_feature_length_direction(self):
        # Longer policy answer → positive length feature.
        feat = _compute_raw_style_feature("a " * 200, "b")
        assert feat[0] > 0

    def test_compute_raw_style_feature_identical_texts(self):
        # Identical texts → zero differentials.
        text = "## Hello\n- item\n**bold**\n" * 10
        feat = _compute_raw_style_feature(text, text)
        assert np.allclose(feat[0], 0.0, atol=1e-6)
        assert np.allclose(feat[1:], 0.0, atol=1e-6)

    def test_compute_raw_style_feature_both_empty(self):
        # Both texts empty → all features zero (no division by zero).
        feat = _compute_raw_style_feature("", "")
        assert feat.shape == (4,)
        assert np.allclose(feat, 0.0)

    def test_style_constants_shapes(self):
        # Style constants are loaded from config; verify that the test reference values
        # produce correctly shaped numpy arrays.
        for cat in _TEST_STYLE_NORM_MEAN:
            mean = np.array(_TEST_STYLE_NORM_MEAN[cat])
            std = np.array(_TEST_STYLE_NORM_STD[cat])
            coefs = np.array(_TEST_STYLE_COEFS[cat])
            assert mean.shape == (4,)
            assert std.shape == (4,)
            assert coefs.shape == (4,)
            assert (std > 0).all()

    def test_fit_bt_with_offset_all_wins(self):
        # All outcomes = 1 → θ should be large positive → expit(θ) ≈ 1.
        from scipy.special import expit

        offsets = np.zeros(50)
        scores = np.ones(50)
        theta = _fit_bt_with_offset(offsets, scores)
        assert expit(theta) > 0.9

    def test_fit_bt_with_offset_all_losses(self):
        from scipy.special import expit

        offsets = np.zeros(50)
        scores = np.zeros(50)
        theta = _fit_bt_with_offset(offsets, scores)
        assert expit(theta) < 0.1

    def test_bootstrap_plain_mean_returns_ci(self):
        scores = np.full(100, 0.6)
        pt_est, ci_lower, ci_upper = _bootstrap(scores, offsets=None, n_rounds=20)
        assert pt_est == approx(0.6, abs=0.05)
        assert ci_lower <= pt_est <= ci_upper

    def test_bootstrap_with_offset_shape(self):
        rng = np.random.RandomState(0)
        scores = rng.uniform(0, 1, 200)
        offsets = rng.normal(0, 0.1, 200)
        pt_est, ci_lower, ci_upper = _bootstrap(scores, offsets=offsets, n_rounds=20)
        assert 0.0 < pt_est < 1.0
        assert ci_lower <= pt_est <= ci_upper


class TestStripThinkingBlocks:
    def test_strips_think_block(self):
        assert _strip_thinking_blocks("<think>reasoning</think>answer") == "answer"

    def test_strips_thinking_block(self):
        assert _strip_thinking_blocks("<thinking>deep thought</thinking>answer") == "answer"

    def test_strips_multiline_block(self):
        assert _strip_thinking_blocks("<think>\nline1\nline2\n</think>result") == "result"

    def test_no_block_unchanged(self):
        assert _strip_thinking_blocks("plain text") == "plain text"

    def test_empty_string(self):
        assert _strip_thinking_blocks("") == ""


class TestExtractThinkingContent:
    def test_extracts_think_block(self):
        assert _extract_thinking_content("<think>step 1</think>answer") == "step 1"

    def test_extracts_thinking_block(self):
        assert _extract_thinking_content("<thinking>deep thought</thinking>answer") == "deep thought"

    def test_extracts_multiple_blocks(self):
        result = _extract_thinking_content("<think>first</think>text<think>second</think>")
        assert result == "first\n\nsecond"

    def test_no_block_returns_empty_string(self):
        assert _extract_thinking_content("plain text") == ""

    def test_empty_string_returns_empty(self):
        assert _extract_thinking_content("") == ""

    def test_empty_think_block_ignored(self):
        # Whitespace-only content is stripped and skipped.
        assert _extract_thinking_content("<think>   </think>answer") == ""


class TestExtractResponseParts:
    """Unit tests for ArenaResourcesServer._extract_response_parts."""

    def _make_reasoning_item(self, summary_texts: list[str]) -> NeMoGymResponseReasoningItem:
        summaries = [{"text": t, "type": "summary_text"} for t in summary_texts]
        return NeMoGymResponseReasoningItem.model_validate({"id": "r1", "summary": summaries, "type": "reasoning"})

    def _make_response_with_items(self, *output_items) -> NeMoGymResponse:
        return NeMoGymResponse.model_validate(
            {
                "id": "test",
                "created_at": 0.0,
                "model": "m",
                "object": "response",
                "output": [item.model_dump() for item in output_items],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            }
        )

    def test_plain_text_no_reasoning(self):
        resp = _make_model_response("Hello world.")
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer == "Hello world."
        assert reasoning is None

    def test_think_block_stripped_from_answer(self):
        resp = _make_model_response("<think>internal</think>Paris.")
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer == "Paris."
        assert reasoning == "internal"

    def test_multiple_think_blocks_concatenated(self):
        resp = _make_model_response("<think>step1</think>mid<think>step2</think>end")
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer == "midend"
        assert reasoning == "step1\n\nstep2"

    def test_reasoning_item_summary_extracted(self):
        reasoning_item = self._make_reasoning_item(["chain of thought"])
        text_item = _make_output_message("Final answer.")
        resp = self._make_response_with_items(reasoning_item, text_item)
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer == "Final answer."
        assert reasoning == "chain of thought"

    def test_reasoning_item_multiple_summaries(self):
        reasoning_item = self._make_reasoning_item(["part one", "part two"])
        text_item = _make_output_message("Answer.")
        resp = self._make_response_with_items(reasoning_item, text_item)
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer == "Answer."
        assert reasoning == "part one\n\npart two"

    def test_reasoning_item_and_think_block_combined(self):
        # Both o-series reasoning summary and <think> block in output_text.
        reasoning_item = self._make_reasoning_item(["summary reasoning"])
        text_item = _make_output_message("<think>inline thought</think>Answer.")
        resp = self._make_response_with_items(reasoning_item, text_item)
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer == "Answer."
        assert "summary reasoning" in reasoning
        assert "inline thought" in reasoning

    def test_empty_response_returns_none_none(self):
        resp = self._make_response_with_items()
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer is None
        assert reasoning is None

    def test_only_think_block_returns_none_answer(self):
        # Policy response is entirely reasoning — nothing left after stripping.
        resp = _make_model_response("<think>only reasoning</think>")
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer is None
        assert reasoning == "only reasoning"

    def test_whitespace_only_returns_none_answer(self):
        resp = _make_model_response("   \n  ")
        answer, reasoning = ArenaResourcesServer._extract_response_parts(resp)
        assert answer is None
        assert reasoning is None


# ── ArenaRunRequest validator ─────────────────────────────────────────────────


class TestArenaRunRequest:
    def _make_request(self, baseline_answer):
        return ArenaVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_model_response("answer"),
            question_id="q1",
            question="Q?",
            baseline_answer=baseline_answer,
        )

    def test_baseline_answer_plain_string(self):
        req = self._make_request("plain text answer")
        assert req.baseline_answer == "plain text answer"

    def test_baseline_answer_dict_format(self):
        req = self._make_request({"answer": "dict-wrapped answer"})
        assert req.baseline_answer == "dict-wrapped answer"


# ── Server tests ──────────────────────────────────────────────────────────────


class TestArenaResourcesServer:
    @fixture
    def config(self) -> ArenaResourcesServerConfig:
        return ArenaResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template="Question: {question}\nA: {answer_a}\nB: {answer_b}",
            judge_system_message="You are an impartial judge.",
            style_norm_mean=_TEST_STYLE_NORM_MEAN,
            style_norm_std=_TEST_STYLE_NORM_STD,
            style_coefs=_TEST_STYLE_COEFS,
        )

    @fixture
    def server(self, config: ArenaResourcesServerConfig) -> ArenaResourcesServer:
        mock_client = MagicMock(spec=ServerClient)
        return ArenaResourcesServer(config=config, server_client=mock_client)

    def _make_verify_request(
        self,
        policy_response_text: str,
        question: str = "What is the capital of France?",
        baseline_answer: str = "Paris.",
        question_id: str = "test-id-001",
    ) -> ArenaVerifyRequest:
        return ArenaVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": question}]
            ),
            response=_make_model_response(policy_response_text),
            question_id=question_id,
            question=question,
            baseline_answer=baseline_answer,
        )

    def _setup_judge_responses(self, server: ArenaResourcesServer, verdicts: list[str]) -> None:
        """Mock server_client.post to return judge responses with the given verdict labels."""
        post_mocks = []
        for i, verdict in enumerate(verdicts):
            msg = _make_output_message(f"My analysis. Final verdict: {verdict}")
            post_mocks.append(_make_post_mock(json.dumps(_make_response(f"judge-{i}", msg))))
        server.server_client.post = AsyncMock(side_effect=post_mocks)

    # ── verify() ──────────────────────────────────────────────────────────────

    async def test_verify_empty_response_returns_zero_reward(self, server: ArenaResourcesServer):
        request = self._make_verify_request("")
        # Override with completely empty output
        request.response.output = []
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.policy_answer is None
        assert result.games is None
        # No judge calls should have been made
        server.server_client.post.assert_not_called()

    async def test_verify_whitespace_only_response_returns_zero_reward(self, server: ArenaResourcesServer):
        request = self._make_verify_request("   \n  ")
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.policy_answer is None
        # No judge calls should have been made for an empty/whitespace response
        server.server_client.post.assert_not_called()

    async def test_verify_policy_wins_both_rounds(self, server: ArenaResourcesServer):
        # Game 1 (policy=A): [[A>B]] → score 1.0
        # Game 2 (baseline=A): [[B>A]] → score 1.0 from B's perspective
        self._setup_judge_responses(server, ["[[A>B]]", "[[B>A]]"])

        request = self._make_verify_request("The capital of France is Paris.")
        result = await server.verify(request)

        assert result.reward == approx(1.0)
        assert result.policy_answer == "The capital of France is Paris."
        assert len(result.games) == 2
        assert result.games[0].verdict == "[[A>B]]"
        assert result.games[1].verdict == "[[B>A]]"

    async def test_verify_policy_loses_both_rounds(self, server: ArenaResourcesServer):
        # Game 1: [[B>A]] → score 0.0. Game 2: [[A>B]] → score 0.0
        self._setup_judge_responses(server, ["[[B>A]]", "[[A>B]]"])

        result = await server.verify(self._make_verify_request("I'm not sure."))
        assert result.reward == approx(0.0)

    async def test_verify_both_rounds_tie(self, server: ArenaResourcesServer):
        # Game 1: [[A=B]] → 0.5. Game 2: [[A=B]] → 0.5
        self._setup_judge_responses(server, ["[[A=B]]", "[[A=B]]"])

        result = await server.verify(self._make_verify_request("Paris is the capital."))
        assert result.reward == approx(0.5)

    async def test_verify_win_plus_tie(self, server: ArenaResourcesServer):
        # Game 1: [[A>>B]] (strong, weight=3) → [1.0, 1.0, 1.0]
        # Game 2: [[A=B]] (tie, weight=1)      → [0.5]
        # Combined: [1.0, 1.0, 1.0, 0.5] → mean = 3.5 / 4 = 0.875
        self._setup_judge_responses(server, ["[[A>>B]]", "[[A=B]]"])

        result = await server.verify(self._make_verify_request("Answer."))
        assert result.reward == approx(0.875)

    async def test_verify_strong_loss_plus_strong_win(self, server: ArenaResourcesServer):
        # Game 1: [[B>>A]] → 0.0. Game 2: [[B>>A]] → 1.0 (policy=B wins strongly)
        self._setup_judge_responses(server, ["[[B>>A]]", "[[B>>A]]"])

        result = await server.verify(self._make_verify_request("Answer."))
        assert result.reward == approx(0.5)

    async def test_verify_both_bad_verdict(self, server: ArenaResourcesServer):
        # [[BB]] → 0.0 for both A and B positions
        self._setup_judge_responses(server, ["[[BB]]", "[[BB]]"])

        result = await server.verify(self._make_verify_request("Answer."))
        assert result.reward == approx(0.0)

    async def test_verify_unparseable_judge_output(self, server: ArenaResourcesServer):
        # Judge doesn't output a valid verdict label → 0.0 for both rounds
        self._setup_judge_responses(server, ["I cannot decide.", "unclear"])

        result = await server.verify(self._make_verify_request("Answer."))
        assert result.reward == approx(0.0)
        assert result.games[0].verdict is None
        assert result.games[1].verdict is None

    async def test_verify_strips_thinking_blocks_before_judge(self, server: ArenaResourcesServer):
        # Policy response contains thinking blocks — they should be stripped before judging
        # but preserved in policy_reasoning for debugging.
        self._setup_judge_responses(server, ["[[A>B]]", "[[B>A]]"])

        request = self._make_verify_request("<think>internal reasoning</think>Paris.")
        result = await server.verify(request)

        assert result.policy_answer == "Paris."
        assert result.policy_reasoning == "internal reasoning"
        assert result.reward == approx(1.0)

    async def test_verify_response_preserves_request_fields(self, server: ArenaResourcesServer):
        self._setup_judge_responses(server, ["[[A=B]]", "[[A=B]]"])

        question = "Who invented calculus?"
        baseline = "Newton and Leibniz independently."
        request = self._make_verify_request("Newton.", question=question, baseline_answer=baseline, question_id="q-42")
        result = await server.verify(request)

        assert result.question_id == "q-42"
        assert result.question == question
        assert result.baseline_answer == baseline
        assert sorted(result.model_dump().keys()) == sorted(
            [
                "responses_create_params",
                "response",
                "reward",
                "question_id",
                "question",
                "baseline_answer",
                "category",
                "policy_answer",
                "policy_reasoning",
                "games",
                "self_comparison",
            ]
        )

    async def test_verify_response_has_multiple_output_items(self, server: ArenaResourcesServer):
        """Multi-turn / reasoning model: reasoning item + multiple messages concatenated."""
        self._setup_judge_responses(server, ["[[A>B]]", "[[B>A]]"])

        model_response = NeMoGymResponse.model_validate(
            {
                "id": "multi",
                "created_at": 0.0,
                "model": "m",
                "object": "response",
                "output": [
                    NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning").model_dump(),
                    _make_output_message("Part 1 ").model_dump(),
                    _make_output_message("Part 2.").model_dump(),
                    NeMoGymResponseOutputMessage(
                        id="refusal-id",
                        content=[NeMoGymResponseOutputRefusal(refusal="n/a", type="refusal")],
                        role="assistant",
                        status="completed",
                        type="message",
                    ).model_dump(),
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            }
        )
        request = ArenaVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q"}]),
            response=model_response,
            question_id="id",
            question="Q",
            baseline_answer="B",
        )
        result = await server.verify(request)

        # Reasoning item and refusal items are skipped; text items are concatenated.
        assert result.policy_answer == "Part 1 Part 2."
        assert result.reward == approx(1.0)

    # ── _run_judge_game() edge cases ──────────────────────────────────────────

    async def test_run_judge_game_empty_output(self, server: ArenaResourcesServer):
        """If the judge returns an empty output list, verdict should be None."""
        empty_response = NeMoGymResponse(
            id="empty",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        post_mock = _make_post_mock(json.dumps(empty_response.model_dump()))
        server.server_client.post = AsyncMock(return_value=post_mock)

        game = await server._run_judge_game(
            "Q?",
            "A",
            "B",
            system_message=server.config.judge_system_message,
            prompt_template=server.config.judge_prompt_template,
        )
        assert game.verdict is None

    async def test_run_judge_game_non_message_output(self, server: ArenaResourcesServer):
        """Non-message output item → verdict is None."""
        reasoning_response = NeMoGymResponse(
            id="reasoning",
            created_at=0.0,
            model="m",
            object="response",
            output=[NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        post_mock = _make_post_mock(json.dumps(reasoning_response.model_dump()))
        server.server_client.post = AsyncMock(return_value=post_mock)

        game = await server._run_judge_game(
            "Q?",
            "A",
            "B",
            system_message=server.config.judge_system_message,
            prompt_template=server.config.judge_prompt_template,
        )
        assert game.verdict is None

    async def test_run_judge_game_refusal_content(self, server: ArenaResourcesServer):
        """Refusal content (non-output_text) → verdict is None."""
        refusal_msg = NeMoGymResponseOutputMessage(
            id="ref",
            content=[NeMoGymResponseOutputRefusal(refusal="refused", type="refusal")],
            role="assistant",
            status="completed",
            type="message",
        )
        refusal_response = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[refusal_msg],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        post_mock = _make_post_mock(json.dumps(refusal_response.model_dump()))
        server.server_client.post = AsyncMock(return_value=post_mock)

        game = await server._run_judge_game(
            "Q?",
            "A",
            "B",
            system_message=server.config.judge_system_message,
            prompt_template=server.config.judge_prompt_template,
        )
        assert game.verdict is None

    async def test_run_judge_game_injects_system_prompt(self, server: ArenaResourcesServer):
        """Judge game must inject system message and formatted prompt."""
        msg = _make_output_message("verdict [[A=B]]")
        post_mock = _make_post_mock(json.dumps(_make_response("j", msg)))
        server.server_client.post = AsyncMock(return_value=post_mock)

        await server._run_judge_game(
            "My question",
            "Answer A",
            "Answer B",
            system_message=server.config.judge_system_message,
            prompt_template=server.config.judge_prompt_template,
        )

        call_kwargs = server.server_client.post.call_args
        sent_params: NeMoGymResponseCreateParamsNonStreaming = call_kwargs.kwargs["json"]
        messages = sent_params.input
        assert messages[0].role == "system"
        assert server.config.judge_system_message in messages[0].content
        assert messages[1].role == "user"
        assert "My question" in messages[1].content
        assert "Answer A" in messages[1].content
        assert "Answer B" in messages[1].content

    # ── compute_metrics() ─────────────────────────────────────────────────────

    def test_compute_metrics_empty(self, server: ArenaResourcesServer):
        assert server.compute_metrics([]) == {}

    def _rollout(self, v1: str, v2: str, reward: float = 0.5, category: str | None = None) -> dict:
        """Build a minimal rollout dict with two verdict games and stub text answers."""
        r: dict = {
            "reward": reward,
            "policy_answer": "Policy answer text for style feature extraction.",
            "baseline_answer": "Baseline answer text for style feature extraction.",
            "games": [{"verdict": v1}, {"verdict": v2}],
        }
        if category is not None:
            r["category"] = category
        return r

    @fixture
    def server_no_style(self, config: ArenaResourcesServerConfig) -> ArenaResourcesServer:
        """Server with style_control=False for testing the plain bootstrap path."""
        config.style_control = False
        return ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @fixture
    def server_two_categories(self, config: ArenaResourcesServerConfig) -> ArenaResourcesServer:
        """Server with two-category style constants (creative_writing + hard_prompt) from arena_hard_v2."""
        config.style_norm_mean = _TEST_MULTI_STYLE_NORM_MEAN
        config.style_norm_std = _TEST_MULTI_STYLE_NORM_STD
        config.style_coefs = _TEST_MULTI_STYLE_COEFS
        return ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_compute_metrics_all_wins(self, server: ArenaResourcesServer):
        rollout = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server.compute_metrics([[rollout], [rollout]])
        assert metrics["rollout_failure_rate"] == approx(0.0)
        assert metrics["style_control"] is True
        # Style-controlled win rate ≈ 1.0 when all battles are wins.
        assert metrics["win_rate"] == approx(1.0, abs=0.05)
        assert metrics["win_rate_ci_lower"] <= metrics["win_rate"]
        assert metrics["win_rate_ci_upper"] >= metrics["win_rate"]

    def test_compute_metrics_all_losses(self, server: ArenaResourcesServer):
        rollout = self._rollout("[[B>A]]", "[[A>B]]", reward=0.0)
        metrics = server.compute_metrics([[rollout]])
        assert metrics["win_rate"] == approx(0.0, abs=0.05)
        assert metrics["rollout_failure_rate"] == approx(0.0)

    def test_compute_metrics_plain_bootstrap_weighted(self, server_no_style: ArenaResourcesServer):
        """With style_control=False, win_rate is a plain bootstrap mean of weighted scores."""
        # Strong loss (6 items × 0) + weak win (2 items × 1) → plain mean = 0.25
        rollout_strong_loss = {
            "reward": 0.0,
            "policy_answer": "p",
            "baseline_answer": "b",
            "games": [{"verdict": "[[B>>A]]"}, {"verdict": "[[A>>B]]"}],
        }
        rollout_weak_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server_no_style.compute_metrics([[rollout_strong_loss], [rollout_weak_win]])
        assert metrics["style_control"] is False
        assert metrics["win_rate"] == approx(0.25, abs=0.05)

    def test_compute_metrics_parse_failures(self, server: ArenaResourcesServer):
        # Disable the failure-rate guard so we can inspect the metric values directly.
        server.config.max_rollout_failure_rate = 1.0
        rollout = {
            "reward": 0.0,
            "policy_answer": "x",
            "baseline_answer": "y",
            "games": [{"verdict": None}, {"verdict": None}],
        }
        metrics = server.compute_metrics([[rollout]])
        assert metrics["rollout_failure_rate"] == approx(1.0)
        assert "win_rate" not in metrics

    def test_compute_metrics_no_games(self, server: ArenaResourcesServer):
        # Disable the failure-rate guard: a single games=None rollout is 100% failure.
        server.config.max_rollout_failure_rate = 1.0
        rollout = {"reward": 0.0, "policy_answer": None, "baseline_answer": "y", "games": None}
        metrics = server.compute_metrics([[rollout]])
        assert metrics == {}

    def test_compute_metrics_skips_both_bad(self, server_no_style: ArenaResourcesServer):
        """[[BB]] rollouts are excluded from win_rate but don't affect rollout_failure_rate."""
        rollout_bb = {
            "reward": 0.0,
            "policy_answer": "x",
            "baseline_answer": "y",
            "games": [{"verdict": "[[BB]]"}, {"verdict": "[[BB]]"}],
        }
        rollout_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server_no_style.compute_metrics([[rollout_bb], [rollout_win]])
        assert metrics["rollout_failure_rate"] == approx(0.0)
        # Only the win rollout contributes to win_rate.
        assert metrics["win_rate"] == approx(1.0, abs=0.05)

    def test_compute_metrics_skips_partial_bb(self, server_no_style: ArenaResourcesServer):
        """A rollout where only one game has [[BB]] is excluded from win_rate."""
        rollout_partial_bb = {
            "reward": 0.0,
            "policy_answer": "x",
            "baseline_answer": "y",
            "games": [{"verdict": "[[A>B]]"}, {"verdict": "[[BB]]"}],
        }
        rollout_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server_no_style.compute_metrics([[rollout_partial_bb], [rollout_win]])
        assert metrics["win_rate"] == approx(1.0, abs=0.05)

    def test_compute_metrics_single_game_rollout_excluded(self, server_no_style: ArenaResourcesServer):
        """Rollouts with only one game are not accumulated into win_rate."""
        rollout_one_game = {
            "reward": 0.5,
            "policy_answer": "x",
            "baseline_answer": "y",
            "games": [{"verdict": "[[A>B]]"}],
        }
        rollout_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server_no_style.compute_metrics([[rollout_one_game], [rollout_win]])
        assert metrics["win_rate"] == approx(1.0, abs=0.05)

    def test_compute_metrics_raises_on_high_failure_rate(self, server: ArenaResourcesServer):
        """Raises ValueError when failed rollouts exceed max_rollout_failure_rate — no score returned."""
        good = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        # 2 good + 1 answer-failure = 33% failure rate, exceeds 1% default
        failed = {"reward": 0.0, "policy_answer": None, "baseline_answer": "b", "games": None}
        with raises(ValueError, match="max_rollout_failure_rate"):
            server.compute_metrics([[good], [good], [failed]])

    def test_compute_metrics_raises_on_high_judge_failure_rate(self, server: ArenaResourcesServer):
        """Judge parse failures also count toward the failure rate and trigger ValueError."""
        good = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        judge_fail = {
            "reward": 0.0,
            "policy_answer": "x",
            "baseline_answer": "y",
            "games": [{"verdict": None}, {"verdict": None}],
        }
        with raises(ValueError, match="max_rollout_failure_rate"):
            server.compute_metrics([[good], [good], [judge_fail]])

    def test_compute_metrics_within_failure_tolerance(self, server: ArenaResourcesServer):
        """No error when failure rate is within max_rollout_failure_rate."""
        good = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        failed = {"reward": 0.0, "policy_answer": None, "baseline_answer": "b", "games": None}
        # 1 failure in 200 rollouts = 0.5% < 1% default
        server.config.max_rollout_failure_rate = 0.01
        metrics = server.compute_metrics([[good]] * 199 + [[failed]])
        assert "win_rate" in metrics

    def test_compute_metrics_style_control_true(self, server: ArenaResourcesServer):
        """style_control=True produces a BT win probability with CI."""
        rollout = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server.compute_metrics([[rollout] * 20])
        assert metrics["style_control"] is True
        assert 0.0 < metrics["win_rate"] < 1.0
        assert metrics["win_rate_ci_lower"] <= metrics["win_rate"]
        assert metrics["win_rate_ci_upper"] >= metrics["win_rate"]

    def test_compute_metrics_multi_category_unweighted_mean(self, server_no_style: ArenaResourcesServer):
        """Win rate is the unweighted mean across categories, not sample-proportional.

        With 20 losses in cat_a and 5 wins in cat_b:
        - Sample-weighted mean = (0.0 × 20 + 1.0 × 5) / 25 = 0.2
        - Category-unweighted mean = (0.0 + 1.0) / 2 = 0.5
        """
        loss = self._rollout("[[B>A]]", "[[A>B]]", reward=0.0, category="cat_a")
        win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0, category="cat_b")
        metrics = server_no_style.compute_metrics([[loss]] * 20 + [[win]] * 5)
        assert metrics["win_rate"] == approx(0.5, abs=0.05)

    def test_compute_metrics_get_style_constants_falls_back_to_default(self, server: ArenaResourcesServer):
        """_get_style_constants falls back to 'default' when the exact category is not found."""
        # `server` is initialized with lmarena style constants keyed by "default" only.
        result_unknown = server._get_style_constants("unknown_category")
        result_default = server._get_style_constants("default")
        assert result_unknown is not None
        assert np.array_equal(result_unknown[0], result_default[0])
        assert np.array_equal(result_unknown[1], result_default[1])
        assert np.array_equal(result_unknown[2], result_default[2])

    def test_compute_metrics_get_style_constants_no_default_returns_none(self, config: ArenaResourcesServerConfig):
        """Unknown category with no 'default' fallback returns None."""
        # Only 'creative_writing' is configured — no 'default' key.
        config.style_norm_mean = {"creative_writing": [0.1, 0.0, 0.0, 0.0]}
        config.style_norm_std = {"creative_writing": [1.0, 1.0, 1.0, 1.0]}
        config.style_coefs = {"creative_writing": [0.3, 0.1, -0.2, 0.0]}
        partial_server = ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert partial_server._get_style_constants("hard_prompt") is None
        assert partial_server._get_style_constants("creative_writing") is not None

    def test_compute_metrics_multi_category_style_controlled(self, server_two_categories: ArenaResourcesServer):
        """Style-controlled win rate runs per-category BT fitting and returns the unweighted mean."""
        rollout_cw = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0, category="creative_writing")
        rollout_hp = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0, category="hard_prompt")
        metrics = server_two_categories.compute_metrics([[rollout_cw] * 10 + [rollout_hp] * 10])
        assert metrics["style_control"] is True
        assert metrics["win_rate"] == approx(1.0, abs=0.1)
        assert metrics["win_rate_ci_lower"] <= metrics["win_rate"]
        assert metrics["win_rate_ci_upper"] >= metrics["win_rate"]

    # ── get_key_metrics() ─────────────────────────────────────────────────────

    def test_get_key_metrics(self, server: ArenaResourcesServer):
        agent_metrics = {
            "mean/reward": 0.65,
            "win_rate": 0.55,
            "win_rate_ci_lower": 0.49,
            "win_rate_ci_upper": 0.61,
            "style_control": True,
            "mean/input_tokens": 512.0,
            "mean/output_tokens": 256.0,
            "rollout_failure_rate": 0.0,
            "std/reward": 0.1,
            "something_else": 42,
        }
        key = server.get_key_metrics(agent_metrics)
        assert set(key.keys()) == {
            "mean/reward",
            "win_rate",
            "win_rate_ci_lower",
            "win_rate_ci_upper",
            "style_control",
            "mean/input_tokens",
            "mean/output_tokens",
            "rollout_failure_rate",
        }
        assert key["mean/reward"] == approx(0.65)
        assert key["win_rate"] == approx(0.55)
        assert key["style_control"] is True
        assert key["rollout_failure_rate"] == approx(0.0)

    def test_get_key_metrics_missing_keys(self, server: ArenaResourcesServer):
        """Missing keys are silently omitted."""
        key = server.get_key_metrics({"mean/reward": 0.5})
        assert key == {"mean/reward": approx(0.5)}
