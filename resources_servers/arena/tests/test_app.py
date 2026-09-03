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
import tiktoken
import yaml
from openai.types.responses.response import IncompleteDetails
from pydantic import ValidationError
from pytest import approx, fixture, mark, raises

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
    _extract_style_counts,
    _extract_thinking_content,
    _extract_verdict,
    _fit_bt_with_offset,
    _score_verdict_as_a,
    _score_verdict_as_b,
    _strip_thinking_blocks,
    _weighted_scores_as_a,
    _weighted_scores_as_b,
)
from resources_servers.arena.metrics import ArenaMetrics
from resources_servers.arena.scripts.remove_failed_rollouts import is_failed_rollout


def _arena_config(name: str) -> dict:
    path = Path(__file__).parent.parent / "configs" / f"{name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)[name]["resources_servers"]["arena"]


_LMARENA_V2_CFG = _arena_config("lmarena_v2")
_LMARENA_V3_CFG = _arena_config("lmarena_v3")
_TEST_STYLE_NORM_MEAN: dict[str, list[float]] = _LMARENA_V2_CFG["style_norm_mean"]
_TEST_STYLE_NORM_STD: dict[str, list[float]] = _LMARENA_V2_CFG["style_norm_std"]
_TEST_STYLE_COEFS: dict[str, list[float]] = _LMARENA_V2_CFG["style_coefs"]
_TEST_ENCODING = tiktoken.encoding_for_model("gpt-4o")


# _score_verdict_as_a/b and _weighted_scores_as_a/b are used by unit tests below.


# ── Helpers ───────────────────────────────────────────────────────────────────


def test_lmarena_versions_are_explicit() -> None:
    assert _LMARENA_V2_CFG["verdict_weight"] == 3
    assert _LMARENA_V2_CFG["score_both_bad_as_tie"] is False
    assert _LMARENA_V3_CFG["verdict_weight"] == 1
    assert _LMARENA_V3_CFG["score_both_bad_as_tie"] is True
    assert _LMARENA_V3_CFG["style_control_method"] == "reference_length"
    assert _LMARENA_V3_CFG["style_length_ratio_range"] == [0.5, 1.75]
    assert _LMARENA_V3_CFG["style_short_reference_max_tokens"] == 100
    assert _LMARENA_V3_CFG["style_short_response_max_tokens"] == 175
    assert _LMARENA_V2_CFG["policy_model_aliases"] == []
    assert _LMARENA_V3_CFG["policy_model_aliases"] == []


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
        assert _score_verdict_as_a("[[BB]]") == approx(0.5)

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
        assert _score_verdict_as_b("[[BB]]") == approx(0.5)

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
    def test_extract_style_counts_plain_text(self):
        counts = _extract_style_counts("hello world", _TEST_ENCODING)
        assert counts[0] > 0
        assert np.array_equal(counts[1:], [0, 0, 0])

    def test_extract_style_counts_markdown(self):
        text = "## Header\n- item one\n- item two\n**bold text**"
        counts = _extract_style_counts(text, _TEST_ENCODING)
        assert np.array_equal(counts[1:], [1, 2, 1])

    def test_extract_style_counts_strips_code_blocks(self):
        # Code blocks should be stripped before counting markdown elements.
        text = "```\n## fake header\n- fake list\n```\n## Real header"
        counts = _extract_style_counts(text, _TEST_ENCODING)
        assert np.array_equal(counts[1:3], [1, 0])

    def test_compute_raw_style_feature_shape(self):
        feat = _compute_raw_style_feature(
            "short answer", "a much longer baseline answer with lots of words", _TEST_ENCODING
        )
        assert feat.shape == (4,)

    def test_compute_raw_style_feature_length_direction(self):
        # Longer policy answer → positive length feature.
        feat = _compute_raw_style_feature("a " * 200, "b", _TEST_ENCODING)
        assert feat[0] > 0

    def test_compute_raw_style_feature_identical_texts(self):
        # Identical texts → zero differentials.
        text = "## Hello\n- item\n**bold**\n" * 10
        feat = _compute_raw_style_feature(text, text, _TEST_ENCODING)
        assert np.allclose(feat[0], 0.0, atol=1e-6)
        assert np.allclose(feat[1:], 0.0, atol=1e-6)

    def test_compute_raw_style_feature_both_empty(self):
        # Both texts empty → all features zero (no division by zero).
        feat = _compute_raw_style_feature("", "", _TEST_ENCODING)
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

    def test_bootstrap_resamples_individual_games(self):
        score_groups = [np.array([0.0, 1.0]) for _ in range(100)]
        pt_est, ci_lower, ci_upper = _bootstrap(score_groups, n_rounds=20)
        assert pt_est == approx(0.49625)
        assert ci_lower == approx(0.42375)
        assert ci_upper == approx(0.545)

    def test_bootstrap_with_offset_shape(self):
        rng = np.random.RandomState(0)
        scores = rng.uniform(0, 1, 200)
        offsets = rng.normal(0, 0.1, 200)
        score_groups = [scores[index : index + 2] for index in range(0, len(scores), 2)]
        offset_groups = [offsets[index : index + 2] for index in range(0, len(offsets), 2)]
        pt_est, ci_lower, ci_upper = _bootstrap(score_groups, offset_groups, n_rounds=20)
        assert 0.0 < pt_est < 1.0
        assert ci_lower <= pt_est <= ci_upper


class TestStripThinkingBlocks:
    def test_strips_think_block(self):
        assert _strip_thinking_blocks("<think>reasoning</think>answer") == "answer"

    def test_strips_thinking_block(self):
        assert _strip_thinking_blocks("<thinking>deep thought</thinking>answer") == "answer"

    def test_strips_multiline_block(self):
        assert _strip_thinking_blocks("<think>\nline1\nline2\n</think>result") == "result"

    @mark.parametrize("tag", ["<think>", "<thinking>"])
    def test_unclosed_block_hides_remaining_text(self, tag):
        assert _strip_thinking_blocks(f"answer{tag}private reasoning") == "answer"

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

    @mark.parametrize("tag", ["<think>", "<thinking>"])
    def test_extracts_unclosed_block(self, tag):
        assert _extract_thinking_content(f"answer{tag}private reasoning") == "private reasoning"

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
            category="lmarena_v2",
        )

    def test_baseline_answer_plain_string(self):
        req = self._make_request("plain text answer")
        assert req.baseline_answer == "plain text answer"


class TestRemoveFailedRollouts:
    def test_v3_max_output_tokens_is_preserved(self):
        row = {
            "category": "lmarena_v3",
            "response": {"incomplete_details": {"reason": "max_output_tokens"}},
            "games": None,
        }
        assert is_failed_rollout(row) is False

    def test_v2_max_output_tokens_is_removed(self):
        row = {
            "category": "lmarena_v2",
            "response": {"incomplete_details": {"reason": "max_output_tokens"}},
            "games": None,
        }
        assert is_failed_rollout(row) is True

    def test_missing_games_is_removed(self):
        assert is_failed_rollout({"category": "lmarena_v3", "games": None}) is True


# ── Server tests ──────────────────────────────────────────────────────────────


class TestArenaResourcesServer:
    @fixture
    def config(self) -> ArenaResourcesServerConfig:
        return ArenaResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            num_workers=None,
            entrypoint="",
            domain="rlhf",
            name="",
            verified=False,
            description="Test Arena benchmark",
            value="Test pairwise responses",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            max_rollout_failure_rate=0.01,
            judge_concurrency=16,
            generation_only=False,
            policy_model_aliases=[],
            verdict_weight=3,
            score_both_bad_as_tie=True,
            style_control_method="bradley_terry",
            style_length_ratio_range=None,
            style_short_reference_max_tokens=None,
            style_short_response_max_tokens=None,
            judge_prompt_template="Question: {question}\nA: {answer_a}\nB: {answer_b}",
            judge_system_message="You are an impartial judge.",
            judge_system_message_by_category={},
            judge_timeout_secs=1800,
            tokenizer_model="gpt-4o",
            bootstrap_rounds=100,
            bootstrap_seed=42,
            style_norm_mean=_TEST_STYLE_NORM_MEAN,
            style_norm_std=_TEST_STYLE_NORM_STD,
            style_coefs=_TEST_STYLE_COEFS,
        )

    @fixture
    def server(self, config: ArenaResourcesServerConfig) -> ArenaResourcesServer:
        mock_client = MagicMock(spec=ServerClient)
        return ArenaResourcesServer(config=config, server_client=mock_client)

    @mark.parametrize(
        ("field", "value"),
        [
            ("verdict_weight", 0),
            ("max_rollout_failure_rate", -0.1),
            ("max_rollout_failure_rate", 1.1),
            ("judge_concurrency", 0),
            ("judge_timeout_secs", 0),
            ("bootstrap_rounds", 0),
        ],
    )
    def test_invalid_numeric_config_is_rejected(self, config, field, value):
        with raises(ValidationError):
            ArenaResourcesServerConfig.model_validate(config.model_dump() | {field: value})

    def _make_verify_request(
        self,
        policy_response_text: str,
        question: str = "What is the capital of France?",
        baseline_answer: str = "Paris.",
        baseline_model: str | None = None,
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
            baseline_model=baseline_model,
            category="lmarena_v2",
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

    async def test_verify_self_comparison_from_response_model_skips_judge(self, server: ArenaResourcesServer):
        server.config.style_control_method = "reference_length"
        request = self._make_verify_request("Paris.", baseline_model="test-model")

        result = await server.verify(request)

        assert result.self_comparison is True
        assert result.reward == approx(0.0)
        assert result.games is None
        server.server_client.post.assert_not_called()

    async def test_verify_self_comparison_from_configured_alias_skips_judge(self, server: ArenaResourcesServer):
        server.config.style_control_method = "reference_length"
        server.config.policy_model_aliases = ["policy-alias"]
        request = self._make_verify_request("Paris.", baseline_model="policy-alias")

        result = await server.verify(request)

        assert result.self_comparison is True
        server.server_client.post.assert_not_called()

    async def test_verify_different_baseline_model_is_judged(self, server: ArenaResourcesServer):
        server.config.style_control_method = "reference_length"
        self._setup_judge_responses(server, ["[[A=B]]", "[[A=B]]"])
        request = self._make_verify_request("Paris.", baseline_model="different-model")

        result = await server.verify(request)

        assert result.self_comparison is False
        assert len(result.games) == 2

    async def test_verify_same_model_suffix_in_different_namespace_is_judged(self, server: ArenaResourcesServer):
        server.config.style_control_method = "reference_length"
        self._setup_judge_responses(server, ["[[A=B]]", "[[A=B]]"])
        request = self._make_verify_request("Paris.", baseline_model="meta-llama/Llama-3.1-70B-Instruct")
        request.response = request.response.model_copy(update={"model": "my-finetunes/Llama-3.1-70B-Instruct"})

        result = await server.verify(request)

        assert result.self_comparison is False
        assert len(result.games) == 2

    async def test_verify_v2_preserves_model_match_scoring(self, server: ArenaResourcesServer):
        self._setup_judge_responses(server, ["[[A=B]]", "[[A=B]]"])
        request = self._make_verify_request("Paris.", baseline_model="test-model")

        result = await server.verify(request)

        assert result.self_comparison is False
        assert len(result.games) == 2

    async def test_verify_whitespace_only_response_returns_zero_reward(self, server: ArenaResourcesServer):
        request = self._make_verify_request("   \n  ")
        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.policy_answer is None
        # No judge calls should have been made for an empty/whitespace response
        server.server_client.post.assert_not_called()

    async def test_verify_v3_max_token_response_returns_zero_without_judging(self, server: ArenaResourcesServer):
        server.config.style_control_method = "reference_length"
        request = self._make_verify_request("Truncated answer")
        request.response.incomplete_details = IncompleteDetails(reason="max_output_tokens")

        result = await server.verify(request)

        assert result.reward == approx(0.0)
        assert result.policy_answer == "Truncated answer"
        assert result.games is None
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

    async def test_verify_retains_compact_prompt_slices(self, server: ArenaResourcesServer):
        request = self._make_verify_request("Paris.")
        request.metadata = {
            "user_language": "en",
            "tags": {"coding_v2": {"value": True}},
            "taxonomy": [{"natural_language": "English", "task_type": "Fact-seeking QA"}],
        }
        request.is_lmarena_v2_prompt = True
        self._setup_judge_responses(server, ["[[A=B]]", "[[A=B]]"])

        result = await server.verify(request)

        assert result.prompt_slices == {
            "arena": ["coding", "english"],
            "taxonomy-language": ["English"],
            "taxonomy-task-type": ["Fact-seeking QA"],
        }
        assert result.is_lmarena_v2_prompt is True
        assert "metadata" not in result.model_dump()

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
        # [[BB]] → 0.5 for both A and B positions
        self._setup_judge_responses(server, ["[[BB]]", "[[BB]]"])

        result = await server.verify(self._make_verify_request("Answer."))
        assert result.reward == approx(0.5)

    async def test_verify_unparseable_judge_output(self, server: ArenaResourcesServer):
        # Judge doesn't output a valid verdict label → 0.0 for both rounds
        self._setup_judge_responses(server, ["I cannot decide.", "unclear"])

        result = await server.verify(self._make_verify_request("Answer."))
        assert result.reward == approx(0.0)
        assert result.games[0].verdict is None
        assert result.games[1].verdict is None

    async def test_verify_generation_only_skips_judge(self, server: ArenaResourcesServer):
        server.config.generation_only = True

        result = await server.verify(self._make_verify_request("<think>hidden</think>Answer."))

        assert result.reward == approx(0.0)
        assert result.policy_answer == "Answer."
        assert result.policy_reasoning == "hidden"
        assert result.games is None
        server.server_client.post.assert_not_called()

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
                "failure_reason",
                "question_id",
                "question",
                "baseline_answer",
                "baseline_model",
                "category",
                "style_reference_token_count",
                "policy_answer",
                "policy_reasoning",
                "games",
                "self_comparison",
                "prompt_slices",
                "is_lmarena_v2_prompt",
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
            category="lmarena_v2",
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

    def _rollout(self, v1: str, v2: str, reward: float = 0.5, category: str = "lmarena_v2") -> dict:
        """Build a minimal rollout dict with two verdict games and stub text answers."""
        r: dict = {
            "reward": reward,
            "policy_answer": "Policy answer text for style feature extraction.",
            "policy_reasoning": "Policy reasoning text.",
            "baseline_answer": "Baseline answer text for style feature extraction.",
            "games": [{"verdict": v1}, {"verdict": v2}],
        }
        r["category"] = category
        return r

    @fixture
    def server_no_sc(self, config: ArenaResourcesServerConfig) -> ArenaResourcesServer:
        """Server fixture for tests that inspect no-style-control metrics."""
        return ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_compute_metrics_all_wins(self, server: ArenaResourcesServer):
        rollout = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server.compute_metrics([[rollout], [rollout]])
        assert metrics["rollout_failure_rate"] == approx(0.0)
        # Style-controlled win rate ≈ 1.0 when all battles are wins.
        assert metrics["win_rate"] == approx(1.0, abs=0.05)
        assert metrics["win_rate_ci95_lower"] <= metrics["win_rate"]
        assert metrics["win_rate_ci95_upper"] >= metrics["win_rate"]

    def test_compute_metrics_all_losses(self, server: ArenaResourcesServer):
        rollout = self._rollout("[[B>A]]", "[[A>B]]", reward=0.0)
        metrics = server.compute_metrics([[rollout]])
        assert metrics["win_rate"] == approx(0.0, abs=0.05)
        assert metrics["rollout_failure_rate"] == approx(0.0)

    def test_compute_metrics_token_stats(self, server_no_sc: ArenaResourcesServer):
        server_no_sc.config.max_rollout_failure_rate = 1.0
        rollout_with_reasoning = {
            "reward": 1.0,
            "policy_answer": "Final answer.",
            "policy_reasoning": "Internal reasoning.",
            "baseline_answer": "Baseline answer.",
            "category": "lmarena_v2",
            "games": [{"verdict": "[[A>B]]"}, {"verdict": "[[B>A]]"}],
        }
        rollout_empty = {
            "reward": 0.0,
            "policy_answer": "",
            "policy_reasoning": None,
            "baseline_answer": "Baseline answer.",
            "category": "lmarena_v2",
            "games": None,
        }
        metrics = server_no_sc.compute_metrics([[rollout_with_reasoning], [rollout_empty]])

        encoding = tiktoken.encoding_for_model("gpt-4o")
        response_counts = np.array([len(encoding.encode("Final answer.")), 0])
        reasoning_counts = np.array([len(encoding.encode("Internal reasoning.")), 0])
        for name, counts in (("response_tokens", response_counts), ("reasoning_tokens", reasoning_counts)):
            assert metrics[f"{name}/mean"] == approx(float(np.mean(counts)))
            assert metrics[f"{name}/median"] == approx(float(np.median(counts)))
            assert metrics[f"{name}/min"] == int(np.min(counts))
            assert metrics[f"{name}/max"] == int(np.max(counts))
            assert metrics[f"{name}/p5"] == approx(float(np.percentile(counts, 5)))
            assert metrics[f"{name}/p95"] == approx(float(np.percentile(counts, 95)))

    def test_compute_metrics_generation_only_reports_token_stats_only(self, server: ArenaResourcesServer):
        server.config.generation_only = True
        rollout = {
            "reward": 0.0,
            "policy_answer": "Final answer.",
            "policy_reasoning": "Internal reasoning.",
            "baseline_answer": "Baseline answer.",
            "games": None,
        }

        metrics = server.compute_metrics([[rollout]])

        assert "response_tokens/mean" in metrics
        assert "reasoning_tokens/mean" in metrics
        assert "mean/reward" not in metrics
        assert "rollout_failure_rate" not in metrics
        assert "win_rate" not in metrics

    def test_compute_metrics_plain_bootstrap_weighted(self, server_no_sc: ArenaResourcesServer):
        """win_rate_no_SC is a plain bootstrap mean of weighted scores."""
        # Strong loss (6 items × 0) + weak win (2 items × 1) → plain mean = 0.25
        rollout_strong_loss = {
            "reward": 0.0,
            "policy_answer": "p",
            "policy_reasoning": None,
            "baseline_answer": "b",
            "category": "lmarena_v2",
            "games": [{"verdict": "[[B>>A]]"}, {"verdict": "[[A>>B]]"}],
        }
        rollout_weak_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server_no_sc.compute_metrics([[rollout_strong_loss], [rollout_weak_win]])
        assert metrics["win_rate_no_SC"] == approx(0.25, abs=0.05)

    def test_compute_metrics_parse_failures(self, server: ArenaResourcesServer):
        # Disable the failure-rate guard so we can inspect the metric values directly.
        server.config.max_rollout_failure_rate = 1.0
        rollout = {
            "reward": 0.0,
            "policy_answer": "x",
            "policy_reasoning": "",
            "baseline_answer": "y",
            "games": [
                {"response": {"output": [{"type": "message"}]}, "verdict": None},
                {"response": {"output": [{"type": "message"}]}, "verdict": None},
            ],
        }
        metrics = server.compute_metrics([[rollout]])
        assert metrics["rollout_failure_rate"] == approx(1.0)
        assert metrics["missing_judgment_rate"] == approx(0.0)
        assert metrics["parse_failure_rate"] == approx(1.0)
        assert "win_rate" not in metrics

    def test_compute_metrics_missing_judgment_takes_precedence(self, server: ArenaResourcesServer):
        server.config.max_rollout_failure_rate = 1.0
        rollout = {
            "reward": 0.0,
            "policy_answer": "x",
            "policy_reasoning": "",
            "baseline_answer": "y",
            "games": [
                {"response": {"output": [{"type": "message"}]}, "verdict": None},
                {"response": {"output": []}, "verdict": None},
            ],
        }
        metrics = server.compute_metrics([[rollout]])
        assert metrics["rollout_failure_rate"] == approx(1.0)
        assert metrics["missing_judgment_rate"] == approx(1.0)
        assert metrics["parse_failure_rate"] == approx(0.0)

    def test_compute_metrics_no_games(self, server: ArenaResourcesServer):
        # Disable the failure-rate guard: a single games=None rollout is 100% failure.
        server.config.max_rollout_failure_rate = 1.0
        rollout = {
            "reward": 0.0,
            "policy_answer": None,
            "policy_reasoning": "",
            "baseline_answer": "y",
            "games": None,
        }
        metrics = server.compute_metrics([[rollout]])
        assert metrics["rollout_failure_rate"] == approx(1.0)
        assert metrics["missing_judgment_rate"] == approx(1.0)
        assert metrics["parse_failure_rate"] == approx(0.0)

    def test_compute_metrics_counts_both_bad_as_tie(self, server_no_sc: ArenaResourcesServer):
        """[[BB]] rollouts count as ties and are still tracked separately."""
        rollout_bb = {
            "reward": 0.0,
            "policy_answer": "x",
            "policy_reasoning": "",
            "baseline_answer": "y",
            "category": "lmarena_v2",
            "games": [{"verdict": "[[BB]]"}, {"verdict": "[[BB]]"}],
        }
        rollout_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server_no_sc.compute_metrics([[rollout_bb], [rollout_win]])
        assert metrics["rollout_failure_rate"] == approx(0.0)
        assert metrics["any_both_bad_rate"] == approx(0.5)
        assert metrics["any_tie_rate"] == approx(0.5)
        assert metrics["win_rate_no_SC"] == approx(0.75, abs=0.05)

    def test_compute_metrics_v2_excludes_both_bad_from_win_rates(self, server_no_sc: ArenaResourcesServer):
        """Preserve v2: verification rewards [[BB]] as 0.5, but win rates exclude it."""
        server_no_sc.config.score_both_bad_as_tie = False
        rollout_bb = self._rollout("[[BB]]", "[[BB]]", reward=0.5)
        rollout_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)

        metrics = server_no_sc.compute_metrics([[rollout_bb], [rollout_win]])

        assert metrics["any_both_bad_rate"] == approx(0.5)
        assert metrics["win_rate_no_SC"] == approx(1.0)
        assert metrics["win_rate"] == approx(1.0)

    def test_compute_metrics_counts_partial_bb_as_tie(self, server_no_sc: ArenaResourcesServer):
        """A rollout where only one game has [[BB]] keeps the non-BB game and a tie game."""
        rollout_partial_bb = {
            "reward": 0.0,
            "policy_answer": "x",
            "policy_reasoning": "",
            "baseline_answer": "y",
            "category": "lmarena_v2",
            "games": [{"verdict": "[[A>B]]"}, {"verdict": "[[BB]]"}],
        }
        rollout_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server_no_sc.compute_metrics([[rollout_partial_bb], [rollout_win]])
        assert metrics["any_both_bad_rate"] == approx(0.5)
        assert metrics["any_tie_rate"] == approx(0.5)
        assert metrics["win_rate_no_SC"] == approx(0.875, abs=0.05)

    def test_compute_metrics_single_game_rollout_excluded(self, server_no_sc: ArenaResourcesServer):
        """Rollouts with only one game are not accumulated into aggregate win rates."""
        server_no_sc.config.max_rollout_failure_rate = 1.0
        rollout_one_game = {
            "reward": 0.5,
            "policy_answer": "x",
            "policy_reasoning": "",
            "baseline_answer": "y",
            "games": [{"verdict": "[[A>B]]"}],
        }
        rollout_win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server_no_sc.compute_metrics([[rollout_one_game], [rollout_win]])
        assert metrics["rollout_failure_rate"] == approx(0.5)
        assert metrics["missing_judgment_rate"] == approx(0.5)
        assert metrics["parse_failure_rate"] == approx(0.0)
        assert metrics["win_rate_no_SC"] == approx(1.0, abs=0.05)

    def test_compute_metrics_raises_on_high_failure_rate(self, server: ArenaResourcesServer):
        """Raises ValueError when failed rollouts exceed max_rollout_failure_rate — no score returned."""
        good = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        # 2 good + 1 answer-failure = 33% failure rate, exceeds 1% default
        failed = {
            "reward": 0.0,
            "policy_answer": None,
            "policy_reasoning": "",
            "baseline_answer": "b",
            "games": None,
        }
        with raises(ValueError, match="max_rollout_failure_rate"):
            server.compute_metrics([[good], [good], [failed]])

    def test_compute_metrics_raises_on_high_judge_failure_rate(self, server: ArenaResourcesServer):
        """Judge parse failures also count toward the failure rate and trigger ValueError."""
        good = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        judge_fail = {
            "reward": 0.0,
            "policy_answer": "x",
            "policy_reasoning": "",
            "baseline_answer": "y",
            "games": [{"verdict": None}, {"verdict": None}],
        }
        with raises(ValueError, match="max_rollout_failure_rate"):
            server.compute_metrics([[good], [good], [judge_fail]])

    def test_compute_metrics_within_failure_tolerance(self, server: ArenaResourcesServer):
        """No error when failure rate is within max_rollout_failure_rate."""
        good = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        failed = {
            "reward": 0.0,
            "policy_answer": None,
            "policy_reasoning": "",
            "baseline_answer": "b",
            "games": None,
        }
        # 1 failure in 200 rollouts = 0.5% < 1% default
        server.config.max_rollout_failure_rate = 0.01
        metrics = server.compute_metrics([[good]] * 199 + [[failed]])
        assert "win_rate" in metrics

    def test_compute_metrics_style_controlled_win_rate(self, server: ArenaResourcesServer):
        """win_rate is a style-controlled BT win probability with CI."""
        rollout = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        metrics = server.compute_metrics([[rollout] * 20])
        assert 0.0 < metrics["win_rate"] < 1.0
        assert metrics["win_rate_ci95_lower"] <= metrics["win_rate"]
        assert metrics["win_rate_ci95_upper"] >= metrics["win_rate"]

    def test_compute_metrics_reference_length(self, config: ArenaResourcesServerConfig):
        config.style_control_method = "reference_length"
        config.style_length_ratio_range = (0.5, 1.5)
        config.style_norm_mean = None
        config.style_norm_std = None
        config.style_coefs = None
        reference_server = ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        accepted = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        accepted["style_reference_token_count"] = 10
        rejected = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        rejected["policy_answer"] = "word " * 100
        rejected["style_reference_token_count"] = 10

        metrics = reference_server.compute_metrics([[accepted], [rejected]])

        assert metrics["win_rate_no_SC"] == approx(1.0, abs=0.05)
        assert metrics["win_rate"] == approx(0.5, abs=0.05)
        assert metrics["verbosity_acceptance_rate"] == approx(0.5)

    def test_reference_length_boundaries_are_exclusive(self, config: ArenaResourcesServerConfig):
        config.style_control_method = "reference_length"
        config.style_length_ratio_range = (0.5, 1.75)
        config.style_short_reference_max_tokens = None
        config.style_short_response_max_tokens = None
        metrics = ArenaMetrics(config)

        assert not metrics._is_comparable_to_reference_length(50, 100)
        assert metrics._is_comparable_to_reference_length(51, 100)
        assert metrics._is_comparable_to_reference_length(174, 100)
        assert not metrics._is_comparable_to_reference_length(175, 100)

    def test_compute_metrics_reference_length_short_rule(self, config: ArenaResourcesServerConfig):
        config.style_control_method = "reference_length"
        config.style_length_ratio_range = (0.5, 1.75)
        config.style_short_reference_max_tokens = 100
        config.style_short_response_max_tokens = 175
        config.style_norm_mean = None
        config.style_norm_std = None
        config.style_coefs = None
        reference_server = ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        accepted = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        accepted["style_reference_token_count"] = 10
        rejected = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        rejected["policy_answer"] = "word " * 500
        rejected["style_reference_token_count"] = 10

        metrics = reference_server.compute_metrics([[accepted], [rejected]])

        assert metrics["win_rate"] == approx(0.5, abs=0.05)
        assert metrics["verbosity_acceptance_rate"] == approx(0.5)

    def test_compute_metrics_v3_max_token_response_scores_zero(self, config: ArenaResourcesServerConfig):
        config.style_control_method = "reference_length"
        config.style_length_ratio_range = (0.5, 1.75)
        config.style_short_reference_max_tokens = 100
        config.style_short_response_max_tokens = 175
        config.style_norm_mean = None
        config.style_norm_std = None
        config.style_coefs = None
        reference_server = ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        win = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        win["style_reference_token_count"] = 10
        truncated = {
            "reward": 0.0,
            "response": {
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"output_tokens": 32768},
            },
            "policy_answer": "Truncated answer",
            "policy_reasoning": "",
            "baseline_answer": "Baseline answer",
            "category": "lmarena_v2",
            "style_reference_token_count": 10,
            "games": None,
        }
        context_exceeded = truncated | {
            "response": {"incomplete_details": {"reason": "max_output_tokens"}, "usage": None}
        }

        metrics = reference_server.compute_metrics([[win], [truncated], [context_exceeded]])

        assert metrics["max_token_reached_rate"] == approx(1 / 3)
        assert metrics["context_window_exceeded_rate"] == approx(1 / 3)
        assert metrics["rollout_failure_rate"] == approx(0.0)
        assert metrics["missing_judgment_rate"] == approx(0.0)
        assert metrics["win_rate_no_SC"] == approx(1 / 3, abs=0.05)
        assert metrics["win_rate"] == approx(1 / 3, abs=0.05)
        assert metrics["verbosity_acceptance_rate"] == approx(1 / 3)

    def test_compute_metrics_v2_max_token_response_remains_failure(self, server: ArenaResourcesServer):
        server.config.max_rollout_failure_rate = 1.0
        truncated = {
            "reward": 0.0,
            "response": {"incomplete_details": {"reason": "max_output_tokens"}},
            "policy_answer": "Truncated answer",
            "policy_reasoning": "",
            "baseline_answer": "Baseline answer",
            "category": "lmarena_v2",
            "games": None,
        }

        metrics = server.compute_metrics([[truncated]])

        assert metrics["rollout_failure_rate"] == approx(1.0)
        assert "max_token_reached_rate" not in metrics
        assert "context_window_exceeded_rate" not in metrics

    def test_compute_metrics_prompt_slices(self, server: ArenaResourcesServer):
        rollout = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        rollout["prompt_slices"] = {
            "arena": ["english"],
            "taxonomy-language": ["English"],
            "taxonomy-task-type": ["Fact-seeking QA"],
        }
        rollout["is_lmarena_v2_prompt"] = True

        metrics = server.compute_metrics([[rollout]] * 50)

        assert metrics["total_prompts"] == 50
        assert metrics["arena/english/prompts"] == 50
        assert metrics["taxonomy-language/English/prompts"] == 50
        assert metrics["taxonomy-task-type/Fact-seeking QA/prompts"] == 50
        assert metrics["arena/english/win_rate_no_SC"] == approx(1.0, abs=0.05)
        assert metrics["win_rate_lmarena_v2_prompts"] == approx(1.0, abs=0.05)
        assert metrics["win_rate_no_SC_lmarena_v2_prompts"] == approx(1.0, abs=0.05)

    def test_compute_metrics_prompt_slice_verbosity(self, config: ArenaResourcesServerConfig):
        config.style_control_method = "reference_length"
        config.style_length_ratio_range = (0.5, 1.75)
        config.style_short_reference_max_tokens = 100
        config.style_short_response_max_tokens = 175
        config.style_norm_mean = None
        config.style_norm_std = None
        config.style_coefs = None
        reference_server = ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        accepted = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        accepted["style_reference_token_count"] = 10
        accepted["prompt_slices"] = {"arena": ["english"]}
        rejected = self._rollout("[[A>B]]", "[[B>A]]", reward=1.0)
        rejected["policy_answer"] = "word " * 500
        rejected["style_reference_token_count"] = 10
        rejected["prompt_slices"] = {"arena": ["english"]}

        metrics = reference_server.compute_metrics([[accepted]] * 25 + [[rejected]] * 25)

        assert metrics["arena/overall/verbosity_acceptance_rate"] == approx(0.5)
        assert metrics["arena/english/verbosity_acceptance_rate"] == approx(0.5)

    def test_bradley_terry_style_constants_require_exact_category(self, server: ArenaResourcesServer):
        with raises(ValueError, match="unknown_category"):
            server._metrics._get_bradley_terry_constants("unknown_category")
        server._metrics._get_bradley_terry_constants("lmarena_v2")

    def test_bradley_terry_style_constants_have_no_implicit_default(self, config: ArenaResourcesServerConfig):
        """Unknown category with no explicit fallback is a configuration error."""
        config.style_norm_mean = {"lmarena_v2": [0.1, 0.0, 0.0, 0.0]}
        config.style_norm_std = {"lmarena_v2": [1.0, 1.0, 1.0, 1.0]}
        config.style_coefs = {"lmarena_v2": [0.3, 0.1, -0.2, 0.0]}
        metrics = ArenaMetrics(config)
        with raises(ValueError, match="unknown_category"):
            metrics._get_bradley_terry_constants("unknown_category")
        metrics._get_bradley_terry_constants("lmarena_v2")

    # ── get_key_metrics() ─────────────────────────────────────────────────────

    def test_get_key_metrics(self, server: ArenaResourcesServer):
        agent_metrics = {
            "mean/reward": 0.65,
            "win_rate": 0.55,
            "win_rate_ci95_lower": 0.49,
            "win_rate_ci95_upper": 0.61,
            "win_rate_no_SC": 0.58,
            "win_rate_no_SC_ci95_lower": 0.51,
            "win_rate_no_SC_ci95_upper": 0.65,
            "win_rate_lmarena_v2_prompts": 0.56,
            "win_rate_no_SC_lmarena_v2_prompts": 0.59,
            "mean/input_tokens": 512.0,
            "mean/output_tokens": 256.0,
            "response_tokens/mean": 300.0,
            "response_tokens/median": 200.4,
            "response_tokens/min": 0,
            "response_tokens/max": 1000,
            "response_tokens/p5": 10.0,
            "response_tokens/p95": 900.0,
            "reasoning_tokens/mean": 40.0,
            "reasoning_tokens/median": 20.0,
            "reasoning_tokens/min": 0,
            "reasoning_tokens/max": 300,
            "reasoning_tokens/p5": 0.0,
            "reasoning_tokens/p95": 250.123456,
            "max_token_reached_rate": 0.03,
            "context_window_exceeded_rate": 0.02,
            "rollout_failure_rate": 0.0,
            "missing_judgment_rate": 0.0,
            "parse_failure_rate": 0.0,
            "any_both_bad_rate": 0.1,
            "any_tie_rate": 0.2,
            "verbosity_acceptance_rate": 0.8,
            "total_prompts": 100,
            "arena/overall/win_rate": 0.55,
            "arena/overall/win_rate_no_SC": 0.58,
            "arena/overall/verbosity_acceptance_rate": 0.8,
            "taxonomy-language/English/prompts": 60,
            "taxonomy-language/English/win_rate": 0.6,
            "taxonomy-language/English/win_rate_no_SC": 0.62,
            "std/reward": 0.1,
            "something_else": 42,
        }
        key = server.get_key_metrics(agent_metrics)
        assert set(key.keys()) == {
            "mean/reward",
            "win_rate",
            "win_rate_ci95_lower",
            "win_rate_ci95_upper",
            "win_rate_no_SC",
            "win_rate_no_SC_ci95_lower",
            "win_rate_no_SC_ci95_upper",
            "win_rate_lmarena_v2_prompts",
            "win_rate_no_SC_lmarena_v2_prompts",
            "response_tokens/mean",
            "response_tokens/median",
            "response_tokens/min",
            "response_tokens/max",
            "response_tokens/p5",
            "response_tokens/p95",
            "reasoning_tokens/mean",
            "reasoning_tokens/median",
            "reasoning_tokens/min",
            "reasoning_tokens/max",
            "reasoning_tokens/p5",
            "reasoning_tokens/p95",
            "max_token_reached_rate",
            "context_window_exceeded_rate",
            "rollout_failure_rate",
            "missing_judgment_rate",
            "parse_failure_rate",
            "any_both_bad_rate",
            "any_tie_rate",
            "verbosity_acceptance_rate",
            "total_prompts",
            "arena/overall/win_rate",
            "arena/overall/win_rate_no_SC",
            "arena/overall/verbosity_acceptance_rate",
            "taxonomy-language/English/prompts",
            "taxonomy-language/English/win_rate",
            "taxonomy-language/English/win_rate_no_SC",
        }
        assert list(key)[:9] == [
            "total_prompts",
            "win_rate",
            "win_rate_ci95_lower",
            "win_rate_ci95_upper",
            "win_rate_no_SC",
            "win_rate_no_SC_ci95_lower",
            "win_rate_no_SC_ci95_upper",
            "win_rate_lmarena_v2_prompts",
            "win_rate_no_SC_lmarena_v2_prompts",
        ]
        assert key["mean/reward"] == approx(0.65)
        assert key["win_rate"] == approx(0.55)
        assert key["response_tokens/median"] == 200
        assert key["reasoning_tokens/p95"] == 250
        assert "style_control" not in key
        assert key["rollout_failure_rate"] == approx(0.0)

    def test_get_key_metrics_missing_keys(self, server: ArenaResourcesServer):
        """Missing keys are silently omitted."""
        key = server.get_key_metrics({"mean/reward": 0.5})
        assert key == {"mean/reward": approx(0.5)}

    def test_get_key_metrics_generation_only(self, server: ArenaResourcesServer):
        server.config.generation_only = True
        key = server.get_key_metrics(
            {
                "mean/reward": 0.0,
                "win_rate": 0.5,
                "response_tokens/mean": 10.0,
                "reasoning_tokens/mean": 2.0,
            }
        )
        assert key == {"response_tokens/mean": approx(10.0), "reasoning_tokens/mean": approx(2.0)}
