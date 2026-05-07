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
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.arena_judge.app import (
    ArenaJudgeConfig,
    ArenaJudgeServer,
    ArenaJudgeVerifyRequest,
    sanitize_generation,
)


def _make_response(text: str, model: str = "mock_model") -> NeMoGymResponse:
    """Build a NeMoGymResponse mock — used as the POLICY model's output
    (the candidate answer the judge scores). The policy server is
    expected to return a Responses API shape, whatever backend it runs.
    """
    return NeMoGymResponse(
        id="mock_resp",
        created_at=0.0,
        model=model,
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="mock_msg",
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


def _make_chat_completion(text: str, model: str = "mock_judge") -> dict:
    """Build a raw ChatCompletion dict — what server_client.post returns
    for the JUDGE call via /v1/chat/completions."""
    return {
        "id": "mock_chat",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class TestSanitizeGeneration:
    """Helper that scrubs UTF-8 surrogate halves + NULs from generations
    before they're embedded in a judge prompt."""

    def test_clean_text_passthrough(self) -> None:
        assert sanitize_generation("hello world") == "hello world"

    def test_strips_lone_surrogates(self) -> None:
        result = sanitize_generation("hello \ud83d world")
        assert "\ud83d" not in result
        assert "hello" in result and "world" in result

    def test_preserves_valid_surrogate_pair(self) -> None:
        # 🐀 is U+1F400 — a complete surrogate pair, not a lone half.
        # Sanitization should leave valid characters untouched.
        assert sanitize_generation("hello 🐀 world") == "hello 🐀 world"

    def test_strips_lone_surrogate_pair_halves(self) -> None:
        # Both halves of a "broken" surrogate pair (high without low, low
        # without high) must be scrubbed — these are the multilingual
        # decode artifacts that trip OpenAI-compatible judge HTTP layers.
        result = sanitize_generation("\ud83d mixed \udc00")
        assert "\ud83d" not in result and "\udc00" not in result
        assert "mixed" in result


class TestParseVerdict:
    """Contract tests for _parse_verdict — mirrors arena-hard-auto's
    per-judgment score extraction (regex match, set-uniqueness, valid
    label whitelist)."""

    def test_slight_win(self) -> None:
        assert ArenaJudgeServer._parse_verdict("my final verdict is [[A>B]]") == "A>B"

    def test_strict_win(self) -> None:
        assert ArenaJudgeServer._parse_verdict("verdict: [[A>>B]]") == "A>>B"

    def test_tie(self) -> None:
        assert ArenaJudgeServer._parse_verdict("close call, [[A=B]]") == "A=B"

    def test_strict_loss(self) -> None:
        assert ArenaJudgeServer._parse_verdict("[[B>>A]]") == "B>>A"

    def test_slight_loss(self) -> None:
        assert ArenaJudgeServer._parse_verdict("[[B>A]]") == "B>A"

    def test_multiple_matches_same_verdict_ok(self) -> None:
        # len(set(matches)) == 1 is acceptable.
        assert ArenaJudgeServer._parse_verdict("first: [[A>B]] then again [[A>B]]") == "A>B"

    def test_multiple_distinct_matches_invalid(self) -> None:
        # ambiguity → None.
        assert ArenaJudgeServer._parse_verdict("[[A>B]] ... [[B>A]]") is None

    def test_no_match_invalid(self) -> None:
        assert ArenaJudgeServer._parse_verdict("no verdict was given") is None

    def test_empty_text_invalid(self) -> None:
        assert ArenaJudgeServer._parse_verdict("") is None

    def test_malformed_verdict_invalid(self) -> None:
        # Matches the regex but isn't one of the five valid labels.
        assert ArenaJudgeServer._parse_verdict("my verdict: [[A=]]") is None


class TestExtractOutputText:
    def test_response_concatenates_text_chunks(self) -> None:
        response = _make_response("Hello world")
        assert ArenaJudgeServer._extract_response_output_text(response) == "Hello world"

    def test_response_empty_output(self) -> None:
        response = NeMoGymResponse(
            id="empty",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        assert ArenaJudgeServer._extract_response_output_text(response) == ""

    def test_chat_completion_extracts_content(self) -> None:
        from nemo_gym.openai_utils import NeMoGymChatCompletion

        completion = NeMoGymChatCompletion.model_validate(_make_chat_completion("judge verdict text"))
        assert ArenaJudgeServer._extract_chat_completion_text(completion) == "judge verdict text"


class TestArenaJudgeServer:
    @fixture
    def config(self) -> ArenaJudgeConfig:
        return ArenaJudgeConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="arena_judge",
            judge_model_server=ModelServerRef(
                type="responses_api_models",
                name="arena_judge_judge_model",
            ),
            judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
        )

    def _mock_judge_sequence(self, server_mock: MagicMock, judge_texts: list[str]) -> None:
        """Configure server_client.post to return chat.completion responses
        in order — matches what /v1/chat/completions actually returns.

        The judge code path calls ``get_response_json(response)`` which
        runs ``orjson.loads(await response.read())`` — so ``read()`` must
        return the JSON-encoded bytes, not a dict.
        """
        import json as _json

        response_mocks = []
        for text in judge_texts:
            resp = AsyncMock()
            resp.read = AsyncMock(return_value=_json.dumps(_make_chat_completion(text)).encode())
            response_mocks.append(resp)
        server_mock.post = AsyncMock(side_effect=response_mocks)

    @pytest.mark.asyncio
    async def test_candidate_strict_win_gives_reward_one(self, config: ArenaJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        # First call is gen-base (A=candidate), second is base-gen (A=baseline).
        # A strict win in gen-base is consistent with B>>A in base-gen (swap).
        self._mock_judge_sequence(
            server_mock,
            ["verdict: [[A>>B]]", "verdict: [[B>>A]]"],
        )
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        result = await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("detailed candidate answer"),
                question="q",
                baseline_answer="baseline",
                category="hard_prompt",
            )
        )
        assert result.reward == approx(1.0)
        assert result.verdict_gen_base == "A>>B"
        assert result.verdict_base_gen == "B>>A"
        assert result.invalid_gen_base is False
        assert result.invalid_base_gen is False
        assert result.category == "hard_prompt"

    @pytest.mark.asyncio
    async def test_candidate_slight_win_gives_reward_one(self, config: ArenaJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        self._mock_judge_sequence(server_mock, ["[[A>B]]", "[[B>A]]"])
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        result = await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("ok"),
                question="q",
                baseline_answer="b",
                category="hard_prompt",
            )
        )
        assert result.reward == approx(1.0)
        assert result.verdict_gen_base == "A>B"

    @pytest.mark.asyncio
    async def test_tie_gives_reward_zero(self, config: ArenaJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        self._mock_judge_sequence(server_mock, ["[[A=B]]", "[[A=B]]"])
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        result = await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("ok"),
                question="q",
                baseline_answer="b",
                category="hard_prompt",
            )
        )
        assert result.reward == approx(0.0)
        assert result.verdict_gen_base == "A=B"

    @pytest.mark.asyncio
    async def test_loss_gives_reward_zero(self, config: ArenaJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        self._mock_judge_sequence(server_mock, ["[[B>>A]]", "[[A>>B]]"])
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        result = await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("ok"),
                question="q",
                baseline_answer="b",
                category="hard_prompt",
            )
        )
        assert result.reward == approx(0.0)
        assert result.verdict_gen_base == "B>>A"

    @pytest.mark.asyncio
    async def test_invalid_gen_base_verdict_gives_reward_zero(self, config: ArenaJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        # gen-base yields no verdict → invalid → reward 0
        self._mock_judge_sequence(server_mock, ["lots of text but no verdict", "[[A>B]]"])
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        result = await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("ok"),
                question="q",
                baseline_answer="b",
                category="hard_prompt",
            )
        )
        assert result.reward == approx(0.0)
        assert result.verdict_gen_base is None
        assert result.invalid_gen_base is True
        assert result.invalid_base_gen is False

    @pytest.mark.asyncio
    async def test_judge_call_exception_returns_invalid(self, config: ArenaJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock(side_effect=RuntimeError("judge timeout"))
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        result = await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("ok"),
                question="q",
                baseline_answer="b",
                category="hard_prompt",
            )
        )
        assert result.reward == approx(0.0)
        assert result.invalid_gen_base is True
        assert result.invalid_base_gen is True

    @pytest.mark.asyncio
    async def test_unknown_category_falls_back_to_default(self, config: ArenaJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        self._mock_judge_sequence(server_mock, ["[[A>B]]", "[[B>A]]"])
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        result = await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("ok"),
                question="q",
                baseline_answer="b",
                category="does_not_exist",
            )
        )
        # Falls back to default_category="hard_prompt".
        assert result.category == "hard_prompt"
        assert result.reward == approx(1.0)

    @pytest.mark.asyncio
    async def test_missing_category_uses_default(self, config: ArenaJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        self._mock_judge_sequence(server_mock, ["[[A>B]]", "[[B>A]]"])
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        result = await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("ok"),
                question="q",
                baseline_answer="b",
                # category omitted entirely
            )
        )
        assert result.category == "hard_prompt"
        assert result.reward == approx(1.0)

    @pytest.mark.asyncio
    async def test_sanitize_generations_strips_bad_unicode_before_judging(self, config: ArenaJudgeConfig) -> None:
        """When ``sanitize_generations`` is enabled, lone UTF-8 surrogate halves
        and NULs in candidate/baseline must be scrubbed before the judge
        prompt is rendered — otherwise the surrogate trips the judge HTTP layer."""
        import json as _json

        config.sanitize_generations = True
        server_mock = MagicMock(spec=ServerClient)
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_json.dumps(_make_chat_completion("[[A>B]]")).encode())
        server_mock.post = AsyncMock(return_value=resp)
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        bad_candidate = "candidate \ud83d answer"
        bad_baseline = "baseline \udc00 answer"
        await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response(bad_candidate),
                question="q",
                baseline_answer=bad_baseline,
                category="hard_prompt",
            )
        )

        for call in server_mock.post.call_args_list:
            request_params = call.kwargs["json"]
            for message in request_params.messages:
                content = message["content"]
                assert "\ud83d" not in content
                assert "\udc00" not in content

    @pytest.mark.asyncio
    async def test_creative_writing_category_uses_creative_prompt(self, config: ArenaJudgeConfig) -> None:
        # Verify that distinct prompts are loaded per category by capturing
        # the messages sent to the judge.
        import json as _json

        server_mock = MagicMock(spec=ServerClient)
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_json.dumps(_make_chat_completion("[[A>B]]")).encode())
        server_mock.post = AsyncMock(return_value=resp)
        server = ArenaJudgeServer(config=config, server_client=server_mock)

        await server.verify(
            ArenaJudgeVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("ok"),
                question="write a haiku",
                baseline_answer="baseline haiku",
                category="creative_writing",
            )
        )
        # The creative prompt system message does NOT contain "generating
        # your own answer"; the default arena.yaml prompt does.
        all_calls = server_mock.post.call_args_list
        assert len(all_calls) == 2
        for call in all_calls:
            request_params = call.kwargs["json"]
            system_messages = [m for m in request_params.messages if m["role"] == "system"]
            assert len(system_messages) == 1
            assert "generating your own answer" not in system_messages[0]["content"]


class TestComputeAndKeyMetrics:
    """Cover the non-_score_fn branches of compute_metrics + get_key_metrics."""

    def _cfg(self) -> ArenaJudgeConfig:
        return ArenaJudgeConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="arena_judge",
            judge_model_server=ModelServerRef(type="responses_api_models", name="m"),
            judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
        )

    def test_compute_metrics_includes_category_subsets_and_arena_elo(self) -> None:
        server = ArenaJudgeServer(config=self._cfg(), server_client=MagicMock(spec=ServerClient))
        # 4 tasks per category with varied outcomes — the logistic
        # regression needs at least two classes of winners to fit.
        tasks = [
            [  # candidate strong win
                {"verdict_gen_base": "A>>B", "verdict_base_gen": "B>>A", "category": "hard_prompt"},
            ],
            [  # tie
                {"verdict_gen_base": "A=B", "verdict_base_gen": "A=B", "category": "hard_prompt"},
            ],
            [  # candidate loss
                {"verdict_gen_base": "B>>A", "verdict_base_gen": "A>>B", "category": "hard_prompt"},
            ],
            [  # creative: candidate slight win
                {"verdict_gen_base": "A>B", "verdict_base_gen": "B>A", "category": "creative_writing"},
            ],
            [  # creative: loss
                {"verdict_gen_base": "B>A", "verdict_base_gen": "A>B", "category": "creative_writing"},
            ],
        ]
        metrics = server.compute_metrics(tasks)
        # pass@k on verdict-decomposition scores.
        assert any(k.startswith("pass@1[avg-of-") and k.endswith("/wins") for k in metrics), metrics
        # Per-category pass@k subsets.
        assert any(k.startswith("hard_prompt/") for k in metrics)
        assert any(k.startswith("creative_writing/") for k in metrics)
        # Arena-Elo headline + per-category.
        assert "arena_elo/score" in metrics
        assert "arena_elo/ci_lower" in metrics
        assert "arena_elo/ci_upper" in metrics
        assert "arena_elo/invalid_scores" in metrics
        assert "arena_elo/hard_prompt/score" in metrics
        assert "arena_elo/creative_writing/score" in metrics
        # Score is a win-rate percentage, so 0-100.
        assert 0.0 <= metrics["arena_elo/score"] <= 100.0

    def test_compute_metrics_degenerate_sweep_all_losses(self) -> None:
        """Regression test: if every battle has the same winner,
        logistic regression can't fit — the server should return a
        degenerate 0% / 100% headline instead of crashing."""
        server = ArenaJudgeServer(config=self._cfg(), server_client=MagicMock(spec=ServerClient))
        tasks = [
            [{"verdict_gen_base": "B>>A", "verdict_base_gen": "A>>B", "category": "hard_prompt"}],
            [{"verdict_gen_base": "B>>A", "verdict_base_gen": "A>>B", "category": "hard_prompt"}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["arena_elo/score"] == approx(0.0)

    def test_compute_metrics_invalid_judge_outputs_yield_nan(self) -> None:
        server = ArenaJudgeServer(config=self._cfg(), server_client=MagicMock(spec=ServerClient))
        tasks = [
            [{"verdict_gen_base": None, "verdict_base_gen": None, "category": "hard_prompt"}],
        ]
        metrics = server.compute_metrics(tasks)
        # NaN propagates when there are no usable battles.
        import math as _math

        assert _math.isnan(metrics["arena_elo/score"])
        assert metrics["arena_elo/invalid_scores"] == 2

    def test_get_key_metrics_picks_token_means_and_pass_k(self) -> None:
        server = ArenaJudgeServer(config=self._cfg(), server_client=MagicMock(spec=ServerClient))
        agent_metrics = {
            "mean/input_tokens": 100.0,
            "mean/output_tokens": 200.0,
            "pass@1[avg-of-4]/wins": 0.5,
            "pass@4/wins": 0.75,
            "pass@4/no_answer": 0.01,  # excluded by exclude_names
        }
        key = server.get_key_metrics(agent_metrics)
        assert key["mean/input_tokens"] == 100.0
        assert key["mean/output_tokens"] == 200.0
        assert "pass@1[avg-of-4]/wins" in key
        assert "pass@4/wins" in key
        assert "pass@4/no_answer" not in key


class TestExtractChatCompletionEdgeCases:
    def test_no_choices_returns_empty(self) -> None:
        from nemo_gym.openai_utils import NeMoGymChatCompletion

        completion = NeMoGymChatCompletion.model_validate(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [],
            }
        )
        assert ArenaJudgeServer._extract_chat_completion_text(completion) == ""


class TestModelPostInitValidation:
    def test_bad_default_category_raises(self) -> None:
        cfg = ArenaJudgeConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="arena_judge",
            judge_model_server=ModelServerRef(type="responses_api_models", name="m"),
            judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
            default_category="nonexistent",
        )
        # judge_prompt_paths defaults to {hard_prompt, creative_writing};
        # default_category="nonexistent" should fail init.
        with pytest.raises(ValueError, match="default_category"):
            ArenaJudgeServer(config=cfg, server_client=MagicMock(spec=ServerClient))


class TestScoreFn:
    def test_strict_win(self) -> None:
        scores = ArenaJudgeServer._arena_score_fn({"verdict_gen_base": "A>>B", "verdict_base_gen": "B>>A"})
        assert scores["wins"] == approx(1.0)
        assert scores["strict_wins"] == approx(1.0)
        assert scores["ties"] == approx(0.0)
        assert scores["losses"] == approx(0.0)
        assert scores["double_wins"] == approx(1.0)

    def test_slight_win(self) -> None:
        scores = ArenaJudgeServer._arena_score_fn({"verdict_gen_base": "A>B", "verdict_base_gen": "A>B"})
        assert scores["wins"] == approx(1.0)
        assert scores["strict_wins"] == approx(0.0)
        # Base-gen agrees candidate is A (favored), but here with A (baseline)
        # winning — no double-win.
        assert scores["double_wins"] == approx(0.0)

    def test_tie(self) -> None:
        scores = ArenaJudgeServer._arena_score_fn({"verdict_gen_base": "A=B", "verdict_base_gen": "A=B"})
        assert scores["wins"] == approx(0.0)
        assert scores["ties"] == approx(1.0)
        assert scores["losses"] == approx(0.0)

    def test_loss(self) -> None:
        scores = ArenaJudgeServer._arena_score_fn({"verdict_gen_base": "B>>A", "verdict_base_gen": "A>>B"})
        assert scores["wins"] == approx(0.0)
        assert scores["losses"] == approx(1.0)

    def test_invalid_verdict(self) -> None:
        scores = ArenaJudgeServer._arena_score_fn(
            {"verdict_gen_base": None, "verdict_base_gen": None, "invalid_gen_base": True}
        )
        assert scores["wins"] == approx(0.0)
        assert scores["losses"] == approx(0.0)
        assert scores["invalid_gen_base"] == approx(1.0)
