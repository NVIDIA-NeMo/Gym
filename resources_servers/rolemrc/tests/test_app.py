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
import json
import math
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.rolemrc.app import (
    RoleMRCResourcesServer,
    RoleMRCResourcesServerConfig,
    RoleMRCVerifyRequest,
    _build_conversation_text,
    _build_judge_prompts,
    _coerce_text,
    _compute_bertscore,
    _compute_bleu,
    _compute_meteor,
    _compute_rouge,
    _conversation_messages,
    _ensure_nltk_data,
    _extract_nested_content,
    _input_messages,
    _judge_rollups,
    _parse_judge_score,
    _response_text,
    _safe_call,
    _strip_think,
    _task_dimension,
)


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
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


def _judge_response_bytes(text: str) -> bytes:
    """A `/v1/chat/completions` reply — the default `judge_api`."""
    return json.dumps(
        {
            "id": "chatcmpl",
            "object": "chat.completion",
            "created": 0,
            "model": "judge_model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
        }
    ).encode()


def _judge_responses_api_bytes(text: str) -> bytes:
    """A `/v1/responses` reply — only for `judge_api: responses`."""
    return json.dumps(_make_response(text).model_dump()).encode()


# ── Pure helpers ─────────────────────────────────────────────────────────


class TestTaskDimension:
    def test_default_on_scene(self) -> None:
        assert _task_dimension("role_related_mrc_answer_with_narration") == "on_scene_dialogue"

    def test_multi_turn_suffixes(self) -> None:
        assert _task_dimension("x-2ndrefused") == "multi_turn"
        assert _task_dimension("x-2ndanswer") == "multi_turn"

    def test_nested_suffixes(self) -> None:
        assert _task_dimension("x-special-content") == "nested_instruction"
        assert _task_dimension("x-special-format") == "nested_instruction"

    def test_priority_suffix(self) -> None:
        assert _task_dimension("x-refused") == "instruction_priority"


class TestStripThink:
    def test_no_think(self) -> None:
        assert _strip_think("plain answer") == "plain answer"

    def test_paired_tags(self) -> None:
        assert _strip_think("<think>reason</think>answer").strip() == "answer"

    def test_orphan_closing_tag(self) -> None:
        assert _strip_think("reasoning text</think>final").strip() == "final"

    def test_empty(self) -> None:
        assert _strip_think("") == ""


class TestParseJudgeScore:
    """Strip `Score:`, then the remainder must be a bare integer or it is bad."""

    def test_one(self) -> None:
        assert _parse_judge_score("Score: 1") == (1, False)

    def test_zero(self) -> None:
        assert _parse_judge_score("Score: 0") == (0, False)

    def test_bare_number(self) -> None:
        assert _parse_judge_score("1") == (1, False)

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert _parse_judge_score("\n 1 \n") == (1, False)

    def test_unparseable_is_a_bad_response_scored_zero(self) -> None:
        assert _parse_judge_score("no number here") == (0, True)

    def test_prose_around_the_digit_is_bad_not_one(self) -> None:
        # A regex search would score this 1; a bare-integer parse rejects it.
        assert _parse_judge_score("The score is 1") == (0, True)

    def test_float_is_bad_not_rounded(self) -> None:
        # int("0.9") raises, so a float verdict is a bad response, not a 1.
        assert _parse_judge_score("Score: 0.9") == (0, True)

    def test_out_of_range_integer_is_not_clamped(self) -> None:
        # The parsed integer is used as-is, so an out-of-range verdict rides
        # through to the aspect mean instead of being silently capped.
        assert _parse_judge_score("2") == (2, False)

    def test_empty(self) -> None:
        assert _parse_judge_score("") == (0, True)


class TestJudgePromptBuilding:
    def test_conversation_text(self) -> None:
        msgs = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi?"},
            {"role": "assistant", "content": "Yo."},
        ]
        text = _build_conversation_text(msgs)
        assert 'System Instruction: "Be terse."' in text
        assert 'User Query: "Hi?"' in text
        assert 'LLM Response: "Yo."' in text

    def test_extract_nested_content_strips_lead(self) -> None:
        # The lead-in has no trailing space, so the space it did not consume
        # stays in the rendered prompt: `** end every reply…**`.
        sys = "You are a bot. You must end every reply with 'Indeed'."
        assert _extract_nested_content(sys) == " end every reply with 'Indeed'"

    def test_extract_nested_content_strips_one_trailing_period(self) -> None:
        assert _extract_nested_content("A bot. You will add a joke..") == " add a joke."

    def test_extract_nested_content_replaces_lead_anywhere(self) -> None:
        # Lead-ins are replaced wherever they occur, not just as a prefix.
        assert _extract_nested_content("A bot. Answer, and You must be terse") == "Answer, and  be terse"

    def test_extract_nested_content_falls_back_without_a_sentence_split(self) -> None:
        # No `". "` to split on: fall back to the whole string rather than raise.
        assert _extract_nested_content("You will be terse") == " be terse"

    def test_two_aspects_for_answer_with_narration(self) -> None:
        prompts = _build_judge_prompts(
            "role_related_mrc_answer_with_narration",
            conversation_text="conv",
            system_content="sys",
            response="resp",
        )
        names = [name for name, _ in prompts]
        assert names == ["knowledge_range", "style_compliance"]

    def test_unknown_task_yields_no_prompts(self) -> None:
        assert _build_judge_prompts("not_a_task", "c", "s", "r") == []


# ── Server construction + verify ─────────────────────────────────────────


def _reference_config(include_bertscore: bool = False) -> RoleMRCResourcesServerConfig:
    return RoleMRCResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="rolemrc",
        mode="reference",
        include_bertscore=include_bertscore,
    )


def _judge_config() -> RoleMRCResourcesServerConfig:
    return RoleMRCResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="rolemrc",
        mode="judge",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_chat_completion_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
    )


def _judge_config_responses_api() -> RoleMRCResourcesServerConfig:
    config = _judge_config()
    config.judge_api = "responses"
    config.judge_chat_completion_create_params = None
    config.judge_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    return config


class TestServerConstruction:
    def test_reference_sanity(self) -> None:
        RoleMRCResourcesServer(config=_reference_config(), server_client=MagicMock(spec=ServerClient))

    def test_judge_sanity(self) -> None:
        RoleMRCResourcesServer(config=_judge_config(), server_client=MagicMock(spec=ServerClient))

    def test_judge_mode_requires_judge_server(self) -> None:
        bad = RoleMRCResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="rolemrc", mode="judge")
        with pytest.raises(ValueError):
            RoleMRCResourcesServer(config=bad, server_client=MagicMock(spec=ServerClient))


class TestReferenceVerify:
    @fixture(autouse=True)
    def _patch_optional_metrics(self, monkeypatch) -> None:
        # Avoid sacrebleu/nltk (and their network downloads) in unit tests;
        # ROUGE is exercised for real below.
        import resources_servers.rolemrc.app as app

        monkeypatch.setattr(app, "_compute_bleu", lambda r, ref: 0.0)
        monkeypatch.setattr(app, "_compute_meteor", lambda r, ref: 0.0)

    async def test_exact_match_rouge_l_is_reward(self) -> None:
        pytest.importorskip("rouge_score")
        server = RoleMRCResourcesServer(config=_reference_config(), server_client=MagicMock(spec=ServerClient))
        gold = "The intruder left through the window."
        request = RoleMRCVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(gold),
            reference=gold,
            task="role_related_mrc_answer_with_narration",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.dimension == "on_scene_dialogue"
        assert result.rougeL == approx(1.0)

    async def test_think_is_stripped_before_scoring(self) -> None:
        pytest.importorskip("rouge_score")
        server = RoleMRCResourcesServer(config=_reference_config(), server_client=MagicMock(spec=ServerClient))
        gold = "Answer text."
        request = RoleMRCVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("<think>secret reasoning</think>Answer text."),
            reference=gold,
            task="role_related_mrc_answer_no_narration",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert "think" not in result.generation


class TestJudgeVerify:
    def _server(self) -> tuple[RoleMRCResourcesServer, MagicMock]:
        mock = MagicMock(spec=ServerClient)
        server = RoleMRCResourcesServer(config=_judge_config(), server_client=mock)
        return server, mock

    def _request(self, task: str) -> RoleMRCVerifyRequest:
        return RoleMRCVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {"role": "system", "content": "You are a detective."},
                    {"role": "user", "content": "Passage: clue. Where did they go?"},
                ]
            ),
            response=_make_response("They went out the window."),
            reference="They went out the window.",
            task=task,
        )

    async def test_all_aspects_score_one(self) -> None:
        server, mock = self._server()
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_response_bytes("Score: 1"))
        mock.post = AsyncMock(return_value=resp)

        result = await server.verify(self._request("role_related_mrc_answer_with_narration"))
        # 2 aspects (knowledge_range + style), both 1 -> reward 1.0.
        assert result.reward == approx(1.0)
        assert result.n_aspects == 2
        assert mock.post.await_count == 2
        assert result.aspects == {"knowledge_range": 1, "style_compliance": 1}

    async def test_all_aspects_score_zero(self) -> None:
        server, mock = self._server()
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_response_bytes("Score: 0"))
        mock.post = AsyncMock(return_value=resp)

        result = await server.verify(self._request("role_related_mrc_answer_no_narration"))
        # 1 aspect (knowledge_range), 0 -> reward 0.0.
        assert result.reward == approx(0.0)
        assert result.n_aspects == 1

    async def test_judge_call_failure_raises_judge_error(self) -> None:
        server, mock = self._server()
        mock.post = AsyncMock(side_effect=RuntimeError("judge down"))

        with pytest.raises(JudgeError, match="knowledge_range"):
            await server.verify(self._request("role_related_mrc_answer_no_narration"))

    async def test_reasoning_model_think_tags_stripped_before_scoring(self) -> None:
        # If the judge is a reasoning model, <think> blocks may contain numbers
        # that would corrupt _SCORE_RE's first-match lookup without stripping.
        server, mock = self._server()
        resp = AsyncMock()
        resp.read = AsyncMock(
            return_value=_judge_response_bytes("<think>The passage mentions 0 errors and 3 facts.</think>\nScore: 1")
        )
        mock.post = AsyncMock(return_value=resp)

        result = await server.verify(self._request("role_related_mrc_answer_no_narration"))
        assert result.reward == approx(1.0), "think-block numbers must not pollute score parsing"


class TestAggregation:
    def test_compute_metrics_by_dimension(self) -> None:
        server = RoleMRCResourcesServer(config=_reference_config(), server_client=MagicMock(spec=ServerClient))
        tasks = [
            [{"reward": 1.0, "dimension": "on_scene_dialogue"}],
            [{"reward": 0.0, "dimension": "on_scene_dialogue"}],
            [{"reward": 0.5, "dimension": "multi_turn"}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["mean_reward"] == approx(0.5)
        assert metrics["dimension/on_scene_dialogue/mean_reward"] == approx(0.5)
        assert metrics["dimension/on_scene_dialogue/count"] == 2
        assert metrics["dimension/multi_turn/mean_reward"] == approx(0.5)

    def test_compute_metrics_by_aspect(self) -> None:
        server = RoleMRCResourcesServer(config=_judge_config(), server_client=MagicMock(spec=ServerClient))
        tasks = [
            [{"reward": 1.0, "dimension": "on_scene_dialogue", "aspect_style_compliance": 1.0}],
            [{"reward": 0.0, "dimension": "on_scene_dialogue", "aspect_style_compliance": 0.0}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["aspect/style_compliance/mean"] == approx(0.5)
        assert metrics["aspect/style_compliance/count"] == 2

    def test_compute_metrics_aggregates_auto_metrics(self) -> None:
        """Every reference metric gets a corpus mean, not just the reward."""
        server = RoleMRCResourcesServer(config=_reference_config(), server_client=MagicMock(spec=ServerClient))
        tasks = [
            [{"reward": 0.2, "dimension": "on_scene_dialogue", "rouge1": 0.4, "bleu": 0.02, "meteor": 0.3}],
            [{"reward": 0.4, "dimension": "on_scene_dialogue", "rouge1": 0.6, "bleu": 0.04, "meteor": 0.5}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["auto/rouge1/mean"] == approx(0.5)
        assert metrics["auto/bleu/mean"] == approx(0.03)
        assert metrics["auto/meteor/mean"] == approx(0.4)
        assert metrics["auto/rouge1/count"] == 2
        # Metrics absent from the rows must not be invented.
        assert "auto/bertscore_f1/mean" not in metrics

    def test_compute_metrics_judge_rollups(self) -> None:
        """AvgSimple / AvgWeighted / AvgS(noMT) are aspect-macro, not row-macro."""
        server = RoleMRCResourcesServer(config=_judge_config(), server_client=MagicMock(spec=ServerClient))
        # 3 rows, 4 judge calls: knowledge fires twice (1.0, 0.0), style once
        # (1.0), multi-turn once (0.0).
        tasks = [
            [
                {
                    "reward": 1.0,
                    "dimension": "on_scene_dialogue",
                    "aspect_knowledge_range": 1.0,
                    "aspect_style_compliance": 1.0,
                }
            ],
            [{"reward": 0.0, "dimension": "on_scene_dialogue", "aspect_knowledge_range": 0.0}],
            [{"reward": 0.0, "dimension": "multi_turn", "aspect_multi_turn_instruction": 0.0}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["judge/n_calls"] == 4
        # aspect means: knowledge 0.5, style 1.0, multi-turn 0.0
        assert metrics["judge/avg_simple"] == approx((0.5 + 1.0 + 0.0) / 3)
        # weighted by call count: (0.5*2 + 1.0*1 + 0.0*1) / 4
        assert metrics["judge/avg_weighted"] == approx(0.5)
        assert metrics["judge/avg_simple_no_mt"] == approx((0.5 + 1.0) / 2)
        # The row-macro mean matches none of them -- that is the whole point.
        assert metrics["mean_reward"] == approx(1.0 / 3)

    def test_judge_rollups_all_multi_turn_has_no_no_mt(self) -> None:
        rollups = _judge_rollups({"multi_turn_instruction": [1.0, 0.0]})
        assert rollups["judge/avg_simple"] == approx(0.5)
        assert "judge/avg_simple_no_mt" not in rollups

    def test_judge_rollups_empty(self) -> None:
        assert _judge_rollups({}) == {}

    def test_get_key_metrics_promotes_headline_first(self) -> None:
        """AvgS(noMT) is RoleMRC's published metric, so it leads the key metrics."""
        server = RoleMRCResourcesServer(config=_judge_config(), server_client=MagicMock(spec=ServerClient))
        key_metrics = server.get_key_metrics(
            {
                "mean_reward": 0.5,
                "judge/avg_simple": 0.6,
                "judge/avg_weighted": 0.7,
                "judge/avg_simple_no_mt": 0.8,
                "judge/n_calls": 4,
                "auto/bleu/mean": 0.02,
                "aspect/style_compliance/mean": 0.9,
            }
        )
        assert list(key_metrics)[0] == "judge/avg_simple_no_mt"
        assert key_metrics["auto/bleu/mean"] == approx(0.02)
        assert key_metrics["aspect/style_compliance/mean"] == approx(0.9)
        assert "judge/n_calls" not in key_metrics

    def test_compute_metrics_empty(self) -> None:
        server = RoleMRCResourcesServer(config=_reference_config(), server_client=MagicMock(spec=ServerClient))
        assert server.compute_metrics([]) == {}

    def test_get_key_metrics_selects_headline_and_breakdowns(self) -> None:
        server = RoleMRCResourcesServer(config=_reference_config(), server_client=MagicMock(spec=ServerClient))
        agent_metrics = {
            "mean_reward": 0.5,
            "count": 10,
            "dimension/multi_turn/mean_reward": 0.4,
            "dimension/multi_turn/count": 3,
            "aspect/style_compliance/mean": 0.6,
            "aspect/style_compliance/count": 3,
        }
        assert server.get_key_metrics(agent_metrics) == {
            "mean_reward": 0.5,
            "dimension/multi_turn/mean_reward": 0.4,
            "aspect/style_compliance/mean": 0.6,
        }


# ── Text / extraction helpers ─────────────────────────────────────────────


class TestCoerceText:
    def test_plain_string(self) -> None:
        assert _coerce_text("hi") == "hi"

    def test_list_of_dicts(self) -> None:
        assert _coerce_text([{"text": "a"}, {"text": "b"}]) == "ab"

    def test_list_of_objects_and_bare_strings(self) -> None:
        assert _coerce_text([SimpleNamespace(text="x"), "y", {"no_text": 1}]) == "xy"

    def test_none_and_scalar(self) -> None:
        assert _coerce_text(None) == ""
        assert _coerce_text(123) == "123"


class TestInputMessages:
    def test_string_input(self) -> None:
        assert _input_messages(SimpleNamespace(input="hello")) == [{"role": "user", "content": "hello"}]

    def test_none_input(self) -> None:
        assert _input_messages(SimpleNamespace(input=None)) == []

    def test_dict_items_lowercased_and_flattened(self) -> None:
        params = SimpleNamespace(input=[{"role": "SYSTEM", "content": [{"text": "s"}]}])
        assert _input_messages(params) == [{"role": "system", "content": "s"}]

    def test_object_items(self) -> None:
        params = SimpleNamespace(input=[SimpleNamespace(role="User", content="q")])
        assert _input_messages(params) == [{"role": "user", "content": "q"}]


class TestConversationMessages:
    """A runner that does not deliver ``responses_create_params`` to /verify must
    still get the conversation, via ``verifier_metadata``."""

    CONVERSATION = [
        {"role": "system", "content": "You are a helpful AI assistant. You must shout"},
        {"role": "user", "content": "hi"},
    ]

    def _body(self, **extra: object) -> SimpleNamespace:
        degenerate = SimpleNamespace(input=[{"role": "user", "content": ""}])
        return SimpleNamespace(responses_create_params=degenerate, **extra)

    def test_prefers_verifier_metadata_over_degenerate_params(self) -> None:
        body = self._body(verifier_metadata={"conversation": self.CONVERSATION})
        assert _conversation_messages(body) == self.CONVERSATION

    def test_falls_back_to_params_when_metadata_absent(self) -> None:
        assert _conversation_messages(self._body()) == [{"role": "user", "content": ""}]

    @pytest.mark.parametrize("meta", [None, {}, {"conversation": []}, {"conversation": None}, "nope"])
    def test_falls_back_on_unusable_metadata(self, meta: object) -> None:
        body = self._body(verifier_metadata=meta)
        assert _conversation_messages(body) == [{"role": "user", "content": ""}]

    def test_system_instruction_survives_into_the_judge_prompt(self) -> None:
        """The regression: without this the nested rubric renders as ``(****)``."""
        body = self._body(verifier_metadata={"conversation": self.CONVERSATION})
        msgs = _conversation_messages(body)
        system = next(m["content"] for m in msgs if m["role"] == "system")
        prompts = dict(
            _build_judge_prompts(
                "role_related_mrc_answer_with_narration-special-format",
                _build_conversation_text(msgs),
                system,
                "HI",
            )
        )
        assert "(****)" not in prompts["nested_instruction"]
        assert "** shout**" in prompts["nested_instruction"]


class TestResponseText:
    def test_output_text_fast_path(self) -> None:
        assert _response_text(SimpleNamespace(output_text="fast")) == "fast"

    def test_fallback_walks_message_output(self) -> None:
        resp = SimpleNamespace(
            output_text=None,
            output=[
                SimpleNamespace(type="reasoning", content="ignored"),
                SimpleNamespace(type="message", content=[{"text": "hello"}]),
            ],
        )
        assert _response_text(resp) == "hello"


class TestSafeCall:
    def test_returns_value(self) -> None:
        assert _safe_call("x", lambda a: a + 1, 1) == 2

    def test_swallows_exception(self) -> None:
        def boom() -> None:
            raise RuntimeError("nope")

        assert _safe_call("x", boom) is None


# ── Reference metrics ─────────────────────────────────────────────────────


class TestReferenceMetrics:
    def test_rouge_exact_match(self) -> None:
        pytest.importorskip("rouge_score")
        assert _compute_rouge("the cat sat", "the cat sat")["rougeL"] == approx(1.0)

    def test_rouge_failure_returns_zeros(self, monkeypatch) -> None:
        import resources_servers.rolemrc.app as app

        class Bad:
            def score(self, *_a):
                raise RuntimeError("boom")

        monkeypatch.setattr(app, "_rouge_scorer", lambda: Bad())
        assert _compute_rouge("a", "b") == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    def test_bleu_empty_is_zero(self) -> None:
        assert _compute_bleu("", "ref") == 0.0

    def test_bleu_exact_match_is_one(self) -> None:
        pytest.importorskip("sacrebleu")
        # Needs >= 4 tokens for the 4-gram precision to be non-zero.
        assert _compute_bleu("the cat sat on the mat", "the cat sat on the mat") == approx(1.0)

    def test_bleu_is_unsmoothed(self) -> None:
        """A missing n-gram order zeroes the score, matching evaluate.load("bleu").

        This is the whole reason we do not use ``sacrebleu.sentence_bleu``: its
        exponential smoothing would return a non-zero score here.
        """
        pytest.importorskip("sacrebleu")
        # Exact match but only 3 tokens -> no 4-gram exists -> precision[3] == 0.
        assert _compute_bleu("the cat sat", "the cat sat") == 0.0
        assert _compute_bleu("a dog ran through the park", "the cat sat on the mat") == 0.0

    def test_bleu_brevity_penalty(self) -> None:
        """All precisions 1.0, hypothesis 4 tokens vs reference 6 -> bp = exp(1 - 6/4)."""
        pytest.importorskip("sacrebleu")
        assert _compute_bleu("the cat sat on", "the cat sat on the mat") == approx(math.exp(1 - 6 / 4))

    def test_bleu_failure_returns_zero(self, monkeypatch) -> None:
        import resources_servers.rolemrc.app as app

        def boom(*_a):
            raise RuntimeError("tokenizer exploded")

        monkeypatch.setattr(app, "_bleu_score", boom)
        assert _compute_bleu("the cat sat on the mat", "the cat sat on the mat") == 0.0

    def test_meteor_exact_match(self) -> None:
        pytest.importorskip("nltk")
        assert _compute_meteor("the cat sat", "the cat sat") > 0.0

    def test_meteor_holds_the_wordnet_lock(self, monkeypatch) -> None:
        """METEOR must run under `_WORDNET_LOCK`; NLTK's WordNet reader is not thread-safe.

        Asserted directly rather than by racing threads: a test that has to win
        a race passes just as happily with the bug present.
        """
        pytest.importorskip("nltk")
        import nltk.translate.meteor_score as meteor_module

        import resources_servers.rolemrc.app as app

        held = []

        def fake_meteor_score(references, hypothesis):
            held.append(app._WORDNET_LOCK.locked())
            return 0.5

        monkeypatch.setattr(meteor_module, "meteor_score", fake_meteor_score)
        assert _compute_meteor("the cat sat", "the cat sat") == approx(0.5)
        assert held == [True], "meteor_score must be called while holding _WORDNET_LOCK"
        assert not app._WORDNET_LOCK.locked(), "lock must be released afterwards"

    def test_meteor_tokenizes_outside_the_lock(self, monkeypatch) -> None:
        """Tokenization is pure once punkt is warm, so it must not extend the critical section."""
        pytest.importorskip("nltk")
        import nltk.tokenize as tokenize_module

        import resources_servers.rolemrc.app as app

        real_word_tokenize = tokenize_module.word_tokenize
        held = []

        def fake_word_tokenize(text):
            held.append(app._WORDNET_LOCK.locked())
            return real_word_tokenize(text)

        monkeypatch.setattr(tokenize_module, "word_tokenize", fake_word_tokenize)
        _compute_meteor("the cat sat", "the cat sat")
        assert held and not any(held), "tokenization should happen outside _WORDNET_LOCK"

    def test_meteor_lock_serializes_concurrent_calls(self) -> None:
        """Smoke test: concurrent METEOR yields one consistent value per input.

        Cannot reliably reproduce the race itself (see
        `test_meteor_holds_the_wordnet_lock`), but catches a deadlocking lock.
        """
        pytest.importorskip("nltk")
        pair = (
            "The knight swiftly departed the castle and rode toward the northern hills.",
            "The warrior quickly left the fortress and travelled to the northern mountains.",
        )
        expected = _compute_meteor(*pair)
        assert expected > 0.0

        results: list[float] = []
        collect_lock = threading.Lock()

        def worker() -> None:
            for _ in range(10):
                value = _compute_meteor(*pair)
                with collect_lock:
                    results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in threads), "_WORDNET_LOCK deadlocked"
        assert results == [expected] * 80

    def test_ensure_nltk_data_warms_lazy_loaders(self) -> None:
        """Warm-up must materialize the lazy loaders, not just locate the files."""
        pytest.importorskip("nltk")
        _ensure_nltk_data()
        from nltk.corpus import wordnet

        # LazyCorpusLoader swaps in the real reader on first attribute access;
        # after warm-up the synset path must work without triggering that.
        assert wordnet.synsets("castle")

    def test_bertscore_mocked(self, monkeypatch) -> None:
        import resources_servers.rolemrc.app as app

        monkeypatch.setattr(app, "_bert_scorer", lambda: SimpleNamespace(score=lambda c, r: ([0.8], [0.7], [0.75])))
        assert _compute_bertscore("a", "b")["bertscore_f1"] == approx(0.75)

    def test_bertscore_load_failure_returns_zeros(self, monkeypatch) -> None:
        import resources_servers.rolemrc.app as app

        def boom() -> None:
            raise RuntimeError("no torch")

        monkeypatch.setattr(app, "_bert_scorer", boom)
        assert _compute_bertscore("a", "b") == {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }

    def test_bertscore_score_failure_returns_zeros(self, monkeypatch) -> None:
        import resources_servers.rolemrc.app as app

        def raise_score(_c, _r):
            raise RuntimeError("cuda oom")

        monkeypatch.setattr(app, "_bert_scorer", lambda: SimpleNamespace(score=raise_score))
        assert _compute_bertscore("a", "b")["bertscore_f1"] == 0.0

    async def test_reference_verify_includes_bertscore(self, monkeypatch) -> None:
        pytest.importorskip("rouge_score")
        import resources_servers.rolemrc.app as app

        monkeypatch.setattr(app, "_compute_bleu", lambda r, ref: 0.0)
        monkeypatch.setattr(app, "_compute_meteor", lambda r, ref: 0.0)
        monkeypatch.setattr(app, "_bert_scorer", lambda: SimpleNamespace(score=lambda c, r: ([0.9], [0.9], [0.9])))
        server = RoleMRCResourcesServer(
            config=_reference_config(include_bertscore=True), server_client=MagicMock(spec=ServerClient)
        )
        request = RoleMRCVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("some answer"),
            reference="some answer",
            task="role_related_mrc_answer_with_narration",
        )
        result = await server.verify(request)
        assert result.bertscore_f1 == approx(0.9)


# ── Judge edge cases ──────────────────────────────────────────────────────


class TestJudgeEdgeCases:
    def _server(self) -> tuple[RoleMRCResourcesServer, MagicMock]:
        mock = MagicMock(spec=ServerClient)
        return RoleMRCResourcesServer(config=_judge_config(), server_client=mock), mock

    def _request(self, task: str) -> RoleMRCVerifyRequest:
        return RoleMRCVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {"role": "system", "content": "You are a detective. You must be terse."},
                    {"role": "user", "content": "Where did they go?"},
                ]
            ),
            response=_make_response("Out the window."),
            reference="Out the window.",
            task=task,
        )

    async def test_unknown_task_is_skipped(self) -> None:
        server, _ = self._server()
        result = await server.verify(self._request("not_a_real_task"))
        assert result.reward == approx(0.0)
        assert result.judge_skipped is True

    async def test_unparseable_score_marks_bad_aspect(self) -> None:
        server, mock = self._server()
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_response_bytes("no number here"))
        mock.post = AsyncMock(return_value=resp)
        result = await server.verify(self._request("role_related_mrc_answer_no_narration"))
        assert result.reward == approx(0.0)
        assert result.bad_aspects == ["knowledge_range"]

    async def test_empty_judge_response_is_a_failure_not_a_zero(self) -> None:
        """Empty judge output means no verdict was received, so it must not score 0.

        A reasoning judge spends `max_output_tokens` on reasoning before emitting
        the score; an exhausted budget returns a well-formed response with no
        text. Scoring that 0 would silently understate every affected row.
        """
        server, mock = self._server()
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_response_bytes(""))
        mock.post = AsyncMock(return_value=resp)
        with pytest.raises(JudgeError) as excinfo:
            await server.verify(self._request("role_related_mrc_answer_no_narration"))
        assert "empty response text" in str(excinfo.value)

    def test_nested_task_injects_extracted_content(self) -> None:
        prompts = _build_judge_prompts(
            "role_related_mrc_answer_with_narration-special-content",
            conversation_text="conv",
            system_content="You are a detective. You must be terse.",
            response="resp",
        )
        assert prompts[0][0] == "nested_instruction"
        assert "be terse" in prompts[0][1]


class TestJudgeFailureReporting:
    async def test_empty_judge_text_is_a_judge_failure(self, monkeypatch) -> None:
        """An exhausted reasoning budget returns no text — a failure, not a 0 verdict."""
        server = RoleMRCResourcesServer(config=_judge_config(), server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(server, "_call_judge", AsyncMock(return_value=(None, "empty response text")))
        request = RoleMRCVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("a reply"),
            task="role_related_mrc_answer_no_narration",
        )
        with pytest.raises(JudgeError) as excinfo:
            await server.verify(request)
        assert "empty response text" in str(excinfo.value)

    async def test_judge_error_message_carries_the_reason(self, monkeypatch) -> None:
        """The underlying cause must reach the failure record, not just the aspect name."""
        server = RoleMRCResourcesServer(config=_judge_config(), server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(
            server,
            "_call_judge",
            AsyncMock(return_value=(None, "ClientResponseError: 500, message='Internal Server Error'")),
        )
        request = RoleMRCVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("a reply"),
            task="role_related_mrc_answer_no_narration",
        )
        with pytest.raises(JudgeError) as excinfo:
            await server.verify(request)
        message = str(excinfo.value)
        assert "knowledge_range" in message
        assert "500" in message and "Internal Server Error" in message


# ── Judge sampling parameters ────────────────────────────────────────────


class TestJudgeSamplingParams:
    """Reasoning judges reject `temperature`/`top_p`, so omitting them must actually omit them."""

    def _server(self, **params: object) -> tuple[RoleMRCResourcesServer, MagicMock]:
        config = _judge_config()
        config.judge_chat_completion_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[], **params
        )
        mock = MagicMock(spec=ServerClient)
        return RoleMRCResourcesServer(config=config, server_client=mock), mock

    async def _payload(self, server: RoleMRCResourcesServer, mock: MagicMock) -> dict:
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_response_bytes("Score: 1"))
        mock.post = AsyncMock(return_value=resp)
        await server.verify(
            RoleMRCVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response("a reply"),
                task="role_related_mrc_answer_no_narration",  # one aspect, one call
            )
        )
        return mock.post.await_args_list[0].kwargs["json"]

    async def test_configured_params_are_sent(self) -> None:
        server, mock = self._server(temperature=0.0, top_p=1.0)
        payload = await self._payload(server, mock)
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0

    async def test_unset_params_are_not_sent(self) -> None:
        server, mock = self._server()
        assert not {"temperature", "top_p"} & set(await self._payload(server, mock))

    async def test_null_params_are_not_sent(self) -> None:
        """`temperature: null` is *set*, so without the null filter it would be sent as null."""
        server, mock = self._server(temperature=None, top_p=1.0)
        payload = await self._payload(server, mock)
        assert "temperature" not in payload
        assert payload["top_p"] == 1.0

    async def test_reasoning_effort_and_budget_reach_the_endpoint(self) -> None:
        """A reasoning judge sends `reasoning_effort` + `max_completion_tokens`."""
        server, mock = self._server(reasoning_effort="low", max_completion_tokens=2048, n=1)
        payload = await self._payload(server, mock)
        assert payload["reasoning_effort"] == "low"
        assert payload["max_completion_tokens"] == 2048
        assert payload["n"] == 1


# ── Judge API surface ────────────────────────────────────────────────────


class TestJudgeApiSurface:
    """`judge_api` picks the surface the aspect prompts are posted to; the two
    score the same model differently, so the default must not drift."""

    def _request(self) -> RoleMRCVerifyRequest:
        return RoleMRCVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("a reply"),
            task="role_related_mrc_answer_no_narration",  # one aspect, one call
        )

    async def test_default_posts_chat_completions_with_a_user_message(self) -> None:
        config = _judge_config()
        assert config.judge_api == "chat_completions"
        mock = MagicMock(spec=ServerClient)
        server = RoleMRCResourcesServer(config=config, server_client=mock)
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_response_bytes("1"))
        mock.post = AsyncMock(return_value=resp)

        result = await server.verify(self._request())

        call = mock.post.await_args_list[0].kwargs
        assert call["url_path"] == "/v1/chat/completions"
        assert call["json"]["messages"][0]["role"] == "user"
        assert "knowledge range" in call["json"]["messages"][0]["content"]
        assert "input" not in call["json"]
        assert result.reward == approx(1.0)

    async def test_responses_api_mode_still_works(self) -> None:
        mock = MagicMock(spec=ServerClient)
        server = RoleMRCResourcesServer(config=_judge_config_responses_api(), server_client=mock)
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_responses_api_bytes("1"))
        mock.post = AsyncMock(return_value=resp)

        result = await server.verify(self._request())

        call = mock.post.await_args_list[0].kwargs
        assert call["url_path"] == "/v1/responses"
        assert call["json"]["input"][0]["role"] == "user"
        assert result.reward == approx(1.0)

    def test_missing_params_for_the_selected_surface_is_rejected(self) -> None:
        config = _judge_config()
        config.judge_chat_completion_create_params = None
        config.judge_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        with pytest.raises(ValueError, match="judge_chat_completion_create_params"):
            RoleMRCResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


# ── Response text handed to the judge ────────────────────────────────────


class TestJudgeSeesReasoningStrippedText:
    async def _judged_response_text(self, generation: str) -> str:
        mock = MagicMock(spec=ServerClient)
        server = RoleMRCResourcesServer(config=_judge_config(), server_client=mock)
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_response_bytes("1"))
        mock.post = AsyncMock(return_value=resp)
        await server.verify(
            RoleMRCVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_make_response(generation),
                task="role_related_mrc_answer_no_narration",
            )
        )
        return mock.post.await_args_list[0].kwargs["json"]["messages"][0]["content"]

    async def test_think_trace_is_dropped(self) -> None:
        prompt = await self._judged_response_text("<think>secret</think>the answer")
        assert "secret" not in prompt
        assert "the answer" in prompt

    async def test_channel_marker_is_dropped(self) -> None:
        # '<channel|>' is an end tag too; _strip_think alone would not cut it.
        prompt = await self._judged_response_text("analysis noise<channel|>the answer")
        assert "analysis noise" not in prompt
        assert "the answer" in prompt

    async def test_full_response_is_sent_not_the_500_char_log_field(self) -> None:
        long_answer = "word " * 400
        prompt = await self._judged_response_text(long_answer)
        assert long_answer.strip() in prompt
