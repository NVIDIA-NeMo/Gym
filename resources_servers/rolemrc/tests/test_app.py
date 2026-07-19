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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
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
    _extract_nested_content,
    _input_messages,
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
    def test_one(self) -> None:
        assert _parse_judge_score("Score: 1") == 1

    def test_zero(self) -> None:
        assert _parse_judge_score("Score: 0") == 0

    def test_bare_number(self) -> None:
        assert _parse_judge_score("1") == 1

    def test_unparseable_defaults_zero(self) -> None:
        assert _parse_judge_score("no number here") == 0

    def test_float_rounds(self) -> None:
        assert _parse_judge_score("Score: 0.9") == 1


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
        sys = "You are a bot. You must end every reply with 'Indeed'."
        assert _extract_nested_content(sys) == "end every reply with 'Indeed'"

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
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


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

    async def test_judge_call_failure_counts_as_zero(self) -> None:
        server, mock = self._server()
        mock.post = AsyncMock(side_effect=RuntimeError("judge down"))

        result = await server.verify(self._request("role_related_mrc_answer_no_narration"))
        assert result.reward == approx(0.0)
        assert result.judge_errors == ["knowledge_range"]

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
        assert _compute_bleu("the cat sat", "the cat sat") == approx(1.0)

    def test_meteor_exact_match(self) -> None:
        pytest.importorskip("nltk")
        assert _compute_meteor("the cat sat", "the cat sat") > 0.0

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

    async def test_empty_judge_response_scores_zero(self) -> None:
        server, mock = self._server()
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_judge_response_bytes(""))
        mock.post = AsyncMock(return_value=resp)
        result = await server.verify(self._request("role_related_mrc_answer_no_narration"))
        assert result.reward == approx(0.0)

    def test_nested_task_injects_extracted_content(self) -> None:
        prompts = _build_judge_prompts(
            "role_related_mrc_answer_with_narration-special-content",
            conversation_text="conv",
            system_content="You are a detective. You must be terse.",
            response="resp",
        )
        assert prompts[0][0] == "nested_instruction"
        assert "be terse" in prompts[0][1]
