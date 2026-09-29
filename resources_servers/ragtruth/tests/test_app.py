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
from types import SimpleNamespace
from unittest.mock import MagicMock

from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.ragtruth.app import (
    RagtruthResourcesServer,
    RagtruthResourcesServerConfig,
    RagtruthVerifyRequest,
    _has_hallucination,
    _parse_response,
    _response_text,
    _strip_json_fence,
    _strip_think,
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


def _config() -> RagtruthResourcesServerConfig:
    return RagtruthResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="ragtruth")


def _request(text: str, *, is_halu: bool, task_type: str = "QA") -> RagtruthVerifyRequest:
    return RagtruthVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        is_halu=is_halu,
        task_type=task_type,
    )


def _server() -> RagtruthResourcesServer:
    return RagtruthResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))


# ── Pure helpers ─────────────────────────────────────────────────────────


class TestStripThink:
    def test_no_think(self) -> None:
        assert _strip_think("plain") == "plain"

    def test_paired_tags(self) -> None:
        assert _strip_think("<think>reasoning</think>answer") == "answer"

    def test_orphan_closing_tag(self) -> None:
        assert _strip_think("reasoning</think>answer") == "answer"

    def test_empty(self) -> None:
        assert _strip_think("") == ""


class TestStripJsonFence:
    def test_whole_fence(self) -> None:
        assert _strip_json_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_no_fence_passthrough(self) -> None:
        assert _strip_json_fence('{"a": 1}') == '{"a": 1}'

    def test_inner_fence(self) -> None:
        assert _strip_json_fence('prefix ```{"a": 1}``` suffix') == '{"a": 1}'

    def test_unclosed_fence_passthrough(self) -> None:
        # A fence with no closing ``` matches neither pattern; return as-is.
        assert _strip_json_fence('```json\n{"a": 1}') == '```json\n{"a": 1}'


class TestParseResponse:
    def test_valid_dict(self) -> None:
        assert _parse_response('{"hallucination list": []}') == {"hallucination list": []}

    def test_fenced(self) -> None:
        assert _parse_response('```json\n{"hallucination list": ["x"]}\n```') == {"hallucination list": ["x"]}

    def test_invalid_json_returns_none(self) -> None:
        assert _parse_response("not json") is None

    def test_non_dict_returns_none(self) -> None:
        assert _parse_response("[1, 2, 3]") is None

    def test_empty_returns_none(self) -> None:
        assert _parse_response("") is None


class TestHasHallucination:
    def test_non_empty_list(self) -> None:
        assert _has_hallucination({"hallucination list": ["span"]}) is True

    def test_empty_list(self) -> None:
        assert _has_hallucination({"hallucination list": []}) is False

    def test_missing_key(self) -> None:
        assert _has_hallucination({}) is False

    def test_none(self) -> None:
        assert _has_hallucination(None) is False


# ── verify() ─────────────────────────────────────────────────────────────


class TestVerify:
    async def test_correct_positive(self) -> None:
        result = await _server().verify(_request('{"hallucination list": ["van Gogh"]}', is_halu=True))
        assert result.reward == approx(1.0)
        assert result.pred_halu == 1
        assert result.is_halu == 1
        assert result.parse_fail == 0

    async def test_correct_negative(self) -> None:
        result = await _server().verify(_request('{"hallucination list": []}', is_halu=False))
        assert result.reward == approx(1.0)
        assert result.pred_halu == 0
        assert result.is_halu == 0

    async def test_false_negative(self) -> None:
        result = await _server().verify(_request('{"hallucination list": []}', is_halu=True))
        assert result.reward == approx(0.0)
        assert result.pred_halu == 0
        assert result.is_halu == 1

    async def test_parse_failure_counts_as_no_hallucination(self) -> None:
        result = await _server().verify(_request("garbage output", is_halu=True))
        assert result.parse_fail == 1
        assert result.pred_halu == 0
        assert result.reward == approx(0.0)

    async def test_think_stripped_before_parse(self) -> None:
        result = await _server().verify(
            _request('<think>let me check</think>{"hallucination list": ["x"]}', is_halu=True)
        )
        assert result.reward == approx(1.0)
        assert result.pred_halu == 1


# ── compute_metrics() ──────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_accuracy_and_f1(self) -> None:
        # 2 QA, 2 Summary. TP=1, FP=1, FN=1, TN=1 overall.
        tasks = [
            [{"reward": 1.0, "task_type": "QA", "is_halu": 1, "pred_halu": 1, "parse_fail": 0}],
            [{"reward": 0.0, "task_type": "QA", "is_halu": 1, "pred_halu": 0, "parse_fail": 0}],
            [{"reward": 0.0, "task_type": "Summary", "is_halu": 0, "pred_halu": 1, "parse_fail": 0}],
            [{"reward": 1.0, "task_type": "Summary", "is_halu": 0, "pred_halu": 0, "parse_fail": 0}],
        ]
        metrics = _server().compute_metrics(tasks)
        assert metrics["mean_reward"] == approx(0.5)
        assert metrics["count"] == 4
        # precision = 1/(1+1) = 0.5, recall = 1/(1+1) = 0.5, f1 = 0.5
        assert metrics["precision"] == approx(0.5)
        assert metrics["recall"] == approx(0.5)
        assert metrics["f1"] == approx(0.5)
        assert metrics["task_type/QA/accuracy"] == approx(0.5)
        assert metrics["task_type/QA/count"] == 2
        assert metrics["parse_fail_rate"] == approx(0.0)

    def test_empty(self) -> None:
        assert _server().compute_metrics([]) == {}


# ── _response_text() ───────────────────────────────────────────────────────


class TestResponseText:
    def test_output_text_fast_path(self) -> None:
        assert _response_text(_make_response("hello")) == "hello"

    def test_fallback_joins_message_content(self) -> None:
        # No output_text -> iterate output, skipping non-message items and
        # joining text from both object- and dict-shaped content parts.
        message = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(text="a"), {"text": "b"}],
        )
        reasoning = SimpleNamespace(type="reasoning", content="ignored")
        response = SimpleNamespace(output_text=None, output=[reasoning, message])
        assert _response_text(response) == "ab"

    def test_fallback_string_content(self) -> None:
        message = SimpleNamespace(type="message", content="plain")
        response = SimpleNamespace(output_text="", output=[message])
        assert _response_text(response) == "plain"


# ── get_key_metrics() ────────────────────────────────────────────────────────


class TestGetKeyMetrics:
    def test_selects_headline_and_per_slice(self) -> None:
        agent_metrics = {
            "mean_reward": 0.5,
            "f1": 0.4,
            "precision": 0.3,
            "recall": 0.2,
            "count": 10,
            "task_type/QA/f1": 0.6,
            "task_type/QA/accuracy": 0.7,
            "task_type/QA/count": 5,
        }
        out = _server().get_key_metrics(agent_metrics)
        assert out["mean_reward"] == approx(0.5)
        assert out["f1"] == approx(0.4)
        assert out["task_type/QA/f1"] == approx(0.6)
        assert out["task_type/QA/accuracy"] == approx(0.7)
        # Non-headline / count fields are dropped.
        assert "count" not in out
        assert "task_type/QA/count" not in out
