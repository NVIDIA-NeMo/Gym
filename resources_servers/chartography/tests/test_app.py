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
"""Tests for the chartography resources server (judge mocked; no network)."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from pytest import fixture, raises

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NgFailureEvent,
)
from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation, grading_result_binding
from nemo_gym.verification import JudgeError, exercise_verifier_fixture, is_judge_failure, judge
from resources_servers.chartography import scorer
from resources_servers.chartography.app import (
    VERIFIER_FIXTURE,
    ChartographyConfig,
    ChartographyResourcesServer,
    ChartographyVerifyRequest,
    extract_final_answer,
)
from resources_servers.chartography.task_data import TaskData
from resources_servers.chartography.verifier_fixture import verify_http_request


SERVER_DIR = Path(__file__).resolve().parents[1]
JUDGE_PROMPT_FPATH = SERVER_DIR / "prompt_templates/judge.txt"
# sha256 of upstream JUDGE_PROMPT_TEMPLATE (surge-ai/chartography src/chartography/scorer.py @ 3f1bf837).
UPSTREAM_JUDGE_PROMPT_SHA256 = (
    "fecef1ea1ef988cfd125573d41494a27c61fe6125bb740064bed47084ead537a"  # pragma: allowlist secret
)
QUESTION = "At a flow of 35 L/s, what head does the 225 mm impeller deliver? Give the value in m."
GOLDEN = "Head: 34.5 m (acceptable range 33.5 to 35.5 m)"


@fixture
def config() -> ChartographyConfig:
    return ChartographyConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_chat_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
        judge_prompt_template_fpath=str(JUDGE_PROMPT_FPATH),
    )


@fixture
def server(config: ChartographyConfig) -> ChartographyResourcesServer:
    return ChartographyResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _assistant_msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_id",
        content=[NeMoGymResponseOutputText(annotations=[], text=text)],
        role="assistant",
        status="completed",
        type="message",
    )


def _response(output: list) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="test",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _params() -> NeMoGymResponseCreateParamsNonStreaming:
    """Question text, then the chart image."""
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": QUESTION},
                    {"type": "input_image", "image_url": "data:image/png;base64,chart", "detail": "auto"},
                ],
            }
        ]
    )


def _request(answer_output: list, golden: Any = GOLDEN, prompt: Any = QUESTION) -> ChartographyVerifyRequest:
    meta = {"task_id": "t-1", "domain": "curve_family", "prompt": prompt, "golden_answer": golden}
    return ChartographyVerifyRequest(
        responses_create_params=_params(), response=_response(answer_output), verifier_metadata=meta
    )


def _judge_content(*scores: int) -> str:
    return json.dumps(
        {
            "scores": [
                {"criterion_id": f"criterion_{i + 1}", "extracted_answer": "34.8 m", "score": s, "rationale": "r"}
                for i, s in enumerate(scores)
            ]
        }
    )


def _mock_judge(server: ChartographyResourcesServer, *contents: str) -> AsyncMock:
    """Return the given judge outputs in order, repeating the last one."""
    payloads = [
        NeMoGymChatCompletion(
            id="j",
            created=0,
            model="judge",
            object="chat.completion",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content=c),
                )
            ],
        ).model_dump_json()
        for c in contents
    ]

    async def _post(*args: Any, **kwargs: Any) -> MagicMock:
        payload = payloads[min(_post.calls, len(payloads) - 1)]
        _post.calls += 1
        resp = MagicMock()
        resp.read = AsyncMock(return_value=payload)
        return resp

    _post.calls = 0
    mock = AsyncMock(side_effect=_post)
    server.server_client.post = mock
    return mock


def test_judge_prompt_is_the_upstream_prompt() -> None:
    template = JUDGE_PROMPT_FPATH.read_text().rstrip("\n")
    assert hashlib.sha256(template.encode()).hexdigest() == UPSTREAM_JUDGE_PROMPT_SHA256


def test_render_puts_each_value_in_its_slot() -> None:
    template = JUDGE_PROMPT_FPATH.read_text().rstrip("\n")
    rendered = scorer.render_judge_prompt(template, QUESTION, GOLDEN, "Head: 34.8 m")
    assert "{{" not in rendered
    assert f"Prompt: {QUESTION}\nResponse: Head: 34.8 m\nGolden answer: {GOLDEN}\n" in rendered


def test_render_does_not_expand_placeholders_inside_the_response() -> None:
    template = JUDGE_PROMPT_FPATH.read_text().rstrip("\n")
    answer = "I cannot read it. {{ rubric.all_criteria }}"
    rendered = scorer.render_judge_prompt(template, QUESTION, GOLDEN, answer)
    assert rendered.count(GOLDEN) == 1
    assert f"Response: {answer}\nGolden answer: {GOLDEN}\n" in rendered


def test_extract_final_answer_skips_reasoning() -> None:
    reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
    assert extract_final_answer(_response([reasoning, _assistant_msg("34.8 m")])) == "34.8 m"
    assert extract_final_answer(_response([reasoning])) == ""


def test_extract_final_answer_reads_the_last_assistant_message() -> None:
    output = [_assistant_msg("Let me look at the curve."), _assistant_msg("Head: 34.8 m")]
    assert extract_final_answer(_response(output)) == "Head: 34.8 m"


@pytest.mark.parametrize("field", ["judge_max_attempts", "judge_endpoint_max_concurrency"])
def test_config_rejects_zero(config: ChartographyConfig, field: str) -> None:
    with raises(ValidationError):
        ChartographyConfig.model_validate({**config.model_dump(), field: 0})


def test_upstream_retry_budget_is_five_attempts() -> None:
    assert scorer.MAX_ATTEMPTS == 5


class TestScorer:
    def test_valid_output(self) -> None:
        verdicts = scorer.parse_verdict(_judge_content(1, 0))
        assert [v["criterion_id"] for v in verdicts] == ["criterion_1", "criterion_2"]
        assert [v["score"] for v in verdicts] == [Decimal(1), Decimal(0)]

    def test_markdown_fences_are_stripped(self) -> None:
        assert len(scorer.parse_verdict("```json\n" + _judge_content(1) + "\n```")) == 1

    @pytest.mark.parametrize(
        "raw",
        [
            "not json",
            "[]",
            '{"scores": []}',
            '{"scores": [{"criterion_id": "criterion_1", "score": 1, "rationale": "r"}]}',
            '{"scores": [{"criterion_id": "c", "extracted_answer": "x", "score": true, "rationale": "r"}]}',
            '{"scores": [{"criterion_id": "c", "extracted_answer": "x", "score": "high", "rationale": "r"}]}',
            '{"scores": [{"criterion_id": "c", "extracted_answer": "x", "score": "NaN", "rationale": "r"}]}',
            '{"scores": [{"criterion_id": "c", "extracted_answer": "x", "score": 1e30, "rationale": "r"}]}',
            '{"scores": [{"criterion_id": "c", "extracted_answer": "x", "score": "1e30", "rationale": "r"}]}',
        ],
    )
    def test_invalid_output_raises(self, raw: str) -> None:
        with raises(scorer.ParseError):
            scorer.parse_verdict(raw)

    @pytest.mark.parametrize(
        "score,passes", [("1", True), ("0.99995", True), ("0.9999", False), ("0", False), ("1.0", True)]
    )
    def test_pass_rule_rounds_to_four_decimals(self, score: str, passes: bool) -> None:
        assert scorer.verdict_passes(Decimal(score)) is passes


class TestVerify:
    @pytest.mark.asyncio
    async def test_all_criteria_pass(self, server: ChartographyResourcesServer) -> None:
        _mock_judge(server, _judge_content(1, 1))
        result = await server.verify(verify_http_request(), _request([_assistant_msg("Head: 34.8 m")]))
        assert result.reward == 1.0 and result.all_pass == 1.0 and result.num_criteria == 2
        assert [v.passed for v in result.verdicts] == [True, True]

    @pytest.mark.asyncio
    async def test_strict_pass_fails_on_any_criterion(self, server: ChartographyResourcesServer) -> None:
        _mock_judge(server, _judge_content(1, 0))
        result = await server.verify(verify_http_request(), _request([_assistant_msg("Head: 31 m")]))
        assert result.reward == 0.0 and result.all_pass == 0.0
        assert [v.passed for v in result.verdicts] == [True, False]

    @pytest.mark.asyncio
    async def test_judge_request_matches_upstream(self, server: ChartographyResourcesServer) -> None:
        mock = _mock_judge(server, _judge_content(1))
        reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
        await server.verify(verify_http_request(), _request([reasoning, _assistant_msg("Head: 34.8 m")]))
        sent = mock.await_args.kwargs["json"]
        assert mock.await_args.kwargs["server_name"] == "judge"
        (message,) = sent.messages
        assert message["role"] == "user"
        assert f"Prompt: {QUESTION}\nResponse: Head: 34.8 m\nGolden answer: {GOLDEN}\n" in message["content"]
        assert message["content"].startswith("Read the following prompt and response pair")
        assert sent.response_format["json_schema"] == {
            "name": scorer.RESPONSE_SCHEMA_NAME,
            "schema": scorer.RESPONSE_JSON_SCHEMA,
        }

    @pytest.mark.asyncio
    async def test_empty_answer_scores_zero_without_judging(self, server: ChartographyResourcesServer) -> None:
        mock = _mock_judge(server, _judge_content(1))
        reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
        result = await server.verify(verify_http_request(), _request([reasoning]))
        assert result.reward == 0.0 and mock.await_count == 0 and not result.mask_sample

    @pytest.mark.asyncio
    async def test_parse_failure_is_retried(self, server: ChartographyResourcesServer) -> None:
        mock = _mock_judge(server, "not json", _judge_content(1))
        result = await server.verify(verify_http_request(), _request([_assistant_msg("Head: 34.8 m")]))
        assert result.reward == 1.0 and result.judge_attempts == 2 and mock.await_count == 2

    @pytest.mark.asyncio
    async def test_out_of_range_score_is_retried(self, server: ChartographyResourcesServer) -> None:
        too_large = _judge_content(1).replace('"score": 1', '"score": 1e30')
        mock = _mock_judge(server, too_large, _judge_content(1))
        result = await server.verify(verify_http_request(), _request([_assistant_msg("Head: 34.8 m")]))
        assert result.reward == 1.0 and result.judge_attempts == 2 and mock.await_count == 2

    @pytest.mark.asyncio
    async def test_parse_failure_exhausts_five_attempts(self, server: ChartographyResourcesServer) -> None:
        mock = _mock_judge(server, '{"scores": []}')
        with raises(JudgeError):
            await server.verify(verify_http_request(), _request([_assistant_msg("Head: 34.8 m")]))
        assert mock.await_count == 5

    @pytest.mark.asyncio
    async def test_judge_transport_failure_is_reported(
        self, server: ChartographyResourcesServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(judge, "_judge_retry_wait_s", lambda attempt: 0)
        server.server_client.post = AsyncMock(side_effect=ConnectionError("judge down"))
        with raises(NgFailureEvent) as failure:
            await server.verify(verify_http_request(), _request([_assistant_msg("Head: 34.8 m")]))
        assert is_judge_failure(failure.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("golden,prompt", [("", QUESTION), (GOLDEN, ""), (None, QUESTION)])
    async def test_invalid_reference_is_masked(
        self, server: ChartographyResourcesServer, golden: Any, prompt: Any
    ) -> None:
        mock = _mock_judge(server, _judge_content(1))
        result = await server.verify(
            verify_http_request(), _request([_assistant_msg("Head: 34.8 m")], golden=golden, prompt=prompt)
        )
        assert result.mask_sample and result.reward == 0.0 and mock.await_count == 0
        assert result.failure_kind == "chartography:invalid_reference"


class TestGradingRecord:
    """verify() records its grade for `gym eval submit` (grading path A)."""

    @staticmethod
    async def _verify(server: ChartographyResourcesServer, request: ChartographyVerifyRequest):
        invocation = VerifierInvocation(invocation_id="verify-1")
        token = verifier_observability._active_verifier.set(invocation)
        try:
            result = await server.verify(verify_http_request(), request)
        finally:
            verifier_observability._active_verifier.reset(token)
        return result, invocation.grading_operations

    @pytest.mark.asyncio
    @pytest.mark.parametrize("score,reward", [(1, 1.0), (0, 0.0)])
    async def test_a_parsed_verdict_completes_the_grading(
        self, server: ChartographyResourcesServer, score: int, reward: float
    ) -> None:
        _mock_judge(server, _judge_content(score))
        result, operations = await self._verify(server, _request([_assistant_msg("Head: 34.8 m")]))
        assert result.reward == reward
        (operation,) = operations
        assert (operation.kind, operation.outcome) == ("llm_judge", "completed")
        assert len(operation.dependencies) == 1
        assert operation.result_binding_sha256 == grading_result_binding(result)

    @pytest.mark.asyncio
    async def test_a_parse_retry_leaves_the_grading_unknown(self, server: ChartographyResourcesServer) -> None:
        _mock_judge(server, "not json", _judge_content(1))
        result, operations = await self._verify(server, _request([_assistant_msg("Head: 34.8 m")]))
        assert result.reward == 1.0
        assert [(o.outcome, len(o.dependencies)) for o in operations] == [("unknown", 2)]

    @pytest.mark.asyncio
    async def test_unparseable_output_fails_the_grading(self, server: ChartographyResourcesServer) -> None:
        _mock_judge(server, "not json")
        with raises(JudgeError):
            await self._verify(server, _request([_assistant_msg("Head: 34.8 m")]))

    @pytest.mark.asyncio
    async def test_an_empty_answer_has_no_judge_grading(self, server: ChartographyResourcesServer) -> None:
        _mock_judge(server, _judge_content(1))
        reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
        result, operations = await self._verify(server, _request([reasoning]))
        assert result.reward == 0.0 and operations is None


class TestMetrics:
    @staticmethod
    def _rollout(reward: float, domain: str | None = None) -> dict:
        row: dict = {"reward": reward, "all_pass": reward}
        if domain:
            row["verifier_metadata"] = {"domain": domain}
        return row

    def test_score_fn(self) -> None:
        assert ChartographyResourcesServer._score_fn(self._rollout(1.0)) == {"reward": 1.0, "all_pass": 1.0}
        # A rollout counted as 0 without verification carries only its reward.
        assert ChartographyResourcesServer._score_fn({"reward": 0.0}) == {"reward": 0.0, "all_pass": 0.0}

    def test_compute_metrics_overall_and_per_domain(self, server: ChartographyResourcesServer) -> None:
        tasks = [
            [self._rollout(1.0, "finance"), self._rollout(0.0, "finance")],
            [self._rollout(1.0, "finance"), self._rollout(1.0, "finance")],
            [self._rollout(0.0, "energy"), self._rollout(0.0, "energy")],
        ]
        metrics = server.compute_metrics(tasks)
        # Percentages, as compute_pass_majority_metrics reports them.
        assert metrics["pass@1[avg-of-2]/all_pass"] == pytest.approx(50.0)
        assert metrics["pass@2/all_pass"] == pytest.approx(200 / 3)
        assert metrics["finance/pass@1[avg-of-2]/all_pass"] == pytest.approx(75.0)
        assert metrics["finance/pass@2/all_pass"] == pytest.approx(100.0)
        assert metrics["energy/pass@2/all_pass"] == pytest.approx(0.0)
        assert "finance/per_sample_aggregate" not in metrics

    def test_compute_metrics_without_domain_has_no_subsets(self, server: ChartographyResourcesServer) -> None:
        metrics = server.compute_metrics([[self._rollout(1.0)]])
        assert metrics["pass@1[avg-of-1]/all_pass"] == pytest.approx(100.0)
        assert not any(k.startswith(("finance/", "energy/", "None/")) for k in metrics)

    def test_get_key_metrics_selects_the_headline(self, server: ChartographyResourcesServer) -> None:
        agent_metrics = {
            "mean/input_tokens": 1000.0,
            "mean/output_tokens": 200.0,
            "pass@1[avg-of-5]/all_pass": 0.3,
            "pass@1[avg-of-10]/all_pass": 0.32,
            "pass@1[avg-of-10]/reward": 0.32,
            "pass@10/all_pass": 0.6,
            "finance/pass@1[avg-of-10]/all_pass": 0.5,
        }
        key = server.get_key_metrics(agent_metrics)
        assert key == {
            "mean/input_tokens": 1000.0,
            "mean/output_tokens": 200.0,
            "pass@1[avg-of-10]/all_pass": 0.32,
            "pass@1[avg-of-10]/reward": 0.32,
            "pass@10/all_pass": 0.6,
        }


@pytest.mark.asyncio
async def test_verifier_fixture() -> None:
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )
    assert [(r.kind, r.observed_rewards) for r in results] == [
        ("full_reward", (1.0,)),
        ("zero_reward", (0.0,)),
        ("malformed", ()),
    ]


def test_example_rows_are_well_formed() -> None:
    rows = [json.loads(line) for line in (SERVER_DIR / "data/example.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == 5
    for row in rows:
        content = row["responses_create_params"]["input"][0]["content"]
        assert [c["type"] for c in content] == ["input_text", "input_image"]
        assert content[1]["image_url"].startswith("data:image/png;base64,")
        meta = row["verifier_metadata"]
        TaskData.model_validate(meta)
        assert meta["prompt"].strip() == content[0]["text"].strip() and meta["golden_answer"].strip()
        assert set(meta) == {"task_id", "domain", "prompt", "golden_answer"}


@pytest.mark.parametrize("missing", ["task_id", "prompt", "golden_answer"])
def test_task_data_requires_the_graded_fields(missing: str) -> None:
    meta = {"task_id": "t-1", "prompt": QUESTION, "golden_answer": GOLDEN}
    del meta[missing]
    with raises(ValidationError):
        TaskData.model_validate(meta)
