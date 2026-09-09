# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.gdp_pdf.app import (
    GdpPdfResourcesServer,
    GdpPdfResourcesServerConfig,
    GdpPdfVerifyRequest,
    extract_response_text,
    parse_judge_verdict,
)


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                type="message",
                role="assistant",
                status="completed",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _empty_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _http_response(text: str) -> AsyncMock:
    response = AsyncMock()
    payload = _response(text).model_dump()
    response.json = AsyncMock(return_value=payload)
    response.read = AsyncMock(return_value=orjson.dumps(payload))
    return response


class TestJudgeVerdict:
    def test_reasoning_is_not_an_answer(self) -> None:
        response = NeMoGymResponse.model_validate(
            _empty_response().model_dump()
            | {"output": [{"type": "reasoning", "id": "r", "summary": [{"type": "summary_text", "text": "PASS"}]}]}
        )
        assert extract_response_text(response) == ""

    def test_exact_pass(self) -> None:
        assert parse_judge_verdict("PASS") == (True, True)

    def test_last_tag_wins(self) -> None:
        assert parse_judge_verdict("PASS was considered, final: FAIL") == (False, True)

    def test_malformed_is_failed_and_visible(self) -> None:
        assert parse_judge_verdict("unclear") == (False, False)


class TestGdpPdfResourcesServer:
    def test_missing_trailing_repeats_remain_in_denominator(self, server) -> None:
        resource, _ = server
        resource.config.expected_num_repeats = 5
        metrics = resource.compute_metrics(
            [
                [
                    {
                        "_ng_rollout_index": 0,
                        "all_pass": 1.0,
                        "mean_pass": 1.0,
                        "verifier_metadata": {"domain": "Finance"},
                    }
                ]
            ]
        )
        assert metrics["pass@1[avg-of-5]/all_pass"] == approx(10.0)
        assert metrics["rollouts/expected"] == 10
        assert metrics["rollouts/scored"] == 1
        assert metrics["domain/legal/pass@1[avg-of-5]/mean_pass"] == 0
        empty = resource.compute_metrics([])
        assert empty["pass@1[avg-of-5]/all_pass"] == 0
        assert empty["rollouts/scored"] == 0

    def test_explicit_k1_and_subset_denominator(self, server) -> None:
        resource, _ = server
        resource.config.expected_num_repeats = 1
        resource.config.expected_task_count = 1
        resource.config.expected_domain_task_counts = {"Finance": 1}
        metrics = resource.compute_metrics(
            [
                [
                    {
                        "_ng_rollout_index": 0,
                        "all_pass": 1.0,
                        "mean_pass": 1.0,
                        "verifier_metadata": {"domain": "Finance"},
                    }
                ]
            ]
        )
        assert metrics["pass@1[avg-of-1]/all_pass"] == 100
        assert metrics["rollouts/expected"] == 1

    async def test_truncated_judge_verdict_is_retried(self, server) -> None:
        resource, client = server
        truncated = _response("PASS").model_copy(update={"status": "incomplete"})
        first = AsyncMock()
        first.read = AsyncMock(return_value=orjson.dumps(truncated.model_dump()))
        client.post = AsyncMock(side_effect=[first, _http_response("FAIL")])
        result = await resource._judge_criterion("Task", "Answer", {"id": "c1", "criterion": "Correct."})
        assert result.passed is False
        assert client.post.await_count == 2

    @fixture
    def server(self) -> tuple[GdpPdfResourcesServer, MagicMock]:
        config = GdpPdfResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="gdp_pdf",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            expected_task_count=2,
            expected_domain_task_counts={"Finance": 1, "Legal": 1},
        )
        client = MagicMock(spec=ServerClient)
        return GdpPdfResourcesServer(config=config, server_client=client), client

    async def test_independent_criteria_and_dense_reward(
        self, server: tuple[GdpPdfResourcesServer, MagicMock]
    ) -> None:
        resource, client = server
        client.post = AsyncMock(side_effect=[_http_response("PASS"), _http_response("FAIL")])
        request = GdpPdfVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            response=_response("candidate"),
            verifier_metadata={
                "task_id": "t1",
                "task_prompt": "Analyze the filing.",
                "domain": "Finance",
                "rubric_criteria": [
                    {"id": "rubric-1", "criterion": "States revenue."},
                    {"id": "rubric-2", "criterion": "States margin."},
                ],
                "secret_document_text": "must never reach the judge",
            },
        )

        result = await resource.verify(request)

        assert result.reward == approx(0.5)
        assert result.reward_components == {"mean_pass": 0.5, "all_pass": 0.0}
        assert result.criteria_passed == 1
        assert result.criteria_total == 2
        assert [evaluation.passed for evaluation in result.criterion_evaluations] == [True, False]
        assert client.post.await_count == 2
        prompts = [call.kwargs["json"].input[0].content for call in client.post.await_args_list]
        assert all("Analyze the filing." in prompt and "candidate" in prompt for prompt in prompts)
        assert "States revenue." in prompts[0] and "States margin." not in prompts[0]
        assert "States margin." in prompts[1] and "States revenue." not in prompts[1]
        assert all("must never reach the judge" not in prompt for prompt in prompts)

    async def test_empty_policy_output_is_zero_without_judge_call(
        self, server: tuple[GdpPdfResourcesServer, MagicMock]
    ) -> None:
        resource, client = server
        client.post = AsyncMock()
        request = GdpPdfVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            response=_empty_response(),
            verifier_metadata={
                "task_id": "t1",
                "task_prompt": "Analyze the filing.",
                "domain": "Finance",
                "rubric_criteria": [{"id": "rubric-1", "criterion": "States revenue."}],
            },
        )

        result = await resource.verify(request)

        assert result.reward == 0.0
        assert result.all_pass == 0.0
        assert result.mean_pass == 0.0
        client.post.assert_not_awaited()

    async def test_malformed_judge_output_is_retried(self, server: tuple[GdpPdfResourcesServer, MagicMock]) -> None:
        resource, client = server
        client.post = AsyncMock(side_effect=[_http_response("unclear"), _http_response("PASS")])
        request = GdpPdfVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            response=_response("candidate"),
            verifier_metadata={
                "task_id": "t1",
                "task_prompt": "Analyze the filing.",
                "domain": "Finance",
                "rubric_criteria": [{"id": "rubric-1", "criterion": "States revenue."}],
            },
        )

        result = await resource.verify(request)

        assert result.all_pass == 1.0
        assert client.post.await_count == 2

    async def test_malformed_judge_output_fails_after_bounded_retries(
        self, server: tuple[GdpPdfResourcesServer, MagicMock]
    ) -> None:
        resource, client = server
        client.post = AsyncMock(return_value=_http_response("unclear"))
        request = GdpPdfVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            response=_response("candidate"),
            verifier_metadata={
                "task_id": "t1",
                "task_prompt": "Analyze the filing.",
                "domain": "Finance",
                "rubric_criteria": [{"id": "rubric-1", "criterion": "States revenue."}],
            },
        )

        with pytest.raises(JudgeError, match="after 5 attempts"):
            await resource.verify(request)
        assert client.post.await_count == 5

    def test_metrics_are_task_macro_average_not_best_of_k(
        self, server: tuple[GdpPdfResourcesServer, MagicMock]
    ) -> None:
        resource, _ = server
        tasks = [
            [
                {
                    "_ng_rollout_index": 0,
                    "all_pass": 1.0,
                    "mean_pass": 1.0,
                    "verifier_metadata": {"domain": "Finance"},
                },
                {
                    "_ng_rollout_index": 1,
                    "all_pass": 0.0,
                    "mean_pass": 0.5,
                    "verifier_metadata": {"domain": "Finance"},
                },
            ],
            [
                {"_ng_rollout_index": 0, "all_pass": 0.0, "mean_pass": 0.5, "verifier_metadata": {"domain": "Legal"}},
            ],
        ]

        metrics = resource.compute_metrics(tasks)

        assert metrics["pass@1[avg-of-1]/all_pass"] == approx(50.0)
        assert metrics["pass@1[avg-of-1]/mean_pass"] == approx(75.0)
        assert metrics["pass@1[avg-of-2]/all_pass"] == approx(25.0)
        assert metrics["pass@1[avg-of-2]/mean_pass"] == approx(50.0)
        assert metrics["domain/finance/pass@1[avg-of-2]/mean_pass"] == approx(75.0)
        assert metrics["domain/legal/pass@1[avg-of-2]/mean_pass"] == approx(25.0)
