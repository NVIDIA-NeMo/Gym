# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.spatialclaw.app import (
    SpatialClawResourcesServer,
    SpatialClawResourcesServerConfig,
    SpatialClawVerifyRequest,
    _visible_answer,
)


def _response(answer: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        metadata={"spatialclaw_final_answer": answer},
    )


def _server() -> SpatialClawResourcesServer:
    return SpatialClawResourcesServer(
        config=SpatialClawResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )


async def test_verify_mcqa_uses_parsed_spatialclaw_answer() -> None:
    request = SpatialClawVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("ReturnAnswer('C')"),
        expected_answer="C",
        scoring_mode="auto",
    )

    result = await _server().verify(request)

    assert result.reward == 1.0
    assert result.prediction == "ReturnAnswer('C')"
    assert result.scoring_mode_used == "mcqa"


async def test_verify_token_f1_ignores_private_reasoning() -> None:
    request = SpatialClawVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("A black dog jumps over the wooden fence."),
        expected_answer="<think>private</think>The black dog jumps over a fence.",
        scoring_mode="token_f1",
    )

    result = await _server().verify(request)

    assert 0.7 < result.reward < 1.0
    assert _visible_answer("reasoning</think>Visible answer") == "Visible answer"


def test_compute_metrics_uses_native_dataset_aggregation(monkeypatch) -> None:
    samples = [SimpleNamespace(sample_id="one"), SimpleNamespace(sample_id="two")]

    class FakeBenchmark:
        data = samples

        def evaluate(self, predictions, output_dir=None):
            del output_dir
            correct = sum(predictions.get(sample.sample_id) == "correct" for sample in self.data)
            return {"overall_accuracy": correct / len(self.data), "total_samples": len(self.data)}

    monkeypatch.setattr(SpatialClawResourcesServer, "_benchmark", lambda *args: FakeBenchmark())
    tasks = [
        [
            {
                "benchmark": "fake",
                "sample_id": "one",
                "prediction": "correct",
                ROLLOUT_INDEX_KEY_NAME: 0,
            },
            {
                "benchmark": "fake",
                "sample_id": "one",
                "prediction": "correct",
                ROLLOUT_INDEX_KEY_NAME: 1,
            },
        ],
        [
            {
                "benchmark": "fake",
                "sample_id": "two",
                "prediction": "wrong",
                ROLLOUT_INDEX_KEY_NAME: 0,
            },
            {
                "benchmark": "fake",
                "sample_id": "two",
                "prediction": "correct",
                ROLLOUT_INDEX_KEY_NAME: 1,
            },
        ],
    ]

    metrics = _server().compute_metrics(tasks)

    assert metrics["native/fake/repeat_0/overall_accuracy"] == 0.5
    assert metrics["native/fake/repeat_1/overall_accuracy"] == 1.0
    assert metrics["native/fake/overall_accuracy"] == 0.75


def test_partial_grouped_metrics_keep_only_complete_native_groups() -> None:
    samples = [SimpleNamespace(sample_id=str(index), group_type="logic") for index in range(8)]

    selected = SpatialClawResourcesServer._selected_native_samples(
        "videommev2",
        samples,
        {str(index): "A" for index in (0, 1, 2, 3, 4, 6, 7)},
    )

    assert [sample.sample_id for sample in selected] == ["0", "1", "2", "3"]
