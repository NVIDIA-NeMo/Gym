# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.spatialclaw.app import (
    SpatialClawResourcesServer,
    SpatialClawResourcesServerConfig,
    SpatialClawVerifyRequest,
    _flatten_numeric,
)


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _server() -> tuple[SpatialClawResourcesServer, object, object]:
    first = SimpleNamespace(sample_id="one", answer="A")
    second = SimpleNamespace(sample_id="two", answer="B")

    class FakeBenchmark:
        def __init__(self):
            self.data = [first, second]

        def extract_answer(self, prediction):
            return prediction.strip().upper()

        def evaluate_single(self, sample, prediction):
            return float(self.extract_answer(prediction) == sample.answer)

        def evaluate(self, predictions, output_dir=None):
            assert output_dir is None
            scores = [self.evaluate_single(sample, predictions.get(sample.sample_id, "")) for sample in self.data]
            return {
                "overall_accuracy": sum(scores) / len(scores),
                "total_samples": len(scores),
                "per_category": {"spatial": {"accuracy": sum(scores) / len(scores)}},
                "detailed_results": [{"sample_id": sample.sample_id} for sample in self.data],
            }

    config = SpatialClawResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="spatialclaw_test",
        dataset_config="fake",
    )
    server = SpatialClawResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._benchmark_instance = FakeBenchmark()
    server._sample_by_id = {"one": first, "two": second}
    return server, first, second


def test_flatten_numeric_drops_details() -> None:
    assert _flatten_numeric({"overall": 0.5, "nested": {"count": 2}, "detailed_results": [{"score": 1}]}) == {
        "overall": 0.5,
        "nested/count": 2,
    }


@pytest.mark.asyncio
async def test_verify_uses_native_per_sample_scorer() -> None:
    server, _, _ = _server()
    body = SpatialClawVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("A"),
        sample_id="one",
        answer="A",
    )
    verified = await server.verify(body)
    assert verified.reward == 1.0
    assert verified.native_score == 1.0
    assert verified.extracted_answer == "A"
    assert verified.scored is True


def test_compute_metrics_uses_native_subset_and_restores_dataset() -> None:
    server, first, second = _server()
    original_data = server._benchmark_instance.data
    metrics = server.compute_metrics(
        [[{"sample_id": "two", "prediction": "B", "extracted_answer": "B", "native_score": 1.0, "reward": 1.0}]]
    )
    assert metrics["spatialclaw/overall_accuracy"] == 1.0
    assert metrics["spatialclaw/total_samples"] == 1
    assert metrics["spatialclaw/per_category/spatial/accuracy"] == 1.0
    assert metrics["spatialclaw/num_evaluated_tasks"] == 1
    assert server._benchmark_instance.data is original_data
    assert server._benchmark_instance.data == [first, second]
