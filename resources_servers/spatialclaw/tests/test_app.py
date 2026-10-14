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
    assert verified.evaluation_completed is True


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


def test_masked_infrastructure_failure_is_not_a_native_wrong_answer() -> None:
    server, first, second = _server()
    metrics = server.compute_metrics(
        [
            [{"sample_id": "one", "prediction": "A", "extracted_answer": "A", "native_score": 1.0, "reward": 1.0}],
            [{"sample_id": "two", "prediction": "", "reward": 0.0, "mask_sample": True}],
        ]
    )
    assert metrics["spatialclaw/overall_accuracy"] == 1.0
    assert metrics["spatialclaw/num_evaluated_tasks"] == 1
    assert server._benchmark_instance.data == [first, second]
    empty = server.compute_metrics([[{"sample_id": "two", "mask_sample": True, "reward": 0.0}]])
    assert empty["spatialclaw/num_evaluated_tasks"] == 0
    assert "spatialclaw/overall_accuracy" not in empty


@pytest.mark.parametrize("prediction,reward", [("B", 0.0), ("", 0.0), (" a ", 1.0)])
async def test_native_wrong_and_empty_answers_remain_scored(prediction, reward):
    server, _, _ = _server()
    result = await server.verify(
        SpatialClawVerifyRequest(
            responses_create_params={"input": "question"}, response=_response(prediction), sample_id="one"
        )
    )
    assert result.reward == reward and result.evaluation_completed
    assert not result.mask_sample
    with pytest.raises(ValueError, match="Unknown SpatialClaw sample"):
        await server.verify(
            SpatialClawVerifyRequest(
                responses_create_params={"input": "question"}, response=_response(prediction), sample_id="missing"
            )
        )


async def test_native_unscored_result_is_distinguished_from_incorrect_answer():
    server, _, _ = _server()
    server._benchmark_instance.evaluate_single = lambda *args: None
    result = await server.verify(
        SpatialClawVerifyRequest(
            responses_create_params={"input": "question"}, response=_response("A"), sample_id="one"
        )
    )
    assert result.native_score is None and not result.scored
    assert result.reward == 0.0 and result.evaluation_completed


async def test_load_benchmark_once_and_reject_partial_invalid_loads(tmp_path, monkeypatch):
    import asyncio
    import sys
    from types import ModuleType

    from resources_servers.spatialclaw import app

    factory_module = ModuleType("spatial_agent.evals.factory")
    config_module = ModuleType("spatial_agent.config")

    class Config:
        benchmark = "native-name"
        question_type = None

        def _load_from_envs(self):
            pass

        def update_from_dataset_json(self, path):
            assert path == "dataset.json"

    config_module.SpatialAgentConfig = Config
    config_module.set_config = MagicMock()
    native = SimpleNamespace(data=[SimpleNamespace(sample_id=1)])
    factory = MagicMock(return_value=native)
    factory_module.BenchmarkFactory = SimpleNamespace(create_benchmark=factory)
    monkeypatch.setitem(sys.modules, "spatial_agent.config", config_module)
    monkeypatch.setitem(sys.modules, "spatial_agent.evals.factory", factory_module)
    monkeypatch.setattr(app, "_config_path", lambda *args: "dataset.json")
    monkeypatch.setattr(app, "_install_source_path", lambda *args: None)
    monkeypatch.setattr(app, "_validate_spatialclaw_checkout", lambda root, commit: root)
    monkeypatch.setattr(app, "ensure_spatialclaw_checkout", lambda url, commit, cache: tmp_path)
    server, _, _ = _server()
    server._benchmark_instance = None
    server._load_lock = None
    server.config.spatialclaw_root = str(tmp_path)
    results = await asyncio.gather(server._load_benchmark(), server._load_benchmark())
    assert results == [native, native]
    factory.assert_called_once_with("native-name", data_root=str(tmp_path / "data"), question_type=None)
    assert server._sample_by_id == {"1": native.data[0]}
    server._benchmark_instance = None
    server.config.spatialclaw_root = None
    server.config.source_cache_root = str(tmp_path / "cache")
    server.config.data_root = str(tmp_path / "dataset")
    native.data.append(SimpleNamespace(sample_id=1))
    with pytest.raises(RuntimeError, match="Duplicate"):
        await server._load_benchmark()
    assert server._benchmark_instance is None
    factory.return_value = None
    with pytest.raises(RuntimeError, match="selected no benchmark"):
        await server._load_benchmark()
    native.data.pop()
    factory.return_value = native
    assert await server._load_benchmark() is native
    assert factory.call_args.kwargs["data_root"] == str(tmp_path / "dataset")


def test_dataset_metrics_handle_empty_unknown_rows_and_select_native_headlines():
    server, _, _ = _server()
    assert (
        server.compute_metrics([[], [{"sample_id": "unknown", "reward": 0}]])["spatialclaw/num_evaluated_tasks"] == 0
    )
    metrics = {"spatialclaw/overall_accuracy": 0.5, "internal_count": 3, "pass@1[avg-of-1]/native_score": 50}
    assert server.get_key_metrics(metrics) == {
        "spatialclaw/overall_accuracy": 0.5,
        "pass@1[avg-of-1]/native_score": 50,
    }
    assert _flatten_numeric({"evaluated": True, "label": "ignored"}) == {"evaluated": 1}
