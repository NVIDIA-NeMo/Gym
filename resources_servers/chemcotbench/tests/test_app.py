# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows
from resources_servers.chemcotbench.app import (
    ChemCoTBenchResourcesServer,
    ChemCoTBenchResourcesServerConfig,
    ChemCoTBenchVerifyRequest,
)
from resources_servers.chemcotbench.scoring_pool import ScoringWorkerError
from resources_servers.chemcotbench.setup_upstream import ensure_data, ensure_repository
from resources_servers.chemcotbench.task_data import TaskData
from resources_servers.chemcotbench.worker import json_safe


ROOT = Path(__file__).parents[1]


def response(text):
    return NeMoGymResponse(
        id="test",
        created_at=0,
        model="test",
        object="response",
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        output=[]
        if text is None
        else [
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


def request(text):
    row = json.loads((ROOT / "data/example.jsonl").read_text().splitlines()[0])
    return ChemCoTBenchVerifyRequest(**row, response=response(text))


@pytest.fixture
async def server(monkeypatch, tmp_path):
    monkeypatch.setattr("resources_servers.chemcotbench.app.ensure_repository", lambda *args: tmp_path)
    monkeypatch.setattr("resources_servers.chemcotbench.app.ensure_data", lambda *args: tmp_path)
    instance = ChemCoTBenchResourcesServer(
        config=ChemCoTBenchResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="chemcotbench",
            enable_molopt=False,
            timeout_seconds=0.02,
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    yield instance
    await instance.close()


@pytest.mark.parametrize("text", [None, "", "<think>Answer: C</think>", "<thinking>Answer: C"])
async def test_empty_output(server, text):
    result = await server.verify(request(text))
    assert result.reward == 0 and result.parse_ok is False and result.scoring_error is None
    assert not result.mask_sample


async def test_subprocess_protocol(server, monkeypatch):
    score = AsyncMock(
        return_value={
            "reward": 1.0,
            "layer1_correct": True,
            "predicted_answer": "CCO",
            "layer2_state_score": 0.5,
        }
    )
    monkeypatch.setattr(server._pool, "score", score)
    body = request("<think>not final</think>Answer: CCO")
    before = body.model_dump()
    result = await server.verify(body)
    assert result.reward == 1.0 and result.layer2_state_score == 0.5
    command, payload = score.call_args.args
    assert payload["generation"] == "Answer: CCO"
    assert payload["metadata"] == body.verifier_metadata.model_dump()
    assert payload["run_layer3"] is True
    assert "--data-dir" in command and "--serve" in command
    assert body.model_dump() == before


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (ScoringWorkerError("bad output"), "upstream_error"),
        (ScoringWorkerError("bad JSON", code="invalid_result"), "invalid_result"),
        (OSError("worker cannot start"), "upstream_error"),
        (TimeoutError(), "timeout"),
    ],
)
async def test_worker_failure(server, monkeypatch, failure, error):
    monkeypatch.setattr(server._pool, "score", AsyncMock(side_effect=failure))
    result = await server.verify(request("Answer: invalid"))
    assert result.reward == 0 and result.scoring_error == error and result.failure_reason
    assert result.mask_sample


async def test_invalid_worker_response(server, monkeypatch):
    monkeypatch.setattr(server._pool, "score", AsyncMock(return_value={"reward": "bad"}))
    result = await server.verify(request("Answer: invalid"))
    assert result.scoring_error == "invalid_result"
    assert result.mask_sample and result.failure_reason


async def test_disabled_optimization_is_masked(server):
    body = request("Answer: CCCC")
    body.verifier_metadata = TaskData(
        id="synthetic", task_family="mol_opt", subtask="logp", upstream_record={"src": "CCO", "tgt": "CCCC"}
    )
    result = await server.verify(body)
    assert result.reward == 0 and result.mask_sample
    assert result.scoring_error == "molopt_disabled" and result.failure_reason


async def test_aggregation_excludes_failures_but_keeps_wrong_answers(server, monkeypatch):
    monkeypatch.setattr(
        server._pool, "score", AsyncMock(side_effect=[{"reward": 1.0}, TimeoutError(), {"reward": 0.0}])
    )
    rows = []
    for index in range(3):
        result = await server.verify(request("Answer: CCO"))
        rows.append(result.model_dump() | {"_ng_task_index": index, "_ng_rollout_index": 0})
    assert not rows[0]["mask_sample"] and not rows[2]["mask_sample"]
    for selected, expected_mean, measured in [(rows[:2], 1.0, 1), (rows, 0.5, 2)]:
        metrics = (await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=selected))).agent_metrics
        assert metrics["mean/reward"] == expected_mean
        assert metrics["coverage/measured_rollouts"] == measured
        assert metrics["coverage/masked_rollouts"] == 1


async def test_cancellation_propagates(server, monkeypatch):
    monkeypatch.setattr(server._pool, "score", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await server.verify(request("Answer: C"))


async def test_lifespan_closes_workers(server, monkeypatch):
    close = AsyncMock()
    monkeypatch.setattr(server._pool, "close", close)
    app = server.setup_webserver()
    async with app.router.lifespan_context(app):
        close.assert_not_awaited()
    close.assert_awaited_once()


@pytest.mark.parametrize(("field", "value"), [("subtask", "../../other"), ("upstream_record", {}), ("id", "wrong")])
def test_invalid_task(field, value):
    row = request("").verifier_metadata.model_dump()
    row[field] = value
    with pytest.raises(ValidationError):
        TaskData(**row)


def test_example_schema():
    path = ROOT / "data/example.jsonl"
    report = validate_jsonl_rows("chemcotbench", load_task_data_schema(ROOT), str(path), path.read_text().splitlines())
    assert report.rows == 5 and report.error_rows == 0 and not report.unknown_keys


def test_json_safe():
    import numpy as np

    assert json_safe({"a": np.int64(2), "b": (np.float64(0.5), float("nan"), float("inf"))}) == {
        "a": 2,
        "b": [0.5, None, None],
    }


@pytest.mark.skipif(
    importlib.util.find_spec("rdkit") is None, reason="RDKit is required for upstream chemistry verification"
)
async def test_real_upstream_reference_cases(monkeypatch):
    repo = ensure_repository(os.environ.get("CHEMCOTBENCH_TEST_REPO"))
    data = ensure_data(os.environ.get("CHEMCOTBENCH_TEST_DATA"))
    server = ChemCoTBenchResourcesServer(
        config=ChemCoTBenchResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="chemcotbench",
            repo_path=str(repo),
            enable_molopt=False,
            data_dir=str(data),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    try:
        cases = [json.loads(line) for line in (ROOT / "tests/verifier_cases.jsonl").read_text().splitlines()]
        monkeypatch.syspath_prepend(str(repo))
        monkeypatch.setenv("CHEMCOT_DATA_DIR", str(data))
        from resources_servers.chemcotbench.worker import score_record

        for case in cases:
            body = ChemCoTBenchVerifyRequest.model_validate(case["request"])
            result = await server.verify(body)
            assert result.scoring_error is None, result.failure_reason
            assert result.reward == case["expected_reward"], case["name"]
            metadata = body.verifier_metadata.model_dump()
            before = copy.deepcopy(metadata)
            direct = score_record(metadata, body.response.output_text, True)
            assert direct["reward"] == result.reward
            assert metadata == before
            for key, value in case.get("expected_metrics", {}).items():
                assert getattr(result, key) == value, (case["name"], key)
        # Exercise the real JSON worker entry point, including stdout/stderr separation.
        from resources_servers.chemcotbench.worker import main

        body = ChemCoTBenchVerifyRequest.model_validate(cases[0]["request"])
        output = io.StringIO()
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", ["worker.py", "--repo", str(repo), "--data-dir", str(data)])
            context.setattr(
                sys,
                "stdin",
                io.StringIO(
                    json.dumps(
                        {
                            "metadata": body.verifier_metadata.model_dump(),
                            "generation": body.response.output_text,
                            "run_layer3": True,
                        }
                    )
                ),
            )
            context.setattr(sys, "stdout", output)
            main()
        assert json.loads(output.getvalue())["reward"] == 1.0
        # Turning off Layer 3 retains the answer score and exposes absent secondary metrics.
        server.config.run_layer3 = False
        server._data = None
        result = await server.verify(ChemCoTBenchVerifyRequest.model_validate(cases[0]["request"]))
        assert result.reward == 1 and result.layer3_type1 is None and result.layer3_type2 is None
    finally:
        await server.close()


@pytest.mark.skipif(
    importlib.util.find_spec("rdkit") is None, reason="RDKit is required for upstream chemistry verification"
)
@pytest.mark.parametrize(
    "subtask,reference,text,predicted,reward",
    [
        ("yield_pred", {"gt_float": 98.6922}, "Answer: 98.6922", "98.6922", 1.0),
        ("yield_pred", {"gt_float": 50.0}, "Answer: 0", "0", 0.0),
        ("yield_pred", {"gt_float": 50.0}, "I cannot answer.", None, 0.0),
        ("retro", {"gt_reactants": "CCO"}, "Answer: OCC", "OCC", 1.0),
        ("retro", {"gt_reactants": "CCO"}, "Answer: CCC", "CCC", 0.0),
        ("retro", {"gt_reactants": "CCO"}, "I cannot answer.", None, 0.0),
    ],
)
async def test_real_upstream_task_specific_predictions(subtask, reference, text, predicted, reward):
    repo = ensure_repository(os.environ.get("CHEMCOTBENCH_TEST_REPO"))
    server = ChemCoTBenchResourcesServer(
        config=ChemCoTBenchResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="chemcotbench",
            repo_path=str(repo),
            enable_molopt=False,
            run_layer3=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    try:
        body = ChemCoTBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Synthetic chemistry question"}]},
            response=response(text),
            verifier_metadata={
                "id": "synthetic",
                "task_family": "rxn_pred",
                "subtask": subtask,
                "upstream_record": reference,
            },
        )
        result = await server.verify(body)
        assert result.scoring_error is None, result.failure_reason
        assert result.predicted_answer == predicted
        assert result.reward == reward
        assert not result.mask_sample
    finally:
        await server.close()


async def test_workload_verifier_fixture():
    from nemo_gym.verifier_fixture import exercise_verifier_fixture
    from resources_servers.chemcotbench.app import VERIFIER_FIXTURE

    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )
    assert {result.kind for result in results} >= {"full_reward", "zero_reward", "malformed"}
