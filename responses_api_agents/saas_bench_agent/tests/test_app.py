# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External-harness boundary tests; no private benchmark data or app images required."""

import asyncio
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest, ModelServerRef
from responses_api_agents.saas_bench_agent.app import SaasBenchAgent, SaasBenchConfig, SaasBenchRunRequest


@pytest.fixture
def agent(tmp_path, monkeypatch):
    source = tmp_path / "source"
    package = source / "saas_bench/opencode"
    package.mkdir(parents=True)
    for path in [source / "saas_bench/__init__.py", package / "__init__.py"]:
        path.write_text("")
    (package / "worker.py").write_text("import time\ntime.sleep(3600)\n")
    core = types.ModuleType("saas_bench.opencode.core")
    core.HARNESS_VERSION = "2.1.1"
    core.measured_score = lambda result: result["score"]
    core.response_items = lambda events: []
    observer = types.ModuleType("saas_bench.opencode.observability")
    observer.native_trajectory = lambda *args: None
    monkeypatch.setitem(sys.modules, "saas_bench.opencode.core", core)
    monkeypatch.setitem(sys.modules, "saas_bench.opencode.observability", observer)
    config = SaasBenchConfig(
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        name="saas_test",
        source_root=str(source),
        output_root=str(tmp_path / "episodes"),
        worker_python=sys.executable,
        command="/explicit/opencode",
        model_name="policy",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    return SaasBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))


def request(identity="repeat-0"):
    return SaasBenchRunRequest.model_validate(
        {
            "responses_create_params": {"input": "Generated fixture task"},
            "_ng_rollout_id": identity,
            "verifier_metadata": {
                "task_id": "fixture",
                "category_id": "Fixture",
                "source_sha256": "source-hash",
                "harness_version": "2.1.1",
            },
        }
    )


def result(score=0.25, masked=False):
    return {
        "artifacts": "/private/episode",
        "agent_output": "Done",
        "trajectory": [],
        "mask_sample": masked,
        "verify": {"status": "FAIL", "score": score},
    }


@pytest.mark.parametrize("score", [0.0, 0.25, 1.0])
def test_native_partial_and_zero_scores_remain_unmasked_with_grading_artifact(agent, score):
    response = agent._response(request(), result(score))
    assert response.reward == score and response.mask_sample is False
    assert response.native_metrics == {"weighted_score": score, "execution": 1.0}
    assert response.native_diagnostics["graded_response_id"] == response.response.id
    assert (
        response.native_diagnostics["graded_response_sha256"]
        == hashlib.sha256(response.response.model_dump_json().encode()).hexdigest()
    )
    assert response.response.usage is None


def test_infrastructure_failure_is_masked_and_keeps_native_diagnostics(agent):
    native = result(masked=True)
    native.update(failure_kind="saas_bench:verifier", error="readback unavailable")
    response = agent._response(request(), native)
    assert response.mask_sample is True and response.reward == 0.0
    assert response.failure_kind == "verifier_error"
    assert response.native_failure_kind == "saas_bench:verifier"
    assert not response.native_metrics


@pytest.mark.asyncio
async def test_metrics_exclude_masked_but_include_legitimate_zero(agent):
    rows = [
        {"reward": reward, "mask_sample": masked, "category_id": "Fixture", "verify_status": "FAIL"}
        for reward, masked in [(0.0, False), (0.5, False), (0.0, True)]
    ]
    metrics = (await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))).agent_metrics
    assert metrics["saas_bench/mean_score"] == 0.25
    assert metrics["saas_bench/num_scored"] == 2 and metrics["saas_bench/num_masked"] == 1


@pytest.mark.asyncio
async def test_duplicate_http_requests_share_episode_and_repeats_do_not(agent, monkeypatch):
    calls = []

    async def episode(body, output, fingerprint, identity):
        calls.append((identity, output))
        await asyncio.sleep(0)
        return agent._response(body, result())

    monkeypatch.setattr(agent, "_episode", episode)
    one, two = await asyncio.gather(agent.run(request()), agent.run(request()))
    assert one is two and len(calls) == 1
    await agent.run(request("repeat-1"))
    assert len(calls) == 2 and calls[0][1] != calls[1][1]


@pytest.mark.asyncio
async def test_persisted_result_is_reused_and_binding_changes_are_refused(agent, monkeypatch):
    body = request()
    identity = hashlib.sha256(b"repeat-0").hexdigest()
    output = Path(agent.config.output_root) / identity
    output.mkdir(parents=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {"request": body.model_dump(), "config": agent.config.model_dump(), "harness_version": "2.1.1"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (output / "binding.json").write_text(json.dumps({"fingerprint": fingerprint}))
    (output / "result.json").write_text(json.dumps(result()))

    async def unexpected(*args, **kwargs):
        pytest.fail("cached episode launched a worker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected)
    assert (await agent.run(body)).reward == 0.25
    changed = body.model_copy(deep=True)
    changed.verifier_metadata["task_id"] = "other"
    with pytest.raises(HTTPException, match="different request"):
        await agent.run(changed)


@pytest.mark.asyncio
async def test_missing_identity_is_rejected_before_worker(agent):
    with pytest.raises(HTTPException, match="_ng_rollout_id"):
        await agent.run(request(None))


@pytest.mark.asyncio
async def test_cancellation_reaps_actual_worker_process(agent, monkeypatch):
    created = asyncio.Event()
    child = []
    original = asyncio.create_subprocess_exec

    async def launch(*args, **kwargs):
        process = await original(*args, **kwargs)
        child.append(process)
        created.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(type(agent), "resolve_model_base_url", lambda *args: "http://localhost:9/v1")
    task = asyncio.create_task(agent.run(request()))
    await asyncio.wait_for(created.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert child[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child[0].pid, 0)
