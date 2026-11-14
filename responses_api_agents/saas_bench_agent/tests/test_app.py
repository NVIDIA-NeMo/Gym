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

from environment_servers.saas_bench_environment.app import SaasBenchEnvironmentConfig, SaasBenchEnvironmentServer
from nemo_gym.client import ServerClient
from nemo_gym.config import AgentServerRef, AggregateMetricsRequest, ModelServerRef
from responses_api_agents.saas_bench_agent.app import SaasBenchAgent, SaasBenchConfig
from responses_api_agents.saas_bench_agent.schemas import SaasBenchRunRequest


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from responses_api_agents.saas_bench_agent.runtime import BENCHMARK_ROOT

    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr("responses_api_agents.saas_bench_agent.app.check_bundle_code", lambda source: None)

    monkeypatch.syspath_prepend(str(BENCHMARK_ROOT))
    source = tmp_path / "source"
    package = source / "saas_bench/opencode"
    package.mkdir(parents=True)
    for path in [source / "saas_bench/__init__.py", package / "__init__.py"]:
        path.write_text("")
    (package / "worker.py").write_text("import time\ntime.sleep(3600)\n")
    protocol = types.ModuleType("saas_bench.opencode.protocol")
    protocol_path = BENCHMARK_ROOT / "saas_bench/opencode/protocol.py"
    exec(compile(protocol_path.read_text(), str(protocol_path), "exec"), protocol.__dict__)
    monkeypatch.setitem(sys.modules, "saas_bench.opencode.protocol", protocol)
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
        cache_root=str(tmp_path / "cache"),
        output_root=str(tmp_path / "episodes"),
        worker_python=sys.executable,
        command="/explicit/opencode",
        model_name="policy",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    instance = SaasBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))
    instance._prepare_runtime()
    return instance


@pytest.fixture
def grader(agent):
    return SaasBenchEnvironmentServer(
        config=SaasBenchEnvironmentConfig(
            name="saas_environment",
            host="127.0.0.1",
            port=1235,
            entrypoint="app.py",
            agent_server=AgentServerRef(type="responses_api_agents", name=agent.config.name),
            harness=agent.config.harness,
            model_name=agent.config.model_name,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


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
def test_native_partial_and_zero_scores_remain_unmasked_with_grading_artifact(grader, score):
    response = grader._response(request(), result(score))
    assert response.reward == score and response.mask_sample is False
    assert response.native_metrics == {"weighted_score": score, "execution": 1.0}
    assert response.native_diagnostics["graded_response_id"] == response.response.id
    assert (
        response.native_diagnostics["graded_response_sha256"]
        == hashlib.sha256(response.response.model_dump_json().encode()).hexdigest()
    )
    assert response.response.usage is None


def test_infrastructure_failure_is_masked_and_keeps_native_diagnostics(grader):
    native = result(masked=True)
    native.update(failure_kind="saas_bench:verifier", error="readback unavailable")
    response = grader._response(request(), native)
    assert response.mask_sample is True and response.reward == 0.0
    assert response.failure_kind == "verifier_error"
    assert response.native_failure_kind == "saas_bench:verifier"
    assert not response.native_metrics


@pytest.mark.asyncio
async def test_metrics_exclude_masked_but_include_legitimate_zero(grader):
    rows = [
        {"reward": reward, "mask_sample": masked, "category_id": "Fixture", "verify_status": "FAIL"}
        for reward, masked in [(0.0, False), (0.5, False), (0.0, True)]
    ]
    metrics = (await grader.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))).agent_metrics
    assert metrics["saas_bench/mean_score"] == 0.25
    assert metrics["saas_bench/num_scored"] == 2 and metrics["saas_bench/num_masked"] == 1


@pytest.mark.asyncio
async def test_duplicate_http_requests_share_episode_and_repeats_do_not(agent, monkeypatch):
    calls = []

    async def episode(body, output, fingerprint, identity):
        calls.append((identity, output))
        await asyncio.sleep(0)
        return result()

    monkeypatch.setattr(agent, "_episode", episode)
    one, two = await asyncio.gather(agent._execute(request()), agent._execute(request()))
    assert one is two and len(calls) == 1
    await agent._execute(request("repeat-1"))
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
    assert (await agent._execute(body))["verify"]["score"] == 0.25
    changed = body.model_copy(deep=True)
    changed.verifier_metadata["task_id"] = "other"
    with pytest.raises(HTTPException, match="different request"):
        await agent._execute(changed)


@pytest.mark.asyncio
async def test_missing_identity_is_rejected_before_worker(agent):
    with pytest.raises(HTTPException, match="_ng_rollout_id"):
        await agent._execute(request(None))


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
    task = asyncio.create_task(agent._execute(request()))
    await asyncio.wait_for(created.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert child[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child[0].pid, 0)


@pytest.mark.parametrize(
    "ending,kind",
    [
        ("malformed_tool_call", "agent_invalid_output"),
        ("model_protocol_error", "agent_invalid_output"),
        ("output_token_limit", "agent_limit"),
        ("context_limit", "model_context_limit"),
        ("timeout", "agent_timeout"),
        ("agent_error", "agent_invalid_output"),
        ("permission_abort", "agent_command_not_allowed"),
    ],
)
def test_model_endings_keep_the_measured_grade(grader, ending, kind):
    from responses_api_agents.saas_bench_agent.schemas import SaasBenchVerifyResponse

    native = result(score=0.75, masked=True)
    native.update(
        failure_kind="saas_bench:model_ending", execution_health={"termination": ending}, model_turn_observed=True
    )
    response = grader._response(request(), native)
    assert response.reward == 0.75 and response.mask_sample is False
    assert response.failure_kind == kind
    assert response.failure_outcome_action == "verify"
    assert response.failure_outcome_attribution == "model"
    restored = SaasBenchVerifyResponse.model_validate(response.model_dump(mode="json"))
    assert restored.failure_outcome_action == "verify"
    assert restored.failure_outcome_attribution == "model"


@pytest.mark.parametrize("kind", ["judge_failed", "judge_unparseable"])
def test_exhausted_judge_fault_wins_and_reports_terminal_evaluation_failure(grader, kind):
    native = result(score=0.75)
    native["verify"]["judge_failures"] = [{"failure_kind": kind, "reason": "fixture"}]
    native["execution_health"] = {"termination": "timeout"}
    response = grader._response(request(), native)
    assert response.mask_sample is True and response.reward == 0
    assert response.failure_kind == kind
    assert response.failure_outcome_action == "fail"
    assert response.failure_outcome_attribution == "evaluation"
    assert response.native_metrics == {}


def test_timeout_flag_with_model_turn_keeps_canonical_failure_kind(grader):
    native = result(score=0.75)
    native["timed_out"] = True
    native["model_turn_observed"] = True
    response = grader._response(request(), native)
    assert response.failure_kind == "agent_timeout"
    assert response.failure_outcome_action == "verify"
    assert response.failure_outcome_attribution == "model"
    assert response.reward == 0.75 and response.mask_sample is False


@pytest.mark.parametrize("infra_kind", [None, "saas_bench:infrastructure"])
def test_timeout_without_model_turn_stays_retryable(grader, infra_kind):
    native = result(score=0.75)
    native.update(
        timed_out=True, failure_kind=infra_kind, native_trajectory=[], execution_health={"termination": "timeout"}
    )
    response = grader._response(request(), native)
    assert response.mask_sample and response.failure_outcome_action == "retry"
    assert response.failure_outcome_attribution == "evaluation"
    assert response.failure_kind == "agent_run_error"


@pytest.mark.parametrize("origin,masked", [(None, True), ("model", False)])
def test_browser_model_ending_requires_explicit_origin_even_for_cached_results(grader, origin, masked):
    grader.config.harness = "browser_use"
    native = result(score=0.75)
    native.update(
        failure_kind="saas_bench:model_ending", failure_origin=origin, execution_health={"termination": "agent_error"}
    )
    response = grader._response(request(), native)
    assert response.mask_sample is masked
    assert response.failure_outcome_attribution == ("evaluation" if masked else "model")


def test_start_failure_is_recoverable_without_a_saved_generation(grader):
    from nemo_gym.rollouts.rollout_recovery import masked_infrastructure_kind

    native = {
        "status": "error",
        "artifacts": "fixture",
        "trajectory": [],
        "native_trajectory": [],
        "agent_output": "",
        "mask_sample": True,
        "failure_kind": "saas_bench:infrastructure",
    }
    row = grader._response(request(), native).model_dump(mode="json")
    assert row["response"]["output"] == []
    assert masked_infrastructure_kind(row) == "agent_run_error"


def test_missing_judge_endpoint_fails_before_episodes(agent, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("MINDRA_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        fresh = SaasBenchAgent(config=agent.config, server_client=MagicMock(spec=ServerClient))
        fresh._prepare_runtime()


@pytest.mark.asyncio
async def test_cancellation_stops_detached_apps(agent, monkeypatch, tmp_path):
    import shutil
    import signal

    from responses_api_agents.saas_bench_agent.runtime import BENCHMARK_ROOT

    source = Path(agent.config.source_root)
    shutil.copy(BENCHMARK_ROOT / "saas_bench/enroot_slot.py", source / "saas_bench/enroot_slot.py")
    # enroot remove is external; the real PID teardown must still reap a detached keeper.
    executable = tmp_path / "enroot"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    (source / "saas_bench/opencode/worker.py").write_text(
        "import json, subprocess, sys, time\nfrom pathlib import Path\n"
        "o=Path(json.loads(Path(sys.argv[1]).read_text())['output'])\n"
        "(o/'enroot-state').mkdir(exist_ok=True)\n"
        "p=subprocess.Popen(['sleep','600'],start_new_session=True)\n"
        "(o/'enroot-state/fixture.enroot.pid').write_text(str(p.pid))\n"
        "(o/'app.pid').write_text(str(p.pid))\ntime.sleep(600)\n"
    )
    monkeypatch.setattr(type(agent), "resolve_model_base_url", lambda *args: "http://localhost:9/v1")
    output = Path(agent.config.output_root) / hashlib.sha256(b"repeat-0").hexdigest()
    task = asyncio.create_task(agent._execute(request()))

    async def wait_marker():
        while not (output / "app.pid").exists():
            await asyncio.sleep(0.05)

    await asyncio.wait_for(wait_marker(), 20)
    pid = int((output / "app.pid").read_text())
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 60)
        path = Path(f"/proc/{pid}/stat")
        assert not path.exists() or path.read_text().rpartition(")")[2].split()[0] == "Z"
    finally:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
