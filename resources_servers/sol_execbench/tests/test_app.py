# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic command-contract tests; no SOL installation or GPU is needed."""

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ConfigDict, ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.sol_execbench.app import (
    EvaluationManifest,
    EvaluatorResult,
    SolExecBenchResourcesServer,
    SolExecBenchResourcesServerConfig,
    SolExecBenchVerifyRequest,
    canonical_json,
)


PROTOCOL = "a" * 64
GPU = "GPU-00000000-0000-0000-0000-000000000000"
RAW_TEXT = "reasoning\r\n<answer>\rλ\n</answer>"

# This trusted synthetic command records actual process overlap and device environment.
RUNNER = r"""
import argparse, json, os, pathlib, signal, subprocess, sys, time
p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
p.add_argument('--result', required=True)
a = p.parse_args()
request = json.loads(pathlib.Path(a.request).read_bytes())
metadata = request['verifier_metadata']
root = pathlib.Path(a.request).parent.parent
lock = root / 'synthetic.lock'
with lock.open('x'):
    pass
with (root / 'executions.jsonl').open('a') as stream:
    stream.write(json.dumps({'request_id': request['request_id'], 'pid': os.getpid()}) + '\n')
mode = metadata.get('mode', 'pass')
if mode in ('hang', 'exit_leader'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    if mode == 'exit_leader':
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    pathlib.Path(a.request).with_name('child.pid').write_text(str(child.pid))
    time.sleep(60)
if mode == 'detached':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    def terminate(*_):
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=1)
        pathlib.Path(a.request).with_name('detached-cleaned').write_text('yes')
        sys.exit(0)
    signal.signal(signal.SIGTERM, terminate)
    pathlib.Path(a.request).with_name('child.pid').write_text(str(child.pid))
    time.sleep(60)
time.sleep(0.05)
if mode == 'exit':
    sys.exit(7)
if mode == 'missing':
    lock.unlink()
    sys.exit(0)
outcome = {'timeout': 'EVALUATION_TIMEOUT', 'infra': 'ENVIRONMENT_FAILURE',
           'candidate': 'CANDIDATE_FAILED'}.get(mode, 'PASSED')
infra = mode in ('timeout', 'infra')
result = {k: request[k] for k in ('request_id', 'task_id', 'protocol_sha256')}
result.update(outcome=outcome, infrastructure_error=infra, solved=outcome == 'PASSED',
              sol_score=None if infra else (0.25 if outcome == 'PASSED' else 0.0),
              native_result={'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'raw': request['response']})
if mode == 'identity':
    result['request_id'] = 'f' * 64
if mode == 'bool':
    result['infrastructure_error'] = 'false'
if mode == 'nan':
    result['sol_score'] = float('nan')
if mode == 'nested_nan':
    result['native_result']['detail'] = float('nan')
pathlib.Path(a.result).write_text(json.dumps(result))
lock.unlink()
"""


def request(response_id: str = "response-one", **metadata) -> SolExecBenchVerifyRequest:
    response = NeMoGymResponse(
        id=response_id,
        created_at=0,
        model="synthetic",
        object="response",
        output=[
            {
                "id": "message-one",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": RAW_TEXT, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return SolExecBenchVerifyRequest(
        task_id="synthetic-task",
        verifier_metadata=metadata,
        responses_create_params={"input": "Synthetic prompt"},
        response=response,
    )


@pytest.fixture
def server(tmp_path: Path, monkeypatch) -> SolExecBenchResourcesServer:
    monkeypatch.setattr("resources_servers.sol_execbench.app.TERMINATION_GRACE_S", 0.2)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "protocol_sha256": PROTOCOL,
                "samples_per_task": 2,
                "tasks": [{"task_id": "synthetic-task"}],
            }
        )
    )
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER)
    config = SolExecBenchResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="sol_execbench",
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        evaluator_command=[sys.executable, str(runner)],
        artifact_root=tmp_path / "artifacts",
        gpu_uuid=GPU,
    )
    return SolExecBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def executions(server: SolExecBenchResourcesServer) -> list[dict]:
    path = server.config.artifact_root / "executions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.asyncio
async def test_concurrent_retries_run_once_and_different_requests_serialize(server):
    first, duplicate, second = await asyncio.gather(
        server.verify(request()),
        server.verify(request()),
        server.verify(request("response-two")),
    )
    assert first == duplicate
    assert first.request_id != second.request_id
    assert len(executions(server)) == 2
    assert all(item.outcome == "PASSED" and item.reward == 0.25 for item in (first, second))
    assert first.native_result["gpu_uuid"] == GPU
    assert first.native_result["raw"]["output"][0]["content"][0]["text"] == RAW_TEXT
    restarted = SolExecBenchResourcesServer(config=server.config, server_client=server.server_client)
    assert (await restarted.verify(request())).request_id == first.request_id
    assert len(executions(server)) == 2


@pytest.mark.asyncio
async def test_metadata_is_data_and_changes_request_identity(server):
    first = await server.verify(request(command="touch forbidden", provenance={"sample": 0}))
    second = await server.verify(request(command="touch forbidden", provenance={"sample": 1}))
    assert first.request_id != second.request_id
    saved = json.loads((server.config.artifact_root / first.request_id / "request.json").read_bytes())
    assert saved["verifier_metadata"] == {
        "command": "touch forbidden",
        "provenance": {"sample": 0},
    }
    assert first.verifier_metadata == saved["verifier_metadata"]


@pytest.mark.asyncio
async def test_prompt_sampling_and_trusted_argv_are_part_of_identity(server):
    body = request()
    first = await server.verify(body)
    body.responses_create_params.input = "A different prompt"
    second = await server.verify(body)
    body.responses_create_params.temperature = 0.75
    third = await server.verify(body)
    server.config.evaluator_command.insert(1, "-u")
    fourth = await server.verify(body)
    assert len({row.request_id for row in (first, second, third, fourth)}) == 4
    saved = json.loads((server.config.artifact_root / third.request_id / "request.json").read_bytes())
    assert saved["responses_create_params"]["input"] == "A different prompt"
    assert saved["responses_create_params"]["temperature"] == 0.75
    assert len(executions(server)) == 4


@pytest.mark.asyncio
async def test_native_timeout_stays_null_masked_and_worker_continues(server):
    timeout = await server.verify(request(mode="timeout"))
    success = await server.verify(request("response-two"))
    assert timeout.outcome == "EVALUATION_TIMEOUT"
    assert timeout.sol_score is None and timeout.reward == 0 and timeout.mask_sample
    assert not timeout.solved and timeout.failure_kind == "sol_execbench:evaluation_timeout"
    assert success.outcome == "PASSED" and len(executions(server)) == 2


@pytest.mark.asyncio
async def test_candidate_failure_is_measured_zero(server):
    result = await server.verify(request(mode="candidate"))
    assert result.sol_score == 0 and result.reward == 0 and not result.mask_sample
    assert not result.solved and result.failure_kind is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, expected",
    [
        ("infra", "ENVIRONMENT_FAILURE"),
        ("missing", "INVALID_RESULT"),
        ("identity", "INVALID_RESULT"),
        ("bool", "INVALID_RESULT"),
        ("nan", "INVALID_RESULT"),
        ("nested_nan", "INVALID_RESULT"),
        ("exit", "RUNNER_FAILED"),
    ],
)
async def test_infrastructure_failures_stop_queued_and_restarted_worker(server, mode, expected):
    failed, queued = await asyncio.gather(server.verify(request(mode=mode)), server.verify(request("response-two")))
    assert failed.outcome == expected and failed.mask_sample and failed.sol_score is None
    assert queued.outcome == "WORKER_STOPPED" and queued.mask_sample
    assert len(executions(server)) == 1
    restarted = SolExecBenchResourcesServer(config=server.config, server_client=server.server_client)
    assert (await restarted.verify(request("response-three"))).outcome == "WORKER_STOPPED"
    assert (await restarted.verify(request(mode=mode))).outcome == expected
    assert len(executions(server)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_request", [False, True])
async def test_uncertain_attempt_never_reruns(server, corrupt_request):
    result = await server.verify(request())
    attempt = server.config.artifact_root / result.request_id
    (attempt / "accepted.json").unlink()
    if corrupt_request:
        (attempt / "request.json").write_text("{}")
    unknown = await server.verify(request())
    assert unknown.outcome == "ATTEMPT_UNRESOLVED" and unknown.sol_score is None
    assert len(executions(server)) == 1
    assert (await server.verify(request("response-two"))).outcome == "WORKER_STOPPED"


async def wait_for_child(server) -> Path:
    for _ in range(200):
        children = list(server.config.artifact_root.glob("*/child.pid"))
        if children:
            return children[0]
        await asyncio.sleep(0.01)
    raise AssertionError("Synthetic child did not start")


def assert_reaped(attempt: Path, expected_returncode: int = -9):
    process = json.loads((attempt / "process.json").read_bytes())
    assert process["returncode"] == expected_returncode
    with pytest.raises(ProcessLookupError):
        os.kill(process["pid"], 0)
    child = int((attempt / "child.pid").read_text())
    # An orphan may briefly remain a zombie, but must not be executing.
    for _ in range(100):
        stat = Path(f"/proc/{child}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    raise AssertionError("Synthetic child still executing after process-group cleanup")


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Evaluator process-group isolation requires POSIX")
async def test_transport_timeout_reaps_descendants_and_stops_worker(server):
    server.config.runner_timeout_s = 0.4
    result = await server.verify(request(mode="hang"))
    assert result.outcome == "RUNNER_TIMEOUT" and result.mask_sample and result.sol_score is None
    assert_reaped(server.config.artifact_root / result.request_id)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Evaluator process-group isolation requires POSIX")
@pytest.mark.parametrize("mode", ["detached", "exit_leader"])
async def test_graceful_shutdown_cleans_detached_child_and_group_after_leader_exit(server, mode):
    server.config.runner_timeout_s = 0.4
    result = await server.verify(request(mode=mode))
    assert result.outcome == "RUNNER_TIMEOUT" and result.mask_sample
    attempt = server.config.artifact_root / result.request_id
    assert_reaped(attempt, expected_returncode=0)
    if mode == "detached":
        assert (attempt / "detached-cleaned").read_text() == "yes"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Evaluator process-group isolation requires POSIX")
async def test_disconnected_caller_does_not_cancel_shared_work_but_shutdown_reaps_it(
    server,
):
    app = server.setup_webserver()
    async with app.router.lifespan_context(app):
        caller = asyncio.create_task(server.verify(request(mode="hang")))
        child = await wait_for_child(server)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert len(server._inflight) == 1
    accepted = json.loads((child.parent / "accepted.json").read_bytes())
    assert accepted["outcome"] == "RUNNER_CANCELLED" and accepted["sol_score"] is None
    assert_reaped(child.parent)


@pytest.mark.asyncio
async def test_manifest_rejects_unknown_task_without_execution(server):
    body = request().model_copy(update={"task_id": "unselected-task"})
    result = await server.verify(body)
    assert result.outcome == "UNKNOWN_TASK" and result.mask_sample and result.sol_score is None
    assert not executions(server)


@pytest.mark.asyncio
async def test_persistence_failure_halts_worker_before_another_launch(server, monkeypatch):
    from resources_servers.sol_execbench import app

    original_write = app.write_json

    def fail_accept(path, value):
        if path.name == "accepted.json":
            raise OSError("Synthetic persistence failure")
        original_write(path, value)

    monkeypatch.setattr(app, "write_json", fail_accept)
    failed = await server.verify(request())
    assert failed.outcome == "PERSISTENCE_FAILURE" and failed.mask_sample
    monkeypatch.setattr(app, "write_json", original_write)
    assert (await server.verify(request("response-two"))).outcome == "WORKER_STOPPED"
    assert len(executions(server)) == 1


def test_manifest_pin_and_strict_schema(server):
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        SolExecBenchResourcesServer(
            config=server.config.model_copy(update={"manifest_sha256": "b" * 64}),
            server_client=server.server_client,
        )
    manifest = json.loads(server.config.manifest_path.read_bytes())
    for patch in (
        {"schema_version": True},
        {"samples_per_task": True},
        {"tasks": []},
        {"tasks": [{"task_id": "same"}, {"task_id": "same"}]},
        {"tasks": [{"task_id": " "}]},
        {"extra": "forbidden"},
    ):
        with pytest.raises(ValidationError):
            EvaluationManifest.model_validate(manifest | patch)


@pytest.mark.parametrize("workers", [None, 0, 2, -1, True, "1"])
def test_server_rejects_multiple_or_ambiguous_workers(server, workers):
    with pytest.raises(ValidationError):
        SolExecBenchResourcesServerConfig.model_validate(server.config.model_dump() | {"num_workers": workers})


def test_one_process_and_no_ray_are_explicit(server):
    from nemo_gym.server_utils import _server_uses_ray

    assert server.config.num_workers == 1
    assert _server_uses_ray(SolExecBenchResourcesServer) is False


@pytest.mark.parametrize(
    "patch",
    [
        {"solved": False},
        {"infrastructure_error": True},
        {"sol_score": None},
        {"sol_score": True},
        {"solved": 1},
        {"outcome": "FAILED"},
        {"outcome": "EVALUATION_TIMEOUT", "solved": False, "sol_score": 0.0},
        {"outcome": "UNRECOGNIZED", "solved": False, "sol_score": 0.0},
    ],
)
def test_result_contract_rejects_inconsistent_native_outcomes(patch):
    result = {
        "request_id": "b" * 64,
        "task_id": "synthetic-task",
        "protocol_sha256": PROTOCOL,
        "outcome": "PASSED",
        "solved": True,
        "infrastructure_error": False,
        "sol_score": 0.25,
    }
    with pytest.raises(ValidationError):
        EvaluatorResult.model_validate(result | patch)


def test_real_fastapi_request_and_response_schema(server):
    with TestClient(server.setup_webserver()) as client:
        assert client.get("/reverify_mode").json() == "stateless"
        response = client.post("/verify", json=request().model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert response.json()["outcome"] == "PASSED"
        assert response.json()["response"]["output"][0]["content"][0]["text"] == RAW_TEXT
        invalid = request().model_dump(mode="json")
        invalid["response"] = {"id": "incomplete"}
        assert client.post("/verify", json=invalid).status_code == 422


@pytest.mark.asyncio
async def test_evaluator_and_gym_result_fields_override_request_extras(server):
    class ExtraRequest(SolExecBenchVerifyRequest):
        model_config = ConfigDict(extra="allow")

    body = ExtraRequest.model_validate(
        request(mode="candidate").model_dump()
        | {
            "outcome": "PASSED",
            "reward": 99.0,
            "sol_score": 99.0,
            "solved": True,
            "request_id": "f" * 64,
            "mask_sample": True,
            "native_result": {"forged": True},
            "failure_kind": "forged",
            "failure_reason": "forged",
        }
    )
    result = await server.verify(body)
    assert result.outcome == "CANDIDATE_FAILED" and not result.solved
    assert result.reward == result.sol_score == 0.0
    assert result.request_id != "f" * 64 and not result.mask_sample
    assert "forged" not in result.native_result
    assert result.failure_kind is None and result.failure_reason is None


@pytest.mark.asyncio
async def test_aggregate_override_receives_masked_rows_and_external_denominator(server, monkeypatch):
    from nemo_gym.config_types import AggregateMetrics
    from resources_servers.sol_execbench import app

    seen = {}

    def aggregate(rows, **kwargs):
        seen.update(rows=rows, **kwargs)
        return AggregateMetrics(agent_metrics={"official": None})

    monkeypatch.setattr(app, "aggregate_sol_results", aggregate)
    masked = {"task_id": "synthetic-task", "mask_sample": True, "sol_score": None}
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[masked]))
    assert seen == {
        "rows": [masked],
        "task_ids": ["synthetic-task"],
        "samples_per_task": 2,
        "protocol_sha256": PROTOCOL,
        "timeout_zero_sensitivity": False,
    }
    assert result.agent_metrics["official"] is None
