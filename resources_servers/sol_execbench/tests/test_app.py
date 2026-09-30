# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native result classification and mocked OpenSandbox lifecycle contracts."""

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.sol_execbench.app import (
    VERIFIER_FIXTURE,
    SolExecBenchResourcesServer,
    SolExecBenchResourcesServerConfig,
    SolExecBenchVerifyRequest,
    classify_native_result,
)
from resources_servers.sol_execbench.fixture import synthetic_problem
from resources_servers.sol_execbench.problem_store import (
    NATIVE_REVISION,
    Asset,
    canonical_json,
    checked_asset,
    load_manifest,
    problem_digest,
    safe_relative_path,
)


def trace(status="PASSED"):
    payload = json.loads((Path(__file__).parent / "verifier_cases.jsonl").read_text().splitlines()[0])["request"][
        "traces"
    ][0]
    payload["evaluation"]["status"] = status
    if status != "PASSED":
        payload["evaluation"]["performance"] = None
        if status != "INCORRECT_NUMERICAL":
            payload["evaluation"]["correctness"] = None
    return payload


def request(name="synthetic_solution", text=None):
    solution = {
        "name": name,
        "definition": "synthetic_identity",
        "author": "synthetic",
        "spec": {"languages": ["triton"], "target_hardware": ["B200"], "entry_point": "main.py::run"},
        "sources": [{"path": "main.py", "content": "def run(x): return x"}],
    }
    return SolExecBenchVerifyRequest(
        verifier_metadata={
            "task_id": synthetic_problem().task_id,
            "problem_digest": synthetic_problem().problem_digest,
        },
        responses_create_params={"input": "Original synthetic test prompt"},
        response={
            "id": name,
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "output": [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text if text is not None else json.dumps(solution),
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
    )


class FakeSandbox:
    instances = []
    mode = "pass"

    def __init__(self, provider, spec):
        self.provider, self.spec = provider, spec
        self.files, self.commands = {}, []
        self.stopped = False
        self.instances.append(self)

    async def start(self):
        if self.mode == "create_failure":
            raise RuntimeError("provider unavailable")

    async def upload(self, local, remote):
        self.files[remote] = Path(local).read_bytes()

    async def download(self, remote, local):
        Path(local).write_bytes(self.files[remote])

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command.startswith("mkdir"):
            return SandboxExecResult("", "", 0)
        await asyncio.sleep(0.01)
        if self.mode == "timeout":
            return SandboxExecResult("partial", "watchdog", 125, error_type="timeout")
        solution = json.loads(self.files["/sol-eval/solution.json"])
        native_trace = trace("INCORRECT_NUMERICAL" if self.mode == "candidate_failure" else "PASSED")
        native_trace["solution"] = solution["name"]
        records = {
            "hardware.json": {
                "nvidia_smi": "NVIDIA B200, GPU-synthetic, synthetic",
                "native_revision": NATIVE_REVISION,
            },
            "validation.json": {"valid": True},
            "execution.json": {
                "return_code": 1 if self.mode == "candidate_failure" else 0,
                "native_schema_validated": True,
                "native_workloads_validated": True,
            },
        }
        for name, data in records.items():
            self.files[f"/sol-eval/{name}"] = canonical_json(data)
        if self.mode != "missing_trace":
            self.files["/sol-eval/trace.jsonl"] = canonical_json(native_trace) + b"\n"
        self.files["/sol-eval/native.stdout"] = b"native stdout"
        self.files["/sol-eval/native.stderr"] = b"native stderr"
        return SandboxExecResult("runner stdout", "runner stderr", 0)

    async def stop(self):
        self.stopped = True
        if self.mode == "cleanup_failure":
            raise RuntimeError("sandbox cleanup not confirmed")


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr("resources_servers.sol_execbench.app.AsyncSandbox", FakeSandbox)
    FakeSandbox.instances = []
    FakeSandbox.mode = "pass"
    path = tmp_path / "problem_manifest.json"
    path.write_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "source": {"repository": "synthetic", "revision": "synthetic"},
                "native_revision": NATIVE_REVISION,
                "problems": [synthetic_problem().model_dump()],
            }
        )
    )
    config = SolExecBenchResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="sol_execbench",
        problem_manifest_path=path,
        problem_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        artifact_root=tmp_path / "artifacts",
        sandbox_image="synthetic/image@sha256:" + "a" * 64,
        sandbox_provider={"opensandbox": {"operations": {"command_retries": 0}}},
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return SolExecBenchResourcesServer(config=config, server_client=client)


@pytest.mark.asyncio
async def test_native_success_uses_gpu_latency_records_evidence_and_replays(server):
    first, duplicate = await asyncio.gather(server.verify(request()), server.verify(request()))
    assert first == duplicate
    assert first.reward == 1 and not first.mask_sample and first.sol_score is None
    assert first.latency_ms == {"synthetic-workload": 0.01}
    assert first.reference_latency_ms == {}
    assert len(FakeSandbox.instances) == 1
    sandbox = FakeSandbox.instances[0]
    assert sandbox.stopped and sandbox.spec.resources.gpu == 1
    assert sandbox.spec.image.endswith("@sha256:" + "a" * 64)
    assert sandbox.spec.entrypoint == ["/bin/sh", "-c", "exec sleep infinity"]
    assert sandbox.provider["opensandbox"]["operations"]["command_retries"] == 0
    assert sandbox.commands[-1] == ("/venv/bin/python /sol-eval/native_runner.py", {"timeout_s": 900})
    assert json.loads(sandbox.files["/sol-eval/config.json"])["benchmark_reference"] is False
    assert json.loads(sandbox.files["/sol-eval/protocol.json"])["native_revision"] == NATIVE_REVISION
    attempt = Path(first.artifact_path)
    assert (attempt / "native.stdout").read_text() == "native stdout"
    assert (attempt / "native.stderr").read_text() == "native stderr"
    assert json.loads((attempt / "trace.jsonl").read_bytes())["evaluation"]["status"] == "PASSED"
    assert (await server.verify(request())).reward == 1
    assert len(FakeSandbox.instances) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,masked,reward",
    [
        ("candidate_failure", False, 0),
        ("missing_trace", True, 0),
        ("timeout", True, 0),
        ("create_failure", True, 0),
        ("cleanup_failure", True, 0),
    ],
)
async def test_native_failure_lifecycle(server, mode, masked, reward):
    FakeSandbox.mode = mode
    result = await server.verify(request())
    assert result.mask_sample is masked and result.reward == reward
    assert result.infrastructure_error is masked and result.sol_score is None
    assert FakeSandbox.instances[0].stopped
    assert json.loads((Path(result.artifact_path) / "result.json").read_bytes())["outcome"] == result.outcome


@pytest.mark.asyncio
async def test_bad_output_does_not_allocate_gpu_and_metadata_cannot_choose_paths(server):
    result = await server.verify(request(text="not JSON"))
    assert result.outcome == "INVALID_SOLUTION" and result.reward == 0 and not result.mask_sample
    assert not FakeSandbox.instances
    body = request().model_dump()
    body["verifier_metadata"]["path"] = "/untrusted"
    with pytest.raises(ValidationError, match="Extra inputs"):
        SolExecBenchVerifyRequest.model_validate(body)
    bad = request()
    bad.verifier_metadata.problem_digest = "f" * 64
    with pytest.raises(ValueError, match="server-owned"):
        await server.verify(bad)
    assert not FakeSandbox.instances


@pytest.mark.asyncio
async def test_incomplete_cached_attempt_is_not_retried(server):
    result = await server.verify(request())
    (Path(result.artifact_path) / "result.json").unlink()
    repeated = await server.verify(request())
    assert repeated.mask_sample and repeated.outcome == "ATTEMPT_UNRESOLVED"
    assert len(FakeSandbox.instances) == 1


@pytest.mark.parametrize(
    "traces,code,outcome",
    [
        ([], 0, "INCOMPLETE_TRACE"),
        ([trace(), trace()], 0, "INCOMPLETE_TRACE"),
        ([trace("RUNTIME_ERROR")], 1, "NATIVE_UNRESOLVED"),
        ([trace("INVALID_REFERENCE")], 1, "NATIVE_UNRESOLVED"),
        ([trace("TIMEOUT")], 1, "NATIVE_UNRESOLVED"),
        ([trace("INCORRECT_NUMERICAL")], 1, "CANDIDATE_FAILED"),
        ([trace()], 1, "INVALID_EXIT_STATUS"),
    ],
)
def test_native_status_and_exact_workload_coverage(traces, code, outcome):
    result = classify_native_result(
        problem=synthetic_problem(),
        solution_name="synthetic_solution",
        return_code=code,
        traces=traces,
        benchmark_reference=False,
    )
    assert result.outcome == outcome
    assert result.infrastructure_error == (outcome != "CANDIDATE_FAILED")


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -1, 0, True])
def test_invalid_native_latency_is_unresolved(latency):
    native_trace = trace()
    native_trace["evaluation"]["performance"]["latency_ms"] = latency
    result = classify_native_result(
        problem=synthetic_problem(),
        solution_name="synthetic_solution",
        return_code=0,
        traces=[native_trace],
        benchmark_reference=False,
    )
    assert result.infrastructure_error


def test_manifest_digest_and_safe_asset_paths(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_relative_path("../outside")
    blob = tmp_path / "blob.safetensors"
    blob.write_bytes(b"synthetic content, not a corpus asset")
    asset = Asset(path=blob.name, sha256=hashlib.sha256(blob.read_bytes()).hexdigest())
    assert checked_asset(tmp_path, asset) == blob
    blob.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checked_asset(tmp_path, asset)
    outside = tmp_path.parent / "outside.sol-test"
    outside.write_bytes(b"outside")
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        checked_asset(tmp_path, Asset(path="link", sha256=hashlib.sha256(b"outside").hexdigest()))
    assert problem_digest("a", {"inputs": {"x": 1, "y": 2}}, [], []) != problem_digest(
        "a", {"inputs": {"y": 2, "x": 1}}, [], []
    )
    with pytest.raises(ValueError, match="manifest SHA256"):
        load_manifest(blob, "a" * 64)


@pytest.mark.asyncio
async def test_verifier_fixture_contract():
    cases = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=[0, 1], determinism="unknown")
    assert [case.kind for case in cases] == ["full_reward", "zero_reward", "malformed"]


async def test_missing_workload_payload_validation_is_unresolved(server):
    response = await server.verify(request())
    attempt = Path(response.artifact_path)
    execution = json.loads((attempt / "execution.json").read_text())
    del execution["native_workloads_validated"]
    (attempt / "execution.json").write_text(json.dumps(execution))
    solution = json.loads((attempt / "solution.json").read_text())
    result = server._read_result(attempt, synthetic_problem(), solution)
    assert result.infrastructure_error
    assert result.outcome == "INVALID_NATIVE_RESULT"
    assert "workload payload validation" in result.detail
