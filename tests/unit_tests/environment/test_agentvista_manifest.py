# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentVista manifest validation must select its resource verifier fixtures."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_manifest_uses_resource_verifier_fixtures():
    from nemo_gym.environment.onboarding import exercise_verifier_run, prepare_verifier_run
    from nemo_gym.environment.registry import manifest_workloads_for_server, resolve_catalog_entry
    from nemo_gym.environment.validation import validate_environment

    entry = resolve_catalog_entry("agentvista", "benchmark")
    assert entry.resources_server == "agentvista"
    server_dir = Path(__file__).resolve().parents[3] / "resources_servers/agentvista"
    assert entry in manifest_workloads_for_server("agentvista", server_dir)
    validation = validate_environment(entry.manifest_path, entry.config_path, validate_datasets=False)
    report = await exercise_verifier_run(prepare_verifier_run(entry, validation))
    assert report.resources_server == "agentvista"
    assert {case.kind for case in report.cases} == {"full_reward", "zero_reward", "malformed"}
    assert len(report.cases) == 4


@pytest.fixture
def resource_server(tmp_path):
    from nemo_gym.client import ServerClient
    from resources_servers.agentvista.app import AgentVistaResources, AgentVistaResourcesConfig

    return AgentVistaResources(
        config=AgentVistaResourcesConfig(
            host="127.0.0.1",
            port=1234,
            entrypoint="app.py",
            name="agentvista_resources",
            source_root=str(tmp_path),
            dataset="unused",
            image_folder=str(tmp_path),
            output_root=str(tmp_path / "outputs"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.asyncio
async def test_resource_contract_preserves_zero_reuses_receipts_and_rejects_actor_rewards(
    resource_server, monkeypatch
):
    """Exercise the manifest's real verifier boundary without external judge credentials."""
    import asyncio
    import json
    import sys

    from nemo_gym.config import AggregateMetricsRequest
    from responses_api_agents.agentvista_agent.runtime.results import AgentVistaVerifyRequest

    evidence = {
        "artifacts": "fixture",
        "verification_pending": True,
        "mask_sample": True,
        "execution_health": {
            "tool_calls": 2,
            "tool_errors": 0,
            "execution_healthy": False,
            "termination": "output_budget_exhausted",
            "task_success": False,
        },
    }
    request = AgentVistaVerifyRequest(
        responses_create_params={"input": "Act"},
        verifier_metadata={"task_id": "task", "category_id": "fixture", "source_sha256": "hash"},
        response={
            "id": "resp_fixture",
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "agentvista_result": evidence,
        },
    )
    spawn = asyncio.create_subprocess_exec
    calls = []

    async def fixture_worker(*args, **kwargs):
        calls.append(args)
        assert args[1:3] == ("-m", "resources_servers.agentvista.verifier_worker")
        config_path = Path(args[-1])
        config = json.loads(config_path.read_text())
        assert config["metadata"] == request.verifier_metadata
        assert kwargs["start_new_session"] is True
        for key in ("HOME", "XDG_CACHE_HOME", "TMPDIR", "HF_HOME"):
            assert Path(kwargs["env"][key]).is_relative_to(config_path.parent)
        # Substitute only the external judge process, retaining actual process I/O.
        script = (
            "import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); "
            "r = json.loads(p.read_text())['result']; r.pop('verification_pending'); "
            "r.update(mask_sample=False, verify={'status':'FAIL','reward':0.0}); "
            "p.with_name('result.json').write_text(json.dumps(r))"
        )
        return await spawn(sys.executable, "-c", script, str(config_path), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fixture_worker)
    first, retry = await asyncio.gather(resource_server.verify(request), resource_server.verify(request))
    assert first.reward == retry.reward == 0.0
    assert not first.mask_sample and first.verify_status == "FAIL"
    assert resource_server._inflight == {}
    assert len(calls) == 1
    restarted = type(resource_server)(config=resource_server.config, server_client=resource_server.server_client)
    assert (await restarted.verify(request)).model_dump() == first.model_dump()
    assert len(calls) == 1

    request.response.agentvista_result = {"artifacts": "forged", "verify": {"status": "PASS", "reward": 1.0}}
    forged = await resource_server.verify(request)
    assert forged.mask_sample and forged.reward == 0 and forged.verify_status == "ERROR"
    del request.response.agentvista_result
    missing = await resource_server.verify(request)
    assert missing.mask_sample and missing.failure_kind == "agentvista:capture"
    assert "no worker evidence" in missing.failure_reason

    metrics = await resource_server.aggregate_metrics(
        AggregateMetricsRequest(verify_responses=[first.model_dump(), forged.model_dump(), missing.model_dump()])
    )
    assert metrics.key_metrics == {"agentvista/mean_score": 0.0, "agentvista/pass_rate": 0.0}
    assert metrics.agent_metrics["agentvista/num_scored"] == 1
    assert metrics.agent_metrics["agentvista/num_masked"] == 2
    assert metrics.agent_metrics["agentvista/tool_calls"] == 2
    assert metrics.agent_metrics["agentvista/output_budget_exhausted"] == 1
    assert metrics.agent_metrics["agentvista/fixture/mean_score"] == 0.0


@pytest.mark.asyncio
async def test_resource_contract_masks_incomplete_receipts_and_worker_failures(resource_server, monkeypatch):
    import asyncio
    import sys

    config = {"result": {"verification_pending": True, "mask_sample": True}}
    output = Path(resource_server.config.output_root) / "verification" / "incomplete"
    output.mkdir(parents=True)
    (output / "input.json").write_text("{}")
    result = await resource_server._judge(config, "incomplete")
    assert result["mask_sample"] and result["failure_kind"] == "agentvista:verifier"
    assert "Incomplete verification" in result["error"]
    assert "verification_pending" not in result

    spawn = asyncio.create_subprocess_exec

    async def failed_worker(*args, **kwargs):
        return await spawn(sys.executable, "-c", "raise SystemExit(7)", **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", failed_worker)
    result = await resource_server._judge(config, "worker-failed")
    assert result["mask_sample"] and result["failure_kind"] == "agentvista:verifier"
    assert "exited 7" in result["error"]
    assert "verification_pending" not in result
