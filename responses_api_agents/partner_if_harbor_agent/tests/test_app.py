# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytest.importorskip("harbor")

from pydantic import ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.server_utils import (
    IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME,
    IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME,
    ServerClient,
    SimpleServer,
)
from resources_servers.partner_if.engine import evaluate
from resources_servers.partner_if.grading_service import GradeResponse, contract_hash
from resources_servers.partner_if.tests.test_shared_adapter import shared_request
from responses_api_agents.harbor_agent_general.app import HarborRunRequest, HarborVerifyResponse
from responses_api_agents.partner_if_harbor_agent.app import (
    PartnerIFHarborAgent,
    PartnerIFHarborConfig,
    validate_openhands_completion,
)


def test_openhands_zero_exit_is_not_enough_without_finished_event(tmp_path):
    events = tmp_path / "agent/sessions/run/events"
    events.mkdir(parents=True)
    (events / "1.json").write_text(
        json.dumps({"observation": "agent_state_changed", "extras": {"agent_state": "running"}})
    )
    with pytest.raises(RuntimeError, match="completion state"):
        validate_openhands_completion(tmp_path)
    (events / "2.json").write_text(
        json.dumps({"observation": "agent_state_changed", "extras": {"agent_state": "finished"}})
    )
    validate_openhands_completion(tmp_path)


@pytest.mark.parametrize("exists", [True, False])
async def test_harbor_delegates_visible_evidence_to_shared_grade_endpoint(tmp_path, exists):
    body = shared_request()
    contract = body.verifier_metadata.partner_if
    contract.evidence_mode = "live_agent"
    contract.rubrics = [contract.rubrics[0]]
    contract.rubrics[0].verifier.params["source"] = "artifact:submission"
    (tmp_path / "native.json").write_text(
        json.dumps({"contract": contract.model_dump(), "documents": {"skill": "Say HELLO"}})
    )
    run_body = HarborRunRequest.model_validate(
        {**body.model_dump(exclude={"response"}), "task_name": "native", "task_index": 0, "rollout_index": 0}
    )
    native = HarborVerifyResponse.model_validate(
        {
            **body.model_dump(),
            "reward": 1,
            "partner_if_artifact": {"exists": exists, "text": "HELLO" if exists else ""},
        }
    )
    calls = []

    async def post(**kwargs):
        from resources_servers.partner_if.grading_service import GradeRequest

        payload = GradeRequest.model_validate(kwargs["json"])
        calls.append((kwargs, payload))
        result = await evaluate(payload.contract, payload.evidence, judge=AsyncMock())
        receipt = GradeResponse(task_id=contract.task_id, rubric_hash=contract_hash(contract), grading_result=result)
        return MagicMock(json=AsyncMock(return_value=receipt.model_dump(mode="json")))

    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(side_effect=post)
    config = PartnerIFHarborConfig(
        name="native",
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        harbor_jobs_dir=tmp_path / "jobs",
        task_bundles_dir=tmp_path,
        harbor_dataset={"path": str(tmp_path / "tasks")},
        model_server={"type": "responses_api_models", "name": "partner_if_harbor_model"},
        grading_server={"type": "resources_servers", "name": "common"},
    )
    with patch("responses_api_agents.harbor_agent_general.app.get_global_config_dict", return_value={}):
        agent = PartnerIFHarborAgent(config=config, server_client=client)
    with (
        patch("responses_api_agents.harbor_agent_general.app.HarborAgent.run", AsyncMock(return_value=native)),
        patch("responses_api_agents.partner_if_harbor_agent.app.raise_for_status", AsyncMock()),
    ):
        result = await agent.run(run_body)
    assert calls[0][0]["server_name"] == "common" and calls[0][0]["url_path"] == "/grade"
    assert calls[0][1].evidence.artifacts["submission"].status == ("present" if exists else "absent")
    assert result.reward == int(exists) and not result.mask_sample
    assert result.model_dump()["grading_result"]["dimensions"]["if"]["partial_score"] == int(exists)


def harbor_config(tmp_path, **overrides):
    return PartnerIFHarborConfig.model_validate(
        {
            "name": "partner_if_markdown",
            "host": "127.0.0.1",
            "port": 8000,
            "entrypoint": "app.py",
            "harbor_jobs_dir": str(tmp_path / "jobs"),
            "task_bundles_dir": str(tmp_path),
            "harbor_dataset": {"path": str(tmp_path / "tasks")},
            "harbor_agent": {
                "name": "openhands",
                "model_name": "openai/policy",
                "kwargs": {"version": "0.62.0", "max_iterations": 30},
            },
            "model_server": {"type": "responses_api_models", "name": "partner_if_harbor_model"},
            "grading_server": {"type": "resources_servers", "name": "partner_if"},
            **overrides,
        }
    )


def test_job_config_points_openhands_at_the_sandbox_reachable_model_server(tmp_path, monkeypatch):
    requested = []

    def server_url(name):
        requested.append(name)
        return "http://192.0.2.10:9000/"

    monkeypatch.setattr("nemo_gym.sandbox.agent_tools.get_server_url", server_url)
    config = harbor_config(tmp_path)
    job = config.build_job_config("native", "job-1")
    [agent] = job.agents
    assert agent.kwargs == {"version": "0.62.0", "max_iterations": 30, "api_base": "http://192.0.2.10:9000/v1"}
    assert (agent.name, agent.model_name) == ("openhands", "openai/policy")
    assert requested == ["partner_if_harbor_model"]
    # Only the endpoint is added: the inherited job keeps its task, and the configured agent is not mutated.
    assert job.job_name == "job-1" and job.datasets[0].task_names == ["native"]
    assert config.harbor_agent.kwargs == {"version": "0.62.0", "max_iterations": 30}


@pytest.mark.parametrize("url", ["http://127.0.0.1:9000", "http://localhost:9000", "http://0.0.0.0:9000"])
def test_job_config_rejects_a_model_server_the_sandbox_cannot_reach(tmp_path, monkeypatch, url):
    monkeypatch.setattr("nemo_gym.sandbox.agent_tools.get_server_url", lambda name: url)
    with pytest.raises(ValueError, match="sandbox-reachable"):
        harbor_config(tmp_path).build_job_config("native", "job-1")


def test_model_server_is_required_and_relative_package_paths_resolve_under_the_gym_roots(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="model_server"):
        harbor_config(tmp_path, model_server=None)
    monkeypatch.chdir(tmp_path)
    relative = {"task_bundles_dir": "resources_servers/partner_if/data", "harbor_dataset": {"path": "benchmarks"}}
    built_in = harbor_config(tmp_path, **relative)
    assert built_in.task_bundles_dir == PARENT_DIR / "resources_servers/partner_if/data"
    assert built_in.harbor_dataset.path == PARENT_DIR / "benchmarks"
    # A benchmark plugin root (--search-dir) is searched first.
    plugin = tmp_path / "plugin"
    (plugin / "benchmarks").mkdir(parents=True)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(plugin))
    assert harbor_config(tmp_path, **relative).harbor_dataset.path == plugin / "benchmarks"
    assert harbor_config(tmp_path).harbor_dataset.path == tmp_path / "tasks"


def test_uvicorn_workers_import_a_module_level_app(monkeypatch):
    # With num_workers > 1, each uvicorn worker imports `responses_api_agents.partner_if_harbor_agent.app:app`.
    path, name = str(Path(__file__).parents[1] / "app.py"), "responses_api_agents.partner_if_harbor_agent.app"
    started = MagicMock(return_value=object())
    monkeypatch.setattr(SimpleServer, "run_webserver", started)
    for key in (IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME):
        monkeypatch.delenv(key, raising=False)
    assert "app" not in runpy.run_path(path, run_name=name)
    started.assert_not_called()
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME, "responses_api_agents/partner_if_harbor_agent/app.py")
    assert runpy.run_path(path, run_name=name)["app"] is started.return_value
    started.assert_called_once_with()
