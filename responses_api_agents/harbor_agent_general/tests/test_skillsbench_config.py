# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from nemo_gym.base_responses_api_model import CaptureStore, ModelCallCaptureConfig, install_model_call_capture
from nemo_gym.server_runtime.ray_support import entrypoint_ray_enabled
from responses_api_agents.harbor_agent_general import app, skillsbench_app
from responses_api_agents.harbor_agent_general.app import HarborRunRequest, temporary_worker_env
from responses_api_agents.harbor_agent_general.skillsbench_app import (
    SkillsBenchHarborAgent,
    SkillsBenchHarborAgentConfig,
)


ROOT = Path(__file__).resolve().parents[3]


def _config(opensandbox: bool, max_input_tokens: int = 131072) -> SkillsBenchHarborAgentConfig:
    configs = [OmegaConf.load(ROOT / "benchmarks/skillsbench/harbor.yaml")]
    if opensandbox:
        configs.append(OmegaConf.load(ROOT / "benchmarks/skillsbench/harbor_opensandbox.yaml"))
    merged = OmegaConf.merge(
        *configs, {"policy_model_name": "test-policy", "skillsbench_max_input_tokens": max_input_tokens}
    )
    raw = OmegaConf.to_container(
        merged.skillsbench_harbor_agent.responses_api_agents.harbor_agent_general, resolve=True
    )
    return SkillsBenchHarborAgentConfig.model_validate(
        raw | {"name": "skillsbench_harbor_agent", "host": "127.0.0.1", "port": 8080}
    )


@pytest.mark.parametrize("opensandbox", [False, True])
def test_profiles_route_model_and_keep_secrets_out_of_harbor_artifacts(tmp_path, monkeypatch, opensandbox):
    config = _config(opensandbox)
    global_config = OmegaConf.create(
        {
            "num_samples_in_parallel": 1,
            "model_call_capture_dir": str(tmp_path / "model-captures"),
            "policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.2", "port": 9010}}},
            "sandbox": {
                "default_metadata": {"owner": "synthetic-user"},
                "opensandbox": {"connection": {"domain": "sandbox.example", "api_key": "synthetic-secret"}},
            },
        }
    )
    monkeypatch.setattr(skillsbench_app, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr("responses_api_agents.harbor_agent_general.app.get_global_config_dict", lambda: global_config)
    agent = SkillsBenchHarborAgent.model_construct(config=config)
    job = agent.config.build_job_config("example-task", "t0-r0")
    assert job.agents[0].kwargs["api_base"] == "http://127.0.0.2:9010/v1"
    assert job.agents[0].model_name == "test-policy"
    assert job.datasets[0].task_names == ["example-task"]
    assert job.jobs_dir == tmp_path / "harbor-jobs"
    assert job.environment.delete
    assert "synthetic-secret" not in job.model_dump_json()
    if opensandbox:
        from benchmarks.skillsbench.environment import SkillsBenchSandboxEnvironment

        provider = json.loads(agent.config.harbor_worker_env["SKILLSBENCH_SANDBOX_CONFIG"])
        assert provider["opensandbox"]["connection"]["api_key"] == "synthetic-secret"
        assert provider["default_metadata"] == {"owner": "synthetic-user"}
        with temporary_worker_env(agent.config.harbor_worker_env):
            with patch("benchmarks.skillsbench.environment.NemoGymSandboxEnvironment.__init__") as parent:
                SkillsBenchSandboxEnvironment(skill_mode="with-skill")
        assert parent.call_args.kwargs["sandbox_provider"] == provider
    else:
        assert agent.config.harbor_worker_env == {}


def test_entrypoint_starts_ray_and_rows_follow_general_adapter_contract():
    assert entrypoint_ray_enabled(ROOT / "responses_api_agents/harbor_agent_general/skillsbench_app.py")
    row = HarborRunRequest.model_validate(
        {
            "task_name": "example-task",
            "task_index": 0,
            "rollout_index": 0,
            "responses_create_params": {"input": []},
        }
    )
    assert row.task_name == "example-task"


def test_flat_context_limit_override_reaches_native_agent_list():
    config = _config(opensandbox=True, max_input_tokens=65536)
    job = config.build_job_config("example-task", "t0-r0")
    assert job.agents[0].kwargs["model_info"]["max_input_tokens"] == 65536


def test_opensandbox_profile_inherits_shared_aiohttp_transport(monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "synthetic-key")
    merged = OmegaConf.merge(
        OmegaConf.load(ROOT / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"),
        OmegaConf.load(ROOT / "benchmarks/skillsbench/harbor.yaml"),
        OmegaConf.load(ROOT / "benchmarks/skillsbench/harbor_opensandbox.yaml"),
    )
    assert merged.sandbox.opensandbox.connection.transport_backend == "aiohttp"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_concurrent_dispatch_uses_per_row_model_capture_urls(tmp_path, monkeypatch, enabled):
    from urllib.parse import urlsplit

    from fastapi import FastAPI
    from starlette.testclient import TestClient

    global_config = {
        "num_samples_in_parallel": 4,
        "observability_enabled": enabled,
        "global_aiohttp_connector_limit_per_host": 17,
        "policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.2", "port": 9010}}},
    }
    monkeypatch.setattr(skillsbench_app, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: global_config)
    agent = SkillsBenchHarborAgent.model_construct(
        config=_config(False), server_client=SimpleNamespace(global_config_dict=global_config)
    )
    template = agent.config.job.model_dump(mode="json")
    submitted = []

    async def finish():
        await asyncio.sleep(0)
        return str(tmp_path)

    def remote(job, name, env, http_config):
        submitted.append(job)
        assert http_config["global_aiohttp_connector_limit_per_host"] == 17
        return asyncio.create_task(finish())

    monkeypatch.setattr(
        app, "harbor_job_worker", SimpleNamespace(options=Mock(return_value=SimpleNamespace(remote=remote)))
    )
    monkeypatch.setattr(SkillsBenchHarborAgent, "success_response", lambda self, body, trial_dir: body)
    identities = [
        {"_ng_task_index": 3, "_ng_rollout_index": 0},
        {"_ng_task_index": 4, "_ng_rollout_index": 1},
        {"_ng_task_index": 3, "_ng_rollout_index": 0, "_ng_attempt_index": 2},
        {"_ng_task_index": 3, "_ng_rollout_index": 0, "_ng_rollout_id": "explicit"},
    ]
    bodies = [
        HarborRunRequest.model_validate({"task_name": "example", "responses_create_params": {"input": []}, **identity})
        for identity in identities
    ]
    bodies[0].responses_create_params.temperature = 0
    bodies[0].responses_create_params.top_p = 1
    bodies[0].responses_create_params.max_output_tokens = 1024
    assert await asyncio.gather(*(agent.run(body) for body in bodies)) == bodies
    assert len(submitted) == 4
    assert len({job["job_name"] for job in submitted}) == 4
    assert agent.config.job.model_dump(mode="json") == template
    assert submitted[0]["agents"][0]["kwargs"]["responses_create_params"] == {
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 1024,
    }
    for job in submitted[1:]:
        kwargs = job["agents"][0]["kwargs"]
        assert kwargs["responses_create_params"] == {}
        assert kwargs["temperature"] == 1.0
        assert kwargs["top_p"] == 0.95
        assert kwargs["model_info"]["max_output_tokens"] == 32768

    model_app = FastAPI()

    @model_app.post("/v1/chat/completions")
    async def completions():
        return {"id": "completion", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    install_model_call_capture(
        model_app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path),
        model_server_name="policy_model",
    )
    with TestClient(model_app) as client:
        for job, capture_key in zip(submitted, ["3-0", "4-1", "3-0-a2", "explicit"], strict=True):
            url = job["agents"][0]["kwargs"]["api_base"]
            suffix = f"/ng-rollout/{capture_key}/v1" if enabled else "/v1"
            assert url == "http://127.0.0.2:9010" + suffix
            response = client.post(urlsplit(url).path + "/chat/completions", json={"messages": []})
            assert response.status_code == 200
            records = CaptureStore(tmp_path).read(capture_key)
            assert len(records) == (1 if enabled else 0)
