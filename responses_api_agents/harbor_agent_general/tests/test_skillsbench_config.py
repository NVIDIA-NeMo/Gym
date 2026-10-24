# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_gym.server_utils import entrypoint_ray_enabled
from responses_api_agents.harbor_agent_general import skillsbench_app
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
    merged = OmegaConf.merge(*configs, {"policy_model_name": "test-policy", "skillsbench_max_input_tokens": max_input_tokens})
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
