# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf
from ray.exceptions import RayTaskError

from nemo_gym.models import CaptureStore, ModelCallCaptureConfig, install_model_call_capture
from nemo_gym.server_runtime import entrypoint_ray_enabled
from responses_api_agents.harbor_agent_general import app, skillsbench_app
from responses_api_agents.harbor_agent_general.app import HarborRunRequest, temporary_worker_env
from responses_api_agents.harbor_agent_general.skillsbench_app import (
    SkillsBenchHarborAgent,
    SkillsBenchHarborAgentConfig,
)


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "exception_type",
    [
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "EnvironmentStartTimeoutError",
        "RuntimeError",
        "SkillsBenchSandboxError",
        "VerifierInfrastructureError",
        "AddTestsDirError",
        "DownloadVerifierDirError",
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_failure_classification_keeps_infrastructure_out_of_timeout_scores(monkeypatch, exception_type, wrapped):
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(SkillsBenchHarborAgent, "model_post_init", lambda self, context: None)
    agent = SkillsBenchHarborAgent.model_construct(config=_config(opensandbox=False))
    # The misleading text must not turn a transport failure into a task zero.
    error = app.HarborTrialError(exception_type, "AgentTimeoutError mentioned by provider", "/evidence/trial")
    error = pickle.loads(pickle.dumps(error))
    if wrapped:
        error = RayTaskError("worker", "traceback", error).as_instanceof_cause()
    body = HarborRunRequest(task_name="task", task_index=1, rollout_index=2, responses_create_params={"input": []})
    result = agent.failure_response(body, error).model_dump()
    deadline = exception_type == "AgentTimeoutError"
    assert result.get("_ng_failure_class") == (None if deadline else "skillsbench_infrastructure")
    assert bool(result.get("evaluation_completed")) == deadline
    assert not result.get("_ng_failure_terminal")
    assert result["evaluation_outcome"]["verification_completed"] is False
    assert result["evaluation_outcome"]["exception_type"] == exception_type
    assert result["evaluation_outcome"]["trial_dir"] == "/evidence/trial"
    assert (result["evaluation_outcome"].get("score_origin") == "timeout_policy") == deadline


def _config(opensandbox: bool, max_input_tokens: int = 131072, **overrides) -> SkillsBenchHarborAgentConfig:
    configs = [OmegaConf.load(ROOT / "benchmarks/skillsbench/harbor.yaml")]
    if opensandbox:
        configs.append(OmegaConf.load(ROOT / "benchmarks/skillsbench/harbor_opensandbox.yaml"))
    merged = OmegaConf.merge(
        *configs, {"policy_model_name": "test-policy", "skillsbench_max_input_tokens": max_input_tokens, **overrides}
    )
    raw = OmegaConf.to_container(
        merged.skillsbench_harbor_agent.responses_api_agents.harbor_agent_general, resolve=True
    )
    return SkillsBenchHarborAgentConfig.model_validate(
        raw | {"name": "skillsbench_harbor_agent", "host": "127.0.0.1", "port": 8080}
    )


def test_opensandbox_omits_unrelated_platform_s3_mount():
    config = _config(opensandbox=True)
    assert config.job.environment.kwargs["sandbox_metadata"]["nemo.nvidia.com/s3-hostpath"] == "false"


@pytest.mark.parametrize("failure", ["status: 429", "Attempted read from closed stream", "status code: 503"])
def test_verifier_transfer_failure_preserves_evidence_and_becomes_retryable(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(SkillsBenchHarborAgent, "model_post_init", lambda self, context: None)
    agent = SkillsBenchHarborAgent.model_construct(config=_config(opensandbox=False))
    body = HarborRunRequest(
        task_name="flink-query", task_index=32, rollout_index=0, responses_create_params={"input": []}
    )
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "tests": [
                        {
                            "name": "test_maven_build",
                            "status": "failed",
                            "trace": f"E         stdout: Could not transfer artifact example:jar:1 from/to central: {failure}",
                        }
                    ]
                }
            }
        )
    )
    original = '{"verifier_result": {"rewards": {"reward": 0.0}}}'
    (tmp_path / "result.json").write_text(original)
    with pytest.raises(app.HarborTrialError) as raised:
        agent.success_response(body, tmp_path)
    response = agent.failure_response(body, raised.value).model_dump()
    assert response["_ng_failure_class"] == "skillsbench_infrastructure"
    assert not response.get("_ng_failure_terminal")
    assert response["evaluation_outcome"]["exception_type"] == "VerifierInfrastructureError"
    assert not response["evaluation_outcome"]["verification_completed"]
    assert (tmp_path / "result.failed.json").read_text() == original
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "status,trace",
    [
        ("failed", "E AssertionError: expected 11 answers, got 4"),
        ("failed", "E Could not transfer artifact wrong:jar:1 from/to central: status: 404"),
        ("failed", "E Compilation failure: cannot find symbol"),
        ("passed", "E Could not transfer artifact example:jar:1 from/to central: status: 429"),
        ("failed", "print('Could not transfer artifact example:jar:1 from/to central: status: 429')"),
    ],
)
def test_verifier_guard_preserves_task_failures_and_ignores_unrelated_output(tmp_path, monkeypatch, status, trace):
    monkeypatch.setattr(SkillsBenchHarborAgent, "model_post_init", lambda self, context: None)
    agent = SkillsBenchHarborAgent.model_construct(config=_config(opensandbox=False))
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "tests": [
                        {
                            "name": "test_output",
                            "status": status,
                            "trace": trace,
                        }
                    ]
                }
            }
        )
    )
    body = HarborRunRequest(task_name="task", task_index=0, rollout_index=0, responses_create_params={"input": []})
    official = app.HarborVerifyResponse.model_construct(reward=0.0)
    (tmp_path / "result.json").write_text("{}")
    trial = SimpleNamespace(step_results=[], verifier_result=None)
    with (
        patch.object(app.HarborAgent, "success_response", return_value=official) as parent,
        patch.object(app.TrialResult, "model_validate_json", return_value=trial),
    ):
        assert agent.success_response(body, tmp_path).reward == 0.0
    parent.assert_called_once_with(body, tmp_path)


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


def test_long_generation_limits_reach_native_agent_list():
    config = _config(opensandbox=True, skillsbench_max_output_tokens=131072, skillsbench_llm_timeout_s=1800)
    kwargs = config.build_job_config("example-task", "t0-r0").agents[0].kwargs
    assert kwargs["model_info"] == {"max_input_tokens": 131072, "max_output_tokens": 131072}
    assert kwargs["llm_timeout_s"] == 1800
    defaults = _config(opensandbox=True).job.agents[0].kwargs
    assert defaults["model_info"]["max_output_tokens"] == 32768
    assert defaults["llm_timeout_s"] == 3600


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
    bodies[0].responses_create_params.metadata = {"extra_body": '{"seed": 0, "unrelated": "not forwarded"}'}
    assert await asyncio.gather(*(agent.run(body) for body in bodies)) == bodies
    assert len(submitted) == 4
    assert len({job["job_name"] for job in submitted}) == 4
    assert agent.config.job.model_dump(mode="json") == template
    assert submitted[0]["agents"][0]["kwargs"]["responses_create_params"] == {
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 1024,
        "seed": 0,
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


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "exception_type,failure_class",
    [
        ("SkillsBenchResourceUnattributedError", "skillsbench_resource_unattributed"),
    ],
)
def test_resource_failures_are_terminal_without_fabricating_verifier_reward(
    monkeypatch, wrapped, exception_type, failure_class
):
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(SkillsBenchHarborAgent, "model_post_init", lambda self, context: None)
    agent = SkillsBenchHarborAgent.model_construct(config=_config(opensandbox=False))
    error = pickle.loads(pickle.dumps(app.HarborTrialError(exception_type, "evidence", "/trial")))
    if wrapped:
        error = RayTaskError("worker", "traceback", error).as_instanceof_cause()
    body = HarborRunRequest(task_name="task", task_index=1, rollout_index=0, responses_create_params={"input": []})
    result = agent.failure_response(body, error).model_dump()
    assert result["_ng_failure_class"] == failure_class
    assert result["_ng_failure_terminal"]
    assert result["mask_sample"]
    assert not result["evaluation_outcome"]["verification_completed"]
    if failure_class == "skillsbench_resource_unattributed":
        assert result["evaluation_outcome"]["requires_review"]
        assert "score_origin" not in result["evaluation_outcome"]


def test_the_leaf_sets_the_llm_request_timeout_and_keeps_the_task_agent_budgets():
    config = OmegaConf.load(ROOT / "benchmarks/skillsbench/harbor.yaml")
    job = config.skillsbench_harbor_agent.responses_api_agents.harbor_agent_general.job
    kwargs = job.agents[0].kwargs
    assert (kwargs.llm_timeout_s, kwargs.llm_num_retries) == (3600, 5)
    # Each task's own [agent] timeout_sec is the benchmark's budget; Harbor scales it by this unchanged multiplier.
    assert job.agent_timeout_multiplier == 1.0 and "override_timeout_sec" not in job.agents[0]


def test_opensandbox_recovers_process_oom_without_changing_task_limits():
    config = _config(opensandbox=True)
    metadata = config.job.environment.kwargs["sandbox_metadata"]
    assert metadata["nemo.nvidia.com/oom-kill"] == "process"
    raw = OmegaConf.load(ROOT / "benchmarks/skillsbench/harbor_opensandbox.yaml")
    assert raw.sandbox.opensandbox.setup.commands == ['test "$(cat /sys/fs/cgroup/memory.oom.group)" = 0']
    assert not set(config.job.environment.kwargs) & {"cpus", "memory_mb", "storage_mb"}
