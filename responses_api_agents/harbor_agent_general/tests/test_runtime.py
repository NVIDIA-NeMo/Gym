import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from harbor.models.trial.config import AgentConfig, ArtifactConfig, VerifierConfig
from omegaconf import OmegaConf

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.harbor_agent_general.app import (
    ALERTED_OPENCODE_IMPORT_PATH,
    AUDITED_OPENCODE_IMPORT_PATH,
    FileAccessAuditConfig,
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
    VerifierFileAccessAuditConfig,
    _failure_class_for_error,
    _file_access_audit_metrics,
    _policy_agent_timed_out,
    _policy_alert_metrics,
    _sandbox_cleanup_failed,
    _verifier_file_access_audit_metrics,
)


def test_failure_class_preserves_sandbox_lifecycle_reset_through_wrapper():
    error = RuntimeError(
        "Harbor trial failed with OpenSandboxLifecycleResetError: OpenSandbox background command state disappeared"
    )

    assert _failure_class_for_error(error) == "sandbox_lifecycle_reset"
    assert (
        _failure_class_for_error(RuntimeError("Get command status failed: HTTP 502: backend is unreachable"))
        == "sandbox_backend_unreachable"
    )
    assert _failure_class_for_error(RuntimeError("judge failed")) == "harbor_failed"


def make_config(tmp_path: Path) -> HarborAgentConfig:
    return HarborAgentConfig(
        name="harbor_agent_general",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        token_id_capture=True,
        harbor_jobs_dir=tmp_path / "rollouts.jsonl",
        harbor_dataset={"path": tmp_path / "dataset"},
        harbor_agent={
            "name": "opencode",
            "model_name": "nemo/test-model",
            "env": {"EXISTING": "value"},
        },
        harbor_environment={"type": "docker"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_api_key="test-key",  # pragma: allowlist secret
    )


@pytest.mark.parametrize(
    ("exception_type", "expected"),
    [
        (None, False),
        ("AgentTimeoutError", True),
        ("VerifierTimeoutError", False),
    ],
)
def test_policy_agent_timed_out(exception_type: str | None, expected: bool) -> None:
    exception_info = None if exception_type is None else SimpleNamespace(exception_type=exception_type)
    trial = SimpleNamespace(step_results=[SimpleNamespace(exception_info=exception_info)])

    assert _policy_agent_timed_out(trial) is expected


def test_policy_timeout_accepts_absent_step_results() -> None:
    assert _policy_agent_timed_out(SimpleNamespace(step_results=None)) is False


def test_explicit_null_concurrency_uses_default(tmp_path: Path) -> None:
    with patch(
        "responses_api_agents.harbor_agent_general.app.get_global_config_dict",
        return_value={"num_samples_in_parallel": None},
    ):
        agent = HarborAgent.model_construct(config=make_config(tmp_path), server_client=MagicMock())
    assert not agent._sem.locked()


@pytest.mark.parametrize(
    ("exception_type", "message", "expected"),
    [
        (None, "", False),
        ("TimeoutError", "Timed out during OpenSandbox kill after 30s; sandbox_id='test'", True),
        ("TimeoutError", "another timeout", False),
        ("SandboxInternalException", "Timed out during OpenSandbox kill after 30s", False),
    ],
)
def test_sandbox_cleanup_failed(
    exception_type: str | None,
    message: str,
    expected: bool,
) -> None:
    exception_info = (
        None
        if exception_type is None
        else SimpleNamespace(
            exception_type=exception_type,
            exception_message=message,
        )
    )
    assert _sandbox_cleanup_failed(SimpleNamespace(exception_info=exception_info)) is expected


def test_jobs_dir_uses_sibling_harbor_directory_for_jsonl(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert config.harbor_jobs_dir == tmp_path.resolve() / "harbor"


def test_jobs_dir_defaults_to_ray_tmpdir(monkeypatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    monkeypatch.delenv("HARBOR_JOBS_DIR", raising=False)
    monkeypatch.setenv("RAY_TMPDIR", "/tmp/ray-test")

    config = OmegaConf.merge(
        {"policy_model_name": "test-model"},
        OmegaConf.load(gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent.yaml"),
    )

    assert config.harbor_agent_general.responses_api_agents.harbor_agent_general.harbor_jobs_dir == (
        "/tmp/ray-test/harbor_agent_general_jobs"
    )
    assert config.harbor_agent_general.responses_api_agents.harbor_agent_general.harbor_agent.model_name.startswith(
        "nemo/"
    )
    assert config.harbor_agent_general.responses_api_agents.harbor_agent_general.context_window == 131072
    assert config.policy_model.responses_api_models.vllm_model.reasoning_response_field == "reasoning_content"


def test_agent_for_model_server_injects_route_without_mutating_config(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    agent = config.agent_for_model_server(
        "http://model/ng-rollout/t0-r1/v1",
        auxiliary_base_url="http://model/v1",
    )

    assert agent.env == {
        "EXISTING": "value",
        "OPENAI_BASE_URL": "http://model/ng-rollout/t0-r1/v1",
        "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
    }
    assert agent.kwargs == {
        "opencode_config": {
            "provider": {
                "nemo": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": "http://model/ng-rollout/t0-r1/v1",
                        "apiKey": "EMPTY",  # pragma: allowlist secret
                    },
                    "models": {
                        "test-model": {
                            "name": "test-model",
                            "reasoning": True,
                            "tool_call": True,
                            "interleaved": {"field": "reasoning"},
                            "limit": {"context": 262144, "output": 131072},
                        }
                    },
                },
                "nemo-auxiliary": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": "http://model/v1",
                        "apiKey": "EMPTY",  # pragma: allowlist secret
                    },
                    "models": {"test-model": {"name": "test-model"}},
                },
            },
            "small_model": "nemo-auxiliary/test-model",
        }
    }
    assert config.harbor_agent.env == {"EXISTING": "value"}
    assert config.harbor_agent.kwargs == {}


def test_policy_alert_deadline_also_sets_harbor_timeout(tmp_path: Path) -> None:
    raw_config = make_config(tmp_path).model_dump()
    raw_config["harbor_agent"] = {
        "name": None,
        "import_path": ALERTED_OPENCODE_IMPORT_PATH,
        "model_name": "nemo/test-model",
    }
    raw_config["policy_alerts"] = {
        "deadline_seconds": 3600,
        "alerts": [
            {"remaining_seconds": 600, "message": "ten minutes left"},
            {"remaining_seconds": 300, "message": "five minutes left"},
        ],
    }
    config = HarborAgentConfig.model_validate(raw_config)

    agent = config.agent_for_model_server("http://model/v1")

    assert agent.override_timeout_sec == 3600
    assert agent.kwargs["alert_schedule"] == config.policy_alerts.model_dump(mode="json")


def test_config_rejects_unknown_sandbox_template_before_rollout_dispatch(tmp_path: Path) -> None:
    raw_config = make_config(tmp_path).model_dump()
    raw_config["harbor_environment"]["kwargs"] = {
        "sandbox_provider_options": {
            "volumes": [
                {
                    "name": "problem-data",
                    "host": {"path": "/mnt/s3/data/train/{task_nmae}"},
                    "mountPath": "/app/data",
                }
            ]
        }
    }

    with pytest.raises(ValueError, match=r"environment.kwargs.sandbox_provider_options.*\{task_nmae\}"):
        HarborAgentConfig.model_validate(raw_config)


def test_policy_alert_metrics_summarize_agent_context() -> None:
    trial = SimpleNamespace(
        agent_result=SimpleNamespace(
            metadata={
                "runtime_alerts": [
                    {"delivered": True, "attempts": 1},
                    {"delivered": False, "attempts": 2},
                ]
            }
        ),
        step_results=None,
    )

    assert _policy_alert_metrics(trial) == {
        "policy_alert_count": 2,
        "policy_alert_attempted_count": 2,
        "policy_alert_delivered_count": 1,
        "policy_alert_delivery_rate": 0.5,
        "policy_alert_all_delivered": False,
    }


def test_policy_alert_metrics_distinguish_capture_from_zero_token_length() -> None:
    trial = SimpleNamespace(
        agent_result=SimpleNamespace(
            metadata={
                "runtime_alerts": [
                    {
                        "delivered": True,
                        "attempts": 1,
                        "session_user_turn_recorded": True,
                        "session_assistant_turn_recorded": True,
                        "session_alert_processed": False,
                        "session_zero_token_length": True,
                        "session_compaction_completed": False,
                    }
                ]
            }
        ),
        step_results=None,
    )

    metrics = _policy_alert_metrics(trial)

    assert metrics["policy_alert_session_recorded_count"] == 1
    assert metrics["policy_alert_session_responded_count"] == 0
    assert metrics["policy_alert_zero_token_length_count"] == 1
    assert metrics["policy_alert_compaction_completed_count"] == 0


def test_audited_opencode_uses_gym_model_server_provider(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.harbor_agent = AgentConfig(
        name=None,
        import_path=AUDITED_OPENCODE_IMPORT_PATH,
        model_name="nemo/test-model",
        env={},
    )

    agent = config.agent_for_model_server("http://model/ng-rollout/t0-r1/v1")

    assert agent.import_path == AUDITED_OPENCODE_IMPORT_PATH
    assert agent.kwargs["opencode_config"]["provider"]["nemo"]["options"]["baseURL"] == (
        "http://model/ng-rollout/t0-r1/v1"
    )


def test_file_access_audit_counts_only_honeypot_events(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    trajectory_path = agent_dir / "trajectory.json"
    trajectory_path.write_text("{}")
    (agent_dir / ".policy-fs.trace").write_text(
        'openat(AT_FDCWD</app>, "/app/data/input.csv", O_RDONLY) = 3</app/data/input.csv>\n'
        'openat(AT_FDCWD</app>, "/mnt/s3-database/not-the-honeypot", O_RDONLY) = 3\n'
        'openat(AT_FDCWD</app>, "/mnt/s3-data/private/tests", O_RDONLY) = 3\n'
        'chdir("/mnt/s3-data") = 0\n'
        'openat(AT_FDCWD</app>, "/mnt/efs-data/other-rollout", O_RDONLY) = 3\n'
        'newfstatat(3</mnt>, "s3-data/relative", 0x0, 0) = -1 ENOENT\n'
        'newfstatat(3</mnt>, "s3-database/relative", 0x0, 0) = -1 ENOENT\n'
    )

    metrics = _file_access_audit_metrics(
        [trajectory_path],
        FileAccessAuditConfig(
            honeypot_path="/mnt/s3-data",
            additional_honeypot_paths=["/mnt/efs-data"],
        ),
    )

    assert metrics == {
        "policy_honeypot_accessed": True,
        "policy_honeypot_access_event_count": 4,
        "policy_honeypot_read_accessed": True,
        "policy_honeypot_read_access_event_count": 4,
        "policy_honeypot_write_accessed": False,
        "policy_honeypot_write_access_event_count": 0,
    }


def test_verifier_file_access_audit_separates_local_and_efs_reads_and_writes(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "steps" / "rollout" / "agent" / "trajectory.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text("{}")
    judge_dir = trajectory_path.parent.parent / "verifier" / "judge"
    judge_dir.mkdir(parents=True)
    (judge_dir / ".judge-fs.trace").write_text(
        'openat(AT_FDCWD</judge>, "/app/REPORT.md", O_RDONLY) = 3</app/REPORT.md>\n'
        'openat(AT_FDCWD</judge>, "/app/REPORT.md", O_WRONLY|O_TRUNC) = 3</app/REPORT.md>\n'
        'newfstatat(3</mnt>, "efs-data/other-rollout", 0x0, 0) = 0\n'
        'unlinkat(3</mnt/efs-data>, "other-rollout", AT_REMOVEDIR) = 0\n'
    )

    metrics = _verifier_file_access_audit_metrics(
        [trajectory_path],
        VerifierFileAccessAuditConfig(audited_paths={"policy_artifacts": "/app", "efs": "/mnt/efs-data"}),
    )

    assert metrics == {
        "verifier_policy_artifacts_read_accessed": True,
        "verifier_policy_artifacts_read_access_event_count": 1,
        "verifier_policy_artifacts_write_accessed": True,
        "verifier_policy_artifacts_write_access_event_count": 1,
        "verifier_efs_read_accessed": True,
        "verifier_efs_read_access_event_count": 1,
        "verifier_efs_write_accessed": True,
        "verifier_efs_write_access_event_count": 1,
    }


def test_file_access_audit_fails_when_trace_is_missing(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "agent" / "trajectory.json"
    trajectory_path.parent.mkdir()
    trajectory_path.write_text("{}")

    with pytest.raises(FileNotFoundError, match="audit trace is missing"):
        _file_access_audit_metrics(
            [trajectory_path],
            FileAccessAuditConfig(honeypot_path="/mnt/s3-data"),
        )


def test_build_job_config_scopes_task_and_forces_environment_cleanup(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.environment_build_timeout_multiplier = 3.0
    config.harbor_environment.kwargs = {
        "sandbox_provider": {"opensandbox": {"connection": {"api_key": "resolved-secret"}}}
    }
    config.harbor_verifier = VerifierConfig(
        import_path=("responses_api_agents.harbor_agent_general.agentic_verifier:AgenticVerifier"),
        kwargs={"config": {"judge_agent": {"name": "nop"}}},
    )
    config.artifacts = [ArtifactConfig(source="/app", exclude=["data"])]
    agent = AgentConfig(name="opencode", model_name="nemo/test-model")

    job = config.build_job_config("bbh-task", "t0-r1", agent)

    assert job.job_name == "t0-r1"
    assert job.jobs_dir == tmp_path.resolve() / "harbor"
    assert job.datasets[0].task_names == ["bbh-task"]
    assert job.agents == [agent]
    assert job.environment.delete is True
    assert job.environment.kwargs["sandbox_provider"]["opensandbox"]["connection"]["api_key"] == (
        "${OPENSANDBOX_API_KEY}"
    )
    assert config.harbor_environment.kwargs["sandbox_provider"]["opensandbox"]["connection"]["api_key"] == (
        "resolved-secret"
    )
    assert job.verifier == config.harbor_verifier
    assert job.artifacts == config.artifacts
    assert job.environment_build_timeout_multiplier == 3.0
    assert job.n_attempts == 1
    assert job.n_concurrent_trials == 1


def test_nrl_rollout_id_routes_to_prefixed_gym_model_url(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    server_client = MagicMock()
    server_client.global_config_dict = {
        "token_id_capture": {"enabled": True},
        "policy_model": {"responses_api_models": {"vllm_model": {"host": "model-host", "port": 9000}}},
    }
    server_client._build_server_base_url.return_value = "http://model-host:9000"
    with patch(
        "responses_api_agents.harbor_agent_general.app.get_global_config_dict",
        return_value={},
    ):
        agent = HarborAgent.model_construct(config=config, server_client=server_client)
    body = HarborRunRequest(
        task_name="bbh-task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        **{
            "_ng_task_index": 3,
            "_ng_rollout_index": 1,
            "_ng_rollout_id": "nrl-step7-sample3",
        },
    )

    rollout_id = agent.rollout_id_from_run(body)
    base_url = agent.resolve_model_base_url(config.model_server.name, rollout_id)

    assert rollout_id == "nrl-step7-sample3"
    assert base_url == "http://model-host:9000/ng-rollout/nrl-step7-sample3/training-token-capture/v1"


@patch("responses_api_agents.harbor_agent_general.app.harbor_job_worker")
@pytest.mark.asyncio
async def test_run_scopes_harbor_job_dir_to_rollout_id(harbor_job_worker, tmp_path: Path) -> None:
    config = make_config(tmp_path)
    server_client = MagicMock()
    server_client.global_config_dict = {
        "token_id_capture": {"enabled": True},
        "policy_model": {"responses_api_models": {"vllm_model": {"host": "model-host", "port": 9000}}},
    }
    server_client._build_server_base_url.return_value = "http://model-host:9000"
    with patch(
        "responses_api_agents.harbor_agent_general.app.get_global_config_dict",
        return_value={},
    ):
        agent = HarborAgent.model_construct(config=config, server_client=server_client)
    body = HarborRunRequest(
        task_name="bbh-task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        **{
            "_ng_task_index": 3,
            "_ng_rollout_index": 1,
            "_ng_rollout_id": "nrl-step7-sample3",
        },
    )
    worker = harbor_job_worker.options.return_value
    worker.remote.side_effect = RuntimeError("stop after config capture")

    response = await agent.run(body)

    harbor_job_worker.options.assert_called_once_with(num_cpus=0.25)
    job_config = worker.remote.call_args.args[0]
    assert job_config["job_name"] == "t3-r1-nrl-step7-sample3"
    assert job_config["agents"][0]["kwargs"]["opencode_config"]["small_model"] == ("nemo-auxiliary/test-model")
    assert (
        job_config["agents"][0]["kwargs"]["opencode_config"]["provider"]["nemo-auxiliary"]["options"]["baseURL"]
        == "http://model-host:9000/v1"
    )
    assert response.reward == 0.0


def test_explicit_opencode_small_model_is_preserved(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.harbor_agent.kwargs = {"opencode_config": {"small_model": "custom/title-model"}}

    agent = config.agent_for_model_server(
        "http://model/ng-rollout/t0-r1/v1",
        auxiliary_base_url="http://model/v1",
    )

    opencode_config = agent.kwargs["opencode_config"]
    assert opencode_config["small_model"] == "custom/title-model"
    assert "nemo-auxiliary" not in opencode_config["provider"]


def test_opensandbox_config_separates_requests_from_limits(monkeypatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    dataset_path = "/datasets/bbh-harbor-rl-v0/val"
    jobs_dir = "/shared/harbor/jobs"
    monkeypatch.setenv("HARBOR_DATASET_PATH", dataset_path)
    monkeypatch.setenv("HARBOR_JOBS_DIR", jobs_dir)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-key")
    monkeypatch.setenv("OPENSANDBOX_PROTOCOL", "https")
    monkeypatch.setenv("OPENSANDBOX_USE_SERVER_PROXY", "false")
    monkeypatch.setenv("OPENSANDBOX_CA_BUNDLE", "/certs/opensandbox.pem")
    monkeypatch.setenv("HARBOR_IMAGE_OVERRIDE", "docker.io/example/task@sha256:1234")
    monkeypatch.setenv(
        "HARBOR_SANDBOX_ENTRYPOINT",
        '["sh", "/opt/entrypoint.sh", "tail", "-f", "/dev/null"]',
    )
    monkeypatch.setenv(
        "HARBOR_SANDBOX_PROBE_COMMAND",
        "python -c \"import bbh_mcp.server; print('nemo-gym-sandbox-ready')\"",
    )
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_EXPECTED_STDOUT", "nemo-gym-sandbox-ready")
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_TIMEOUT_S", "180")
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_DEADLINE_S", "240")
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_STABLE_COUNT", "1")
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_STABLE_DELAY_S", "0")
    monkeypatch.setenv("HARBOR_ENABLE_EFS", "true")
    monkeypatch.setenv(
        "HARBOR_SANDBOX_VOLUMES",
        (
            '[{"name":"problem-data","host":{"path":"/mnt/s3/data/train/'
            '{environment_name}/environment/data"},"mountPath":"/data",'
            '"readOnly":true}]'
        ),
    )
    monkeypatch.setenv(
        "HARBOR_SANDBOX_PATH_COPIES",
        '[{"source":"/mnt/task-data","destination":"/app/copied-data"}]',
    )
    monkeypatch.setenv(
        "HARBOR_SANDBOX_PATH_SYMLINKS",
        '[{"source":"/data","destination":"/app/data"}]',
    )

    config = OmegaConf.merge(
        OmegaConf.load(gym_root / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"),
        OmegaConf.load(gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent.yaml"),
        OmegaConf.load(gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent_opensandbox.yaml"),
    )
    agent_config = config.harbor_agent_general.responses_api_agents.harbor_agent_general
    environment = OmegaConf.to_container(agent_config.harbor_environment, resolve=True)

    assert agent_config.harbor_jobs_dir == jobs_dir
    assert OmegaConf.to_container(agent_config.harbor_dataset, resolve=True)["path"] == dataset_path
    assert environment["import_path"].endswith(":NemoGymSandboxEnvironment")
    assert environment["override_cpus"] == 4
    assert environment["override_memory_mb"] == 65536
    assert agent_config.environment_build_timeout_multiplier == 3.0
    assert environment["kwargs"]["sandbox_provider_options"] == {
        "resource_requests": {"cpu": 0.25, "memory_mib": 512},
        "volumes": [
            {
                "name": "problem-data",
                "host": {"path": ("/mnt/s3/data/train/{environment_name}/environment/data")},
                "mountPath": "/data",
                "readOnly": True,
            }
        ],
    }
    assert environment["kwargs"]["sandbox_provider"]["opensandbox"]["connection"]["api_key"] == "test-key"
    assert environment["kwargs"]["sandbox_provider"]["opensandbox"]["connection"]["protocol"] == "https"
    assert environment["kwargs"]["sandbox_provider"]["opensandbox"]["connection"]["use_server_proxy"] is False
    assert (
        environment["kwargs"]["sandbox_provider"]["opensandbox"]["connection"]["ca_bundle_path"]
        == "/certs/opensandbox.pem"
    )
    assert environment["kwargs"]["image_override"] == "docker.io/example/task@sha256:1234"
    assert environment["kwargs"]["entrypoint"] == [
        "sh",
        "/opt/entrypoint.sh",
        "tail",
        "-f",
        "/dev/null",
    ]
    assert environment["kwargs"]["sandbox_provider"]["opensandbox"]["probe"] == {
        "command": "python -c \"import bbh_mcp.server; print('nemo-gym-sandbox-ready')\"",
        "expected_stdout": "nemo-gym-sandbox-ready",
        "timeout_s": 180,
        "deadline_s": 240,
        "stable_count": 1,
        "stable_delay_s": 0,
    }
    operations = environment["kwargs"]["sandbox_provider"]["opensandbox"]["operations"]
    assert operations["background_exec"] is True
    assert operations["background_poll_interval_s"] == 10.0
    assert operations["background_poll_initial_s"] == 0.5
    assert operations["status_poll_timeout_s"] == 30
    assert operations["background_request_retries"] == 0
    assert operations["command_retries"] == 0
    assert environment["kwargs"]["exec_shell"] == "bash -lc"
    assert environment["kwargs"]["sandbox_path_copies"] == [
        {"source": "/mnt/task-data", "destination": "/app/copied-data"}
    ]
    assert environment["kwargs"]["sandbox_path_symlinks"] == [{"source": "/data", "destination": "/app/data"}]
    assert environment["kwargs"]["sandbox_metadata"] == {
        "harbor-benchmark": "harbor",
        "nemo-gym.nvidia.com/user": "unknown",
        "nemo-gym.nvidia.com/run": "unscoped",
        "nemo.nvidia.com/efs-hostpath": "false",
        "nemo.nvidia.com/s3-hostpath": "false",
    }


def test_honeypot_audit_config_is_independent_of_task_data_transport(monkeypatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("HARBOR_DATASET_PATH", "/datasets/test")
    monkeypatch.setenv("HARBOR_HONEYPOT_PATH", "/mnt/s3-data")
    monkeypatch.setenv("HARBOR_ADDITIONAL_HONEYPOT_PATHS", "[/mnt/efs-data]")

    config = OmegaConf.merge(
        {"policy_model_name": "policy-model", "policy_api_key": "policy-key"},
        OmegaConf.load(gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent.yaml"),
        OmegaConf.load(
            gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent_honeypot_audit.yaml"
        ),
    )
    agent_config = OmegaConf.to_container(
        config.harbor_agent_general.responses_api_agents.harbor_agent_general,
        resolve=True,
    )

    assert agent_config["harbor_agent"] == {
        "name": None,
        "model_name": "nemo/policy-model",
        "env": {},
        "import_path": AUDITED_OPENCODE_IMPORT_PATH,
        "kwargs": {"use_preinstalled": True},
    }
    assert agent_config["file_access_audit"] == {
        "honeypot_path": "/mnt/s3-data",
        "additional_honeypot_paths": ["/mnt/efs-data"],
    }


def test_policy_alert_and_honeypot_overlays_compose(monkeypatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("HARBOR_DATASET_PATH", "/datasets/test")
    monkeypatch.setenv("HARBOR_HONEYPOT_PATH", "/mnt/s3-data")
    monkeypatch.setenv("HARBOR_POLICY_TIMEOUT_S", "3600")

    config = OmegaConf.merge(
        {"policy_model_name": "policy-model", "policy_api_key": "policy-key"},
        OmegaConf.load(gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent.yaml"),
        OmegaConf.load(gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent_policy_alerts.yaml"),
        OmegaConf.load(
            gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent_honeypot_audit.yaml"
        ),
    )
    agent_config = OmegaConf.to_container(
        config.harbor_agent_general.responses_api_agents.harbor_agent_general,
        resolve=True,
    )

    assert agent_config["harbor_agent"]["import_path"] == AUDITED_OPENCODE_IMPORT_PATH
    assert agent_config["policy_alerts"]["deadline_seconds"] == 3600
    assert [alert["remaining_seconds"] for alert in agent_config["policy_alerts"]["alerts"]] == [600, 300]


def test_policy_alert_deadline_cannot_exceed_sandbox_exec_timeout(tmp_path: Path) -> None:
    raw_config = make_config(tmp_path).model_dump(mode="python")
    raw_config["harbor_agent"] = {
        "name": None,
        "import_path": ALERTED_OPENCODE_IMPORT_PATH,
        "model_name": "nemo/test-model",
    }
    raw_config["harbor_environment"]["kwargs"] = {"default_exec_timeout_s": 3600}
    raw_config["policy_alerts"] = {
        "deadline_seconds": 6000,
        "alerts": [{"remaining_seconds": 300, "message": "Finish now."}],
    }

    with pytest.raises(
        ValueError,
        match="policy_alerts.deadline_seconds cannot exceed environment.kwargs.default_exec_timeout_s",
    ):
        HarborAgentConfig.model_validate(raw_config)


def test_efs_direct_shared_workspace_config(monkeypatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv(
        "HARBOR_EFS_ARTIFACT_SOURCE_PATH",
        "/mnt/efs/data/shared/example-user/test-run/{context_id}",
    )
    monkeypatch.setenv("HARBOR_EFS_ARTIFACT_TIMEOUT_S", "321")
    monkeypatch.setenv("HARBOR_EFS_CLEANUP_TIMEOUT_S", "654")
    monkeypatch.setenv("HARBOR_EFS_CLEANUP_TTL_S", "987")
    config = OmegaConf.merge(
        OmegaConf.load(gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent_opensandbox.yaml"),
        OmegaConf.load(
            gym_root / "responses_api_agents/harbor_agent_general/configs/harbor_agent_efs_artifact_transfer.yaml"
        ),
    )
    environment = config.harbor_agent_general.responses_api_agents.harbor_agent_general.harbor_environment

    assert OmegaConf.to_container(environment.kwargs.shared_workspace, resolve=True) == {
        "volume": {
            "name": "harbor-policy-workspace",
            "host": {"path": "/mnt/efs/data/shared/example-user/test-run/{context_id}"},
            "mountPath": "/app",
        },
        "handoff_timeout_s": 321,
        "cleanup_timeout_s": 654,
        "cleanup_ttl_s": 987,
    }


@pytest.mark.asyncio
async def test_run_job_recovers_completed_trial_when_only_cleanup_times_out(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    job_config = config.build_job_config(
        "bbh-task",
        "t0-r0",
        AgentConfig(name="opencode", model_name="nemo/test-model"),
    )
    trial_dir = job_config.jobs_dir / job_config.job_name / "trial"
    trajectory_path = trial_dir / "steps" / "rollout" / "agent" / "trajectory.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text("{}")
    (trial_dir / "result.json").write_text("{}")
    trial_result = SimpleNamespace(
        task_name="bbh-task",
        exception_info=SimpleNamespace(
            exception_type="TimeoutError",
            exception_message=("Timed out during OpenSandbox kill after 30s; sandbox_id='test'"),
        ),
        verifier_result=SimpleNamespace(rewards={"reward": 0.5}),
        step_results=[SimpleNamespace(step_name="rollout", exception_info=None)],
    )
    job = SimpleNamespace(run=AsyncMock())

    with (
        patch(
            "responses_api_agents.harbor_agent_general.app.Job.create",
            new=AsyncMock(return_value=job),
        ),
        patch(
            "responses_api_agents.harbor_agent_general.app.TrialResult.model_validate_json",
            return_value=trial_result,
        ),
    ):
        recovered = await HarborAgent.run_job(job_config.model_dump(mode="json"), "bbh-task")

    assert recovered == str(trial_dir.resolve())
    assert (trial_dir / "result.json").is_file()


@pytest.mark.asyncio
async def test_run_job_rejects_opensandbox_infra_failure_recorded_on_step(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    job_config = config.build_job_config(
        "bbh-task",
        "t0-r0",
        AgentConfig(name="opencode", model_name="nemo/test-model"),
    )
    trial_dir = job_config.jobs_dir / job_config.job_name / "trial"
    trial_dir.mkdir(parents=True)
    result_path = trial_dir / "result.json"
    result_path.write_text("{}")
    trial_result = SimpleNamespace(
        task_name="bbh-task",
        exception_info=None,
        verifier_result=SimpleNamespace(rewards={"reward": 0.0}),
        step_results=[
            SimpleNamespace(
                step_name="rollout",
                exception_info=SimpleNamespace(
                    exception_type="SandboxApiException",
                    exception_message=(
                        "Get command status failed: HTTP 502: Could not connect to backend sandbox endpoint"
                    ),
                ),
            )
        ],
    )
    job = SimpleNamespace(run=AsyncMock())

    with (
        patch(
            "responses_api_agents.harbor_agent_general.app.Job.create",
            new=AsyncMock(return_value=job),
        ),
        patch(
            "responses_api_agents.harbor_agent_general.app.TrialResult.model_validate_json",
            return_value=trial_result,
        ),
        pytest.raises(RuntimeError, match="Get command status failed: HTTP 502"),
    ):
        await HarborAgent.run_job(job_config.model_dump(mode="json"), "bbh-task")

    assert not result_path.exists()


@pytest.mark.asyncio
async def test_run_job_does_not_recover_another_tasks_result(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    job_config = config.build_job_config("expected-task", "t0-r0", config.harbor_agent)
    trial_dir = job_config.jobs_dir / job_config.job_name / "old-trial"
    trial_dir.mkdir(parents=True)
    result_path = trial_dir / "result.json"
    result_path.write_text("{}")
    with (
        patch(
            "responses_api_agents.harbor_agent_general.app.Job.create",
            new=AsyncMock(side_effect=RuntimeError("job config changed")),
        ),
        patch(
            "responses_api_agents.harbor_agent_general.app.TrialResult.model_validate_json",
            return_value=SimpleNamespace(task_name="other-task"),
        ),
        pytest.raises(RuntimeError, match="job config changed"),
    ):
        await HarborAgent.run_job(job_config.model_dump(mode="json"), "expected-task")
    assert result_path.is_file()


@pytest.mark.asyncio
async def test_run_cancellation_cancels_its_ray_worker(tmp_path: Path) -> None:
    with patch("responses_api_agents.harbor_agent_general.app.get_global_config_dict", return_value={}):
        agent = HarborAgent.model_construct(config=make_config(tmp_path), server_client=MagicMock())
    body = HarborRunRequest(
        task_name="bbh-task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        **{"_ng_task_index": 0, "_ng_rollout_index": 0},
    )
    job_ref = asyncio.get_running_loop().create_future()
    job_ref.cancel()
    with (
        patch.object(HarborAgent, "rollout_id_from_run", return_value=None),
        patch.object(HarborAgent, "resolve_model_base_url", return_value="http://model/v1"),
        patch("responses_api_agents.harbor_agent_general.app.harbor_job_worker") as worker,
        patch("responses_api_agents.harbor_agent_general.app.ray.cancel") as cancel,
        pytest.raises(asyncio.CancelledError),
    ):
        worker.options.return_value.remote.return_value = job_ref
        await agent.run(body)
    cancel.assert_called_once_with(job_ref, force=True)


@pytest.mark.parametrize("multistep", [False, True])
def test_success_uses_upstream_atif_conversion_and_explicit_reward_key(tmp_path: Path, multistep: bool) -> None:
    config = make_config(tmp_path)
    config.harbor_reward_key = "accuracy"
    with patch("responses_api_agents.harbor_agent_general.app.get_global_config_dict", return_value={}):
        agent = HarborAgent.model_construct(config=config, server_client=MagicMock())
    body = HarborRunRequest(
        task_name="bbh-task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        **{"_ng_task_index": 0, "_ng_rollout_index": 0},
    )
    trial_dir = tmp_path / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text("{}")
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "agent": {"name": "test-agent", "version": "1"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": "question"},
                    {"step_id": 2, "source": "agent", "message": "answer"},
                    {"step_id": 3, "source": "user", "message": "follow-up"},
                ],
            }
        )
    )
    trial = SimpleNamespace(
        id="trial",
        finished_at=None,
        step_results=None,
        agent_result=None,
        exception_info=None,
        verifier_result=SimpleNamespace(rewards={"accuracy": 0.75, "cost": 10}),
        compute_token_cost_totals=lambda: (1, 0, 1, 0.0),
    )
    if multistep:
        trial.step_results = []
        for name, reward in [("first", 0.5), ("second", 1.0)]:
            step_agent_dir = trial_dir / "steps" / name / "agent"
            step_agent_dir.mkdir(parents=True)
            (step_agent_dir / "trajectory.json").write_text((trial_dir / "agent" / "trajectory.json").read_text())
            trial.step_results.append(
                SimpleNamespace(
                    step_name=name,
                    exception_info=None,
                    agent_result=None,
                    verifier_result=SimpleNamespace(rewards={"accuracy": reward}),
                )
            )
    with patch("responses_api_agents.harbor_agent_general.app.TrialResult.model_validate_json", return_value=trial):
        result = agent.success_response(body, trial_dir)
        assert result.reward == 0.75
        assert result.responses_create_params.input[0].content == "question"
        assert result.response.output[-1].role == "user"
        assert result.response.output[-1].content == "follow-up"
        assert len(result.response.output) == (5 if multistep else 2)
        config.harbor_reward_key = "missing"
        with pytest.raises(ValueError, match="no reward key"):
            agent.success_response(body, trial_dir)
