import logging
from asyncio import Semaphore
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist
from pier.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from pier.models.trial.paths import TrialPaths

from responses_api_agents.deep_swe_agent.app import (
    DeepSWEAgent,
    DeepSWEAgentConfig,
    _ensure_pier_litellm_compat,
    _provider_without_secret,
    run_pier_job,
)
from responses_api_agents.deep_swe_agent.opensandbox_environment import (
    INSTALL_EGRESS_TARGETS,
    PierOpenSandboxEnvironment,
)


def _agent(tmp_path: Path) -> DeepSWEAgent:
    config = DeepSWEAgentConfig(
        name="deep_swe_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        description="DeepSWE through Pier",
        value="Repository-scale software engineering",
        datasets=[],
        concurrency=1,
        model_server={"type": "responses_api_models", "name": "policy_model"},
        harbor_datasets={"deep_swe": {"local_dataset_path": str(tmp_path / "tasks"), "workdir": "/app"}},
        harbor_agent_name="mini-swe-agent",
        harbor_agent_kwargs={
            "version": "2.4.6",
            "reasoning_effort": "xhigh",
            "cost_limit": 0,
            "model_class": "litellm_response",
        },
        harbor_agent_env={"OPENAI_BASE_URL": "https://inference-api.nvidia.com/v1"},
        harbor_jobs_dir=str(tmp_path / "jobs"),
        sandbox_provider={"opensandbox": {"connection": {"domain": "sandbox", "api_key": "secret"}}},
        sandbox_spec={"provider_options": {"platform": {"os": "linux", "arch": "amd64"}}},
    )
    return DeepSWEAgent.model_construct(
        config=config,
        server_client=MagicMock(),
        sem=Semaphore(1),
    )


def test_builds_single_task_pier_job_without_serializing_api_key(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    config = agent._build_job_config(
        dataset_alias="deep_swe",
        task_name="abs-module-cache-flags",
        model_name="model",
        api_base="http://policy:8000/v1",
        job_name="job-1",
        jobs_dir=tmp_path / "jobs",
        responses_create_params={"temperature": 0.2},
    )

    assert config["datasets"][0]["task_names"] == ["abs-module-cache-flags"]
    assert config["environment"]["import_path"].endswith(":PierOpenSandboxEnvironment")
    assert "api_key" not in config["environment"]["kwargs"]["provider"]["opensandbox"]["connection"]
    mini = config["agents"][0]
    assert mini["name"] == "mini-swe-agent"
    assert mini["import_path"] is None
    assert mini["model_name"] == "model"
    assert mini["kwargs"] == agent.config.harbor_agent_kwargs
    assert "api_base" not in mini["kwargs"]
    assert "responses_create_params" not in mini["kwargs"]
    assert mini["env"] == {
        "OPENAI_API_KEY": "${POLICY_API_KEY}",
        "OPENAI_BASE_URL": "https://inference-api.nvidia.com/v1",
    }


def test_opensandbox_stream_timeout_exceeds_deep_swe_agent_timeout() -> None:
    config_path = Path(__file__).parents[1] / "configs/deep_swe_opensandbox.yaml"
    config = yaml.safe_load(config_path.read_text())
    agent_config = config["deep_swe_agent"]["responses_api_agents"]["deep_swe_agent"]
    connection = agent_config["sandbox_provider"]["opensandbox"]["connection"]

    assert connection["request_timeout_s"] > 5400
    assert agent_config["harbor_agent_kwargs"]["cost_limit"] == 0


def test_provider_secret_is_returned_separately() -> None:
    sanitized, secret = _provider_without_secret(
        {"opensandbox": {"connection": {"domain": "sandbox", "api_key": "secret"}}}
    )
    assert secret == "secret"
    assert sanitized == {"opensandbox": {"connection": {"domain": "sandbox"}}}


@pytest.mark.asyncio
async def test_pier_job_returns_absolute_trial_path(tmp_path: Path, monkeypatch) -> None:
    from pier.job import Job

    monkeypatch.chdir(tmp_path)
    config = _agent(tmp_path)._build_job_config(
        dataset_alias="deep_swe",
        task_name="abs-module-cache-flags",
        model_name="model",
        api_base="http://policy:8000/v1",
        job_name="job-1",
        jobs_dir=Path("relative-jobs"),
    )
    trial_dir = tmp_path / "relative-jobs" / "job-1" / "trial-1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text("{}")

    class FakeJob:
        async def run(self) -> None:
            return None

    async def fake_create(_config):
        return FakeJob()

    monkeypatch.setattr(Job, "create", fake_create)

    result = await run_pier_job(config)

    assert Path(result).is_absolute()
    assert Path(result) == trial_dir.resolve()


def test_pier_litellm_compat_adds_missing_zai_registry(monkeypatch) -> None:
    import litellm

    monkeypatch.delattr(litellm, "zai_models", raising=False)
    _ensure_pier_litellm_compat()
    assert litellm.zai_models == set()


def test_verifier_dockerfile_resolves_pristine_image_and_deny_network(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text("FROM public.ecr.aws/example/task:v1\nCOPY . /tests\n")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    environment = PierOpenSandboxEnvironment(
        environment_dir=tests_dir,
        environment_name="task-one",
        session_id="trial-one-verifier",
        trial_paths=trial_paths,
        task_env_config=TaskEnvironmentConfig(
            cpus=2,
            memory_mb=8192,
            storage_mb=20480,
        ),
        logger=logging.getLogger(__name__),
        provider={"opensandbox": {"connection": {"domain": "sandbox"}}},
        spec={
            "tmux_bundle_path": "/driver/tmux.tar.gz",
            "provider_options": {"platform": {"os": "linux", "arch": "amd64"}},
        },
    )

    spec = environment._sandbox_spec()
    assert spec.image == "public.ecr.aws/example/task:v1"
    assert spec.resources.cpu == 2
    assert spec.resources.memory_mib == 8192
    assert spec.resources.disk_gib == 20
    assert spec.provider_options["network_policy"] == {"defaultAction": "deny"}
    assert "tmux_bundle_path" not in spec.provider_options


def test_agent_environment_has_temporary_install_and_model_egress(tmp_path: Path) -> None:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    install = AgentInstallSpec(
        agent_name="mini-swe-agent",
        version="2.4.6",
        steps=[InstallStep(run="true")],
    )
    environment = PierOpenSandboxEnvironment(
        environment_dir=environment_dir,
        environment_name="task-one",
        session_id="trial-one-agent",
        trial_paths=trial_paths,
        task_env_config=TaskEnvironmentConfig(
            docker_image="public.ecr.aws/example/task:v1",
            allow_internet=False,
        ),
        logger=logging.getLogger(__name__),
        agent_install_spec=install,
        network_allowlist=NetworkAllowlist(domains=["inference-api.nvidia.com"]),
        provider={"opensandbox": {"connection": {"domain": "sandbox"}}},
        spec={"agent_install_timeout_s": 600},
    )

    assert environment.capabilities.filtered_egress is True
    assert environment.capabilities.preinstall_agents is True
    rules = environment._sandbox_spec().provider_options["network_policy"]
    assert rules["defaultAction"] == "deny"
    targets = {rule["target"] for rule in rules["egress"]}
    assert "inference-api.nvidia.com" in targets
    assert "*.astral.sh" in targets
    assert set(INSTALL_EGRESS_TARGETS) <= targets


@pytest.mark.asyncio
async def test_preinstall_runs_as_declared_users_and_closes_install_egress() -> None:
    environment = object.__new__(PierOpenSandboxEnvironment)
    environment._spec_config = {"agent_install_timeout_s": 321}
    environment.default_user = None
    environment.agent_install_spec = AgentInstallSpec(
        agent_name="mini-swe-agent",
        steps=[
            InstallStep(user="root", run="install root dependency", env={"A": "1"}),
            InstallStep(user="agent", run="install agent", env={"B": "2"}),
        ],
        verification_command="mini-swe-agent --version",
    )
    environment.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=127, stdout="", stderr="missing compiler"),
            SimpleNamespace(return_code=0, stdout="", stderr=""),
            SimpleNamespace(return_code=0, stdout="", stderr=""),
            SimpleNamespace(return_code=0, stdout="2.4.6", stderr=""),
        ]
    )
    endpoint_headers: dict[str, str] = {}
    client_headers: dict[str, str] = {}
    raw = SimpleNamespace(
        id="sandbox-one",
        _egress_service=SimpleNamespace(
            endpoint=SimpleNamespace(headers=endpoint_headers),
            _httpx_client=SimpleNamespace(headers=client_headers),
        ),
        _sandbox_service=SimpleNamespace(
            get_sandbox_endpoint=AsyncMock(
                return_value=SimpleNamespace(
                    headers={"OPENSANDBOX-EGRESS-AUTH": "egress-test-token"},  # pragma: allowlist secret
                )
            )
        ),
        patch_egress_rules=AsyncMock(),
    )
    sandbox = SimpleNamespace(
        _require_handle=lambda: SimpleNamespace(raw=raw),
    )
    environment._sandbox = sandbox

    await environment._preinstall_agent()
    await environment._close_install_egress()

    assert environment.exec.await_args_list[0].kwargs["user"] == "root"
    assert environment.exec.await_args_list[1].kwargs["user"] == "root"
    assert environment.exec.await_args_list[1].kwargs["timeout_sec"] == 321
    assert environment.exec.await_args_list[2].kwargs["user"] is None
    rules = raw.patch_egress_rules.await_args.args[0]
    assert [rule.target for rule in rules] == list(INSTALL_EGRESS_TARGETS)
    assert {rule.action for rule in rules} == {"deny"}
    raw._sandbox_service.get_sandbox_endpoint.assert_awaited_once_with(
        "sandbox-one",
        18080,
        use_server_proxy=False,
    )
    assert set(endpoint_headers) == {"OPENSANDBOX-EGRESS-AUTH"}
    assert set(client_headers) == {"OPENSANDBOX-EGRESS-AUTH"}


@pytest.mark.asyncio
async def test_preinstall_skips_redundant_root_packages_when_tools_exist() -> None:
    environment = object.__new__(PierOpenSandboxEnvironment)
    environment._spec_config = {"agent_install_timeout_s": 321}
    environment.default_user = None
    environment.agent_install_spec = AgentInstallSpec(
        agent_name="mini-swe-agent",
        steps=[
            InstallStep(user="root", run="apt-get update && apt-get install build-essential"),
            InstallStep(user="agent", run="install agent"),
        ],
        verification_command="mini-swe-agent --version",
    )
    environment.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=0, stdout="", stderr=""),
            SimpleNamespace(return_code=0, stdout="", stderr=""),
            SimpleNamespace(return_code=0, stdout="2.4.6", stderr=""),
        ]
    )

    await environment._preinstall_agent()

    commands = [call.args[0] for call in environment.exec.await_args_list]
    assert commands[0].startswith("command -v curl")
    assert not any("apt-get" in command for command in commands)
    assert commands[1].startswith("bash -lc")


@pytest.mark.asyncio
async def test_startup_command_retries_transient_backend_failure(monkeypatch) -> None:
    environment = object.__new__(PierOpenSandboxEnvironment)
    environment.logger = logging.getLogger(__name__)
    environment.exec = AsyncMock(
        side_effect=[
            ConnectionError("All connection attempts failed"),
            SimpleNamespace(return_code=0, stdout="ready", stderr=""),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(
        "responses_api_agents.deep_swe_agent.opensandbox_environment.asyncio.sleep",
        sleep,
    )

    result = await environment._exec_startup("true", user="root")

    assert result.return_code == 0
    assert environment.exec.await_count == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_offline_tmux_bundle_is_uploaded_and_verified(tmp_path: Path) -> None:
    bundle = tmp_path / "tmux.tar.gz"
    bundle.write_bytes(b"fixture")
    environment = object.__new__(PierOpenSandboxEnvironment)
    environment._spec_config = {"tmux_bundle_path": str(bundle)}
    environment.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=127, stdout="", stderr="tmux missing"),
            SimpleNamespace(return_code=0, stdout="tmux 3.2a", stderr=""),
        ]
    )
    environment.upload_file = AsyncMock()

    await environment.ensure_tmux()

    environment.upload_file.assert_awaited_once_with(bundle, "/tmp/deep-swe-tmux.tar.gz")
    install_command = environment.exec.await_args_list[1].args[0]
    assert "/usr/local/bin/tmux -V" in install_command
