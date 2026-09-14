# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.harbor_agent_general.sandbox_environment import HarborSandboxEnvironment


def make_environment(tmp_path: Path, **kwargs) -> HarborSandboxEnvironment:
    config = kwargs.pop("task_env_config", EnvironmentConfig(docker_image="test@sha256:abc", cpus=4, memory_mb=4096))
    return HarborSandboxEnvironment(
        environment_dir=tmp_path,
        environment_name=kwargs.pop("environment_name", "task"),
        session_id="trial-env",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=config,
        sandbox_provider=kwargs.pop("sandbox_provider", {"local": {}}),
        **kwargs,
    )


async def test_offline_verifier_creation_overrides_allow_rules_without_affecting_agent(tmp_path, monkeypatch):
    from responses_api_agents.harbor_agent_general import sandbox_environment as module

    options = {
        "resource_requests": "limits",
        "network_policy": {"defaultAction": "allow", "egress": [{"action": "allow", "target": "example.com"}]},
    }
    original = deepcopy(options)
    created = []

    def create(provider, spec):
        created.append(spec)
        sandbox = AsyncMock()
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
        return sandbox

    monkeypatch.setattr(module, "AsyncSandbox", create)
    for phase in ("agent", "verifier"):
        env = make_environment(
            tmp_path / phase,
            sandbox_provider={"opensandbox": {}},
            sandbox_provider_options=options,
            network_policy=NetworkPolicy(network_mode="no-network" if phase == "verifier" else "public"),
        )
        env._upload_environment_dir_after_start = AsyncMock()
        await env.start(False)
        await env.stop(True)
    assert created[0].provider_options == original
    assert created[1].provider_options == {
        "resource_requests": "limits",
        "network_policy": {"defaultAction": "deny", "egress": []},
    }
    assert options == original


@pytest.mark.parametrize("provider,compose", [("local", False), ("opensandbox", True)])
def test_offline_unsupported_backends_fail_closed(tmp_path, provider, compose):
    if compose:
        (tmp_path / "docker-compose.yaml").write_text("services: {}")
    with pytest.raises(ValueError, match="no-network.*not supported"):
        make_environment(
            tmp_path,
            sandbox_provider={provider: {}},
            compose_image_configs="unused.json",
            network_policy=NetworkPolicy(network_mode="no-network"),
            allow_unenforced_internet_isolation=True,
        )


async def test_allowlists_and_runtime_policy_changes_remain_unsupported(tmp_path):
    with pytest.raises(ValueError, match="allowlist.*not supported"):
        make_environment(
            tmp_path,
            sandbox_provider={"opensandbox": {}},
            network_policy=NetworkPolicy(network_mode="allowlist", allowed_hosts=["example.com"]),
        )
    env = make_environment(tmp_path, sandbox_provider={"opensandbox": {}})
    with pytest.raises(ValueError, match="cannot change network policy"):
        await env.set_network_policy(NetworkPolicy(network_mode="no-network"))


@pytest.mark.parametrize("request_gpu_type", [True, False])
def test_current_resources_env_and_gpu_type(tmp_path, request_gpu_type):
    env = make_environment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="test@sha256:abc",
            cpus=16,
            memory_mb=32768,
            storage_mb=1024000,
            gpus=1,
            gpu_types=["H100"],
            env={"TASK": "value"},
        ),
        persistent_env={"TRIAL": "value"},
        sandbox_env={"EXTRA": "value"},
        sandbox_request_gpu_type=request_gpu_type,
    )
    spec = env._build_spec()
    assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (16, 32768, 1000)
    assert (spec.resources.gpu, spec.resources.gpu_type) == (1, "H100" if request_gpu_type else None)
    assert env.task_env_config.gpu_types == ["H100"]
    assert spec.env == {"TASK": "value", "TRIAL": "value", "EXTRA": "value"}
    assert env.capabilities.gpus and not env.capabilities.mounted
    assert env.resource_capabilities().cpu_limit
    assert env.type() == "nemo-gym-sandbox"


def test_resource_rounding_and_ignore(tmp_path):
    env = make_environment(
        tmp_path,
        task_env_config=EnvironmentConfig(docker_image="image", storage_mb=1025),
        cpu_enforcement_policy="ignore",
        memory_enforcement_policy="ignore",
    )
    spec = env._build_spec()
    assert spec.resources.disk_gib == 2
    assert spec.resources.cpu is None and spec.resources.memory_mib is None
    assert spec.resources.gpu is None and spec.resources.gpu_type is None


def test_task_runtime_env_is_scoped_and_preserves_packages_resources_and_explicit_overrides(tmp_path):
    task_config = EnvironmentConfig(docker_image="image", cpus=2, memory_mb=4096, env={"TASK": "original"})
    overrides = {"nextjs-performance": {"CIRCLE_NODE_TOTAL": "3"}}
    original = deepcopy(overrides)
    env = make_environment(
        tmp_path / "selected",
        environment_name="nextjs-performance",
        task_env_config=task_config,
        sandbox_env_by_task=overrides,
    )
    spec = env._build_spec()
    assert spec.env == {"TASK": "original", "CIRCLE_NODE_TOTAL": "3"}
    assert (spec.resources.cpu, spec.resources.memory_mib) == (2, 4096)
    assert task_config.env == {"TASK": "original"} and overrides == original

    unrelated = make_environment(tmp_path / "other", sandbox_env_by_task=overrides)
    assert "CIRCLE_NODE_TOTAL" not in unrelated._build_spec().env
    explicit = make_environment(
        tmp_path / "explicit",
        environment_name="nextjs-performance",
        sandbox_env_by_task=overrides,
        sandbox_env={"CIRCLE_NODE_TOTAL": "2", "EXTRA": "value"},
    )
    assert explicit._build_spec().env == {"CIRCLE_NODE_TOTAL": "2", "EXTRA": "value"}


@pytest.mark.parametrize("filename", ["docker-compose.yaml"])
def test_compose_requires_recorded_startup_metadata(tmp_path, filename):
    (tmp_path / filename).write_text("services: {}")
    with pytest.raises(ValueError, match="Compose"):
        make_environment(tmp_path)


def test_mounts_and_ambiguous_gpu_rejected(tmp_path):
    with pytest.raises(ValueError, match="log mounts"):
        make_environment(tmp_path, mounts=[{"type": "bind", "source": "/host", "target": "/app"}])
    with pytest.raises(ValueError, match="one explicit GPU type"):
        make_environment(tmp_path, task_env_config=EnvironmentConfig(docker_image="image", gpu_types=["H100", "A100"]))


async def test_exec_preserves_harbor_user_env_and_timeout(tmp_path):
    env = make_environment(tmp_path, persistent_env={"TASK": "base"}, default_exec_timeout_s=64800)
    env._sandbox = AsyncMock()
    env._sandbox.exec.return_value = SandboxExecResult(stdout="result", stderr="", return_code=0)
    env.default_user = "agent"
    result = await env.exec("echo 'quoted'", cwd="/app", env={"TASK": "override"})
    call = env._sandbox.exec.call_args
    assert call.kwargs == {"cwd": "/app", "env": {"TASK": "override"}, "timeout_s": 64800, "user": "agent"}
    assert call.args[0].startswith("bash -ic ")
    assert result.stdout == "result"
    env._exec_shell = None
    await env.exec("true", timeout_sec=9, user=0)
    assert env._sandbox.exec.call_args.args == ("true",)
    assert env._sandbox.exec.call_args.kwargs["user"] == 0
    assert env._sandbox.exec.call_args.kwargs["timeout_s"] == 9


async def test_start_log_paths_and_stop_on_failure(tmp_path, monkeypatch):
    from responses_api_agents.harbor_agent_general import sandbox_environment as module

    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    monkeypatch.setattr(module, "AsyncSandbox", lambda *args: sandbox)
    env = make_environment(tmp_path, mounts=[{"type": "bind", "source": "/host", "target": "/logs/user-agent"}])
    env._upload_environment_dir_after_start = AsyncMock()
    await env.start(False)
    assert "/logs/user-agent" in sandbox.exec.call_args.args[0]
    env._upload_environment_dir_after_start.assert_awaited_once()
    await env.stop(True)
    sandbox.stop.assert_awaited_once()
    with pytest.raises(ValueError, match="force_build"):
        await env.start(True)

    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="denied", return_code=1)
    with pytest.raises(RuntimeError, match="denied"):
        await env.start(False)
    assert env._sandbox is None
    assert sandbox.stop.await_count == 2


def test_credentials_are_resolved_without_mutating_job_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "synthetic-test-key")
    provider = {"opensandbox": {"connection": {"domain": "localhost"}}}
    env = make_environment(tmp_path, sandbox_provider=provider, compose_image_configs="benchmarks/images.json")
    assert "api_key" not in provider["opensandbox"]["connection"]
    assert env._sandbox_provider["opensandbox"]["connection"]["api_key"] == "synthetic-test-key"
    assert env._compose_image_configs.is_absolute()


@pytest.fixture
def split_endpoints(monkeypatch):
    for pool in ("CPU", "GPU"):
        monkeypatch.setenv(f"OPENSANDBOX_DOMAIN_{pool}", f"{pool.lower()}.example.test")
        monkeypatch.setenv(f"OPENSANDBOX_API_KEY_{pool}", f"synthetic-{pool.lower()}-key")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "synthetic-legacy-key")


async def test_cpu_gpu_and_verifier_routing_is_isolated(tmp_path, monkeypatch, split_endpoints):
    from responses_api_agents.harbor_agent_general import sandbox_environment as module

    provider = {"opensandbox": {"connection": {"domain": "legacy.example.test", "tls_verify": False}}}
    original = deepcopy(provider)
    connections = []

    def factory(config, spec):
        connections.append((config, spec))
        sandbox = AsyncMock()
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
        return sandbox

    monkeypatch.setattr(module, "AsyncSandbox", factory)
    envs = []
    # Include an independently configured verifier and an effective GPU override.
    for name, gpus, override, pool in (("cpu", 0, None, "cpu"), ("gpu", 1, None, "gpu"), ("verifier", 0, 1, "gpu")):
        env = make_environment(
            tmp_path / name,
            sandbox_provider=provider,
            sandbox_split_endpoints=True,
            task_env_config=EnvironmentConfig(docker_image="image", gpus=gpus),
            override_gpus=override,
        )
        env._upload_environment_dir_after_start = AsyncMock()
        envs.append(env)
        connection = env._sandbox_provider["opensandbox"]["connection"]
        assert connection == {
            "domain": f"{pool}.example.test",
            "api_key": f"synthetic-{pool}-key",
            "tls_verify": False,
        }
        assert env._build_spec().metadata["nemo-gym.nvidia.com/resource-pool"] == pool
    await asyncio.gather(*(env.start(False) for env in envs))
    assert len(connections) == 3
    assert [config["opensandbox"]["connection"]["domain"] for config, _ in connections] == [
        "cpu.example.test",
        "gpu.example.test",
        "gpu.example.test",
    ]
    assert provider == original
    assert "synthetic-" not in json.dumps(provider)
    assert envs[0]._sandbox_provider["opensandbox"]["connection"]["domain"] == "cpu.example.test"
    await asyncio.gather(*(env.stop(True) for env in envs))


@pytest.mark.parametrize("pool,gpus", [("CPU", 0), ("GPU", 1)])
@pytest.mark.parametrize("field", ["DOMAIN", "API_KEY"])
def test_split_routing_rejects_missing_scoped_settings(tmp_path, monkeypatch, split_endpoints, pool, gpus, field):
    name = f"OPENSANDBOX_{field}_{pool}"
    monkeypatch.delenv(name)
    with pytest.raises(ValueError, match=name):
        make_environment(
            tmp_path,
            sandbox_provider={"opensandbox": {"connection": {"domain": "legacy"}}},
            sandbox_split_endpoints=True,
            task_env_config=EnvironmentConfig(docker_image="image", gpus=gpus),
        )


def test_split_routing_requires_opensandbox(tmp_path, split_endpoints):
    with pytest.raises(ValueError, match="requires the opensandbox provider"):
        make_environment(tmp_path, sandbox_split_endpoints=True)


def test_extra_overlays_must_be_resolved_upstream(tmp_path):
    overlay = tmp_path / "extra.yaml"
    overlay.write_text("services: {}")
    with pytest.raises(ValueError, match="overlays"):
        make_environment(tmp_path, extra_docker_compose=[overlay])


async def test_compose_startup_and_cleanup(tmp_path, monkeypatch, split_endpoints):
    from responses_api_agents.harbor_agent_general import sandbox_environment as module

    (tmp_path / "docker-compose.yaml").write_text("services: {peer: {image: peer}}")
    images = tmp_path / "images.json"
    record = {
        "image": "pinned@sha256:abc",
        "os": "linux",
        "architecture": "amd64",
        "config": {"Cmd": ["sleep", "infinity"]},
    }
    images.write_text(json.dumps({"test@sha256:abc": record, "peer": record}))
    main, peer = AsyncMock(), AsyncMock()
    main.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    compose = AsyncMock()
    compose.services = {"main": main, "peer": peer}
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return compose

    monkeypatch.setattr(module, "AsyncSandboxCompose", factory)
    env = make_environment(
        tmp_path,
        compose_image_configs=images,
        sandbox_provider={"opensandbox": {}},
        sandbox_split_endpoints=True,
    )
    env._upload_environment_dir_after_start = AsyncMock()
    await env.start(False)
    assert env._sandbox is main
    assert calls[0][1]["service_specs"]["main"].resources.cpu == 4
    assert calls[0][1]["service_specs"]["peer"].resources.cpu is None
    assert calls[0][1]["service_specs"]["peer"].metadata["nemo-gym.nvidia.com/resource-pool"] == "cpu"
    assert calls[0][0][0]["opensandbox"]["connection"]["domain"] == "cpu.example.test"
    assert env._sandbox_provider["opensandbox"]["connection"]["domain"] == "cpu.example.test"
    assert calls[0][0][1].is_file()
    await env.stop(True)
    compose.stop.assert_awaited_once()
    assert env._sandbox is None and env._compose is None


async def test_service_context_isolated_across_concurrent_transfers(tmp_path):
    env = make_environment(tmp_path, persistent_env={"MAIN_ONLY": "secret"})
    main, peer = AsyncMock(), AsyncMock()
    env._sandbox = main
    env._compose = SimpleNamespace(services={"main": main, "peer": peer})
    peer.exec.return_value = SandboxExecResult(stdout="ok", stderr="", return_code=0)
    main.exec.return_value = SandboxExecResult(stdout="main", stderr="", return_code=0)
    observed = {}

    async def download(source, target):
        await asyncio.sleep(0)
        observed[source] = env._require_sandbox()

    env.download_dir = download
    env.download_file = download
    await asyncio.gather(
        env.service_download_file("peer", tmp_path, service="peer"),
        env.service_download_dir("main", tmp_path, service="main"),
    )
    assert observed == {"peer": peer, "main": main}
    assert env._require_sandbox() is main
    result = await env.service_exec("true", service="peer", env={"PEER": "value"}, timeout_sec=12)
    assert result.stdout == "ok"
    assert peer.exec.call_args.kwargs["env"] == {"PEER": "value"}
    assert peer.exec.call_args.kwargs["timeout_s"] == 12
    await env.service_exec("true", service="peer")
    assert peer.exec.call_args.kwargs["timeout_s"] == 1800
    assert (await env.service_exec("true")).stdout == "main"
    await env.stop_service("main")
    main.stop.assert_awaited_once()
    peer.stop.assert_not_awaited()
    with pytest.raises(ValueError, match="unavailable"):
        await env.service_exec("true", service="missing")
