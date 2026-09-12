# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.harbor_agent_general.sandbox_environment import HarborSandboxEnvironment


def make_environment(tmp_path: Path, **kwargs) -> HarborSandboxEnvironment:
    config = kwargs.pop("task_env_config", EnvironmentConfig(docker_image="test@sha256:abc", cpus=4, memory_mb=4096))
    return HarborSandboxEnvironment(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="trial-env",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=config,
        sandbox_provider={"local": {}},
        **kwargs,
    )


def test_current_resources_env_and_gpu_type(tmp_path):
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
    )
    spec = env._build_spec()
    assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (16, 32768, 1000)
    assert (spec.resources.gpu, spec.resources.gpu_type) == (1, "H100")
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


@pytest.mark.parametrize("filename", ["docker-compose.yaml"])
def test_compose_is_rejected_until_adapter_enabled(tmp_path, filename):
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
