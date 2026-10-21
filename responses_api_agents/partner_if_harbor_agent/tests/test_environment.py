# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest


pytest.importorskip("harbor")

from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.trial.paths import TrialPaths

import nemo_gym.global_config as global_config
import nemo_gym.server_utils as server_utils
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxStatus
from nemo_gym.sandbox.providers import registry
from responses_api_agents.partner_if_harbor_agent.environment import ArtifactEnvironment, actor_directory


class FakeOpenSandbox:
    """Stands in for the OpenSandbox provider; records what each sandbox was asked to do."""

    instances: list["FakeOpenSandbox"] = []
    failing_command: str | None = None

    def __init__(self, **kwargs):
        self.kwargs, self.commands, self.uploads, self.closed = kwargs, [], [], []
        FakeOpenSandbox.instances.append(self)

    async def create(self, spec):
        return SandboxHandle(sandbox_id="sbx", provider_name="opensandbox", raw=None)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append({"command": command, "env": env, "user": user})
        failed = command == FakeOpenSandbox.failing_command
        return SandboxExecResult(stdout="", stderr="permission denied" if failed else None, return_code=int(failed))

    async def upload_file(self, handle, source_path, target_path):
        self.uploads.append(target_path)

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        self.closed.append(handle.sandbox_id)

    async def aclose(self):
        pass


@pytest.fixture
async def opensandbox(monkeypatch):
    monkeypatch.setitem(registry._PROVIDER_REGISTRY, "opensandbox", FakeOpenSandbox)
    monkeypatch.setattr(FakeOpenSandbox, "instances", [])
    monkeypatch.setattr(FakeOpenSandbox, "failing_command", None)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "worker-key")
    # start() initializes Gym's shared HTTP client; keep it, and the config it reads, to this test.
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    yield FakeOpenSandbox
    if server_utils._GLOBAL_AIOHTTP_CLIENT is not None:
        await server_utils._GLOBAL_AIOHTTP_CLIENT.close()


def artifact_environment(tmp_path, **kwargs):
    environment_dir = tmp_path / "task/environment"
    (environment_dir / "skills").mkdir(parents=True)
    (environment_dir / "skills/SKILL.md").write_text("Greet briefly.")
    (tmp_path / "trial").mkdir()
    return ArtifactEnvironment(
        environment_dir=environment_dir,
        environment_name="native",
        session_id="native__trial",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=TaskEnvironmentConfig(docker_image="docker.io/example/task:1.0"),
        sandbox_provider={"opensandbox": {"connection": {"domain": "sandbox.invalid", "api_key": "placeholder"}}},
        actor_directories={"skills": "/root/skills"},
        output_directory="/root/output",
        exec_shell=None,
        **kwargs,
    )


async def test_start_stages_actor_assets_with_the_worker_key(tmp_path, opensandbox):
    env = artifact_environment(tmp_path, persistent_env={"BASE": "base"})
    await env.start(force_build=False)
    [sandbox] = opensandbox.instances
    # The key comes from the worker's environment, never from the Harbor job config.
    assert sandbox.kwargs["connection"]["api_key"] == "worker-key"
    assert any(path.endswith(".tar.gz") for path in sandbox.uploads)
    assert any("-C /root/skills" in call["command"] for call in sandbox.commands)
    assert sandbox.commands[-1]["command"] == "mkdir -p /root/output" and sandbox.closed == []
    # The inherited exec passes Harbor's user and scoped environment (AgentConfig.env) to the sandbox.
    with env.scoped_exec_env({"LLM_API_KEY": "placeholder"}):
        await env.exec("true", user="root")
    assert sandbox.commands[-1]["user"] == "root"
    assert sandbox.commands[-1]["env"] == {"BASE": "base", "LLM_API_KEY": "placeholder"}


async def test_failed_start_stops_and_deletes_the_sandbox(tmp_path, opensandbox):
    opensandbox.failing_command = "mkdir -p /root/output"
    env = artifact_environment(tmp_path)
    with patch.object(env, "stop", wraps=env.stop) as stop, pytest.raises(RuntimeError, match="output directory"):
        await env.start(force_build=False)
    stop.assert_awaited_once_with(delete=True)
    assert opensandbox.instances[0].closed == ["sbx"]


async def test_missing_worker_key_fails_before_a_sandbox_starts(tmp_path, opensandbox, monkeypatch):
    monkeypatch.delenv("OPENSANDBOX_API_KEY")
    with pytest.raises(KeyError, match="OPENSANDBOX_API_KEY"):
        artifact_environment(tmp_path)
    assert opensandbox.instances == []


def test_asset_paths_cannot_escape_prepared_environment(tmp_path):
    (tmp_path / "skills").mkdir()
    assert actor_directory(tmp_path, "skills") == tmp_path / "skills"
    with pytest.raises(ValueError):
        actor_directory(tmp_path, "../rubrics")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        actor_directory(tmp_path, "escape")
