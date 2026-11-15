# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.task.config import NetworkMode, NetworkPolicy

from benchmarks.skillsbench.environment import SkillsBenchSandboxEnvironment
from responses_api_agents.harbor_agent.tests.test_environment import _make_environment, _provider
from responses_api_agents.harbor_agent.tests.test_environment import _reset_fake_provider as _reset_fake_provider


class TestSkillsBenchSkillModes:
    @pytest.mark.asyncio
    async def test_no_skill_removes_preinstalled_skills(self, tmp_path):
        env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="no-skill")
        await env.start(force_build=False)
        assert "rm -rf /root/.agents/skills" in _provider().exec_calls[-1]["command"]
        await env.stop(delete=True)

    @pytest.mark.asyncio
    async def test_no_skill_failure_stops_sandbox(self, tmp_path):
        env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="no-skill")
        env.exec = AsyncMock(return_value=SimpleNamespace(return_code=1, stderr="denied", stdout=""))
        with pytest.raises(RuntimeError, match="denied"):
            await env.start(force_build=False)
        assert _provider().closed_handles == ["sbx-123"]

    @pytest.mark.asyncio
    async def test_with_skill_uploads_and_copies_skills(self, tmp_path):
        skills = tmp_path / "task" / "environment" / "skills" / "example"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("# Example\n")
        env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="with-skill")
        await env.start(force_build=False)
        assert any(path.endswith(".tar.gz") for path in _provider().uploads)
        assert "cp -a /root/.agents/skills/." in _provider().exec_calls[-1]["command"]
        assert _provider().closed_handles == []
        await env.stop(delete=True)

    @pytest.mark.asyncio
    async def test_with_skill_upload_failure_stops_sandbox(self, tmp_path):
        skills = tmp_path / "task" / "environment" / "skills"
        skills.mkdir(parents=True)
        env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="with-skill")
        env.upload_dir = AsyncMock(side_effect=RuntimeError("upload failed"))
        with pytest.raises(RuntimeError, match="upload failed"):
            await env.start(force_build=False)
        assert _provider().closed_handles == ["sbx-123"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("skill_mode,operation", [("no-skill", "exec"), ("with-skill", "upload_dir")])
    @pytest.mark.parametrize("exception", [RuntimeError, asyncio.CancelledError])
    async def test_setup_failure_and_cancellation_close_sandbox(self, tmp_path, skill_mode, operation, exception):
        (tmp_path / "task" / "environment" / "skills").mkdir(parents=True)
        env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode=skill_mode)
        setattr(env, operation, AsyncMock(side_effect=exception("setup interrupted")))
        with pytest.raises(exception, match="setup interrupted"):
            await env.start(force_build=False)
        assert _provider().closed_handles == ["sbx-123"]
        assert env._sandbox is None

    @pytest.mark.asyncio
    async def test_setup_cleanup_failure_retains_sandbox_for_retry(self, tmp_path, monkeypatch):
        from responses_api_agents.harbor_agent.tests.test_environment import FakeProvider

        env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="no-skill")
        env.exec = AsyncMock(side_effect=RuntimeError("setup failed"))
        with monkeypatch.context() as patcher:
            patcher.setattr(FakeProvider, "close", AsyncMock(side_effect=RuntimeError("stop failed")))
            with pytest.raises(RuntimeError, match="stop failed"):
                await env.start(force_build=False)
        assert env._sandbox is not None
        await env.stop(delete=True)
        assert _provider().closed_handles == ["sbx-123"]


@pytest.mark.parametrize("allow", [False, True])
def test_runtime_no_network_requires_explicit_opt_in(tmp_path, allow):
    kwargs = {
        "environment_class": SkillsBenchSandboxEnvironment,
        "network_policy": NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
        "allow_unenforced_internet_isolation": allow,
    }
    if allow:
        env = _make_environment(tmp_path, **kwargs)
        env.validate_network_policy_support(kwargs["network_policy"])
    else:
        with pytest.raises(ValueError, match="no-network"):
            _make_environment(tmp_path, **kwargs)


@pytest.mark.asyncio
async def test_harbor_verifier_can_chmod_as_root_and_run_as_default_user(tmp_path):
    from pathlib import Path

    from harbor.models.task.task import Task
    from harbor.verifier.verifier import Verifier

    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Create the answer.")
    (task_dir / "task.toml").write_text('version = "1.0"\n[environment]\ndocker_image = "example/task:1.0"\n')
    (task_dir / "tests/test.sh").write_text("#!/bin/bash\necho 1.0 > /logs/verifier/reward.txt\n")
    env = _make_environment(tmp_path)
    env.default_user = "verifier"

    async def download_logs(source_dir, target_dir):
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        (Path(target_dir) / "reward.txt").write_text("1.0")

    env.download_dir = AsyncMock(side_effect=download_logs)
    await env.start(force_build=False)
    try:
        result = await Verifier(task=Task(task_dir=task_dir), trial_paths=env.trial_paths, environment=env).verify()
        assert result.rewards == {"reward": 1.0}
        calls = _provider().exec_calls
        assert any(call["command"].startswith("chmod +x") and call["user"] == "root" for call in calls)
        assert any("/tests/test.sh" in call["command"] and call["user"] == "verifier" for call in calls)
    finally:
        await env.stop(delete=True)
