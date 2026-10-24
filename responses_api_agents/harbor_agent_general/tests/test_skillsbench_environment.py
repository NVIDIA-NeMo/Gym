# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
