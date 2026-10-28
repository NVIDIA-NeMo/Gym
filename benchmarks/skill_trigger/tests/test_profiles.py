# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate actual Harbor agent options and the two narrow guidance overrides."""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.hermes import Hermes
from harbor.models.job.config import JobConfig
from omegaconf import OmegaConf

from benchmarks.skill_trigger.profiles import CodexGuidance, HermesCompat, HermesGuidance
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from responses_api_agents.harbor_agent_general.app import HarborAgentConfig


@pytest.mark.parametrize("profile", ["pi", "codex", "codex_guidance", "hermes", "hermes_guidance"])
def test_configs_use_existing_bridge_and_native_harbor_agents(profile):
    raw = {
        "config_paths": [f"benchmarks/skill_trigger/configs/{profile}/config.yaml"],
        "policy_model_name": "openai/fixture",
        "policy_base_url": "https://example.invalid/v1",
    }
    parsed = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(raw),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    cfg = OmegaConf.to_container(parsed.harbor_agent_general.responses_api_agents.harbor_agent_general, resolve=True)
    config = HarborAgentConfig.model_validate(
        {"name": "harbor_agent_general", "host": "127.0.0.1", "port": 8080, **cfg}
    )
    job = JobConfig.model_validate(config.job.model_dump())
    agent = job.agents[0]
    cls = AgentFactory.get_agent_class_from_config(agent)
    cls.parse_options(agent.kwargs, agent.env)
    assert cls.capabilities.atif and cls.capabilities.skills
    if profile.startswith("hermes"):
        assert agent.kwargs["version"] == "v0.22.2"
    assert job.retry.max_retries == 0
    assert cfg["datasets"][0]["num_repeats"] == 1
    assert not parsed.count_failure_classes_as_zero and not parsed.count_missing_rollouts_as_zero
    assert job.verifier.import_path == "benchmarks.skill_trigger.verifier:SkillOpeningVerifier"


def test_hermes_guidance_changes_only_one_setting():
    base = yaml.safe_load(Hermes._build_config_yaml("fixture", 7))
    actual = yaml.safe_load(HermesGuidance._build_config_yaml("fixture", 7))
    assert actual["agent"].pop("tool_use_enforcement") is True
    assert actual == base


@pytest.mark.asyncio
async def test_codex_guidance_uploads_pinned_catalog_with_one_change(tmp_path):
    catalog = {"models": [{"slug": "fixture", "include_skills_usage_instructions": False, "context_window": 1234}]}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    agent = CodexGuidance(
        logs_dir=tmp_path / "logs",
        model_name="openai/fixture",
        version="0.156.1",
        model_catalog_path=str(path),
        model_catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config={"model_catalog_json": "/tmp/skill-trigger-models.json"},
    )
    captured = {}

    async def upload(src, dest):
        captured[dest] = json.loads(Path(src).read_text())

    environment = AsyncMock()
    environment.upload_file.side_effect = upload
    with patch.object(Codex, "setup", new=AsyncMock()):
        await agent.setup(environment)
    modified = captured["/tmp/skill-trigger-models.json"]
    assert modified["models"][0].pop("include_skills_usage_instructions") is True
    original = json.loads(path.read_text())
    original["models"][0].pop("include_skills_usage_instructions")
    assert modified == original
    path.write_text("changed")
    with patch.object(Codex, "setup", new=AsyncMock()), pytest.raises(ValueError, match="hash"):
        await agent.setup(environment)


@pytest.mark.asyncio
async def test_native_hermes_installer_uses_supported_version_probe(tmp_path):
    agent = HermesCompat(logs_dir=tmp_path, model_name="openai/fixture", version="v0.22.2")
    with (
        patch.object(agent, "ensure_system_dependencies", new=AsyncMock()),
        patch.object(BaseInstalledAgent, "exec_as_agent", new=AsyncMock()) as execute,
    ):
        await agent.install(AsyncMock())
    command = execute.call_args.kwargs["command"]
    assert "--branch v0.22.2" in command
    assert command.endswith("&& hermes --version")
    assert agent.get_version_command().endswith("hermes --version")
