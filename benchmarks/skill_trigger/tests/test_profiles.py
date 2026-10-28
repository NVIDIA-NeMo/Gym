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
from harbor.agents.agent_auth_resolution import ResolvedModelConnection
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.hermes import Hermes
from harbor.models.job.config import JobConfig
from harbor.models.model_api_format import ModelApiFormat
from omegaconf import OmegaConf

from benchmarks.skill_trigger.profiles import CodexCompat, CodexGuidance, HermesCompat, HermesGuidance
from benchmarks.skill_trigger.scoring import EvidenceError, score_trajectory
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
    if profile.startswith("codex"):
        assert issubclass(cls, CodexCompat)
        native = agent.kwargs["config"]
        assert native["model_provider"] == "benchmark"
        assert native["model_providers"]["benchmark"]["supports_websockets"] is False
    assert job.retry.max_retries == 0
    assert cfg["datasets"][0]["num_repeats"] == 1
    assert not parsed.count_failure_classes_as_zero and not parsed.count_missing_rollouts_as_zero
    assert job.verifier.import_path == "benchmarks.skill_trigger.verifier:SkillOpeningVerifier"


@pytest.mark.parametrize("custom_connection", [False, True])
def test_hermes_guidance_changes_only_one_setting(custom_connection):
    connection = (
        ResolvedModelConnection(
            provider="openai",
            model_api_format=ModelApiFormat.OPENAI_CHAT_COMPLETIONS,
            resolved_base_url="https://example.invalid/v1",
        )
        if custom_connection
        else None
    )
    base = yaml.safe_load(Hermes._build_config_yaml("fixture", 7, connection=connection))
    actual = yaml.safe_load(HermesGuidance._build_config_yaml("fixture", 7, connection=connection))
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
    assert execute.call_args.kwargs["env"]["HERMES_HOME"] == "/tmp/hermes"
    assert "/d923a5eb487a17c0a68801bba9ef0f56c81040a8/scripts/install.sh" in command
    assert agent.get_version_command().endswith("hermes --version")


SKILL_TEXT = (
    "Use this unique fixture skill to inspect molecules, compare evidence, and report reproducible findings. " * 3
)


def write_codex_session(folder, session_id, *, parent=None, reads=False, spawn=False):
    metadata = {"id": session_id, "cli_version": "0.156.1", "source": "exec"}
    if parent:
        metadata["source"] = {"subagent": {"thread_spawn": {"parent_thread_id": parent}}}
    items = [
        {"type": "session_meta", "payload": metadata},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect the molecule."}],
            },
        },
    ]
    if reads:
        items.extend(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "read-call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "cat /tmp/skills/fixture/SKILL.md"}),
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "call_id": "read-call", "output": SKILL_TEXT},
                },
            ]
        )
    if spawn:
        items.extend(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "spawn-call",
                        "name": "spawn_agent",
                        "arguments": json.dumps({"task": "inspect"}),
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "spawn-call",
                        "output": json.dumps({"agent_id": "child-session"}),
                    },
                },
            ]
        )
    items.append(
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done."}]},
        }
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (session_id + ".jsonl")).write_text("".join(json.dumps(e) + "\n" for e in items))


@pytest.mark.parametrize("root_reads", [True, False])
def test_codex_selects_cli_root_and_keeps_child_evidence(tmp_path, root_reads):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "codex.txt").write_text(json.dumps({"type": "thread.started", "thread_id": "root-session"}))
    write_codex_session(logs / "sessions/day1", "root-session", reads=root_reads, spawn=True)
    write_codex_session(logs / "sessions/day2", "child-session", parent="root-session", reads=True)
    agent = CodexCompat(logs_dir=tmp_path / "parser", model_name="openai/fixture", version="0.156.1")
    trajectory = agent.convert_trajectory(logs)
    assert trajectory.session_id == "root-session"
    assert trajectory.subagent_trajectories[0].session_id == "child-session"
    refs = [r.subagent_trajectory_ref for s in trajectory.steps if s.observation for r in s.observation.results]
    assert any(refs)
    if root_reads:
        assert score_trajectory(trajectory, sources={"fixture": SKILL_TEXT}, target="fixture", harness="codex") == 1
    else:
        with pytest.raises(EvidenceError, match="delegat|Delegat"):
            score_trajectory(trajectory, sources={"fixture": SKILL_TEXT}, target="fixture", harness="codex")


def test_codex_single_session_control_and_missing_child(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "codex.txt").write_text(json.dumps({"type": "thread.started", "thread_id": "root-session"}))
    write_codex_session(logs / "sessions", "root-session", reads=True)
    agent = CodexCompat(logs_dir=tmp_path / "parser", model_name="openai/fixture", version="0.156.1")
    trajectory = agent.convert_trajectory(logs)
    assert not trajectory.subagent_trajectories
    assert score_trajectory(trajectory, sources={"fixture": SKILL_TEXT}, target="fixture", harness="codex") == 1
    write_codex_session(logs / "sessions", "root-session", spawn=True)
    trajectory = agent.convert_trajectory(logs)
    assert trajectory.agent.extra["uncollected_subagents"] is True


@pytest.mark.parametrize("ids", [[], ["root-session", "other-session"], ["missing-session"]])
def test_codex_rejects_missing_or_ambiguous_root(tmp_path, ids):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "codex.txt").write_text("\n".join(json.dumps({"type": "thread.started", "thread_id": i}) for i in ids))
    write_codex_session(logs / "sessions", "root-session", reads=True)
    agent = CodexCompat(logs_dir=tmp_path / "parser", model_name="openai/fixture", version="0.156.1")
    with pytest.raises(EvidenceError, match="main.session"):
        agent.convert_trajectory(logs)
