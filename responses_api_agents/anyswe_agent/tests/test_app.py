# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for anyswe_agent.

These exercise pure logic (no Apptainer/Docker): runner-script generation,
container discovery, deps-key derivation, and config plumbing. Heavy side
effects in model_post_init (deps + eval harness install) are bypassed via
pydantic's model_construct.
"""

import json
from pathlib import Path

import pytest

from responses_api_agents.anyswe_agent.app import (
    _RUNNER_TEMPLATE,
    AnySweAgent,
    AnySweAgentConfig,
    GymAgentHarnessProcessor,
)


def _config(**overrides) -> AnySweAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="anyswe_agent",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
        container_formatter="/sifs/{instance_id}.sif",
    )
    base.update(overrides)
    return AnySweAgentConfig(**base)


class TestRunnerTemplate:
    def test_renders_valid_python(self) -> None:
        rendered = _RUNNER_TEMPLATE.format(
            agent_module="responses_api_agents.hermes_agent.app",
            agent_class="HermesAgent",
            agent_cfg_class="HermesAgentConfig",
            agent_class_lower="hermesagent",
        )
        # Must be syntactically valid Python and reference the agent class.
        compile(rendered, "<runner>", "exec")
        assert "HermesAgent(config=config" in rendered
        assert '"git", "diff", "HEAD"' in rendered

    def test_patch_extraction_is_git_diff(self) -> None:
        # The runner always extracts the patch via `git diff HEAD`, independent
        # of which agent ran; this is the core agent-agnostic contract.
        assert '"git", "diff", "HEAD"' in _RUNNER_TEMPLATE
        assert "patch.diff" in _RUNNER_TEMPLATE

    def test_sampling_is_forwarded(self) -> None:
        rendered = _RUNNER_TEMPLATE.format(
            agent_module="responses_api_agents.hermes_agent.app",
            agent_class="HermesAgent",
            agent_cfg_class="HermesAgentConfig",
            agent_class_lower="hermesagent",
        )
        compile(rendered, "<runner>", "exec")
        # Read from env, forwarded onto the body, and merged into the config
        # (request value winning over the agent_kwargs default).
        assert "NGSWE_SAMPLING" in rendered
        assert "**SAMPLING," in rendered
        assert "**AGENT_KWARGS, **_cfg_sampling" in rendered
        assert "HermesAgentConfig.model_fields" in rendered


class TestAgentKey:
    def test_key_from_module(self) -> None:
        proc = GymAgentHarnessProcessor(config=_config())
        assert proc._agent_key == "hermes_agent"

    def test_key_for_claude(self) -> None:
        proc = GymAgentHarnessProcessor(
            config=_config(
                agent_server_module="responses_api_agents.claude_code_agent.app",
                agent_server_class="ClaudeCodeAgent",
                agent_config_class="ClaudeCodeAgentConfig",
            )
        )
        assert proc._agent_key == "claude_code_agent"


class TestFindContainer:
    def test_exact_match(self, tmp_path: Path) -> None:
        sif = tmp_path / "astropy__astropy-12907.sif"
        sif.write_text("")
        found = AnySweAgent._find_container(
            {"instance_id": "astropy__astropy-12907", "container_formatter": str(tmp_path / "{instance_id}.sif")}
        )
        assert found == str(sif)

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AnySweAgent._find_container(
                {"instance_id": "nope__nope-1", "container_formatter": str(tmp_path / "{instance_id}.sif")}
            )


class TestSetupScriptsExist:
    def test_supported_agents_have_deps_scripts(self) -> None:
        scripts = Path(__file__).parent.parent / "setup_scripts"
        assert (scripts / "hermes_agent_deps.sh").exists()
        assert (scripts / "claude_code_agent_deps.sh").exists()
        assert (scripts / "_portable_python.sh").exists()


class TestExampleData:
    def test_example_jsonl_parses(self) -> None:
        example = Path(__file__).parent.parent / "data" / "example.jsonl"
        rows = [json.loads(line) for line in example.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            assert "metadata" in row["responses_create_params"]
            assert "instance_id" in row["responses_create_params"]["metadata"]
