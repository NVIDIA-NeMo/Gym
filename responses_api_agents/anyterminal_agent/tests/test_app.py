# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for anyterminal_agent.

Modeled on anyswe_agent's tests: these exercise pure logic (no Apptainer/Docker) —
runner-script generation, container discovery, deps-key derivation, setup-script presence,
and example-data shape. Heavy side effects in model_post_init (deps + harness install) are
bypassed by calling staticmethods/properties directly rather than constructing the agent.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.anyterminal_agent.app import (
    _RUNNER_TEMPLATE,
    AnyTerminalAgent,
    AnyTerminalAgentConfig,
    GymAgentHarnessProcessor,
)


def _config(**overrides) -> AnyTerminalAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="anyterminal_agent",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
    )
    base.update(overrides)
    return AnyTerminalAgentConfig(**base)


class TestRunnerTemplate:
    def _render(self) -> str:
        return _RUNNER_TEMPLATE.format(
            agent_module="responses_api_agents.hermes_agent.app",
            agent_class="HermesAgent",
            agent_cfg_class="HermesAgentConfig",
            agent_class_lower="hermesagent",
        )

    def test_renders_valid_python(self) -> None:
        rendered = self._render()
        # Must be syntactically valid Python and reference the agent class.
        compile(rendered, "<runner>", "exec")
        assert "HermesAgent(config=config" in rendered

    def test_response_is_written_back(self) -> None:
        # The runner's agent-agnostic contract is to persist the response where the host reads it.
        assert "/trajectories_mount/response.json" in _RUNNER_TEMPLATE
        assert "response.model_dump_json()" in _RUNNER_TEMPLATE

    def test_sampling_is_forwarded(self) -> None:
        rendered = self._render()
        compile(rendered, "<runner>", "exec")
        # Read from env, forwarded onto the body, and filtered to the agent config's fields.
        assert "NGTB_SAMPLING" in rendered
        assert "**SAMPLING," in rendered
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
    def _stub(self, **cfg_overrides) -> SimpleNamespace:
        # _find_container only touches self.config.tb_sif_dir; avoid building the whole agent.
        return SimpleNamespace(config=_config(**cfg_overrides))

    def test_prebuilt_sif_exact_match(self, tmp_path: Path) -> None:
        sif = tmp_path / "fix-git.sif"
        sif.write_text("")
        found = AnyTerminalAgent._find_container(self._stub(tb_sif_dir=str(tmp_path)), "fix-git", "ubuntu:22.04")
        assert found == str(sif.resolve())

    def test_sif_name_variant_underscore(self, tmp_path: Path) -> None:
        # Task names with dashes also match an underscored SIF filename.
        sif = tmp_path / "fix_git.sif"
        sif.write_text("")
        found = AnyTerminalAgent._find_container(self._stub(tb_sif_dir=str(tmp_path)), "fix-git", "ubuntu:22.04")
        assert found == str(sif.resolve())

    def test_falls_back_to_docker_uri_when_no_sif(self, tmp_path: Path) -> None:
        found = AnyTerminalAgent._find_container(self._stub(tb_sif_dir=str(tmp_path)), "nope", "ubuntu:22.04")
        assert found == "docker://ubuntu:22.04"

    def test_falls_back_to_docker_uri_when_no_sif_dir(self) -> None:
        found = AnyTerminalAgent._find_container(self._stub(), "nope", "ubuntu:22.04")
        assert found == "docker://ubuntu:22.04"

    def test_existing_docker_uri_passed_through(self) -> None:
        found = AnyTerminalAgent._find_container(self._stub(), "nope", "docker://myrepo/img:tag")
        assert found == "docker://myrepo/img:tag"


class TestSetupScriptsExist:
    def test_supported_agents_have_deps_scripts(self) -> None:
        scripts = Path(__file__).parent.parent / "setup_scripts"
        assert (scripts / "hermes_agent_deps.sh").exists()
        assert (scripts / "_portable_python.sh").exists()


class TestExampleData:
    def _example(self) -> Path:
        # Data files are gitignored; test whichever materialized example is present, else skip.
        data_dir = Path(__file__).parent.parent / "data"
        for name in ("terminal_bench_smoke.jsonl", "terminal_bench_example.jsonl"):
            p = data_dir / name
            if p.exists():
                return p
        candidates = sorted(data_dir.glob("*.jsonl"))
        return candidates[0] if candidates else data_dir / "missing.jsonl"

    def test_example_jsonl_parses(self) -> None:
        example = self._example()
        if not example.exists():
            pytest.skip("no example .jsonl present (data/ is gitignored)")
        rows = [json.loads(line) for line in example.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            rcp = row["responses_create_params"]
            assert "input" in rcp
            assert "metadata" in rcp
            md = rcp["metadata"]
            assert "task_name" in md or "instance_id" in md
