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
"""unit tests for the generic in-sandbox agent runner, pure logic, no apptainer."""

from pathlib import Path

from responses_api_agents.cvdp_agent.in_sandbox_runner import (
    agent_key,
    deps_recipe_key,
    harvest,
    render_runner,
)


class TestAgentKey:
    def test_module_to_deps_key(self) -> None:
        assert agent_key("responses_api_agents.hermes_agent.app") == "hermes_agent"
        assert agent_key("responses_api_agents.claude_code_agent.app") == "claude_code_agent"

    def test_degenerate(self) -> None:
        assert agent_key("flat") == "flat"


class TestRenderRunner:
    def _render(self, cls="HermesAgent", cfg="HermesAgentConfig", mod="responses_api_agents.hermes_agent.app") -> str:
        return render_runner(mod, cls, cfg)

    def test_renders_valid_python(self) -> None:
        rendered = self._render()
        compile(rendered, "<runner>", "exec")  # must be syntactically valid

    def test_references_configured_agent(self) -> None:
        rendered = self._render(
            cls="ClaudeCodeAgent", cfg="ClaudeCodeAgentConfig", mod="responses_api_agents.claude_code_agent.app"
        )
        assert (
            "from responses_api_agents.claude_code_agent.app import ClaudeCodeAgent, ClaudeCodeAgentConfig" in rendered
        )
        assert "ClaudeCodeAgent(config=config" in rendered
        assert 'name="claudecodeagent"' in rendered

    def test_writes_response_and_calls_responses(self) -> None:
        rendered = self._render()
        assert "agent.responses(request=None, body=body)" in rendered
        # trajectory dir is configurable (NGTB_TRAJ_DIR) so it works under any sandbox mount
        assert "NGTB_TRAJ_DIR" in rendered
        assert 'Path(TRAJ_DIR, "response.json")' in rendered

    def test_forwards_model_egress_and_system_prompt(self) -> None:
        rendered = self._render()
        assert "NGTB_MODEL_URL" in rendered
        assert "NGTB_SYSTEM_PROMPT" in rendered
        assert "_resolve_model_base_url" in rendered


class TestDepsRecipeKey:
    def test_changes_with_content(self, tmp_path: Path) -> None:
        a = tmp_path / "a.sh"
        a.write_text("one")
        k1 = deps_recipe_key(a)
        a.write_text("two")
        k2 = deps_recipe_key(a)
        assert k1 != k2 and len(k1) == 64

    def test_missing_paths_tolerated(self, tmp_path: Path) -> None:
        assert deps_recipe_key(tmp_path / "nope.sh")  # falls back to a stable digest


class TestHarvest:
    def test_collects_matching_globs(self, tmp_path: Path) -> None:
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "adder.sv").write_text("module adder; endmodule")
        (tmp_path / "rtl" / "mux.v").write_text("module mux; endmodule")
        (tmp_path / "rtl" / "notes.txt").write_text("ignore me")
        out = harvest(tmp_path, ["rtl/**/*.sv", "rtl/**/*.v"])
        assert set(out) == {"rtl/adder.sv", "rtl/mux.v"}
        assert out["rtl/adder.sv"].startswith("module adder")

    def test_skips_unchanged_seeded_files(self, tmp_path: Path) -> None:
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "given.sv").write_text("seed")
        (tmp_path / "rtl" / "new.sv").write_text("written")
        out = harvest(tmp_path, ["rtl/*.sv"], seeded={"rtl/given.sv": "seed"})
        assert set(out) == {"rtl/new.sv"}  # unchanged seed excluded

    def test_reports_modified_seeded_file(self, tmp_path: Path) -> None:
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "given.sv").write_text("EDITED")
        out = harvest(tmp_path, ["rtl/*.sv"], seeded={"rtl/given.sv": "seed"})
        assert out == {"rtl/given.sv": "EDITED"}

    def test_skips_binary_and_missing(self, tmp_path: Path) -> None:
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "bin.sv").write_bytes(b"\xff\xfe\x00\x01")
        out = harvest(tmp_path, ["rtl/*.sv", "nope/*.sv"])
        assert out == {}  # binary skipped, missing dir tolerated
