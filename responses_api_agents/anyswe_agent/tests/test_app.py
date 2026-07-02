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

Exercise pure logic (no Docker/apptainer): the agent-agnostic runner script, the
dataset-name -> harness-key shim, docker image-name derivation, SweTask construction
from a task row, and config plumbing.
"""

import json
from pathlib import Path

from responses_api_agents.anyswe_agent.app import (
    _RUNNER_TEMPLATE,
    AnySweAgent,
    AnySweAgentConfig,
    GymAgentHarnessProcessor,
    _benchmark_key,
    _build_swetask,
    _instance_image,
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
    )
    base.update(overrides)
    return AnySweAgentConfig(**base)


def _problem_info(**overrides) -> dict:
    inst = {
        "base_commit": "deadbeef",
        "FAIL_TO_PASS": ["test_calc.py::test_add"],
        "PASS_TO_PASS": ["test_calc.py::test_sub"],
        "test_patch": "--- a/t.py\n+++ b/t.py\n",
        "repo": "astropy/astropy",
    }
    base = dict(
        instance_id="astropy__astropy-13453",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        problem_statement="Fix the bug.",
        container_formatter="docker://swebench/sweb.eval.x86_64.{instance_id}",
        instance_dict=json.dumps(inst),
    )
    base.update(overrides)
    return base


class TestRunnerTemplate:
    def test_renders_valid_python(self) -> None:
        rendered = _RUNNER_TEMPLATE.format(
            agent_module="responses_api_agents.hermes_agent.app",
            agent_class="HermesAgent",
            agent_cfg_class="HermesAgentConfig",
            agent_class_lower="hermesagent",
        )
        compile(rendered, "<runner>", "exec")
        assert "HermesAgent(config=config" in rendered
        assert "git add -A && git diff --cached" in rendered

    def test_patch_extraction_stages_then_diffs(self) -> None:
        # The runner stages everything before diffing (`git add -A && git diff --cached`) so
        # newly-created files land in the graded patch, matching SWE-bench's own model-patch
        # extraction. Agent-agnostic: it runs regardless of which agent produced the changes.
        assert "git add -A && git diff --cached" in _RUNNER_TEMPLATE
        assert "patch.diff" in _RUNNER_TEMPLATE

    def test_sampling_is_forwarded(self) -> None:
        rendered = _RUNNER_TEMPLATE.format(
            agent_module="responses_api_agents.hermes_agent.app",
            agent_class="HermesAgent",
            agent_cfg_class="HermesAgentConfig",
            agent_class_lower="hermesagent",
        )
        compile(rendered, "<runner>", "exec")
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


class TestBenchmarkKey:
    def test_verified_maps_to_swe_bench(self) -> None:
        assert _benchmark_key("princeton-nlp/SWE-bench_Verified") == "swe-bench"

    def test_multilingual_and_r2e(self) -> None:
        assert _benchmark_key("princeton-nlp/SWE-bench_Multilingual") == "swe-bench-multilingual"
        assert _benchmark_key("R2E-Gym/R2E-Gym-Subset") == "r2e-gym"

    def test_unknown_defaults_to_swe_bench(self) -> None:
        assert _benchmark_key("some/unknown-dataset") == "swe-bench"


class TestInstanceImage:
    def test_docker_scheme_stripped_and_tag_mangled(self) -> None:
        img = _instance_image("docker://swebench/sweb.eval.x86_64.{instance_id}", "astropy__astropy-13453")
        assert img == "swebench/sweb.eval.x86_64.astropy_1776_astropy-13453:latest"

    def test_list_formatter_and_existing_tag(self) -> None:
        img = _instance_image(["swebench/sweb.eval.x86_64.{instance_id}:v1"], "psf__requests-2317")
        assert img == "swebench/sweb.eval.x86_64.psf_1776_requests-2317:v1"

    def test_default_formatter(self) -> None:
        img = _instance_image(None, "django__django-12345")
        assert img == "swebench/sweb.eval.x86_64.django_1776_django-12345:latest"

    def test_local_sif_path_used_verbatim(self) -> None:
        # A .sif formatter (local apptainer image) resolves to an on-disk path with the raw
        # instance_id (no _1776_ mangle) and no :latest tag, so the apptainer provider can
        # ``instance start`` it directly without a registry pull.
        img = _instance_image("/sifs/sweb.eval.x86_64.{instance_id}.sif", "astropy__astropy-13453")
        assert img == "/sifs/sweb.eval.x86_64.astropy__astropy-13453.sif"


class TestBuildSweTask:
    def test_unpacks_instance_dict(self) -> None:
        task = _build_swetask(_problem_info())
        assert task.instance_id == "astropy__astropy-13453"
        assert task.image == "swebench/sweb.eval.x86_64.astropy_1776_astropy-13453:latest"
        assert task.benchmark == "swe-bench"
        assert task.base_commit == "deadbeef"
        assert task.fail_to_pass == ["test_calc.py::test_add"]
        assert task.pass_to_pass == ["test_calc.py::test_sub"]
        assert task.repo_workdir == "/testbed"

    def test_flat_eval_flag_is_configurable(self) -> None:
        assert _build_swetask(_problem_info()).metadata["flat_eval"] is True
        assert _build_swetask(_problem_info(), flat_eval=False).metadata["flat_eval"] is False


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


class _NoSetupAnySweAgent(AnySweAgent):
    """AnySweAgent with ``model_post_init`` stubbed out (skip deps install + server wiring) so the
    pure provider-config methods can be unit-tested without a live server."""

    def model_post_init(self, context) -> None:
        return None


class TestApptainerGradingProvider:
    """The apptainer grading/agent sandbox must be writable AND isolated from the host $HOME.

    apptainer bind-mounts the host home by default, leaking host dotfiles/caches (e.g. the matplotlib
    font cache) into the eval and flipping image-comparison tests vs docker; --no-mount home prevents
    that.
    """

    def _agent(self, **cfg_overrides) -> AnySweAgent:
        return _NoSetupAnySweAgent.model_construct(config=_config(**cfg_overrides))

    def test_grading_provider_writable_and_host_home_isolated(self) -> None:
        cfg = self._agent(sandbox_provider={"apptainer": {}})._grading_provider()
        args = cfg["apptainer"]["create"]["extra_start_args"]
        assert "--writable-tmpfs" in args
        assert args[args.index("--no-mount") + 1] == "home"

    def test_grading_provider_preserves_user_start_args(self) -> None:
        agent = self._agent(sandbox_provider={"apptainer": {"create": {"extra_start_args": ["--nv"]}}})
        args = agent._grading_provider()["apptainer"]["create"]["extra_start_args"]
        assert "--nv" in args and "--writable-tmpfs" in args and "--no-mount" in args

    def test_non_apptainer_grading_provider_unchanged(self) -> None:
        assert self._agent(sandbox_provider={"docker": {}})._grading_provider() == {"docker": {}}
