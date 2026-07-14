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

These exercise runner generation, image resolution, and configuration.
"""

import base64
import json
from pathlib import Path
from types import SimpleNamespace

from responses_api_agents.anyswe_agent.app import (
    _RUNNER_TEMPLATE,
    AnySweAgent,
    AnySweAgentConfig,
    _classify_agent_error,
    _dataset_family,
    _r2e_resolved,
    _safe_config_json,
    _should_mask_sample,
)
from responses_api_agents.anyswe_agent.prepare import _to_gym_row


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
        container_formatter="swebench/sweb.eval.x86_64.{instance_id}",
        sandbox_provider={"opensandbox": {}},
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
        compile(rendered, "<runner>", "exec")
        assert "HermesAgent(config=config" in rendered
        assert '["git", "add", "-A"]' in rendered
        assert '["git", "diff", "--no-color", "--cached", "HEAD"]' in rendered

    def test_patch_extraction_includes_untracked_files(self) -> None:
        assert '["git", "add", "-A"]' in _RUNNER_TEMPLATE
        assert '["git", "diff", "--no-color", "--cached", "HEAD"]' in _RUNNER_TEMPLATE
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


class TestSandboxAPI:
    def test_default_provider_is_named_sandbox(self) -> None:
        config = AnySweAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="anyswe_agent",
            agent_server_module="responses_api_agents.hermes_agent.app",
            agent_server_class="HermesAgent",
            agent_config_class="HermesAgentConfig",
            container_formatter="registry.example.com/anyswe:{instance_id}",
        )
        assert config.sandbox_provider == "sandbox"
        assert not config.upload_agent_runtime

    def test_image_uses_swebench_tag_format(self) -> None:
        image = AnySweAgent._sandbox_image(
            {
                "instance_id": "Astropy__Astropy-12907",
                "container_formatter": "docker://swebench/sweb.eval.x86_64.{instance_id}",
            }
        )
        assert image == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"

    def test_spec_forwards_public_sandbox_fields(self) -> None:
        params = SimpleNamespace(
            sandbox_spec={
                "ttl_s": 600,
                "ready_timeout_s": 300,
                "resources": {"cpu": 2, "memory_mib": 4096},
                "provider_options": {"platform": {"arch": "amd64"}},
            },
            sandbox_default_metadata={"sandbox-api": "opensandbox-sdk"},
            swebench_agent_timeout=100,
            swebench_tests_timeout=200,
            instance_id="astropy__astropy-12907",
            container="image:tag",
        )
        spec = AnySweAgent._sandbox_spec(params, files={"/tmp/input": "data"})
        assert spec.image == "image:tag"
        assert spec.ttl_s == 600
        assert spec.resources.cpu == 2
        assert spec.metadata["sandbox-api"] == "opensandbox-sdk"
        assert spec.provider_options == {"platform": {"arch": "amd64"}}
        assert spec.files == {"/tmp/input": "data"}

    def test_opencode_model_wiring_is_provider_independent(self) -> None:
        params = SimpleNamespace(
            body=SimpleNamespace(model="model", temperature=1.0, top_p=0.95, max_output_tokens=None),
            agent_kwargs={
                "model": "nemo/model",
                "opencode_config": {
                    "provider": {"nemo": {"npm": "@ai-sdk/openai-compatible", "models": {"model": {}}}}
                },
            },
            model_server_url="http://model-host:8000/v1",
            agent_server_module="responses_api_agents.opencode_agent.app",
        )
        env = AnySweAgent._sandbox_agent_env(params)
        kwargs = json.loads(base64.b64decode(env["NGSWE_AGENT_KWARGS_B64"]))
        assert env["NGSWE_MODEL_NAME"] == "model"
        assert kwargs["model"] == "nemo/model"
        assert kwargs["opencode_config"]["provider"]["nemo"]["options"] == {
            "baseURL": "http://model-host:8000/v1",
            "apiKey": "EMPTY",  # pragma: allowlist secret
        }

    def test_agent_error_classification_matches_swe_agents(self) -> None:
        assert _classify_agent_error("maximum iteration reached") == "max_iteration"
        assert _classify_agent_error("ContextWindowExceeded") == "context_window"
        assert _classify_agent_error("") is None

    def test_masking_matches_swe_agents(self) -> None:
        assert not _should_mask_sample(False, None, False, None)
        assert not _should_mask_sample(False, "max_iteration", False, None)
        assert _should_mask_sample(True, "max_iteration", False, None)
        assert _should_mask_sample(True, "context_window", False, None)
        assert not _should_mask_sample(True, "other", False, None)
        assert _should_mask_sample(False, None, True, None)
        assert _should_mask_sample(False, None, False, "eval_timeout")
        assert _should_mask_sample(False, None, False, "sandbox")
        assert not _should_mask_sample(False, None, False, "eval_error")

    def test_dataset_routes(self) -> None:
        assert _dataset_family("princeton-nlp/SWE-bench_Verified") == "swebench"
        assert _dataset_family("SWE-bench_Multilingual") == "swebench_multilingual"
        assert _dataset_family("R2E-Gym/R2E-Gym-Subset") == "r2e"

    def test_r2e_required_tests(self) -> None:
        instance = {
            "FAIL_TO_PASS": ["tests/test_a.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_b.py::test_stays_green"],
        }
        passing = "\n".join(
            [
                "PASSED tests/test_a.py::test_fix",
                "PASSED tests/test_b.py::test_stays_green",
            ]
        )
        failing = passing.replace("PASSED tests/test_a.py", "FAILED tests/test_a.py")
        skipped = passing.replace("PASSED tests/test_a.py", "SKIPPED tests/test_a.py")
        assert _r2e_resolved(instance, passing)
        assert not _r2e_resolved(instance, failing)
        assert not _r2e_resolved(instance, skipped)

    def test_safe_config_redacts_provider_key(self) -> None:
        class Params:
            def model_dump_json(self) -> str:
                return json.dumps(
                    {"sandbox_provider": {"opensandbox": {"api_key": "secret"}}}  # pragma: allowlist secret
                )

        assert json.loads(_safe_config_json(Params()))["sandbox_provider"]["opensandbox"]["api_key"] == "***"


class TestSetupScriptsExist:
    def test_supported_agents_have_deps_scripts(self) -> None:
        scripts = Path(__file__).parent.parent / "setup_scripts"
        assert (scripts / "hermes_agent_deps.sh").exists()
        assert (scripts / "claude_code_agent_deps.sh").exists()
        assert (scripts / "opencode_agent_deps.sh").exists()
        assert (scripts / "_portable_python.sh").exists()


class TestExampleData:
    def test_prepared_rows_do_not_set_sampling(self) -> None:
        row = _to_gym_row({"instance_id": "repo__repo-1", "problem_statement": "Fix it"}, "test")
        assert set(row["responses_create_params"]) == {"input", "metadata"}

    def test_example_jsonl_parses(self) -> None:
        example = Path(__file__).parent.parent / "data" / "example.jsonl"
        rows = [json.loads(line) for line in example.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            assert "metadata" in row["responses_create_params"]
            assert "instance_id" in row["responses_create_params"]["metadata"]
