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

import asyncio
import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from responses_api_agents.anyswe_agent.agent_runner import _extract_patch, _snapshot_repo
from responses_api_agents.anyswe_agent.app import (
    AnySweAgent,
    AnySweAgentConfig,
    AnySweRunRequest,
    SWEBenchMetrics,
    _classify_agent_error,
    _dataset_family,
    _model_url_for_rollout,
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


class TestAgentRunner:
    @staticmethod
    def _source() -> str:
        return (Path(__file__).parent.parent / "agent_runner.py").read_text()

    def test_is_valid_python(self) -> None:
        source = self._source()
        compile(source, "<runner>", "exec")
        assert 'os.environ["NGSWE_AGENT_MODULE"]' in source
        assert '["git", "add", "-A"]' in source
        assert '["git", "diff", "--no-color", "--cached", baseline_tree]' in source

    def test_patch_extraction_excludes_image_dirt_and_includes_agent_files(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "agent-edited.txt").write_text("committed\n")
        (repo / "image-dirty.txt").write_text("committed\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

        (repo / "agent-edited.txt").write_text("image baseline\n")
        (repo / "image-dirty.txt").write_text("pre-existing task image change\n")
        (repo / "image-untracked.txt").write_text("pre-existing untracked file\n")
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)

        (repo / "agent-edited.txt").write_text("agent change\n")
        (repo / "agent-new.txt").write_text("new from agent\n")
        patch_text = _extract_patch(repo, index_path, baseline_tree)

        assert "agent-edited.txt" in patch_text
        assert "agent change" in patch_text
        assert "agent-new.txt" in patch_text
        assert "new from agent" in patch_text
        assert "image-dirty.txt" not in patch_text
        assert "image-untracked.txt" not in patch_text

    def test_sampling_is_forwarded(self) -> None:
        source = self._source()
        assert "NGSWE_SAMPLING" in source
        assert "**sampling," in source
        assert "**{**agent_kwargs, **config_sampling}" in source
        assert "config_class.model_fields" in source
        assert 'agent_kwargs["model"] = model_name' not in source


class TestSandboxAPI:
    def test_run_request_preserves_rollout_indices(self) -> None:
        request = AnySweRunRequest.model_validate(
            {
                "responses_create_params": {"input": [], "model": "model"},
                "_ng_task_index": 3,
                "_ng_rollout_index": 1,
            }
        )
        assert getattr(request, "_ng_task_index") == 3
        assert getattr(request, "_ng_rollout_index") == 1

    def test_model_url_carries_rollout_correlation(self) -> None:
        assert _model_url_for_rollout("http://model-host:8000", "3-1") == "http://model-host:8000/ng-rollout/3-1"
        assert _model_url_for_rollout("http://model-host:8000", None) == "http://model-host:8000"

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
        assert config.agent_runtime_source == "baked"

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

    def test_agent_config_is_forwarded_without_harness_specific_changes(self) -> None:
        params = SimpleNamespace(
            body=SimpleNamespace(model="model", temperature=1.0, top_p=0.95, max_output_tokens=None),
            agent_kwargs={"model": "configured-model", "custom": {"enabled": True}},
            model_server_url="http://model-host:8000/v1",
            agent_server_module="example.agent",
            agent_server_class="ExampleAgent",
            agent_config_class="ExampleAgentConfig",
        )
        env = AnySweAgent._sandbox_agent_env(params)
        kwargs = json.loads(base64.b64decode(env["NGSWE_AGENT_KWARGS_B64"]))
        assert env["NGSWE_MODEL_NAME"] == "model"
        assert kwargs == params.agent_kwargs

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
                    {
                        "sandbox_provider": {"opensandbox": {"api_key": "secret"}},  # pragma: allowlist secret
                        "agent_runtime_source": "https://example.test/runtime?token=secret",
                        "agent_deps_url": "https://example.test/runtime?token=secret",
                    }
                )

        result = json.loads(_safe_config_json(Params()))
        assert result["sandbox_provider"]["opensandbox"]["api_key"] == "***"
        assert "agent_runtime_source" not in result
        assert "agent_deps_url" not in result


class TestSetupScriptsExist:
    def test_supported_agents_have_deps_scripts(self) -> None:
        scripts = Path(__file__).parent.parent / "setup_scripts"
        assert (scripts / "hermes_agent_deps.sh").exists()
        assert (scripts / "claude_code_agent_deps.sh").exists()
        assert (scripts / "cline_agent_deps.sh").exists()
        assert (scripts / "opencode_agent_deps.sh").exists()
        assert (scripts / "openclaw_agent_deps.sh").exists()
        assert (scripts / "pi_agent_deps.sh").exists()
        assert (scripts / "_portable_python.sh").exists()

    def test_portable_python_meets_project_minimum(self) -> None:
        script = (Path(__file__).parent.parent / "setup_scripts" / "_portable_python.sh").read_text()
        assert 'PYTHON_VERSION="${PYTHON_VERSION:-3.13.14}"' in script
        assert 'PBS_RELEASE="${PBS_RELEASE:-20260805}"' in script


def _fake_bin(tmp_path: Path, machine: str) -> Path:
    """A PATH dir whose `uname -m` reports ``machine`` and whose `curl` logs the URL and fails."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uname").write_text(f'#!/bin/sh\necho "{machine}"\n')
    (bin_dir / "curl").write_text(f'#!/bin/sh\nfor a; do url="$a"; done\necho "$url" >> "{tmp_path}/urls"\nexit 1\n')
    for tool in ("uname", "curl"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir


def _portable_python_arch(
    tmp_path: Path, machine: str, arch_override: str | None = None
) -> subprocess.CompletedProcess:
    script = Path(__file__).parent.parent / "setup_scripts" / "_portable_python.sh"
    env = {**os.environ, "PATH": f"{_fake_bin(tmp_path, machine)}:{os.environ['PATH']}"}
    env.pop("ARCH", None)
    if arch_override:
        env["ARCH"] = arch_override
    return subprocess.run(
        ["bash", "-c", f'source "{script}" && echo "$ARCH"'], env=env, capture_output=True, text=True
    )


class TestSetupScriptsHostArch:
    @pytest.mark.parametrize(
        "machine,expected",
        [
            ("x86_64", "x86_64-unknown-linux-gnu"),
            ("aarch64", "aarch64-unknown-linux-gnu"),
            ("arm64", "aarch64-unknown-linux-gnu"),
        ],
    )
    def test_portable_python_arch_follows_host(self, tmp_path: Path, machine: str, expected: str) -> None:
        result = _portable_python_arch(tmp_path, machine)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected

    def test_portable_python_arch_override_wins(self, tmp_path: Path) -> None:
        result = _portable_python_arch(tmp_path, "aarch64", arch_override="x86_64-unknown-linux-gnu")
        assert result.stdout.strip() == "x86_64-unknown-linux-gnu"

    def test_portable_python_rejects_unknown_arch(self, tmp_path: Path) -> None:
        result = _portable_python_arch(tmp_path, "riscv64")
        assert result.returncode != 0
        assert "unsupported portable python architecture" in result.stderr

    @pytest.mark.parametrize(
        "deps_script",
        [
            "claude_code_agent_deps.sh",
            "cline_agent_deps.sh",
            "opencode_agent_deps.sh",
            "openclaw_agent_deps.sh",
            "pi_agent_deps.sh",
        ],
    )
    @pytest.mark.parametrize("machine,node_arch", [("x86_64", "linux-x64"), ("aarch64", "linux-arm64")])
    def test_node_download_follows_host(self, tmp_path: Path, deps_script: str, machine: str, node_arch: str) -> None:
        # Run the deps script with the portable-python helpers stubbed out; the fake curl records the Node URL.
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        shutil.copy(Path(__file__).parent.parent / "setup_scripts" / deps_script, scripts / deps_script)
        (scripts / "_portable_python.sh").write_text(
            "install_portable_python() { :; }\ninstall_nemo_gym_deps() { :; }\n"
        )
        deps_dir = tmp_path / "deps"
        deps_dir.mkdir()
        env = {
            **os.environ,
            "PATH": f"{_fake_bin(tmp_path, machine)}:{os.environ['PATH']}",
            "DEPS_DIR": str(deps_dir),
            "NEMO_GYM_ROOT": str(tmp_path),
        }
        env.pop("NODE_ARCH", None)

        result = subprocess.run(["bash", str(scripts / deps_script)], env=env, capture_output=True, text=True)

        assert result.returncode != 0  # the fake curl fails right after recording the URL
        urls = (tmp_path / "urls").read_text().split()
        assert len(urls) == 1 and urls[0].startswith("https://nodejs.org/dist/")
        assert urls[0].endswith(f"-{node_arch}.tar.xz")


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


class TestRunReportsMaskSample:
    """`/run` must build a serializable response and carry the masking decision."""

    def _agent(self, *, mask_sample: bool) -> AnySweAgent:
        metrics = SWEBenchMetrics(resolved=True, mask_sample=mask_sample)
        instance_config = {
            "problem_info": {"instance_id": "astropy__astropy-12907"},
            "body": {"input": "fix it"},
            "persistent_dir": "/tmp/anyswe",
            "metrics_fpath": "/tmp/anyswe/metrics.json",
            "container": "swebench/sweb.eval.x86_64.astropy__astropy-12907",
            "run_session_id": "run-1",
            "base_results_dir": "/tmp/anyswe/results",
            "model_server_url": "http://localhost:9000",
            "resolved_sandbox_provider": {"opensandbox": {}},
            "sandbox_config": {},
            "sandbox_default_metadata": {},
            **_config().model_dump(mode="json"),
        }
        response = NeMoGymResponse.model_construct(
            id="resp-1",
            output=[],
            tools=[],
            metadata={
                "metrics": metrics.model_dump_json(),
                "instance_config": json.dumps(instance_config),
                "input": json.dumps("fix it"),
            },
        )

        class _StubbedAgent(AnySweAgent):
            async def _responses(self, *args, **kwargs):
                return response

            def rollout_id_from_run(self, body):
                return None

        agent = _StubbedAgent.__new__(_StubbedAgent)
        object.__setattr__(agent, "__dict__", {"_sem": asyncio.Semaphore(1)})
        object.__setattr__(agent, "__pydantic_fields_set__", set())
        object.__setattr__(agent, "__pydantic_extra__", None)
        object.__setattr__(agent, "__pydantic_private__", {})
        return agent

    def _body(self) -> AnySweRunRequest:
        return AnySweRunRequest(responses_create_params={"input": "fix it"})

    def test_a_masked_run_serializes_and_reports_the_flag_once(self) -> None:
        dumped = asyncio.run(self._agent(mask_sample=True).run(self._body())).model_dump()

        assert dumped["mask_sample"] is True
        assert dumped["reward"] == 1.0
        # The compatibility location keeps reporting the same decision.
        assert dumped["instance_config"]["mask_sample"] is True

    def test_an_unmasked_run_is_not_implicitly_masked(self) -> None:
        dumped = asyncio.run(self._agent(mask_sample=False).run(self._body())).model_dump()

        assert dumped["mask_sample"] is False
        assert dumped["instance_config"]["mask_sample"] is False
