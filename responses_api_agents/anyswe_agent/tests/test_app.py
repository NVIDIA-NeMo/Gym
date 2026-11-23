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
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent, RequestFailureLog
from responses_api_agents.anyswe_agent import agent_runner
from responses_api_agents.anyswe_agent.agent_runner import _REPO_CANDIDATES, _extract_patch, _snapshot_repo
from responses_api_agents.anyswe_agent.app import (
    AnySweAgent,
    AnySweAgentConfig,
    AnySweRunRequest,
    AnySweServerConfig,
    SWEBenchMetrics,
    _classify_agent_error,
    _dataset_family,
    _masked_failure_kind,
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
        assert '["git", "diff", "--no-color", "--cached", baseline_tree, "--", ".", *excludes]' in source

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
        patch_text = _extract_patch(repo, index_path, baseline_tree).decode()

        assert "agent-edited.txt" in patch_text
        assert "agent change" in patch_text
        assert "agent-new.txt" in patch_text
        assert "new from agent" in patch_text
        assert "image-dirty.txt" not in patch_text
        assert "image-untracked.txt" not in patch_text

    def test_patch_is_written_when_the_harness_raises(self, tmp_path, monkeypatch) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "i"],
            cwd=repo,
            check=True,
        )
        mount = tmp_path / "mount"
        mount.mkdir()
        (mount / "instruction.txt").write_text("Fix it")

        class FailingAgent:
            def __init__(self, config, server_client) -> None:
                pass

            async def responses(self, request, body):
                (repo / "fix.py").write_text("partial fix\n")
                raise RuntimeError("max_turns reached")

        class Config:
            model_fields = {}

            def __init__(self, **kwargs) -> None:
                pass

        module = SimpleNamespace(FailingAgent=FailingAgent, Config=Config)
        monkeypatch.setitem(sys.modules, "fake_harness", module)
        monkeypatch.setenv("NGSWE_MODEL_NAME", "policy")
        monkeypatch.setenv("NGSWE_AGENT_MODULE", "fake_harness")
        monkeypatch.setenv("NGSWE_AGENT_CLASS", "FailingAgent")
        monkeypatch.setenv("NGSWE_AGENT_CONFIG_CLASS", "Config")
        monkeypatch.setattr(agent_runner, "_find_repo", lambda: repo)
        monkeypatch.setattr(
            agent_runner, "Path", lambda path: Path(str(path).replace("/trajectories_mount", str(mount)))
        )

        with pytest.raises(RuntimeError, match="max_turns"):
            agent_runner.main()

        assert "partial fix" in (mount / "patch.diff").read_text()
        assert not (mount / "response.json").exists()

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
        # No rollout id and no exported traces: byte-for-byte unchanged.
        assert _model_url_for_rollout("http://model-host:8000", None) == "http://model-host:8000"
        assert _model_url_for_rollout("", None) == ""

    def test_sandbox_model_url_keeps_the_rollout_path(self, tmp_path) -> None:
        # The relay reaches the same model server, which attributes the in-box calls by this path.
        with patch.object(AnySweAgent, "model_post_init", lambda self, context: None):
            agent = AnySweAgent(
                config=_config(sandbox_model_base_url="http://relay:9000"), server_client=MagicMock(spec=ServerClient)
            )
        agent._server = AnySweServerConfig(
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://relay:9000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
        )
        body = NeMoGymResponseCreateParamsNonStreaming(input=[], metadata={"instance_id": "repo__repo-1"})

        assert agent._setup_params(body, "3-1").model_server_url == "http://relay:9000/ng-rollout/3-1"

    def test_model_url_carries_trace_context_without_a_rollout_id(self, monkeypatch) -> None:
        # Capture off, traces on: the harness still gets the current span in the URL.
        from nemo_gym.client import server_client

        monkeypatch.setattr(
            server_client, "trace_context_path_segment", lambda: "/ng-trace/00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        )
        assert (
            _model_url_for_rollout("http://model-host:8000", None)
            == "http://model-host:8000/ng-trace/00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        )
        assert (
            _model_url_for_rollout("http://model-host:8000", "3-1")
            == "http://model-host:8000/ng-rollout/3-1/ng-trace/00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        )
        assert _model_url_for_rollout("", None) == ""

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
        assert config.sanitize_repo_git is True

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

    def test_agent_network_policy_applies_to_agent_box_only(self) -> None:
        policy = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "10.0.0.0/16"}]}
        params = SimpleNamespace(
            sandbox_spec={"provider_options": {"platform": {"arch": "amd64"}}},
            sandbox_default_metadata={},
            swebench_agent_timeout=100,
            swebench_tests_timeout=200,
            instance_id="astropy__astropy-12907",
            container="image:tag",
            resolved_sandbox_provider={"opensandbox": {}},
            agent_network_policy=policy,
        )
        agent_spec = AnySweAgent._sandbox_spec(params, agent=True)
        grading_spec = AnySweAgent._sandbox_spec(params)
        assert agent_spec.provider_options == {"platform": {"arch": "amd64"}, "network_policy": policy}
        assert grading_spec.provider_options == {"platform": {"arch": "amd64"}}

        params.agent_network_policy = policy
        params.resolved_sandbox_provider = {"docker": {}}
        with pytest.raises(ValueError, match="opensandbox"):
            AnySweAgent._sandbox_spec(params, agent=True)

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

    @pytest.mark.parametrize(
        "args,kind",
        [
            ((False, None, False, "sandbox"), "provider_unavailable"),
            ((False, None, False, "sandbox", "provider_command_outcome_unknown"), "provider_command_outcome_unknown"),
            ((False, None, True, None), "agent_timeout"),
            ((False, None, False, "eval_timeout"), "anyswe_agent:eval_timeout"),
            ((True, "max_iteration", False, None), "agent_limit"),
            ((True, "context_window", False, None), "model_context_limit"),
            ((False, None, False, None), None),
            ((True, "other", False, None), None),
            ((False, "max_iteration", False, "eval_error"), None),
        ],
    )
    def test_a_masked_sample_names_its_failure_kind(self, args, kind) -> None:
        assert _masked_failure_kind(*args) == kind
        # A kind is named exactly when the sample is masked.
        assert (kind is not None) is _should_mask_sample(*args[:4])

    def test_dataset_routes(self) -> None:
        assert _dataset_family("princeton-nlp/SWE-bench_Verified") == "swebench"
        assert _dataset_family("SWE-bench_Multilingual") == "swebench_multilingual"
        assert _dataset_family("R2E-Gym/R2E-Gym-Subset") == "r2e"

    @pytest.mark.parametrize("colored", [False, True])
    def test_r2e_required_tests(self, colored: bool) -> None:
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
        if colored:
            passing = passing.replace("PASSED", "\x1b[32mPASSED\x1b[0m")
            passing = passing.replace("::test_fix", "::\x1b[1mtest_fix\x1b[0m")
            passing = passing.replace("::test_stays_green", "::\x1b[1mtest_stays_green\x1b[0m")
        assert _r2e_resolved(instance, passing)
        for status in ("FAILED", "ERROR", "SKIPPED"):
            nonpassing = passing.replace("PASSED", status, 1)
            assert not _r2e_resolved(instance, nonpassing)
        assert _r2e_resolved(instance, passing.replace("PASSED", "XFAIL", 1))
        assert not _r2e_resolved(instance, passing.splitlines()[0])

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


_SANITIZE_SCRIPT = Path(__file__).parent.parent / "setup_scripts" / "sanitize_repo_git.sh"


def _git_env(tmp_path: Path) -> dict[str, str]:
    # The script registers the repo as a safe.directory in the global git config.
    return os.environ | {
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": str(tmp_path / ".gitconfig"),
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }


def _git(repo: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout.strip()


def _has_object(repo: Path, env: dict[str, str], sha: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", sha], cwd=repo, env=env, capture_output=True).returncode == 0


def _repo_with_future_fix(repo: Path, env: dict[str, str]) -> tuple[str, str]:
    """main has a base commit and a later fix commit, HEAD at the fix."""
    repo.mkdir()
    _git(repo, env, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("bug\n")
    _git(repo, env, "add", "-A")
    _git(repo, env, "commit", "-qm", "base")
    base = _git(repo, env, "rev-parse", "HEAD")
    (repo / "a.py").write_text("fix\n")
    _git(repo, env, "commit", "-qam", "upstream fix")
    return base, _git(repo, env, "rev-parse", "HEAD")


def _sanitize(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    # The agent runs the script as root through `su -s /bin/sh`, so it has to stay POSIX sh.
    return subprocess.run(["sh", str(_SANITIZE_SCRIPT), str(repo)], env=env, capture_output=True, text=True)


class TestSetupScriptsExist:
    def test_sanitize_repo_git_keeps_only_head_ancestry(self, tmp_path) -> None:
        env, repo = _git_env(tmp_path), tmp_path / "repo"
        base, fix = _repo_with_future_fix(repo, env)
        _git(repo, env, "tag", "v1.0")
        _git(repo, env, "remote", "add", "origin", "https://example.com/repo.git")
        _git(repo, env, "checkout", "-q", base)

        assert _sanitize(repo, env).returncode == 0
        assert _git(repo, env, "rev-parse", "HEAD") == base
        assert _git(repo, env, "log", "--all", "--format=%H").split() == [base]
        assert _git(repo, env, "remote") == ""
        assert not _has_object(repo, env, fix)

    def test_sanitize_repo_git_keeps_the_branch_next_to_a_same_named_tag(self, tmp_path) -> None:
        env, repo = _git_env(tmp_path), tmp_path / "repo"
        base, fix = _repo_with_future_fix(repo, env)
        _git(repo, env, "tag", "main")
        _git(repo, env, "reset", "-q", "--hard", base)

        assert _sanitize(repo, env).returncode == 0
        assert _git(repo, env, "symbolic-ref", "HEAD") == "refs/heads/main"
        assert _git(repo, env, "rev-parse", "HEAD") == base
        assert _git(repo, env, "for-each-ref", "--format=%(refname)") == "refs/heads/main"
        assert not _has_object(repo, env, fix)

    def test_sanitize_repo_git_copies_the_objects_a_shared_clone_borrows(self, tmp_path) -> None:
        env, source, clone = _git_env(tmp_path), tmp_path / "source", tmp_path / "clone"
        base, fix = _repo_with_future_fix(source, env)
        subprocess.run(["git", "clone", "-q", "--shared", str(source), str(clone)], env=env, check=True)
        _git(clone, env, "checkout", "-q", base)

        assert _sanitize(clone, env).returncode == 0
        assert _git(clone, env, "cat-file", "-p", "HEAD:a.py") == "bug"
        assert not (clone / ".git" / "objects" / "info" / "alternates").exists()
        assert not _has_object(clone, env, fix)
        assert _has_object(source, env, fix)

    def test_sanitize_repo_git_leaves_no_dangling_symref_after_a_plain_clone(self, tmp_path) -> None:
        # A plain clone writes refs/remotes/origin/HEAD as a symref to origin/main. Deleting it must remove the
        # symref itself, not only its target, or git fsck reports an invalid pointer.
        env, source, clone = _git_env(tmp_path), tmp_path / "source", tmp_path / "clone"
        base, _ = _repo_with_future_fix(source, env)
        subprocess.run(["git", "clone", "-q", str(source), str(clone)], env=env, check=True)
        assert _git(clone, env, "symbolic-ref", "refs/remotes/origin/HEAD") == "refs/remotes/origin/main"
        _git(clone, env, "checkout", "-q", base)

        assert _sanitize(clone, env).returncode == 0
        assert not (clone / ".git" / "refs" / "remotes" / "origin" / "HEAD").exists()
        assert _git(clone, env, "for-each-ref", "--format=%(refname)") == ""
        fsck = subprocess.run(["git", "fsck"], cwd=clone, env=env, capture_output=True, text=True)
        assert fsck.returncode == 0, fsck.stderr

    def test_sanitize_repo_git_fails_when_a_ref_cannot_be_deleted(self, tmp_path) -> None:
        env, repo = _git_env(tmp_path), tmp_path / "repo"
        base, _ = _repo_with_future_fix(repo, env)
        _git(repo, env, "checkout", "-q", base)
        (repo / ".git" / "refs" / "heads" / "main.lock").write_text("")

        assert _sanitize(repo, env).returncode != 0

    def test_sanitize_repo_git_rejects_a_gitfile_checkout(self, tmp_path) -> None:
        env, repo = _git_env(tmp_path), tmp_path / "repo"
        subprocess.run(
            ["git", "init", "-q", "--separate-git-dir", str(tmp_path / "meta"), str(repo)], env=env, check=True
        )

        result = _sanitize(repo, env)
        assert result.returncode != 0
        assert "gitfile" in result.stderr

    def test_sanitize_repo_git_defaults_to_the_runner_repo_candidates(self) -> None:
        defaults = next(line for line in _SANITIZE_SCRIPT.read_text().splitlines() if "|| set -- " in line)
        assert tuple(defaults.split("|| set -- ")[1].split()) == _REPO_CANDIDATES

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


class TestSandboxGradingFailures:
    @pytest.mark.parametrize(
        "failure_kind,phase",
        [
            ("provider_unavailable", "grading"),
            ("provider_command_outcome_unknown", "grading"),
            ("model_transport_error", "grading"),
            (None, "grading"),
            ("provider_unavailable", "agent"),
            ("provider_command_outcome_unknown", "agent"),
        ],
    )
    async def test_grading_failure_is_excluded_and_preserves_evidence(
        self, tmp_path, monkeypatch, failure_kind, phase
    ) -> None:
        with patch.object(AnySweAgent, "model_post_init", lambda self, context: None):
            agent = AnySweAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        agent._server = AnySweServerConfig(
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://model:9000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
        )
        agent._sem = asyncio.Semaphore(1)
        monkeypatch.setattr(AnySweAgent, "rollout_id_from_run", lambda self, body: None)

        class Sandbox:
            async def start(self):
                pass

            async def stop(self):
                pass

            async def exec(self, command, **kwargs):
                if phase == "agent" and "agent_runner.py" in command:
                    raise error
                return SimpleNamespace(
                    return_code=1 if command == "test -f /trajectories_mount/response.json" else 0,
                    error_type=None,
                    stdout="",
                    stderr="",
                )

            async def download(self, remote, local):
                local.write_bytes(b"diff --git a/test.py b/test.py\n")

        monkeypatch.setattr("responses_api_agents.anyswe_agent.app.AsyncSandbox", lambda *args: Sandbox())
        error = None
        if failure_kind is not None:
            cause = ConnectionError("grading sandbox disappeared")
            error = NgFailureEvent(
                boundary="model" if failure_kind == "model_transport_error" else "sandbox",
                failure_kind=failure_kind,
                operation="exec",
                failure_reason="Grading failed",
            )
            error.__cause__ = cause
        grade = AsyncMock(side_effect=error if phase == "grading" else None, return_value=(False, None))
        monkeypatch.setattr(AnySweAgent, "_grade_sandbox_patch", grade)
        body = AnySweRunRequest(
            responses_create_params={
                "input": "fix it",
                "metadata": {"instance_id": "repo__repo-1"},
            }
        )
        if failure_kind == "model_transport_error":
            with pytest.raises(NgFailureEvent) as caught:
                await agent.run(body)
            assert caught.value is error
            return
        with RequestFailureLog.capture() as log:
            response = await agent.run(body)
        if phase == "agent":
            grade.assert_not_awaited()
        else:
            grade.assert_awaited_once()
        assert response.reward == 0.0
        assert response.mask_sample is (error is not None)
        assert response.error_kind == ("sandbox" if error is not None else None)
        assert response.instance_config["mask_sample"] is response.mask_sample
        assert response.failure_kind == (failure_kind if error is not None else None)
        assert (response.failure_reason is not None) is (error is not None)
        if error is not None:
            event = log.report.events[error.event_id]
            assert event.failure_kind == failure_kind
            assert event.foreign_error.exception_type == "ConnectionError"
            assert error.__cause__ is cause
        else:
            assert not log.report.events


class TestReturnedGradingFailures:
    @pytest.mark.parametrize(
        "returned,failure_kind,reason_part",
        [
            ("eval_timeout", "anyswe_agent:eval_timeout", "timed out"),
            ("sandbox", "provider_unavailable", "sandbox"),
        ],
    )
    async def test_returned_grading_failure_has_an_evaluation_reason(
        self, tmp_path, monkeypatch, returned, failure_kind, reason_part
    ) -> None:
        with patch.object(AnySweAgent, "model_post_init", lambda self, context: None):
            agent = AnySweAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        agent._server = AnySweServerConfig(
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://model:9000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
        )
        agent._sem = asyncio.Semaphore(1)
        monkeypatch.setattr(AnySweAgent, "rollout_id_from_run", lambda self, body: None)

        class Sandbox:
            async def start(self):
                pass

            async def stop(self):
                pass

            async def exec(self, command, **kwargs):
                return SimpleNamespace(
                    return_code=1 if command == "test -f /trajectories_mount/response.json" else 0,
                    error_type=None,
                    stdout="",
                    stderr="",
                )

            async def download(self, remote, local):
                local.write_bytes(b"diff --git a/test.py b/test.py\n")

        monkeypatch.setattr("responses_api_agents.anyswe_agent.app.AsyncSandbox", lambda *args: Sandbox())
        monkeypatch.setattr(AnySweAgent, "_grade_sandbox_patch", AsyncMock(return_value=(False, returned)))
        body = AnySweRunRequest(
            responses_create_params={"input": "fix it", "metadata": {"instance_id": "repo__repo-1"}}
        )

        response = await agent.run(body)

        assert response.mask_sample is True
        assert response.failure_kind == failure_kind
        assert reason_part in response.failure_reason
        assert "resolved" not in response.failure_reason


class TestRuntimeSetupFailures:
    """A setup command that could not run keeps a provider kind; one that completed and failed is a setup failure."""

    @pytest.mark.parametrize(
        "archive,failing,result,failure_kind,reason_part",
        [
            (False, "test -x", dict(error_type="timeout"), "provider_unavailable", "timeout"),
            (False, "test -x", dict(error_type="sandbox"), "provider_unavailable", "sandbox"),
            (False, "test -x", dict(return_code=1), "anyswe_agent:runtime_missing", "does not contain"),
            (True, "tar -xzf", dict(error_type="sandbox"), "provider_unavailable", "sandbox"),
            (True, "tar -xzf", dict(error_type="timeout"), "provider_unavailable", "timeout"),
            (
                True,
                "tar -xzf",
                dict(return_code=2, stderr="gzip: stdin: not in gzip format"),
                "anyswe_agent:setup_failed",
                "not in gzip format",
            ),
        ],
    )
    async def test_runtime_setup_command_outcome(
        self, tmp_path, monkeypatch, archive, failing, result, failure_kind, reason_part
    ) -> None:
        with patch.object(AnySweAgent, "model_post_init", lambda self, context: None):
            agent = AnySweAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        agent._server = AnySweServerConfig(
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://model:9000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
            agent_deps_archive=tmp_path / "deps.tar.gz" if archive else None,
        )
        agent._sem = asyncio.Semaphore(1)
        monkeypatch.setattr(AnySweAgent, "rollout_id_from_run", lambda self, body: None)
        failing_result = SimpleNamespace(
            **{"return_code": 0, "error_type": None, "stdout": "", "stderr": "", **result}
        )

        class Sandbox:
            async def start(self):
                pass

            async def stop(self):
                pass

            async def upload(self, local, remote):
                pass

            async def exec(self, command, **kwargs):
                if failing in command:
                    return failing_result
                return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

        monkeypatch.setattr("responses_api_agents.anyswe_agent.app.AsyncSandbox", lambda *args: Sandbox())
        body = AnySweRunRequest(
            responses_create_params={"input": "fix it", "metadata": {"instance_id": "repo__repo-1"}}
        )

        response = await agent.run(body)

        assert response.reward == 0.0
        assert response.mask_sample is True
        assert response.failure_kind == failure_kind
        assert reason_part in response.failure_reason
        if failure_kind == "provider_unavailable":
            assert "does not contain" not in response.failure_reason
