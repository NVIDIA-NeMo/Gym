# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from responses_api_agents.deep_swe import app
from responses_api_agents.deep_swe.app import (
    DeepSWEAgent,
    DeepSWEAgentConfig,
    DeepSWERunRequest,
)


def _config(**overrides: Any) -> DeepSWEAgentConfig:
    values: dict[str, Any] = {
        "name": "deep_swe",
        "host": "127.0.0.1",
        "port": 9000,
        "entrypoint": "app.py",
        "model_base_url": "https://model.example",
        "model_api_key": "secret-value",  # pragma: allowlist secret
        "model_name": "test-model",
        "sandbox_provider": {"fake": {}},
        "benchmark_expected_task_count": 1,
    }
    values.update(overrides)
    return DeepSWEAgentConfig(**values)


def _agent(**overrides: Any) -> DeepSWEAgent:
    config = _config(**overrides)
    server = DeepSWEAgent.model_construct(config=config, server_client=MagicMock())
    server._sem = asyncio.Semaphore(config.max_concurrent)
    server._assembly_sem = asyncio.Semaphore(config.max_concurrent_assembly)
    server._checkout_lock = asyncio.Lock()
    server._checkout_cache = None
    server._gym_source_provenance = {
        "repository_url": "https://github.com/NVIDIA-NeMo/Gym",
        "commit": "1" * 40,
        "uv_lock_sha256": "2" * 64,
        "working_tree_clean": True,
    }
    return server


def _request(task_id: str = "task-one") -> DeepSWERunRequest:
    return DeepSWERunRequest(
        responses_create_params={"input": []},
        verifier_metadata={"task_id": task_id},
    )


def test_config_requires_exactly_one_environment() -> None:
    assert _config().sandbox_transfer_timeout_s == 1800
    with pytest.raises(ValidationError, match="sandbox_provider"):
        _config(sandbox_provider=None)
    with pytest.raises(ValidationError, match="DeepSWE requires sandbox_provider"):
        _config(sandbox_provider={})
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _config(max_concurrent=0)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _config(max_concurrent_assembly=0)
    with pytest.raises(ValidationError, match="artifact_max_file_bytes cannot exceed"):
        _config(artifact_max_file_bytes=9, artifact_max_total_bytes=8, trajectory_max_bytes=8, patch_max_bytes=8)
    with pytest.raises(ValidationError, match="trajectory_max_bytes cannot exceed"):
        _config(artifact_max_file_bytes=8, artifact_max_total_bytes=8, trajectory_max_bytes=9, patch_max_bytes=8)
    with pytest.raises(ValidationError, match="patch_max_bytes cannot exceed"):
        _config(artifact_max_file_bytes=8, artifact_max_total_bytes=8, trajectory_max_bytes=8, patch_max_bytes=9)
    with pytest.raises(ValidationError, match="claude_code_env accepts non-secret values only"):
        _config(claude_code_env={"EXTRA_API_TOKEN": "must-not-persist"})
    with pytest.raises(ValidationError, match="greater than 0"):
        _config(sandbox_transfer_timeout_s=0)
    with pytest.raises(ValidationError, match="finite number"):
        _config(sandbox_transfer_timeout_s=float("inf"))
    with pytest.raises(ValidationError, match="String should match pattern"):
        _config(claude_code_version="latest")


def test_gym_source_provenance_binds_clean_commit_remote_and_lock(tmp_path: Path) -> None:
    root = tmp_path / "gym"
    root.mkdir()
    lock_bytes = b"version = 1\n"
    (root / "uv.lock").write_bytes(lock_bytes)

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    git("init")
    git("config", "user.name", "DeepSWE test")
    git("config", "user.email", "deep-swe@example.invalid")
    git("add", "uv.lock")
    git("commit", "-m", "test fixture")
    git(
        "remote",
        "add",
        "origin",
        "https://user:token@github.com/NVIDIA-NeMo/Gym.git?access_token=secret#fragment",  # pragma: allowlist secret
    )

    provenance = app._gym_source_provenance(root)
    assert provenance == {
        "repository_url": "https://github.com/NVIDIA-NeMo/Gym",
        "commit": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "uv_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "working_tree_clean": True,
    }

    (root / "uv.lock").write_bytes(lock_bytes + b"dirty = true\n")
    assert app._gym_source_provenance(root)["working_tree_clean"] is False
    git("checkout", "--", "uv.lock")
    (root / "untracked.py").write_text("raise RuntimeError('untracked source')\n")
    assert app._gym_source_provenance(root)["working_tree_clean"] is False

    assert app._gym_source_provenance(tmp_path / "missing") == {
        "repository_url": None,
        "commit": None,
        "uv_lock_sha256": None,
        "working_tree_clean": None,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "ssh://user:secret@example.com:2222/NVIDIA-NeMo/Gym.git?token=secret#fragment",  # pragma: allowlist secret
            "https://example.com:2222/NVIDIA-NeMo/Gym",
        ),
        ("git+ssh://token@example.com/org/repo.git?credential=secret", "https://example.com/org/repo"),
        ("git@example.com:org/repo.git", "https://example.com/org/repo"),
        ("https://github.com/NVIDIA-NeMo/Gym.git", "https://github.com/NVIDIA-NeMo/Gym"),
        ("git@github.com:NVIDIA-NeMo/Gym.git", "https://github.com/NVIDIA-NeMo/Gym"),
        ("file:///private/secret/repo", None),
        ("ssh://user:secret@example.com:not-a-port/repo", None),  # pragma: allowlist secret
    ],
)
def test_repository_url_provenance_strips_credentials_and_rejects_local_paths(
    value: str,
    expected: str | None,
) -> None:
    assert app._safe_repository_url(value) == expected


def _redact_chunks(chunks: list[bytes], secrets: tuple[bytes, ...]) -> bytes:
    replacements = tuple((secret, b"<redacted>") for secret in secrets)
    output = bytearray()
    carry = b""
    for chunk in chunks:
        redacted, carry, _ = app._redact_stream_buffer(carry + chunk, replacements, final=False)
        output.extend(redacted)
    redacted, carry, _ = app._redact_stream_buffer(carry, replacements, final=True)
    output.extend(redacted)
    assert carry == b""
    return bytes(output)


@pytest.mark.parametrize(
    ("chunks", "secrets", "expected"),
    [
        ([b"tail-abca"], (b"abca",), b"tail-<redacted>"),
        ([b"prefix-abcda", b"-suffix"], (b"abcda",), b"prefix-<redacted>-suffix"),
        ([bytes([value]) for value in b"xabcay"], (b"abca",), b"x<redacted>y"),
        ([b"token-", b"long token"], (b"token", b"token-long"), b"<redacted> <redacted>"),
        ([b"ab", b"aba"], (b"aba", b"ababa"), b"<redacted>"),
    ],
)
def test_stream_redaction_handles_self_overlap_boundaries_and_overlapping_sets(
    chunks: list[bytes],
    secrets: tuple[bytes, ...],
    expected: bytes,
) -> None:
    assert _redact_chunks(chunks, secrets) == expected


def test_stream_redaction_without_secrets_passes_through() -> None:
    assert app._redact_stream_buffer(b"unchanged", (), final=False) == (b"unchanged", b"", False)


def test_text_and_mapping_redaction_prefer_longest_secret_and_sanitize_keys() -> None:
    secrets = ("abc", "abcdef")
    assert app._redact_text("abcdef", secrets) == "<redacted>"
    assert app._sanitize_persisted_value({"abcdef": "abcdef"}, secrets) == {"<redacted>": "<redacted>"}


def test_agent_config_and_trial_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _agent(
        claude_code_kwargs={"reasoning_effort": "high"},
        claude_code_env={"EXTRA": "value", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
    )
    inherited_behavior = {
        "ANTHROPIC_BASE_URL": "https://host-override.example",
        "ANTHROPIC_MODEL": "host-model",
        "AWS_PROFILE": "host-bedrock-profile",
        "AWS_REGION": "host-bedrock-region",
        "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "0",
        "CLAUDE_CODE_EFFORT_LEVEL": "low",
        "CLAUDE_CODE_MAX_TURNS": "1",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CONFIG_DIR": "/host/claude",
        "DISABLE_PROMPT_CACHING": "1",
        "MAX_THINKING_TOKENS": "1",
        "PIER_VIEWER_MODE": "tasks",
    }
    for name, value in inherited_behavior.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("UNRELATED_API_KEY", "must-not-reach-pier")
    monkeypatch.setenv("MODAL_TOKEN_ID", "modal-token-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "modal-token-secret")
    monkeypatch.setenv("DEEPSWE_RUNTIME_MARKER", "preserved")
    monkeypatch.setenv("PATH", "/runtime/bin")
    monkeypatch.setenv("HOME", "/runtime/home")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example")
    config = server._agent_config()
    assert config["name"] == "claude-code"
    assert config["model_name"] == "test-model"
    assert config["kwargs"] == {"reasoning_effort": "high", "version": "2.1.153"}
    assert config["env"]["ANTHROPIC_AUTH_TOKEN"] == "${ANTHROPIC_AUTH_TOKEN}"
    assert config["env"]["ANTHROPIC_BASE_URL"] == "https://model.example"
    assert config["env"]["EXTRA"] == "value"
    assert config["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    process_env = server._pier_process_env()
    assert process_env["ANTHROPIC_AUTH_TOKEN"] == "secret-value"
    assert process_env["MODAL_TOKEN_ID"] == "modal-token-id"
    assert process_env["MODAL_TOKEN_SECRET"] == "modal-token-secret"  # pragma: allowlist secret
    assert process_env["DEEPSWE_RUNTIME_MARKER"] == "preserved"
    assert {name: process_env[name] for name in ("PATH", "HOME", "LANG", "HTTPS_PROXY")} == {
        "PATH": "/runtime/bin",
        "HOME": "/runtime/home",
        "LANG": "C.UTF-8",
        "HTTPS_PROXY": "https://proxy.example",
    }
    assert "UNRELATED_API_KEY" not in process_env
    assert inherited_behavior.keys().isdisjoint(process_env)
    assert {"secret-value", "modal-token-id", "modal-token-secret"} <= set(server._secret_variants())

    npm_config = _agent(claude_code_install_method="npm")._agent_config()
    assert npm_config["import_path"] == app.PIER_CLAUDE_CODE_NPM_AGENT
    assert "name" not in npm_config

    job_dir = Path("/tmp/job")
    assert app._trial_path((job_dir / "trial path").as_uri(), job_dir) == (job_dir / "trial path").resolve()
    with pytest.raises(ValueError, match="local file URI"):
        app._trial_path("https://example/trial", job_dir)
    with pytest.raises(ValueError, match="local file URI"):
        app._trial_path("file://remote/tmp/job/trial", job_dir)
    with pytest.raises(ValueError, match="escapes"):
        app._trial_path(Path("/tmp/outside").as_uri(), job_dir)


@pytest.mark.asyncio
async def test_sandbox_environment_config(monkeypatch: pytest.MonkeyPatch) -> None:
    global_config = {
        "sandbox": {
            "fake": {"credential": "hidden"},
            "default_metadata": {"team": "eval"},
        }
    }
    monkeypatch.setattr(
        app.ServerClient,
        "load_from_global_config",
        lambda: SimpleNamespace(global_config_dict=global_config),
    )
    server = _agent(
        sandbox_provider="sandbox",
        sandbox_supports_disable_internet=True,
        sandbox_supports_filtered_egress=True,
        sandbox_preinstall_agent_in_image=True,
        sandbox_transfer_timeout_s=17.5,
    )
    environment, runtime = server._environment_config()
    assert environment["import_path"] == app.PIER_SANDBOX_ENVIRONMENT
    assert runtime is not None
    assert runtime["provider"] == {"fake": {"credential": "hidden"}}
    assert runtime["provider_metadata"] == {"team": "eval"}
    assert runtime["preinstall_agent_in_image"] is True
    assert runtime["transfer_timeout_s"] == 17.5
    assert runtime["expected_agent_name"] == "claude-code"
    assert runtime["expected_agent_version"] == "2.1.153"

    server.config.sandbox_required_provider = "modal"
    with pytest.raises(ValueError, match="requires sandbox provider 'modal'"):
        server._environment_config()

    from nemo_gym.sandbox.providers.modal import ModalProvider

    preflight_calls: list[bool] = []
    monkeypatch.setattr(ModalProvider, "preflight", lambda: preflight_calls.append(True))
    global_config["sandbox"] = {"modal": {}, "default_metadata": {"team": "eval"}}
    environment, runtime = server._environment_config()
    assert environment["import_path"] == app.PIER_SANDBOX_ENVIRONMENT
    assert runtime is not None and runtime["provider"] == {"modal": {}}
    assert preflight_calls == [True]


@pytest.mark.asyncio
async def test_run_returns_structured_harness_error(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _agent()

    async def fail_checkout() -> Path:
        raise RuntimeError("checkout unavailable")

    monkeypatch.setattr(server, "_checkout", fail_checkout)
    response = await server.run(_request())
    assert response.reward == 0.0
    assert response.status == "harness_error"
    assert response.error_type == "RuntimeError"
    assert response.raw_rollout["trajectory"] is None


@pytest.mark.asyncio
async def test_run_rejects_missing_task_id() -> None:
    server = _agent()
    request = DeepSWERunRequest(responses_create_params={"input": []})
    with pytest.raises(ValueError, match="missing verifier_metadata.task_id"):
        await server.run(request)
    with pytest.raises(NotImplementedError):
        await server.responses()


def test_success_response_preserves_trajectory_patch_and_hashes(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "artifacts").mkdir()
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": "session",
        "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
        "final_metrics": {"total_prompt_tokens": 3, "total_completion_tokens": 1},
    }
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory))
    (trial_dir / "artifacts" / "model.patch").write_text("diff --git a/a b/a\n")
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": 1, "partial": 1.0}},
        "exception_info": None,
        "task_checksum": "checksum",
        "trial_name": "trial",
    }
    server = _agent()
    record = _request().model_dump(mode="json")

    response = server._success_response(
        record=record,
        task_id="task-one",
        instruction="Fix it",
        trial=trial,
        job_dir=tmp_path,
    )
    assert response.reward == 1.0
    assert response.raw_reward == 1.0
    assert response.status == "success"
    assert response.model_patch.startswith("diff --git")
    assert response.raw_rollout["format"] == "ATIF-v1.7"
    assert response.responses_create_params.input[0].content == "Fix it"
    assert {item["path"] for item in response.artifacts} == {
        "agent/trajectory.json",
        "artifacts/model.patch",
    }
    assert response.benchmark_metadata["task_checksum"] == "checksum"
    assert response.benchmark_metadata["pier_source_url"] == app.PIER_SOURCE_URL
    assert response.benchmark_metadata["pier_source_commit"] == app.PIER_SOURCE_COMMIT
    assert len(response.benchmark_metadata["pier_direct_url_sha256"]) == 64
    assert len(response.benchmark_metadata["pier_constraints_sha256"]) == 64
    assert response.benchmark_metadata["pier_runtime_modal_version"] == "1.5.1"
    assert response.benchmark_metadata["agent_install_method"] == "pier"
    assert response.benchmark_metadata["gym_source"] == server._gym_source_provenance
    assert response.benchmark_metadata["sandbox_runtime"]["schema_version"] == 2
    assert response.benchmark_metadata["sandbox_runtime"]["gym_source"] == server._gym_source_provenance
    assert response.benchmark_metadata["sandbox_runtime"]["pier_runtime"]["source_commit"] == app.PIER_SOURCE_COMMIT
    assert response.benchmark_metadata["sandbox_runtime"]["provider"] == "fake"
    assert response.benchmark_metadata["sandbox_runtime"]["transfer_timeout_s"] == 1800
    assert len(response.benchmark_metadata["sandbox_runtime"]["sha256"]) == 64


def test_success_response_classifies_trial_error_and_missing_trajectory(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": {
            "exception_type": "AgentTimeoutError",
            "exception_message": "timeout secret-value",
        },
        "task_checksum": "checksum",
    }
    server = _agent()
    response = server._success_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        instruction="Fix it",
        trial=trial,
        job_dir=tmp_path,
    )
    assert response.reward == 0.0
    assert response.raw_reward == 1.0
    assert response.status == "error"
    assert response.error_type == "AgentTimeoutError"
    assert response.error_message == "timeout <redacted>"
    assert response.raw_rollout["trajectory"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized", ["trajectory", "patch"])
async def test_run_rejects_oversized_inline_evidence_without_returning_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    oversized: str,
) -> None:
    server = _agent(
        artifact_max_file_bytes=1024,
        artifact_max_total_bytes=4096,
        trajectory_max_bytes=32 if oversized == "trajectory" else 512,
        patch_max_bytes=4 if oversized == "patch" else 512,
    )
    checkout = tmp_path / "checkout"
    task_dir = checkout / "tasks" / "task-one"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("")
    (task_dir / "instruction.md").write_text("Fix it")
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "artifacts").mkdir()
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [{"source": "agent", "message": "x" * 64}],
            }
        )
    )
    (trial_dir / "artifacts" / "model.patch").write_text("patch-data")
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
    }

    async def checkout_with_digest() -> tuple[Path, str]:
        return checkout, "digest"

    async def run_pier(**_: Any) -> tuple[dict[str, Any], Path]:
        return trial, job_dir

    monkeypatch.setattr(server, "_checkout_with_digest", checkout_with_digest)
    monkeypatch.setattr(server, "_run_pier", run_pier)
    response = await server.run(_request())

    assert response.status == "harness_error"
    assert response.error_type == "ArtifactLimitError"
    assert response.reward == 0.0
    assert response.pier_result is None
    assert response.verifier_result is None
    assert response.raw_rollout == {"format": None, "trajectory": None, "trajectory_path": None}
    assert response.model_patch is None
    assert response.artifacts == []


@pytest.mark.parametrize(
    ("trajectory_payload", "expected_error"),
    [
        (None, "MissingTrajectory"),
        ("{not-json", "InvalidTrajectory"),
        (json.dumps({"schema_version": "ATIF-v1.7", "steps": []}), "InvalidTrajectory"),
        (json.dumps({"schema_version": "ATIF-v1.7", "steps": "bad"}), "InvalidTrajectory"),
        (json.dumps({"schema_version": "ATIF-v1.7", "steps": ["bad"]}), "InvalidTrajectory"),
        (
            json.dumps(
                {
                    "schema_version": "ATIF-v1.7",
                    "steps": [{"source": "system", "message": "No agent execution occurred"}],
                }
            ),
            "InvalidTrajectory",
        ),
        (
            json.dumps(
                {
                    "schema_version": "ATIF-v1.7",
                    "steps": [{"source": "agent", "tool_calls": ["bad"]}],
                }
            ),
            "InvalidTrajectory",
        ),
    ],
)
def test_success_reward_requires_a_valid_captured_trajectory(
    tmp_path: Path,
    trajectory_payload: str | None,
    expected_error: str,
) -> None:
    trial_dir = tmp_path / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    if trajectory_payload is not None:
        (trial_dir / "agent" / "trajectory.json").write_text(trajectory_payload)
    response = _agent()._success_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        instruction="Fix it",
        trial={
            "trial_uri": trial_dir.as_uri(),
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
        },
        job_dir=tmp_path,
    )
    assert response.raw_reward == 1.0
    assert response.reward == 0.0
    assert response.status == "harness_error"
    assert response.error_type == expected_error


@pytest.mark.asyncio
async def test_run_pier_uses_isolated_runtime_and_private_launcher_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _agent(
        sandbox_provider={"fake": {"api_key": "provider-secret"}},  # pragma: allowlist secret
        work_root=str(tmp_path / "jobs"),
        pier_runtime_dir=str(tmp_path / "runtime"),
    )
    pier_path = tmp_path / "runtime" / "bin" / "pier"

    async def runtime(*_: Any) -> Path:
        return pier_path

    captured: dict[str, Any] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, None]:
            config_path = Path(captured["args"][captured["args"].index("--config") + 1])
            config = json.loads(config_path.read_text())
            captured["config"] = config
            runtime_path = Path(config["environment"]["kwargs"]["runtime_config_path"])
            captured["runtime_mode"] = runtime_path.stat().st_mode & 0o777
            captured["runtime"] = json.loads(runtime_path.read_text())
            job_dir = Path(config["jobs_dir"]) / config["job_name"]
            trial_dir = job_dir / "trial-one"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(
                json.dumps({"trial_uri": trial_dir.as_uri(), "task_checksum": "sum"})
            )
            return b"pier complete", None

    async def create_process(*args: str, **kwargs: Any) -> FakeProcess:
        captured["args"] = list(args)
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        captured["process_env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(
        app.ServerClient,
        "load_from_global_config",
        lambda: SimpleNamespace(global_config_dict={}),
    )
    task = tmp_path / "task"
    task.mkdir()
    trial, job_dir = await server._run_pier(task_path=task, run_id="run-one")

    assert trial["task_checksum"] == "sum"
    assert captured["config"]["n_attempts"] == 1
    assert captured["config"]["tasks"] == [{"path": str(task)}]
    assert captured["config"]["agents"][0]["env"]["ANTHROPIC_AUTH_TOKEN"] == "${ANTHROPIC_AUTH_TOKEN}"
    assert captured["process_env"]["ANTHROPIC_AUTH_TOKEN"] == "secret-value"
    assert captured["runtime"]["provider"] == {
        "fake": {"api_key": "provider-secret"}  # pragma: allowlist secret
    }
    assert captured["runtime"]["expected_agent_name"] == "claude-code"
    assert captured["runtime"]["expected_agent_version"] == "2.1.153"
    assert captured["runtime_mode"] == 0o600
    assert (job_dir / "gym-pier-stdout.log").read_text() == "pier complete"
    provenance = json.loads((job_dir / "gym-runtime-provenance.json").read_text())
    assert provenance["provider"] == "fake"
    assert provenance["provider_config"] == {"fake": {"api_key": "<redacted>"}}
    assert provenance["expected_agent"] == {"name": "claude-code", "version": "2.1.153"}
    assert provenance["capabilities"]["supports_filtered_egress"] is False
    assert len(provenance["sha256"]) == 64
    assert not Path(captured["config"]["environment"]["kwargs"]["runtime_config_path"]).exists()
    assert (job_dir.stat().st_mode & 0o777) == 0o500
    assert ((job_dir / "gym-runtime-provenance.json").stat().st_mode & 0o777) == 0o400


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["chmod", "read", "scrub"])
async def test_run_pier_discards_job_when_artifact_finalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    run_id = f"unsafe-{failure_mode}"
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / run_id
    server = _agent(work_root=str(jobs_dir), pier_runtime_dir=str(tmp_path / "runtime"))

    async def runtime(*_: Any) -> Path:
        return tmp_path / "pier"

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, None]:
            trial_dir = job_dir / "trial"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "trial_uri": trial_dir.as_uri(),
                        "verifier_result": {"rewards": {"reward": 1.0}},
                    }
                )
            )
            (trial_dir / "artifact.txt").write_text("secret-value")
            return b"completed", None

    async def create_process(*_: Any, **__: Any) -> CompletedProcess:
        return CompletedProcess()

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create_process)
    if failure_mode == "chmod":
        monkeypatch.setattr(
            app,
            "_seal_job_tree",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("raw chmod detail must not escape")),
        )
    elif failure_mode == "read":
        real_scrub_file = app._scrub_file

        def fail_artifact_read(path: Path, *args: Any, **kwargs: Any) -> None:
            if path.name == "artifact.txt":
                raise PermissionError("raw read detail must not escape")
            real_scrub_file(path, *args, **kwargs)

        monkeypatch.setattr(app, "_scrub_file", fail_artifact_read)
    else:
        monkeypatch.setattr(
            app,
            "_scrub_job_secrets",
            lambda *_: (_ for _ in ()).throw(RuntimeError("raw scrub detail must not escape")),
        )

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    with pytest.raises(app.UnsafeArtifactError) as exc_info:
        await server._run_pier(task_path=task_dir, run_id=run_id)

    assert "raw" not in str(exc_info.value)
    assert not job_dir.exists()
    assert not list(jobs_dir.glob(".unsafe-*"))


@pytest.mark.asyncio
async def test_unsafe_artifact_error_returns_no_evidence_or_job_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _agent()
    checkout = tmp_path / "checkout"
    task_dir = checkout / "tasks" / "task-one"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("")
    (task_dir / "instruction.md").write_text("Fix it")

    async def checkout_with_digest() -> tuple[Path, str]:
        return checkout, "digest"

    async def unsafe_run(**_: Any) -> tuple[dict[str, Any], Path]:
        raise app.UnsafeArtifactError("Pier artifact finalization failed (PermissionError); evidence discarded")

    monkeypatch.setattr(server, "_checkout_with_digest", checkout_with_digest)
    monkeypatch.setattr(server, "_run_pier", unsafe_run)
    response = await server.run(_request())

    assert response.status == "harness_error"
    assert response.error_type == "UnsafeArtifactError"
    assert response.pier_result is None
    assert response.verifier_result is None
    assert response.raw_rollout == {"format": None, "trajectory": None, "trajectory_path": None}
    assert response.artifacts == []
    assert response.model_patch is None
    assert "job_dir" not in response.benchmark_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_kind", ["secret-path", "symlink"])
async def test_run_pier_discards_secret_paths_and_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / unsafe_kind
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-safe")
    server = _agent(work_root=str(jobs_dir), pier_runtime_dir=str(tmp_path / "runtime"))

    async def runtime(*_: Any) -> Path:
        return tmp_path / "pier"

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, None]:
            trial_dir = job_dir / "trial"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(json.dumps({"trial_uri": trial_dir.as_uri()}))
            if unsafe_kind == "secret-path":
                (trial_dir / "secret-value.txt").write_text("safe contents")
            else:
                (trial_dir / "outside-link").symlink_to(outside)
            return b"completed", None

    async def create_process(*_: Any, **__: Any) -> CompletedProcess:
        return CompletedProcess()

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create_process)
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    with pytest.raises(app.UnsafeArtifactError):
        await server._run_pier(task_path=task_dir, run_id=unsafe_kind)

    assert not job_dir.exists()
    assert not list(jobs_dir.glob(".unsafe-*"))
    assert outside.read_text() == "outside-safe"


@pytest.mark.asyncio
async def test_run_pier_reports_process_and_result_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _agent(work_root=str(tmp_path / "jobs"), pier_runtime_dir=str(tmp_path / "runtime"))

    async def runtime(*_: Any) -> Path:
        return tmp_path / "pier"

    class FailedProcess:
        returncode = 2

        async def communicate(self) -> tuple[bytes, None]:
            return b"failed output", None

    async def failed_process(*_: Any, **__: Any) -> FailedProcess:
        return FailedProcess()

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", failed_process)
    task = tmp_path / "task"
    task.mkdir()
    with pytest.raises(RuntimeError, match="Pier exited with code 2"):
        await server._run_pier(task_path=task, run_id="failed")

    class EmptyProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, None]:
            return b"no result", None

    async def empty_process(*_: Any, **__: Any) -> EmptyProcess:
        return EmptyProcess()

    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", empty_process)
    with pytest.raises(RuntimeError, match="expected one Pier trial result"):
        await server._run_pier(task_path=task, run_id="empty")


@pytest.mark.parametrize("raw_reward", [0.5, float("nan"), "not-a-number", None])
def test_trial_response_rejects_non_binary_or_missing_rewards(tmp_path: Path, raw_reward: Any) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    response = _agent()._success_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        instruction="Fix it",
        trial={
            "trial_uri": trial_dir.as_uri(),
            "verifier_result": {"rewards": {"reward": raw_reward}},
            "exception_info": None,
        },
        job_dir=tmp_path,
    )
    assert response.reward == 0.0
    assert response.status == "error"
    assert response.error_type == "InvalidReward"


def test_trial_response_classifies_missing_verifier_result(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    response = _agent()._success_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        instruction="Fix it",
        trial={
            "trial_uri": trial_dir.as_uri(),
            "verifier_result": None,
            "exception_info": None,
        },
        job_dir=tmp_path,
    )
    assert response.status == "error"
    assert response.error_type == "MissingVerifierResult"
    assert response.error_message == "Pier trial did not produce a verifier result"


def test_partial_trial_error_preserves_trajectory_and_sanitizes_secrets(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [{"step_id": 1, "source": "agent", "message": "partial"}],
    }
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory))
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
        "agent_config": {
            "ANTHROPIC_AUTH_TOKEN": "secret-value",
            "api_key": "secr****lue",  # pragma: allowlist secret
        },
    }
    response = _agent()._error_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        error=app.PierRunError("Pier exited with code 2", tmp_path, trial),
        instruction="Fix it",
        job_dir=tmp_path,
        trial=trial,
    )
    assert response.status == "harness_error"
    assert response.reward == 0.0
    assert response.raw_reward == 1.0
    assert response.raw_rollout["trajectory"] == trajectory
    assert response.pier_result is not None
    assert response.pier_result["agent_config"]["ANTHROPIC_AUTH_TOKEN"] == "<redacted>"
    assert response.pier_result["agent_config"]["api_key"] == "<redacted>"


@pytest.mark.asyncio
async def test_run_pier_cancellation_terminates_and_scrubs_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _agent(
        work_root=str(tmp_path / "jobs"),
        pier_runtime_dir=str(tmp_path / "runtime"),
        pier_cancel_grace_s=1,
    )
    started = asyncio.Event()
    terminated = asyncio.Event()

    async def runtime(*_: Any) -> Path:
        return tmp_path / "pier"

    class RunningProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, None]:
            started.set()
            await terminated.wait()
            return b"token=secret-value", None

        def terminate(self) -> None:
            self.returncode = -15
            terminated.set()

        def kill(self) -> None:
            self.returncode = -9
            terminated.set()

    process = RunningProcess()

    async def create_process(*_: Any, **__: Any) -> RunningProcess:
        return process

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create_process)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    run_task = asyncio.create_task(server._run_pier(task_path=task_dir, run_id="cancelled"))
    await started.wait()
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert process.returncode == -15
    log = tmp_path / "jobs" / "cancelled" / "gym-pier-stdout.log"
    assert log.read_text() == "token=<redacted>"
    assert "secret-value" not in log.read_text()


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_pier_process_drain(tmp_path: Path) -> None:
    server = _agent(pier_cancel_grace_s=1)
    started = asyncio.Event()
    terminated = asyncio.Event()
    release = asyncio.Event()

    class RunningProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, None]:
            started.set()
            await release.wait()
            return b"token=secret-value", None

        def terminate(self) -> None:
            self.returncode = -15
            terminated.set()

        def kill(self) -> None:
            self.returncode = -9
            terminated.set()

    process = RunningProcess()
    communication = asyncio.create_task(
        server._communicate_with_cancellation_cleanup(
            process,
            tmp_path / "repeated-cancel.log",
        )
    )
    await started.wait()
    communication.cancel()
    await terminated.wait()
    communication.cancel()
    await asyncio.sleep(0.01)
    assert not communication.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await communication
    assert process.returncode == -15
    assert (tmp_path / "repeated-cancel.log").read_text() == "token=<redacted>"


@pytest.mark.asyncio
async def test_cancellation_during_io_failure_still_waits_for_process_cleanup(
    tmp_path: Path,
) -> None:
    server = _agent(pier_cancel_grace_s=1)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    class BrokenProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, None]:
            raise OSError("stream failed")

        def terminate(self) -> None:
            cleanup_started.set()

        def kill(self) -> None:
            self.returncode = -9
            cleanup_release.set()

        async def wait(self) -> int:
            await cleanup_release.wait()
            if self.returncode is None:
                self.returncode = -15
            return self.returncode

    communication = asyncio.create_task(
        server._communicate_with_cancellation_cleanup(
            BrokenProcess(),
            tmp_path / "io-failure.log",
        )
    )
    await cleanup_started.wait()
    communication.cancel()
    await asyncio.sleep(0.01)
    assert not communication.done()
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await communication


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_artifact_scrub_to_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "cancel-during-scrub"
    server = _agent(work_root=str(jobs_dir), pier_runtime_dir=str(tmp_path / "runtime"))
    scrub_entered = threading.Event()
    scrub_release = threading.Event()

    async def runtime(*_: Any) -> Path:
        return tmp_path / "pier"

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, None]:
            trial_dir = job_dir / "trial"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(json.dumps({"trial_uri": trial_dir.as_uri()}))
            (trial_dir / "artifact.txt").write_text("secret-value")
            return b"completed", None

    async def create_process(*_: Any, **__: Any) -> CompletedProcess:
        return CompletedProcess()

    real_scrub_file = app._scrub_file

    def blocking_scrub(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "artifact.txt":
            scrub_entered.set()
            assert scrub_release.wait(timeout=5)
        real_scrub_file(path, *args, **kwargs)

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(app, "_scrub_file", blocking_scrub)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    run_task = asyncio.create_task(server._run_pier(task_path=task_dir, run_id="cancel-during-scrub"))
    while not scrub_entered.is_set():
        await asyncio.sleep(0.001)

    run_task.cancel()
    await asyncio.sleep(0)
    run_task.cancel()
    await asyncio.sleep(0.01)
    assert not run_task.done()
    scrub_release.set()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    artifact = job_dir / "trial" / "artifact.txt"
    assert artifact.read_text() == "<redacted>"
    assert (artifact.stat().st_mode & 0o777) == 0o400
    assert (job_dir.stat().st_mode & 0o777) == 0o500


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_unsafe_job_disposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "unsafe-job"
    job_dir.mkdir()
    (job_dir / "raw.txt").write_text("secret-value")
    server = _agent()
    disposal_entered = threading.Event()
    disposal_release = threading.Event()
    real_discard = app._discard_or_quarantine_job

    def fail_finalization(*_: Any) -> None:
        raise RuntimeError("scrub failed")

    def blocking_discard(path: Path) -> str:
        disposal_entered.set()
        assert disposal_release.wait(timeout=5)
        return real_discard(path)

    monkeypatch.setattr(app, "_finalize_job_files", fail_finalization)
    monkeypatch.setattr(app, "_discard_or_quarantine_job", blocking_discard)
    finalization = asyncio.create_task(
        server._finalize_job_evidence(
            job_dir,
            job_dir / "gym-pier-stdout.log",
            b"",
        )
    )
    while not disposal_entered.is_set():
        await asyncio.sleep(0.001)

    finalization.cancel()
    await asyncio.sleep(0)
    finalization.cancel()
    await asyncio.sleep(0.01)
    assert not finalization.done()
    disposal_release.set()
    with pytest.raises(asyncio.CancelledError):
        await finalization
    assert not job_dir.exists()


def test_scrub_job_secrets_rejects_symlinks(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-value")
    (job_dir / "link.txt").symlink_to(outside)

    with pytest.raises(app.ArtifactLimitError, match="entry is unsafe"):
        app._scrub_job_secrets(job_dir, app._redacted_secret_variants("secret-value"))
    assert outside.read_text() == "secret-value"


def test_scrub_job_secrets_redacts_and_seals_tree(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    artifact = job_dir / "artifact.txt"
    artifact.write_bytes(b"x" * (1024 * 1024 - 3) + b"secret-value redacted=secr****lue")

    app._scrub_job_secrets(job_dir, app._redacted_secret_variants("secret-value"))

    scrubbed = artifact.read_bytes()
    assert scrubbed.endswith(b"<redacted> redacted=<redacted>")
    assert b"secret-value" not in scrubbed
    assert (artifact.stat().st_mode & 0o777) == 0o400
    assert (job_dir.stat().st_mode & 0o777) == 0o500
    assert app._redacted_secret_variants("") == ()
    assert app._sanitize_persisted_value([("secret-value",)], ("secret-value",)) == [["<redacted>"]]
    assert app._sanitize_persisted_value(
        {"credentials": ["provider-secret"], "nested": {"api_keys": ["other-secret"]}},
        (),
    ) == {"credentials": "<redacted>", "nested": {"api_keys": "<redacted>"}}
    assert app._sanitize_persisted_value({"api_key": "${RUNTIME_TOKEN}"}, ()) == {"api_key": "${RUNTIME_TOKEN}"}


def test_scrub_job_secrets_fails_closed_when_one_file_cannot_be_restricted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    failed = job_dir / "failed.txt"
    failed.write_text("secret-value")
    real_scrub_file = app._scrub_file

    def fail_one_file(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == failed:
            raise PermissionError("denied")
        real_scrub_file(path, *args, **kwargs)

    monkeypatch.setattr(app, "_scrub_file", fail_one_file)
    with pytest.raises(RuntimeError, match="Failed to scrub or restrict 1"):
        app._scrub_job_secrets(job_dir, ("secret-value",))
    assert failed.read_text() == "secret-value"


def test_scrub_file_redacts_self_overlapping_secret_at_eof_and_chunk_boundary(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    secret = b"abca"
    artifact.write_bytes(b"x" * (1024 * 1024 - len(secret)) + secret + b"-" + secret)

    app._scrub_file(artifact, ((secret, b"<redacted>"),))

    contents = artifact.read_bytes()
    assert secret not in contents
    assert contents.endswith(b"<redacted>-<redacted>")


def test_descriptor_scrub_helpers_fail_closed_on_unsafe_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "artifact.txt"
    artifact.write_text("safe")

    with pytest.raises(app.ArtifactLimitError, match="safe relative path"):
        app._open_artifact_parent(root, Path("../escape"))
    with pytest.raises(FileNotFoundError):
        app._open_artifact_parent(root, Path("missing/artifact.txt"))
    with pytest.raises(app.ArtifactLimitError, match="escapes"):
        app._scrub_file(tmp_path / "outside.txt", (), root=root)
    with pytest.raises(FileNotFoundError):
        app._scrub_file(root / "missing.txt", (), root=root)
    with pytest.raises(app.ArtifactLimitError, match="exceeds 2 bytes"):
        app._scrub_file(artifact, (), max_bytes=2, root=root)

    outside = tmp_path / "outside.txt"
    outside.write_text("safe")
    hard_link = root / "hard-link.txt"
    os.link(outside, hard_link)
    with pytest.raises(app.ArtifactLimitError, match="private regular file"):
        app._scrub_file(hard_link, (), root=root)

    parent_descriptor, name = app._open_artifact_parent(root, Path("artifact.txt"))
    descriptor, metadata = app._open_regular_artifact(parent_descriptor, name, max_bytes=16)
    os.close(descriptor)
    artifact.write_text("changed")
    try:
        with pytest.raises(app.ArtifactLimitError, match="changed while scrubbing"):
            app._verify_artifact_entry(parent_descriptor, name, metadata)
    finally:
        os.close(parent_descriptor)

    monkeypatch.setattr(app.os, "write", lambda *_args, **_kwargs: 0)
    with pytest.raises(OSError, match="Could not write"):
        app._write_all(-1, b"value")


def test_scrub_file_detects_growth_and_metadata_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.txt"

    def changed_metadata(metadata: os.stat_result, **overrides: int) -> SimpleNamespace:
        values = {
            "st_dev": metadata.st_dev,
            "st_ino": metadata.st_ino,
            "st_size": metadata.st_size,
            "st_mtime_ns": metadata.st_mtime_ns,
            "st_ctime_ns": metadata.st_ctime_ns,
            "st_nlink": metadata.st_nlink,
            "st_mode": metadata.st_mode,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    artifact.write_bytes(b"xx")
    real_open = app._open_regular_artifact

    def understate_size(parent: int, name: str, *, max_bytes: int) -> tuple[int, SimpleNamespace]:
        descriptor, metadata = real_open(parent, name, max_bytes=16)
        return descriptor, changed_metadata(metadata, st_size=0)

    with monkeypatch.context() as context:
        context.setattr(app, "_open_regular_artifact", understate_size)
        with pytest.raises(app.ArtifactLimitError, match="exceeds 1 bytes"):
            app._scrub_file(artifact, ((b"x", b"z"),), max_bytes=1)

    artifact.write_bytes(b"safe")
    real_fstat = app.os.fstat
    fstat_calls = 0

    def change_after_scan(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal fstat_calls
        fstat_calls += 1
        metadata = real_fstat(descriptor)
        if fstat_calls == 2:
            return changed_metadata(metadata, st_ctime_ns=metadata.st_ctime_ns + 1)
        return metadata

    with monkeypatch.context() as context:
        context.setattr(app.os, "fstat", change_after_scan)
        with pytest.raises(app.ArtifactLimitError, match="changed while scrubbing"):
            app._scrub_file(artifact, ((b"missing", b"z"),))

    artifact.write_bytes(b"secret")
    fstat_calls = 0

    def change_after_write(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal fstat_calls
        fstat_calls += 1
        metadata = real_fstat(descriptor)
        if fstat_calls == 3:
            return changed_metadata(metadata, st_ctime_ns=metadata.st_ctime_ns + 1)
        return metadata

    with monkeypatch.context() as context:
        context.setattr(app.os, "fstat", change_after_write)
        with pytest.raises(app.ArtifactLimitError, match="changed while scrubbing"):
            app._scrub_file(artifact, ((b"secret", b"<redacted>"),))

    artifact.write_bytes(b"x")
    real_read = app.os.read
    read_calls = 0

    def grow_during_rewrite(descriptor: int, size: int) -> bytes:
        nonlocal read_calls
        read_calls += 1
        if read_calls == 3:
            return b"xx"
        return real_read(descriptor, size)

    with monkeypatch.context() as context:
        context.setattr(app.os, "read", grow_during_rewrite)
        with pytest.raises(app.ArtifactLimitError, match="exceeds 1 bytes"):
            app._scrub_file(artifact, ((b"x", b"z"),), max_bytes=1)


def test_job_finalization_rejects_tree_changes_and_oversized_fallback_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    artifact = job_dir / "artifact.txt"
    artifact.write_text("safe")
    real_scrub_file = app._scrub_file

    def add_file_after_scrub(path: Path, *args: Any, **kwargs: Any) -> None:
        real_scrub_file(path, *args, **kwargs)
        (job_dir / "late.txt").write_text("late")

    monkeypatch.setattr(app, "_scrub_file", add_file_after_scrub)
    with pytest.raises(RuntimeError, match="tree changed"):
        app._scrub_job_secrets(job_dir, ())

    stdout_job = tmp_path / "stdout-job"
    stdout_job.mkdir()
    limits = app.ArtifactLimits(max_files=4, max_file_bytes=4, max_total_bytes=8)
    with pytest.raises(app.ArtifactLimitError, match="stdout exceeds 4 bytes"):
        app._finalize_job_files(
            stdout_job,
            stdout_job / "gym-pier-stdout.log",
            b"12345",
            (),
            limits,
        )


def test_unsafe_job_disposal_removes_quarantines_and_isolates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_rmtree = app.shutil.rmtree
    removed = tmp_path / "removed"
    removed.mkdir()
    locked = removed / "locked"
    locked.write_text("secret")
    retried: list[str] = []

    def rmtree_with_retry(path: Path, *, onexc: Any) -> None:
        onexc(lambda value: retried.append(value), str(locked), PermissionError("denied"))
        original_rmtree(path)

    with monkeypatch.context() as context:
        context.setattr(app.shutil, "rmtree", rmtree_with_retry)
        assert app._discard_or_quarantine_job(removed) == "discarded"
    assert retried == [str(locked)]

    quarantined = tmp_path / "quarantined"
    quarantined.mkdir()
    with monkeypatch.context() as context:
        context.setattr(app.shutil, "rmtree", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")))
        assert app._discard_or_quarantine_job(quarantined) == "quarantined"
    quarantine_paths = list(tmp_path.glob(".unsafe-*"))
    assert len(quarantine_paths) == 1
    original_rmtree(quarantine_paths[0])

    isolated = tmp_path / "isolated"
    isolated.mkdir()
    with monkeypatch.context() as context:
        context.setattr(app.shutil, "rmtree", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")))
        context.setattr(Path, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")))
        assert app._discard_or_quarantine_job(isolated) == "isolated under the private jobs root"


@pytest.mark.asyncio
async def test_checkout_and_digest_are_initialized_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _agent()
    checkout_calls = 0
    digest_calls = 0

    async def checkout() -> Path:
        nonlocal checkout_calls
        checkout_calls += 1
        await asyncio.sleep(0)
        return tmp_path

    def digest(path: Path) -> str:
        nonlocal digest_calls
        assert path == tmp_path
        digest_calls += 1
        return "digest"

    monkeypatch.setattr(server, "_checkout", checkout)
    monkeypatch.setattr(app, "task_tree_digest", digest)
    monkeypatch.setattr(app, "materialize_task_snapshot", lambda path, _: path)
    results = await asyncio.gather(*(server._checkout_with_digest() for _ in range(8)))
    assert results == [(tmp_path, "digest")] * 8
    assert checkout_calls == 1
    assert digest_calls == 1


@pytest.mark.asyncio
async def test_run_retains_known_job_after_response_assembly_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _agent()
    checkout = tmp_path / "checkout"
    task_path = checkout / "tasks" / "task-one"
    task_path.mkdir(parents=True)
    (task_path / "task.toml").write_text("")
    (task_path / "instruction.md").write_text("Fix it")
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    (trial_dir / "artifacts").mkdir(parents=True)
    (trial_dir / "agent").mkdir()
    (trial_dir / "artifacts" / "model.patch").write_text("patch")
    trajectory = {"schema_version": "ATIF-v1.7", "steps": [{"source": "agent", "message": "done"}]}
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory))
    trial = {"trial_uri": trial_dir.as_uri(), "verifier_result": {"rewards": {"reward": 1.0}}}

    async def checkout_with_digest() -> tuple[Path, str]:
        return checkout, "digest"

    async def run_pier(**_: Any) -> tuple[dict[str, Any], Path]:
        return trial, job_dir

    def fail_response(*_: Any, **__: Any) -> Any:
        raise RuntimeError("assembly failed")

    monkeypatch.setattr(server, "_checkout_with_digest", checkout_with_digest)
    monkeypatch.setattr(server, "_run_pier", run_pier)
    monkeypatch.setattr(server, "_success_response", fail_response)
    response = await server.run(_request())
    assert response.status == "harness_error"
    assert response.benchmark_metadata["job_dir"] == str(job_dir)
    assert response.raw_reward == 1.0
    assert response.pier_result == trial
    assert response.verifier_result == trial["verifier_result"]
    assert response.raw_rollout["trajectory"] == trajectory
    assert {artifact["path"] for artifact in response.artifacts} == {
        "trial/agent/trajectory.json",
        "trial/artifacts/model.patch",
    }


def test_partial_trial_reader_and_fallback_errors(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    result = trial_dir / "result.json"
    result.write_text("not json")
    server = _agent()
    assert "could not parse" in str(server._read_partial_trial(tmp_path)[1])
    result.write_text("[]")
    assert server._read_partial_trial(tmp_path)[1] == "Pier trial result is not a JSON object"

    response = server._error_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        error=app.PierRunError("failed", tmp_path),
        instruction="Fix it",
        job_dir=tmp_path,
        trial={"trial_uri": Path("/tmp/outside").as_uri()},
    )
    assert response.status == "harness_error"
    assert "failed to read partial" in str(response.error_message)


def test_bounded_error_fallback_drops_unsafe_or_oversized_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _agent(
        artifact_max_file_bytes=1024,
        artifact_max_total_bytes=4096,
        trajectory_max_bytes=8,
        patch_max_bytes=8,
    )
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    trajectory_path.write_text(json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"source": "agent"}]}))
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": "not-a-number"}},
    }

    response = server._error_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        error=RuntimeError("failed"),
        job_dir=job_dir,
        trial=trial,
    )
    assert response.error_type == "ArtifactLimitError"
    assert response.pier_result is None
    assert response.verifier_result is None
    assert response.raw_reward is None
    assert response.artifacts == []

    oversized_server = _agent(
        artifact_max_file_bytes=16,
        artifact_max_total_bytes=32,
        trajectory_max_bytes=8,
        patch_max_bytes=8,
    )
    response = oversized_server._error_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        error=RuntimeError("failed"),
        job_dir=job_dir,
        trial=trial,
    )
    assert response.error_type == "ArtifactLimitError"
    assert response.pier_result is None
    assert response.artifacts == []

    trajectory_path.unlink()
    (trial_dir / "small.txt").write_text("ok")
    monkeypatch.setattr(app, "artifact_snapshot", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")))
    response = server._error_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        error=RuntimeError("failed"),
        job_dir=job_dir,
        trial=trial,
    )
    assert response.error_type == "PierRunError"
    assert response.pier_result is None
    assert response.artifacts == []


def test_error_fallback_handles_invalid_reward_without_evidence() -> None:
    response = _agent()._error_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        error=RuntimeError("failed"),
        trial={"verifier_result": {"rewards": {"reward": "bad"}}},
    )
    assert response.raw_reward is None
    assert response.pier_result is not None


def test_runtime_provenance_falls_back_after_bounded_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _agent(
        artifact_max_file_bytes=8,
        artifact_max_total_bytes=8,
        trajectory_max_bytes=8,
        patch_max_bytes=8,
    )
    (tmp_path / "gym-runtime-provenance.json").write_text("x" * 9)
    assert server._response_runtime_provenance(tmp_path)["provider"] == "fake"

    monkeypatch.setattr(server, "_environment_config", lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
    assert server._response_runtime_provenance(None) is None


@pytest.mark.asyncio
async def test_run_pier_discards_job_when_runtime_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _agent(
        work_root=str(tmp_path / "jobs"),
        pier_runtime_dir=str(tmp_path / "runtime"),
    )

    async def fail_runtime(*_: Any) -> Path:
        raise RuntimeError("runtime setup failed")

    monkeypatch.setattr(app, "ensure_pier_runtime", fail_runtime)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    with pytest.raises(RuntimeError, match="runtime setup failed"):
        await server._run_pier(task_path=task_dir, run_id="runtime-error")
    assert list((tmp_path / "jobs").iterdir()) == []


@pytest.mark.asyncio
async def test_run_pier_wraps_launch_error_and_kills_after_cancel_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _agent(
        work_root=str(tmp_path / "jobs"),
        pier_runtime_dir=str(tmp_path / "runtime"),
        pier_cancel_grace_s=0.01,
    )

    async def runtime(*_: Any) -> Path:
        return tmp_path / "pier"

    async def fail_create(*_: Any, **__: Any) -> Any:
        raise OSError("launch failed with secret-value")

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", fail_create)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    with pytest.raises(app.PierRunError, match="launch failed") as exc_info:
        await server._run_pier(task_path=task_dir, run_id="launch-error")
    assert exc_info.value.job_dir.name == "launch-error"

    started = asyncio.Event()
    killed = asyncio.Event()

    class StuckProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, None]:
            started.set()
            await killed.wait()
            return b"killed", None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.returncode = -9
            killed.set()

    process = StuckProcess()

    async def create_stuck(*_: Any, **__: Any) -> StuckProcess:
        return process

    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create_stuck)
    run_task = asyncio.create_task(server._run_pier(task_path=task_dir, run_id="kill-timeout"))
    await started.wait()
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task
    assert process.returncode == -9


@pytest.mark.asyncio
async def test_run_pier_terminates_live_process_when_communicate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _agent(
        work_root=str(tmp_path / "jobs"),
        pier_runtime_dir=str(tmp_path / "runtime"),
        pier_cancel_grace_s=1,
    )

    async def runtime(*_: Any) -> Path:
        return tmp_path / "pier"

    class BrokenProcess:
        returncode: int | None = None
        terminated = False

        async def communicate(self) -> tuple[bytes, None]:
            raise OSError("broken pipe")

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = BrokenProcess()

    async def create_broken(*_: Any, **__: Any) -> BrokenProcess:
        return process

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create_broken)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    with pytest.raises(app.PierRunError, match="launch failed"):
        await server._run_pier(task_path=task_dir, run_id="broken-communicate")
    assert process.terminated is True
    assert process.returncode == -15


@pytest.mark.asyncio
async def test_failed_communication_reloads_trial_only_after_scrubbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "failed-after-result"
    server = _agent(
        work_root=str(jobs_dir),
        pier_runtime_dir=str(tmp_path / "runtime"),
        pier_cancel_grace_s=1,
    )

    async def runtime(*_: Any) -> Path:
        return tmp_path / "pier"

    class BrokenProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, None]:
            trial_dir = job_dir / "trial"
            (trial_dir / "agent").mkdir(parents=True)
            (trial_dir / "agent" / "trajectory.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ATIF-v1.7",
                        "steps": [{"source": "agent", "message": "partial"}],
                    }
                )
            )
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "trial_uri": trial_dir.as_uri(),
                        "task_checksum": "secret-value",
                        "exception_info": {
                            "exception_type": "secret-value",
                            "exception_message": "secret-value",
                        },
                        "verifier_result": {"rewards": {"reward": 0.0}},
                    }
                )
            )
            raise OSError("stream failed")

        def terminate(self) -> None:
            self.returncode = -15

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    async def create_process(*_: Any, **__: Any) -> BrokenProcess:
        return BrokenProcess()

    monkeypatch.setattr(app, "ensure_pier_runtime", runtime)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create_process)
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    with pytest.raises(app.PierRunError) as exc_info:
        await server._run_pier(task_path=task_dir, run_id="failed-after-result")

    trial = exc_info.value.trial
    assert trial is not None
    assert trial["task_checksum"] == "<redacted>"
    assert trial["exception_info"]["exception_type"] == "<redacted>"
    response = server._error_response(
        record=_request().model_dump(mode="json"),
        task_id="task-one",
        error=exc_info.value,
        instruction="Fix it",
        job_dir=job_dir,
        trial=trial,
    )
    assert "secret-value" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_failed_process_cleanup_kills_after_grace_timeout() -> None:
    server = _agent(pier_cancel_grace_s=0.01)

    class StubbornProcess:
        returncode: int | None = None
        wait_calls = 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            self.wait_calls += 1
            if self.returncode is None:
                await asyncio.Future()
            assert self.returncode is not None
            return self.returncode

    process = StubbornProcess()
    await server._terminate_failed_process(process)
    assert process.returncode == -9
    assert process.wait_calls == 2


@pytest.mark.asyncio
async def test_cancelled_pier_process_has_bounded_post_kill_drain(tmp_path: Path) -> None:
    server = _agent(pier_cancel_grace_s=0.01)
    started = asyncio.Event()

    class NeverDrainsProcess:
        returncode: int | None = None
        killed = False

        async def communicate(self) -> tuple[bytes, None]:
            started.set()
            await asyncio.Future()
            return b"", None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            await asyncio.Future()
            return -9

    process = NeverDrainsProcess()
    communicate = asyncio.create_task(
        server._communicate_with_cancellation_cleanup(process, tmp_path / "never-drains.log")
    )
    await started.wait()
    communicate.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(communicate, timeout=0.2)
    assert process.killed is True


@pytest.mark.asyncio
async def test_pier_stdout_streams_live_with_bounded_redacted_tail(tmp_path: Path) -> None:
    server = _agent()
    waiting_for_release = asyncio.Event()
    release = asyncio.Event()

    class ChunkedReader:
        reads = 0

        async def read(self, _: int) -> bytes:
            self.reads += 1
            if self.reads == 1:
                return b"progress\n" + b"secre"
            if self.reads == 2:
                waiting_for_release.set()
                await release.wait()
                return b"t-value\n" + b"x" * (128 * 1024)
            return b""

    class StreamingProcess:
        returncode = 0
        stdout = ChunkedReader()

        async def wait(self) -> int:
            return 0

    log_path = tmp_path / "pier.log"
    capture = asyncio.create_task(server._capture_pier_output(StreamingProcess(), log_path))
    await waiting_for_release.wait()
    assert log_path.read_bytes().startswith(b"progress")
    release.set()
    tail = await capture
    content = log_path.read_bytes()
    assert b"secret-value" not in content
    assert b"<redacted>" in content
    assert len(tail) == 64 * 1024
    assert (log_path.stat().st_mode & 0o777) == 0o600


@pytest.mark.asyncio
async def test_pier_stdout_persistence_stops_at_artifact_quota(tmp_path: Path) -> None:
    server = _agent(
        artifact_max_file_bytes=8,
        artifact_max_total_bytes=32,
        trajectory_max_bytes=8,
        patch_max_bytes=8,
    )

    class OversizedReader:
        returned = False

        async def read(self, _: int) -> bytes:
            if self.returned:
                return b""
            self.returned = True
            return b"123456789"

    class StreamingProcess:
        returncode = 0
        stdout = OversizedReader()

        async def wait(self) -> int:
            return 0

    log_path = tmp_path / "pier.log"
    with pytest.raises(app.ArtifactLimitError, match="stdout exceeds 8 bytes"):
        await server._capture_pier_output(StreamingProcess(), log_path)
    assert log_path.stat().st_size <= 8


@pytest.mark.asyncio
async def test_run_success_records_checkout_tree_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _agent()
    checkout = tmp_path / "checkout"
    task_dir = checkout / "tasks" / "task-one"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("schema_version = '1.1'\n")
    (task_dir / "instruction.md").write_text("Fix it")
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"source": "agent", "message": "done"}]})
    )
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
        "task_checksum": "task-sum",
    }

    async def checkout_fn() -> Path:
        return checkout

    async def run_pier(**_: Any) -> tuple[dict[str, Any], Path]:
        return trial, job_dir

    monkeypatch.setattr(server, "_checkout", checkout_fn)
    monkeypatch.setattr(server, "_run_pier", run_pier)
    monkeypatch.setattr(app, "task_tree_digest", lambda _: "a" * 64)
    monkeypatch.setattr(app, "materialize_task_snapshot", lambda checkout, _: checkout)
    response = await server.run(_request())
    assert response.status == "success"
    assert response.reward == 1.0
    assert response.benchmark_metadata["benchmark_task_tree_sha256"] == "a" * 64


@pytest.mark.asyncio
async def test_response_assembly_concurrency_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _agent(max_concurrent_assembly=1)
    checkout = tmp_path / "checkout"
    task_dir = checkout / "tasks" / "task-one"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("")
    (task_dir / "instruction.md").write_text("Fix it")
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"source": "agent", "message": "done"}]})
    )
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
    }

    async def checkout_with_digest() -> tuple[Path, str]:
        return checkout, "digest"

    async def run_pier(**_: Any) -> tuple[dict[str, Any], Path]:
        return trial, job_dir

    entered = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def bounded_to_thread(function: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered.set()
        try:
            await release.wait()
            return function(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(server, "_checkout_with_digest", checkout_with_digest)
    monkeypatch.setattr(server, "_run_pier", run_pier)
    monkeypatch.setattr(app.asyncio, "to_thread", bounded_to_thread)
    first = asyncio.create_task(server.run(_request()))
    second = asyncio.create_task(server.run(_request()))
    await entered.wait()
    await asyncio.sleep(0)
    assert active == 1
    release.set()
    responses = await asyncio.gather(first, second)

    assert max_active == 1
    assert all(response.status == "success" for response in responses)


@pytest.mark.asyncio
async def test_cancelled_response_assembly_keeps_its_semaphore_permit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _agent(max_concurrent_assembly=1)
    checkout = tmp_path / "checkout"
    task_dir = checkout / "tasks" / "task-one"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("")
    (task_dir / "instruction.md").write_text("Fix it")
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"source": "agent", "message": "done"}]})
    )
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
    }

    async def checkout_with_digest() -> tuple[Path, str]:
        return checkout, "digest"

    async def run_pier(**_: Any) -> tuple[dict[str, Any], Path]:
        return trial, job_dir

    entered = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def blocking_to_thread(function: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered.set()
        try:
            await release.wait()
            return function(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(server, "_checkout_with_digest", checkout_with_digest)
    monkeypatch.setattr(server, "_run_pier", run_pier)
    monkeypatch.setattr(app.asyncio, "to_thread", blocking_to_thread)
    first = asyncio.create_task(server.run(_request()))
    await entered.wait()
    first.cancel()
    await asyncio.sleep(0)
    second = asyncio.create_task(server.run(_request()))
    await asyncio.sleep(0.01)

    assert not first.done()
    assert active == 1
    assert max_active == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    response = await second
    assert response.status == "success"
    assert max_active == 1


@pytest.mark.asyncio
async def test_cancelled_assembly_still_discards_newly_unsafe_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _agent(max_concurrent_assembly=1)
    checkout = tmp_path / "checkout"
    task_dir = checkout / "tasks" / "task-one"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("")
    (task_dir / "instruction.md").write_text("Fix it")
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"source": "agent", "message": "done"}]})
    )
    (trial_dir / "secret-value.txt").write_text("safe contents")
    trial = {
        "trial_uri": trial_dir.as_uri(),
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
    }

    async def checkout_with_digest() -> tuple[Path, str]:
        return checkout, "digest"

    async def run_pier(**_: Any) -> tuple[dict[str, Any], Path]:
        return trial, job_dir

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_to_thread(function: Any, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return function(*args, **kwargs)

    monkeypatch.setattr(server, "_checkout_with_digest", checkout_with_digest)
    monkeypatch.setattr(server, "_run_pier", run_pier)
    monkeypatch.setattr(app.asyncio, "to_thread", blocking_to_thread)
    run_task = asyncio.create_task(server.run(_request()))
    await entered.wait()
    run_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await run_task
    assert not job_dir.exists()
