# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import io
import json
import logging
import os
import shlex
import stat
import tarfile
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip("pier", reason="Pier is tested in its isolated DeepSWE runtime")

from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.config import ResourceMode
from pier.models.trial.paths import TrialPaths

from nemo_gym.sandbox import SandboxDownloadLimitExceeded, SandboxExecResult, SandboxStatus
from responses_api_agents.deep_swe import pier_sandbox_environment as adapter


_CLAUDE_VERSION_COMMAND = 'export PATH="$HOME/.local/bin:$PATH"; claude --version'
_CLAUDE_VERSION_EXEC = f"/bin/bash -c {shlex.quote(_CLAUDE_VERSION_COMMAND)}"


class FakeAsyncSandbox:
    instances: list["FakeAsyncSandbox"] = []

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.spec = None
        self.exec_calls: list[dict[str, Any]] = []
        self.exec_results: list[SandboxExecResult] = []
        self.uploads: list[tuple[Path, str]] = []
        self.downloads: list[tuple[str, Path]] = []
        self.download_limits: list[int | None] = []
        self.stopped = False
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tf:
            payload = b"downloaded"
            info = tarfile.TarInfo("result.txt")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        self.download_archive = archive.getvalue()
        self.__class__.instances.append(self)

    async def start(self, spec: Any) -> "FakeAsyncSandbox":
        self.spec = spec
        return self

    @property
    def sandbox_id(self) -> str:
        return f"fake-sandbox-{self.__class__.instances.index(self)}"

    @property
    def provider_name(self) -> str:
        return "fake"

    async def exec(self, command: str, **kwargs: Any) -> SandboxExecResult:
        self.exec_calls.append({"command": command, **kwargs})
        if self.exec_results:
            return self.exec_results.pop(0)
        if command == _CLAUDE_VERSION_EXEC:
            return SandboxExecResult(stdout="2.1.153 (Claude Code)\n", stderr=None, return_code=0)
        if "sha256sum --" in command:
            digest = hashlib.sha256(self.download_archive).hexdigest()
            return SandboxExecResult(
                stdout=f"{len(self.download_archive)}\n{digest}  /tmp/archive\n",
                stderr=None,
                return_code=0,
            )
        return SandboxExecResult(stdout="ok", stderr=None, return_code=0)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        self.uploads.append((Path(local_path), remote_path))

    async def download(self, remote_path: str, local_path: Path, *, max_bytes: int | None = None) -> None:
        self.downloads.append((remote_path, Path(local_path)))
        self.download_limits.append(max_bytes)
        if remote_path.endswith(".tar.gz"):
            assert max_bytes is None or len(self.download_archive) <= max_bytes
            Path(local_path).write_bytes(self.download_archive)
        else:
            Path(local_path).write_text("downloaded")

    async def status(self) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def stop(self) -> None:
        self.stopped = True


def _install_spec() -> AgentInstallSpec:
    return AgentInstallSpec(
        agent_name="claude-code",
        version="2.1.153",
        steps=[
            InstallStep(run="apt-get install -y curl", user="root"),
            InstallStep(run="install-claude", user="agent", env={"MODE": "test"}),
        ],
        verification_command=_CLAUDE_VERSION_COMMAND,
    )


async def _environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime_overrides: dict[str, Any] | None = None,
    task_config: EnvironmentConfig | None = None,
    spec: dict[str, Any] | None = None,
    include_agent_install_spec: bool = True,
) -> adapter.PierSandboxEnvironment:
    monkeypatch.setattr(adapter, "AsyncSandbox", FakeAsyncSandbox)
    runtime = {
        "provider": {"fake": {}},
        "provider_metadata": {"owner": "test"},
        "spec": spec or {"ttl_s": 100, "ready_timeout_s": 5},
        "supports_disable_internet": True,
        "supports_filtered_egress": True,
        "preinstall_agent_in_image": True,
        "expected_agent_name": "claude-code",
        "expected_agent_version": "2.1.153",
    }
    runtime.update(runtime_overrides or {})
    runtime_id = await adapter.register_runtime_config(runtime)
    return adapter.PierSandboxEnvironment(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="session",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=task_config
        or EnvironmentConfig(
            docker_image="registry.example/task:tag",
            cpus=2,
            memory_mb=8192,
            storage_mb=20480,
            allow_internet=False,
        ),
        logger=logging.getLogger("test"),
        agent_install_spec=_install_spec() if include_agent_install_spec else None,
        network_allowlist=NetworkAllowlist(domains=["model.example"]),
        runtime_config_id=runtime_id,
    )


def _expected_version_proof() -> dict[str, str]:
    return {
        "agent": "claude-code",
        "expected": "2.1.153",
        "observed": "2.1.153",
        "verification_command": _CLAUDE_VERSION_COMMAND,
    }


@pytest.mark.asyncio
async def test_start_builds_provider_neutral_spec_and_preinstalls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = await _environment(tmp_path, monkeypatch)
    assert env._transfer_timeout_s == adapter._DEFAULT_TRANSFER_TIMEOUT_S
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]

    assert sandbox.spec.image == "registry.example/task:tag"
    assert sandbox.spec.resources.cpu == 2
    assert sandbox.spec.resources.memory_mib == 8192
    assert sandbox.spec.resources.disk_gib == 20
    assert sandbox.spec.metadata["owner"] == "test"
    options = sandbox.spec.provider_options
    assert options["network_allowlist"] == ["model.example"]
    assert options["block_network"] is False
    assert [step["run"] for step in options["image_setup_steps"]] == [
        "apt-get install -y curl",
        "install-claude",
    ]
    assert [step["shell"] for step in options["image_setup_steps"]] == ["/bin/bash", "/bin/bash"]
    assert not any(call["command"] == "install-claude" for call in sandbox.exec_calls)
    assert any(call["command"] == _CLAUDE_VERSION_EXEC for call in sandbox.exec_calls)

    artifact_target = tmp_path / "agent-artifacts"
    await env.download_dir("/logs/artifacts", artifact_target)
    proof_path = artifact_target / adapter._AGENT_VERSION_PROOF_NAME
    assert json.loads(proof_path.read_text()) == _expected_version_proof()
    assert stat.S_IMODE(proof_path.stat().st_mode) == 0o400
    assert proof_path.stat().st_uid == os.geteuid()
    assert [call["command"] for call in sandbox.exec_calls].count(_CLAUDE_VERSION_EXEC) == 2

    await env.stop(delete=True)
    assert sandbox.stopped is True
    observations = list(
        (env.trial_paths.artifacts_dir / adapter._SANDBOX_OBSERVATIONS_DIR_NAME).glob("sandbox-*.json")
    )
    assert len(observations) == 1
    observation = json.loads(observations[0].read_text())
    assert observation["provider"] == "fake"
    assert observation["sandbox_id"] == sandbox.sandbox_id
    assert observation["requested_image"] == "registry.example/task:tag"
    assert observation["resolved_image_digest"] is None
    assert observation["environment_role"] == "agent"
    await env.stop(delete=True)


@pytest.mark.asyncio
async def test_runtime_install_and_transfer_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={"preinstall_agent_in_image": False},
    )
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]
    assert any(call["command"] == "/bin/bash -c install-claude" for call in sandbox.exec_calls)

    runtime_artifacts = tmp_path / "runtime-agent-artifacts"
    await env.download_dir("/logs/artifacts", runtime_artifacts)
    assert json.loads((runtime_artifacts / adapter._AGENT_VERSION_PROOF_NAME).read_text()) == (
        _expected_version_proof()
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("input")
    await env.upload_dir(source, "/target")
    assert any(remote.startswith("/tmp/pier-upload-") for _, remote in sandbox.uploads)

    target = tmp_path / "download"
    await env.download_dir("/remote", target)
    assert (target / "result.txt").read_text() == "downloaded"

    one_file = tmp_path / "one.txt"
    await env.download_file("/remote/one.txt", one_file)
    assert one_file.read_text() == "downloaded"
    assert sandbox.download_limits == [
        adapter._MAX_DOWNLOAD_ARCHIVE_BYTES,
        adapter._MAX_DOWNLOAD_ARCHIVE_BYTES,
        adapter._MAX_DOWNLOAD_FILE_BYTES,
    ]


@pytest.mark.asyncio
async def test_verifier_environment_does_not_emit_agent_version_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(tmp_path, monkeypatch, include_agent_install_spec=False)
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]

    verifier_artifacts = tmp_path / "verifier-artifacts"
    await env.download_dir("/logs/artifacts", verifier_artifacts)

    assert not (verifier_artifacts / adapter._AGENT_VERSION_PROOF_NAME).exists()
    assert not any(call["command"] == _CLAUDE_VERSION_EXEC for call in sandbox.exec_calls)
    await env.stop(delete=True)
    observations = list(
        (env.trial_paths.artifacts_dir / adapter._SANDBOX_OBSERVATIONS_DIR_NAME).glob("sandbox-*.json")
    )
    assert len(observations) == 1
    assert json.loads(observations[0].read_text())["environment_role"] == "verifier"


@pytest.mark.asyncio
async def test_post_run_agent_version_mismatch_fails_artifact_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(tmp_path, monkeypatch)
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]
    sandbox.exec_results = [
        SandboxExecResult(stdout="2.1.154 (Claude Code)\n", stderr=None, return_code=0),
    ]

    artifact_target = tmp_path / "agent-artifacts"
    with pytest.raises(RuntimeError, match=r"reported version 2\.1\.154, expected 2\.1\.153"):
        await env.download_dir("/logs/artifacts", artifact_target)

    assert not (artifact_target / adapter._AGENT_VERSION_PROOF_NAME).exists()
    assert not any("sha256sum --" in call["command"] for call in sandbox.exec_calls)


@pytest.mark.asyncio
async def test_host_version_proof_replaces_malicious_remote_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(tmp_path, monkeypatch)
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]
    archive = io.BytesIO()
    malicious = b'{"agent":"claude-code","expected":"9.9.9","observed":"9.9.9"}'
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        info = tarfile.TarInfo(adapter._AGENT_VERSION_PROOF_NAME)
        info.size = len(malicious)
        tf.addfile(info, io.BytesIO(malicious))
    sandbox.download_archive = archive.getvalue()

    artifact_target = tmp_path / "agent-artifacts"
    await env.download_dir("/logs/artifacts", artifact_target)

    proof_path = artifact_target / adapter._AGENT_VERSION_PROOF_NAME
    assert json.loads(proof_path.read_text()) == _expected_version_proof()
    assert stat.S_IMODE(proof_path.stat().st_mode) == 0o400
    assert not any(adapter._AGENT_VERSION_PROOF_NAME in call["command"] for call in sandbox.exec_calls)


@pytest.mark.asyncio
async def test_modal_resource_modes_preserve_request_and_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={"provider": {"modal": {}}},
    )
    automatic = env._sandbox_spec()
    assert automatic.resources.cpu == 2
    assert automatic.provider_options["cpu_limit"] == 2
    assert "memory_limit_mib" not in automatic.provider_options

    env._cpu_resource_mode = ResourceMode.LIMIT
    env._memory_resource_mode = ResourceMode.LIMIT
    limited = env._sandbox_spec()
    assert limited.resources.cpu == 0.125
    assert limited.provider_options["cpu_limit"] == 2
    assert limited.resources.memory_mib == 128
    assert limited.provider_options["memory_limit_mib"] == 8192

    env._cpu_resource_mode = ResourceMode.REQUEST
    env._memory_resource_mode = ResourceMode.GUARANTEE
    requested = env._sandbox_spec()
    assert requested.resources.cpu == 2
    assert "cpu_limit" not in requested.provider_options
    assert requested.provider_options["memory_limit_mib"] == 8192


@pytest.mark.asyncio
async def test_task_network_policy_rejects_broader_provider_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = await _environment(
        tmp_path,
        monkeypatch,
        spec={"provider_options": {"network_allowlist": ["model.example"]}},
    )
    assert exact._sandbox_spec().provider_options["network_allowlist"] == ["model.example"]

    broader = await _environment(
        tmp_path,
        monkeypatch,
        spec={"provider_options": {"network_allowlist": ["model.example", "evil.example"]}},
    )
    with pytest.raises(ValueError, match="task-derived network policy"):
        broader._sandbox_spec()

    cidr = await _environment(
        tmp_path,
        monkeypatch,
        spec={"provider_options": {"outbound_cidr_allowlist": ["0.0.0.0/0"]}},
    )
    with pytest.raises(ValueError, match="CIDR egress"):
        cidr._sandbox_spec()

    both_domain_fields = await _environment(
        tmp_path,
        monkeypatch,
        spec={
            "provider_options": {
                "network_allowlist": ["model.example"],
                "outbound_domain_allowlist": ["model.example"],
            }
        },
    )
    with pytest.raises(ValueError, match="only one"):
        both_domain_fields._sandbox_spec()

    inbound = await _environment(
        tmp_path,
        monkeypatch,
        spec={"provider_options": {"inbound_cidr_allowlist": ["10.0.0.0/8"]}},
    )
    with pytest.raises(ValueError, match="inbound access"):
        inbound._sandbox_spec()

    conflicting_block = await _environment(
        tmp_path,
        monkeypatch,
        spec={"provider_options": {"block_network": True}},
    )
    with pytest.raises(ValueError, match="block_network conflicts"):
        conflicting_block._sandbox_spec()


@pytest.mark.asyncio
async def test_provider_without_disk_request_verifies_live_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_start = FakeAsyncSandbox.start

    async def start_with_capacity(self: FakeAsyncSandbox, spec: Any) -> FakeAsyncSandbox:
        await original_start(self, spec)
        self.exec_results = [SandboxExecResult(stdout="52428800\n", stderr="", return_code=0)]
        return self

    monkeypatch.setattr(FakeAsyncSandbox, "start", start_with_capacity)
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={"provider": {"modal": {}}, "supports_disk_resource": False},
    )
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]
    assert sandbox.spec.resources.disk_gib is None
    assert sandbox.exec_calls[0]["command"] == "df -Pk / | awk 'NR == 2 {print $4}'"

    async def start_without_capacity(self: FakeAsyncSandbox, spec: Any) -> FakeAsyncSandbox:
        await original_start(self, spec)
        self.exec_results = [SandboxExecResult(stdout="1024\n", stderr="", return_code=0)]
        return self

    monkeypatch.setattr(FakeAsyncSandbox, "start", start_without_capacity)
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={"provider": {"modal": {}}, "supports_disk_resource": False},
    )
    with pytest.raises(RuntimeError, match="live default is insufficient"):
        await env.start(force_build=False)
    assert FakeAsyncSandbox.instances[-1].stopped is True


@pytest.mark.asyncio
async def test_environment_validation_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_id = await adapter.register_runtime_config(
        {
            "provider": {"fake": {}},
            "supports_disable_internet": True,
            "supports_filtered_egress": True,
            "expected_agent_name": "claude-code",
            "expected_agent_version": "2.1.153",
        }
    )
    missing = EnvironmentConfig(allow_internet=False)
    with pytest.raises(FileNotFoundError):
        adapter.PierSandboxEnvironment(
            environment_dir=tmp_path,
            environment_name="task",
            session_id="session",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=missing,
            runtime_config_id=runtime_id,
        )

    env = await _environment(tmp_path, monkeypatch, spec={"unknown": True})
    with pytest.raises(ValueError, match="Unknown Pier sandbox spec keys"):
        env._sandbox_spec()
    with pytest.raises(ValueError, match="force_build is unsupported"):
        await env.start(force_build=True)
    with pytest.raises(RuntimeError, match="has not been started"):
        env._require_sandbox()
    with pytest.raises(NotImplementedError):
        await env.attach()

    await adapter.unregister_runtime_config(runtime_id)
    with pytest.raises(RuntimeError, match="runtime config is unavailable"):
        adapter._runtime_config(runtime_id)


def test_verifier_dockerfile_template_is_fail_closed(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "\n".join(
            [
                "# verifier",
                "FROM registry.example/task:tag",
                *adapter._DEEPSWE_VERIFIER_INSTRUCTIONS,
            ]
        )
    )
    assert adapter._image_from_dockerfile(dockerfile) == "registry.example/task:tag"

    dockerfile.write_text("FROM registry.example/task:tag\nRUN curl example.com")
    with pytest.raises(ValueError, match="Unsupported verifier Dockerfile"):
        adapter._image_from_dockerfile(dockerfile)

    dockerfile.write_text("FROM ${BASE}\n" + "\n".join(adapter._DEEPSWE_VERIFIER_INSTRUCTIONS))
    with pytest.raises(ValueError, match="must be concrete"):
        adapter._image_from_dockerfile(dockerfile)

    dockerfile.write_text("")
    with pytest.raises(ValueError, match="Empty Dockerfile"):
        adapter._image_from_dockerfile(dockerfile)

    dockerfile.write_text("RUN true")
    with pytest.raises(ValueError, match="must begin"):
        adapter._image_from_dockerfile(dockerfile)


def test_runtime_config_file_and_missing_value(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    config_path.write_text('{"provider": {"fake": {}}}')
    assert adapter._runtime_config(config_path=config_path) == {"provider": {"fake": {}}}
    config_path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        adapter._runtime_config(config_path=config_path)
    with pytest.raises(ValueError, match="requires runtime_config"):
        adapter._runtime_config()
    adapter.PierSandboxEnvironment.preflight()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_timeout", [0, -1, float("inf"), float("nan"), True, "1"])
async def test_transfer_timeout_must_be_positive_and_finite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_timeout: Any,
) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        await _environment(
            tmp_path,
            monkeypatch,
            runtime_overrides={"transfer_timeout_s": invalid_timeout},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_overrides", "message"),
    [
        ({"artifact_max_files": 0}, "artifact_max_files"),
        ({"artifact_max_file_bytes": True}, "artifact_max_file_bytes"),
        ({"artifact_max_total_bytes": -1}, "artifact_max_total_bytes"),
        (
            {"artifact_max_file_bytes": 9, "artifact_max_total_bytes": 8},
            "cannot exceed",
        ),
    ],
)
async def test_artifact_transfer_limits_must_be_positive_and_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await _environment(tmp_path, monkeypatch, runtime_overrides=runtime_overrides)


@pytest.mark.asyncio
async def test_artifact_transfer_limits_derive_from_runtime_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={
            "artifact_max_files": 3,
            "artifact_max_file_bytes": 1024,
            "artifact_max_total_bytes": 4096,
        },
    )
    assert env._max_download_members == 12
    assert env._artifact_max_file_bytes == 1024
    assert env._artifact_max_total_bytes == 4096
    assert env._max_download_archive_bytes == 4096 + adapter._DEFAULT_ARCHIVE_OVERHEAD_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_overrides", "message"),
    [
        ({"expected_agent_name": None}, "expected_agent_name"),
        ({"expected_agent_name": " claude-code"}, "expected_agent_name"),
        ({"expected_agent_version": None}, "expected_agent_version"),
        ({"expected_agent_version": "latest"}, "expected_agent_version"),
    ],
)
async def test_expected_agent_identity_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await _environment(tmp_path, monkeypatch, runtime_overrides=runtime_overrides)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version_stdout", "message"),
    [
        ("Claude Code version unavailable\n", "did not report exactly one x.y.z version"),
        ("2.1.154 (Claude Code)\n", r"reported version 2\.1\.154, expected 2\.1\.153"),
    ],
)
async def test_start_fails_closed_on_unverifiable_agent_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_stdout: str,
    message: str,
) -> None:
    original_start = FakeAsyncSandbox.start

    async def start_with_version(self: FakeAsyncSandbox, spec: Any) -> FakeAsyncSandbox:
        await original_start(self, spec)
        self.exec_results = [
            SandboxExecResult(stdout="ok", stderr=None, return_code=0),
            SandboxExecResult(stdout=version_stdout, stderr=None, return_code=0),
        ]
        return self

    monkeypatch.setattr(FakeAsyncSandbox, "start", start_with_version)
    env = await _environment(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match=message):
        await env.start(force_build=False)

    assert FakeAsyncSandbox.instances[-1].stopped is True


@pytest.mark.asyncio
async def test_start_failure_paths_stop_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = await _environment(tmp_path, monkeypatch, runtime_overrides={"preinstall_agent_in_image": False})
    original_start = FakeAsyncSandbox.start

    async def start_with_init_failure(self: FakeAsyncSandbox, spec: Any) -> FakeAsyncSandbox:
        await original_start(self, spec)
        self.exec_results = [SandboxExecResult(stdout="", stderr="mkdir failed", return_code=1)]
        return self

    monkeypatch.setattr(FakeAsyncSandbox, "start", start_with_init_failure)
    with pytest.raises(RuntimeError, match="initialize Pier log"):
        await env.start(force_build=False)
    assert FakeAsyncSandbox.instances[-1].stopped is True

    async def start_with_install_failure(self: FakeAsyncSandbox, spec: Any) -> FakeAsyncSandbox:
        await original_start(self, spec)
        self.exec_results = [
            SandboxExecResult(stdout="ok", stderr="", return_code=0),
            SandboxExecResult(stdout="", stderr="install failed", return_code=2),
        ]
        return self

    monkeypatch.setattr(FakeAsyncSandbox, "start", start_with_install_failure)
    env = await _environment(tmp_path, monkeypatch, runtime_overrides={"preinstall_agent_in_image": False})
    with pytest.raises(RuntimeError, match="install failed"):
        await env.start(force_build=False)
    assert FakeAsyncSandbox.instances[-1].stopped is True

    async def start_with_verification_failure(self: FakeAsyncSandbox, spec: Any) -> FakeAsyncSandbox:
        await original_start(self, spec)
        self.exec_results = [
            SandboxExecResult(stdout="ok", stderr="", return_code=0),
            SandboxExecResult(stdout="ok", stderr="", return_code=0),
            SandboxExecResult(stdout="ok", stderr="", return_code=0),
            SandboxExecResult(stdout="", stderr="version failed", return_code=3),
        ]
        return self

    monkeypatch.setattr(FakeAsyncSandbox, "start", start_with_verification_failure)
    env = await _environment(tmp_path, monkeypatch, runtime_overrides={"preinstall_agent_in_image": False})
    with pytest.raises(RuntimeError, match="verification failed"):
        await env.start(force_build=False)
    assert FakeAsyncSandbox.instances[-1].stopped is True


@pytest.mark.asyncio
async def test_stalled_startup_upload_times_out_and_stops_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM registry.example/task:tag\n" + "\n".join(adapter._DEEPSWE_VERIFIER_INSTRUCTIONS) + "\n"
    )
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={"transfer_timeout_s": 0.01},
        task_config=EnvironmentConfig(allow_internet=False),
    )
    transfer_started = asyncio.Event()
    transfer_cancelled = asyncio.Event()

    async def stalled_upload(self: FakeAsyncSandbox, local_path: Path, remote_path: str) -> None:
        del self, local_path, remote_path
        transfer_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            transfer_cancelled.set()

    monkeypatch.setattr(FakeAsyncSandbox, "upload", stalled_upload)
    with pytest.raises(TimeoutError, match="Sandbox upload timed out after 0.01 seconds"):
        await env.start(force_build=False)

    assert transfer_started.is_set()
    assert transfer_cancelled.is_set()
    assert FakeAsyncSandbox.instances[-1].stopped is True


@pytest.mark.asyncio
async def test_start_preserves_original_failure_and_cancellation_when_stop_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_stop(self: FakeAsyncSandbox) -> None:
        raise RuntimeError("cleanup-secret-must-not-be-reported")

    monkeypatch.setattr(FakeAsyncSandbox, "stop", failing_stop)

    async def failing_start(self: FakeAsyncSandbox, spec: Any) -> FakeAsyncSandbox:
        self.spec = spec
        raise ValueError("original startup failure")

    monkeypatch.setattr(FakeAsyncSandbox, "start", failing_start)
    env = await _environment(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="original startup failure") as failure:
        await env.start(force_build=False)
    assert failure.value.__notes__ == ["Sandbox cleanup also failed; cleanup details were omitted."]
    assert "cleanup-secret" not in "".join(failure.value.__notes__)

    async def cancelled_start(self: FakeAsyncSandbox, spec: Any) -> FakeAsyncSandbox:
        self.spec = spec
        raise asyncio.CancelledError("original cancellation")

    monkeypatch.setattr(FakeAsyncSandbox, "start", cancelled_start)
    env = await _environment(tmp_path, monkeypatch)
    with pytest.raises(asyncio.CancelledError, match="original cancellation") as cancellation:
        await env.start(force_build=False)
    assert cancellation.value.__notes__ == ["Sandbox cleanup also failed; cleanup details were omitted."]
    assert "cleanup-secret" not in "".join(cancellation.value.__notes__)


def _write_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for info, payload in members:
            tf.addfile(info, io.BytesIO(payload) if payload is not None else None)


def _file_member(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info, payload


def _pax_record(key: str, value: str) -> bytes:
    size = len(key.encode()) + len(value.encode()) + 3
    while True:
        record = f"{size} {key}={value}\n".encode()
        if len(record) == size:
            return record
        size = len(record)


def test_download_archive_rejects_unsafe_members_and_removes_partial_output(
    tmp_path: Path,
) -> None:
    for index, unsafe in enumerate(("../escape", "/absolute")):
        archive = tmp_path / f"unsafe-{index}.tar.gz"
        _write_archive(archive, [_file_member("first.txt", b"first"), _file_member(unsafe, b"bad")])
        target = tmp_path / f"target-{index}"
        target.mkdir()
        (target / "original.txt").write_bytes(b"original")
        with pytest.raises(RuntimeError, match="Unsafe sandbox archive member path"):
            adapter._extract_download_archive(archive, target)
        assert (target / "original.txt").read_bytes() == b"original"
        assert not (target / "first.txt").exists()
        assert not list(tmp_path.glob(f".{target.name}.extract-*"))

    for index, member_type in enumerate((tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE)):
        info = tarfile.TarInfo(f"special-{index}")
        info.type = member_type
        info.linkname = "first.txt"
        archive = tmp_path / f"special-{index}.tar.gz"
        _write_archive(archive, [(info, None)])
        with pytest.raises(RuntimeError, match="link or special file"):
            adapter._extract_download_archive(archive, tmp_path / f"special-target-{index}")


def test_download_archive_enforces_all_size_and_count_budgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "archive.tar.gz"
    _write_archive(archive, [_file_member("one", b"123"), _file_member("two", b"456")])

    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_ARCHIVE_BYTES", archive.stat().st_size - 1)
    with pytest.raises(RuntimeError, match="compressed transfer budget"):
        adapter._extract_download_archive(archive, tmp_path / "compressed")

    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_MEMBERS", 1)
    with pytest.raises(RuntimeError, match="member-count budget"):
        adapter._extract_download_archive(archive, tmp_path / "members")

    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_MEMBERS", 2)
    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_MEMBER_BYTES", 2)
    with pytest.raises(RuntimeError, match="member exceeds expanded-size budget"):
        adapter._extract_download_archive(archive, tmp_path / "member-size")

    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_MEMBER_BYTES", 3)
    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_EXPANDED_BYTES", 5)
    with pytest.raises(RuntimeError, match="cumulative expanded-size budget"):
        adapter._extract_download_archive(archive, tmp_path / "expanded")

    assert not any(path.name.startswith(".") and ".extract-" in path.name for path in tmp_path.iterdir())


def test_download_archive_rejects_hidden_extension_and_sparse_bombs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized_metadata = b"x" * 64
    extension = tarfile.TarInfo("pax-header")
    extension.type = tarfile.XHDTYPE
    extension.size = len(oversized_metadata)
    archive = tmp_path / "oversized-pax.tar.gz"
    _write_archive(archive, [(extension, oversized_metadata)])
    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_METADATA_BYTES", len(oversized_metadata) - 1)
    with pytest.raises(RuntimeError, match="extension metadata exceeds budget"):
        adapter._extract_download_archive(archive, tmp_path / "oversized-pax")

    monkeypatch.setattr(adapter, "_MAX_DOWNLOAD_METADATA_BYTES", 1024 * 1024)
    for index, (key, value) in enumerate((("size", "999999999"), ("GNU.sparse.map", "0,1"))):
        payload = _pax_record(key, value)
        extension = tarfile.TarInfo(f"pax-{index}")
        extension.type = tarfile.XHDTYPE
        extension.size = len(payload)
        archive = tmp_path / f"unsafe-pax-{index}.tar.gz"
        _write_archive(archive, [(extension, payload)])
        with pytest.raises(RuntimeError, match="unsupported sparse or size-override"):
            adapter._extract_download_archive(archive, tmp_path / f"unsafe-pax-{index}")

    sparse = tarfile.TarInfo("sparse")
    sparse.type = tarfile.GNUTYPE_SPARSE
    archive = tmp_path / "sparse.tar.gz"
    _write_archive(archive, [(sparse, None)])
    with pytest.raises(RuntimeError, match="unsupported sparse member"):
        adapter._extract_download_archive(archive, tmp_path / "sparse")


def test_download_archive_rejects_oversized_pax_path(tmp_path: Path) -> None:
    archive = tmp_path / "long-path.tar.gz"
    _write_archive(archive, [_file_member("x" * (adapter._MAX_DOWNLOAD_PATH_BYTES + 1), b"data")])
    with pytest.raises(RuntimeError, match="path exceeds metadata budget|path is too long"):
        adapter._extract_download_archive(archive, tmp_path / "long-path")


def test_download_archive_extracts_incrementally_into_empty_destination(tmp_path: Path) -> None:
    directory = tarfile.TarInfo("nested")
    directory.type = tarfile.DIRTYPE
    archive = tmp_path / "valid.tar.gz"
    _write_archive(archive, [(directory, None), _file_member("nested/result.txt", b"result")])
    target = tmp_path / "target"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "result.txt").write_bytes(b"old")
    (target / "preserved.txt").write_bytes(b"preserved")

    adapter._extract_download_archive(archive, target)

    assert (target / "nested" / "result.txt").read_bytes() == b"result"
    assert (target / "preserved.txt").read_bytes() == b"preserved"
    assert (target / "nested" / "result.txt").stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_transfer_failure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    env = await _environment(tmp_path, monkeypatch)
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("data")

    sandbox.exec_results = [SandboxExecResult(stdout="", stderr="mkdir failed", return_code=1)]
    with pytest.raises(RuntimeError, match="create sandbox directory"):
        await env.upload_file(source / "file.txt", "/target/file.txt")
    with pytest.raises(FileNotFoundError):
        await env.upload_dir(tmp_path / "missing", "/target")

    async def upload_ok(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(env, "upload_file", upload_ok)
    sandbox.exec_results = [SandboxExecResult(stdout="", stderr="extract failed", return_code=1)]
    with pytest.raises(RuntimeError, match="extract upload"):
        await env.upload_dir(source, "/target")

    sandbox.exec_results = [SandboxExecResult(stdout="", stderr="archive failed", return_code=1)]
    with pytest.raises(RuntimeError, match="archive /remote"):
        await env.download_dir("/remote", tmp_path / "download-failed")

    sandbox.exec_results = [
        SandboxExecResult(
            stdout=f"{adapter._MAX_DOWNLOAD_ARCHIVE_BYTES + 1}\n{'0' * 64}  /tmp/archive\n",
            stderr="",
            return_code=0,
        )
    ]
    downloads_before = len(sandbox.downloads)
    with pytest.raises(RuntimeError, match="compressed transfer budget"):
        await env.download_dir("/remote", tmp_path / "download-too-large")
    assert len(sandbox.downloads) == downloads_before

    sandbox.exec_results = [
        SandboxExecResult(
            stdout=f"{len(sandbox.download_archive)}\n{'0' * 64}  /tmp/archive\n",
            stderr="",
            return_code=0,
        )
    ]
    with pytest.raises(RuntimeError, match="changed during download"):
        await env.download_dir("/remote", tmp_path / "download-raced")

    digest = hashlib.sha256(sandbox.download_archive).hexdigest()
    sandbox.exec_results = [
        SandboxExecResult(
            stdout=f"{len(sandbox.download_archive)}\n{digest}  /tmp/archive\n",
            stderr="",
            return_code=0,
        ),
        SandboxExecResult(stdout="", stderr="cleanup failed", return_code=1),
    ]
    with caplog.at_level(logging.WARNING):
        await env.download_dir("/remote", tmp_path / "download")
    assert "Failed to remove sandbox transfer archive" in caplog.text
    assert sandbox.download_limits[-1] == adapter._MAX_DOWNLOAD_ARCHIVE_BYTES


@pytest.mark.asyncio
async def test_download_file_removes_partial_output_on_limit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = await _environment(tmp_path, monkeypatch)
    await env.start(force_build=False)

    async def oversized_download(
        self: FakeAsyncSandbox,
        remote_path: str,
        local_path: Path,
        *,
        max_bytes: int | None = None,
    ) -> None:
        del self, remote_path
        assert max_bytes == adapter._MAX_DOWNLOAD_FILE_BYTES
        Path(local_path).write_bytes(b"partial")
        raise SandboxDownloadLimitExceeded("too large")

    monkeypatch.setattr(FakeAsyncSandbox, "download", oversized_download)
    target = tmp_path / "artifact.patch"
    target.write_bytes(b"original")
    with pytest.raises(SandboxDownloadLimitExceeded, match="too large"):
        await env.download_file("/logs/artifacts/model.patch", target)

    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob(".artifact.patch.download-*"))


@pytest.mark.asyncio
async def test_stalled_download_times_out_without_orphaning_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={"transfer_timeout_s": 0.01},
    )
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]
    transfer_started = asyncio.Event()
    transfer_cancelled = asyncio.Event()

    async def stalled_download(
        self: FakeAsyncSandbox,
        remote_path: str,
        local_path: Path,
        *,
        max_bytes: int | None = None,
    ) -> None:
        del self, remote_path, local_path, max_bytes
        transfer_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            transfer_cancelled.set()

    monkeypatch.setattr(FakeAsyncSandbox, "download", stalled_download)
    target = tmp_path / "artifact.patch"
    target.write_bytes(b"original")
    try:
        with pytest.raises(TimeoutError, match="Sandbox download timed out after 0.01 seconds"):
            await env.download_file("/logs/artifacts/model.patch", target)
    finally:
        # Pier's Trial finally block owns this stop after a transfer error.
        await env.stop(delete=True)

    assert transfer_started.is_set()
    assert transfer_cancelled.is_set()
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob(".artifact.patch.download-*"))
    assert sandbox.stopped is True


@pytest.mark.asyncio
async def test_stalled_directory_download_times_out_and_removes_remote_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={"transfer_timeout_s": 0.01},
    )
    await env.start(force_build=False)
    sandbox = FakeAsyncSandbox.instances[-1]
    transfer_cancelled = asyncio.Event()

    async def stalled_download(
        self: FakeAsyncSandbox,
        remote_path: str,
        local_path: Path,
        *,
        max_bytes: int | None = None,
    ) -> None:
        del self, remote_path, local_path, max_bytes
        try:
            await asyncio.Event().wait()
        finally:
            transfer_cancelled.set()

    monkeypatch.setattr(FakeAsyncSandbox, "download", stalled_download)
    try:
        with pytest.raises(TimeoutError, match="Sandbox download timed out after 0.01 seconds"):
            await env.download_dir("/logs/agent", tmp_path / "agent")
    finally:
        await env.stop(delete=True)

    assert transfer_cancelled.is_set()
    assert any(call["command"].startswith("rm -f ") for call in sandbox.exec_calls)
    assert sandbox.stopped is True


@pytest.mark.asyncio
async def test_caller_cancellation_is_not_converted_to_transfer_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(
        tmp_path,
        monkeypatch,
        runtime_overrides={"transfer_timeout_s": 60},
    )
    await env.start(force_build=False)
    transfer_started = asyncio.Event()
    transfer_cancelled = asyncio.Event()

    async def stalled_download(
        self: FakeAsyncSandbox,
        remote_path: str,
        local_path: Path,
        *,
        max_bytes: int | None = None,
    ) -> None:
        del self, remote_path, local_path, max_bytes
        transfer_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            transfer_cancelled.set()

    monkeypatch.setattr(FakeAsyncSandbox, "download", stalled_download)
    download = asyncio.create_task(env.download_file("/remote/file", tmp_path / "file"))
    await transfer_started.wait()
    download.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await download
    finally:
        await env.stop(delete=True)

    assert transfer_cancelled.is_set()


@pytest.mark.asyncio
async def test_download_cleanup_preserves_original_error_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = await _environment(tmp_path, monkeypatch)
    await env.start(force_build=False)

    async def fail_archive(command: str, **_: Any) -> Any:
        if command.startswith("tar czf"):
            raise ValueError("original archive failure")
        if command.startswith("rm -f"):
            raise RuntimeError("cleanup-secret-must-not-be-reported")
        raise AssertionError(command)

    monkeypatch.setattr(env, "exec", fail_archive)
    with pytest.raises(ValueError, match="original archive failure") as failure:
        await env.download_dir("/remote", tmp_path / "error")
    assert failure.value.__notes__ == ["Sandbox transfer archive cleanup also failed; details were omitted."]
    assert "cleanup-secret" not in "".join(failure.value.__notes__)

    async def cancel_archive(command: str, **_: Any) -> Any:
        if command.startswith("tar czf"):
            raise asyncio.CancelledError("original cancellation")
        if command.startswith("rm -f"):
            raise RuntimeError("cleanup-secret-must-not-be-reported")
        raise AssertionError(command)

    monkeypatch.setattr(env, "exec", cancel_archive)
    with pytest.raises(asyncio.CancelledError, match="original cancellation") as cancellation:
        await env.download_dir("/remote", tmp_path / "cancelled")
    assert cancellation.value.__notes__ == ["Sandbox transfer archive cleanup also failed; details were omitted."]
    assert "cleanup-secret" not in "".join(cancellation.value.__notes__)
