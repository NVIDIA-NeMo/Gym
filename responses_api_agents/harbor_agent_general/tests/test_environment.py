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
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths

from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus, register_provider
from responses_api_agents.harbor_agent_general.custom_envs.nemo_gym_sandbox.environment import (
    NemoGymSandboxEnvironment,
)


PROVIDER_NAME = "nemo_gym_sandbox_test_provider"


class FakeProvider:
    """In-memory SandboxProvider that records every call."""

    name = PROVIDER_NAME
    instances: list["FakeProvider"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.created_specs: list[SandboxSpec] = []
        self.exec_calls: list[dict] = []
        self.uploads: dict[str, bytes] = {}
        self.downloads: dict[str, bytes] = {}
        self.closed_handles: list[str] = []
        self.provider_closed = False
        self.exec_results: list[SandboxExecResult] = []
        FakeProvider.instances.append(self)

    def queue_exec_result(self, result: SandboxExecResult) -> None:
        self.exec_results.append(result)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.created_specs.append(spec)
        return SandboxHandle(sandbox_id="sbx-123", provider_name=self.name, raw=None)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        timeout_s=None,
        user=None,
    ) -> SandboxExecResult:
        self.exec_calls.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
        if self.exec_results:
            return self.exec_results.pop(0)
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self.uploads[target_path] = source_path.read_bytes()

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(self.downloads[source_path])

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        self.closed_handles.append(handle.sandbox_id)

    async def aclose(self) -> None:
        self.provider_closed = True


register_provider(PROVIDER_NAME, FakeProvider, override=True)


@pytest.fixture(autouse=True)
def _reset_fake_provider():
    FakeProvider.instances.clear()
    yield
    FakeProvider.instances.clear()


def _make_environment(
    tmp_path: Path,
    *,
    task_env_config: Optional[TaskEnvironmentConfig] = None,
    environment_dir: Optional[Path] = None,
    session_id: str = "example-task__trial-1",
    **kwargs,
):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        sandbox_provider={PROVIDER_NAME: {}},
        sandbox_ttl_s=1234,
        sandbox_ready_timeout_s=56,
        exec_shell=None,
    )
    defaults.update(kwargs)
    return NemoGymSandboxEnvironment(
        environment_dir=environment_dir or tmp_path / "task" / "environment",
        environment_name="example-task",
        session_id=session_id,
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=task_env_config
        or TaskEnvironmentConfig(docker_image="docker.io/example/task:1.0", cpus=4, memory_mb=8192),
        **defaults,
    )


def _provider() -> FakeProvider:
    assert len(FakeProvider.instances) == 1
    return FakeProvider.instances[0]


class TestValidation:
    def test_requires_sandbox_provider(self, tmp_path):
        with pytest.raises(ValueError, match="sandbox_provider"):
            _make_environment(tmp_path, sandbox_provider=None)

    def test_requires_docker_image(self, tmp_path):
        with pytest.raises(ValueError, match="docker_image"):
            _make_environment(tmp_path, task_env_config=TaskEnvironmentConfig(docker_image=None))

    def test_rejects_internet_isolation_by_default(self, tmp_path):
        policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        with pytest.raises(ValueError, match="no-network"):
            _make_environment(tmp_path, network_policy=policy)

    def test_internet_isolation_opt_in(self, tmp_path):
        policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        env = _make_environment(
            tmp_path,
            network_policy=policy,
            allow_unenforced_internet_isolation=True,
        )
        assert env.can_disable_internet is True

    def test_rejects_unsafe_sandbox_path_copy(self, tmp_path):
        with pytest.raises(ValueError, match="absolute non-root path"):
            _make_environment(
                tmp_path,
                sandbox_path_copies=[{"source": "../dataset", "destination": "/app/data"}],
            )

    def test_shared_workspace_host_path_must_end_in_context_id(self, tmp_path):
        with pytest.raises(ValueError, match=r"end with the \{context_id\} placeholder"):
            _make_environment(
                tmp_path,
                shared_workspace={
                    "volume": {
                        "name": "policy-workspace",
                        "host": {"path": "/mnt/efs/data/shared/example-user/test-run/static"},
                        "mountPath": "/app",
                    }
                },
            )


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_builds_spec_and_creates_log_dirs(self, tmp_path):
        env = _make_environment(tmp_path, sandbox_metadata={"harbor-benchmark": "tb-2-1"})
        await env.start(force_build=False)

        provider = _provider()
        assert len(provider.created_specs) == 1
        spec = provider.created_specs[0]
        assert spec.image == "docker.io/example/task:1.0"
        assert spec.ttl_s == 1234
        assert spec.ready_timeout_s == 56
        assert spec.resources.cpu == 4.0
        assert spec.resources.memory_mib == 8192
        assert spec.metadata["harbor-session"] == "example-task__trial-1"
        assert spec.metadata["harbor-task"] == "example-task"
        assert spec.metadata["harbor-benchmark"] == "tb-2-1"

        assert len(provider.exec_calls) == 1
        assert provider.exec_calls[0]["command"] == "mkdir -p /logs/agent /logs/verifier /logs/artifacts"

    @pytest.mark.asyncio
    async def test_start_passes_provider_options_through(self, tmp_path):
        env = _make_environment(
            tmp_path,
            sandbox_provider_options={"resource_requests": {"cpu": 0.25, "memory_mib": 1024}},
        )
        await env.start(force_build=False)
        spec = _provider().created_specs[0]
        assert spec.provider_options == {"resource_requests": {"cpu": 0.25, "memory_mib": 1024}}

    @pytest.mark.asyncio
    async def test_start_expands_task_templates_in_provider_options(self, tmp_path):
        env = _make_environment(
            tmp_path,
            sandbox_provider_options={
                "volumes": [
                    {
                        "name": "capsules",
                        "host_path": "/mnt/s3/data/train/{task_name}",
                        "mount_path": "/app/data",
                        "sub_path": "CapsuleFolder-{task_id}",
                    }
                ],
                "extensions": {"trial": "{session_id}", "task": "{environment_name}"},
            },
        )
        await env.start(force_build=False)

        assert _provider().created_specs[0].provider_options == {
            "volumes": [
                {
                    "name": "capsules",
                    "host_path": "/mnt/s3/data/train/example-task",
                    "mount_path": "/app/data",
                    "sub_path": "CapsuleFolder-example-task",
                }
            ],
            "extensions": {"trial": "example-task__trial-1", "task": "example-task"},
        }

    @pytest.mark.asyncio
    async def test_start_rejects_unknown_provider_option_template_before_create(self, tmp_path):
        env = _make_environment(
            tmp_path,
            sandbox_provider_options={
                "volumes": [
                    {
                        "name": "capsules",
                        "host_path": "/mnt/s3/data/train/{task_nmae}",
                        "mount_path": "/app/data",
                    }
                ]
            },
        )

        with pytest.raises(ValueError, match=r"Unsupported template placeholder.*\{task_nmae\}"):
            await env.start(force_build=False)

        assert FakeProvider.instances == []

    @pytest.mark.asyncio
    async def test_start_copies_mounted_task_data_before_environment_upload(self, tmp_path):
        environment_dir = tmp_path / "task" / "environment"
        (environment_dir / "data").mkdir(parents=True)
        (environment_dir / "data" / "local.txt").write_text("do not upload")
        (environment_dir / "instruction.txt").write_text("analyze /app/data")
        env = _make_environment(
            tmp_path,
            environment_dir=environment_dir,
            task_env_config=TaskEnvironmentConfig(
                docker_image="docker.io/example/task:1.0",
                workdir="/app",
            ),
            sandbox_path_copies=[
                {
                    "source": "/mounted/tasks/{environment_name}/environment/data",
                    "destination": "/app/data",
                }
            ],
            sandbox_path_copy_timeout_s=321,
            environment_upload_excludes=["data"],
        )

        await env.start(force_build=False)

        provider = _provider()
        assert provider.created_specs[0].env == {}
        copy_call = provider.exec_calls[1]
        assert copy_call == {
            "command": (
                "test -d /mounted/tasks/example-task/environment/data && mkdir -p /app/data "
                "&& cp -a -- /mounted/tasks/example-task/environment/data/. /app/data/"
            ),
            "cwd": "/",
            "env": None,
            "timeout_s": 321,
            "user": None,
        }
        upload_call_index = next(
            index for index, call in enumerate(provider.exec_calls) if "tar -xzf" in call["command"]
        )
        assert upload_call_index > 1
        archive = next(iter(provider.uploads.values()))
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            names = {member.name for member in tar.getmembers()}
        assert "./instruction.txt" in names
        assert not any(name == "./data" or name.startswith("./data/") for name in names)

    @pytest.mark.asyncio
    async def test_start_honours_harbor_resource_overrides(self, tmp_path):
        # Harbor's base constructor applies override_* onto task_env_config, which
        # is what the spec reads. If that ever stops happening the sandbox silently
        # keeps the task's own smaller limits and disk-heavy tasks get evicted.
        env = _make_environment(tmp_path, override_cpus=8, override_memory_mb=16384, override_storage_mb=30720)
        await env.start(force_build=False)
        resources = _provider().created_specs[0].resources
        assert resources.cpu == 8.0
        assert resources.memory_mib == 16384
        assert resources.disk_gib == 30

    @pytest.mark.asyncio
    async def test_start_applies_image_rewrites(self, tmp_path):
        env = _make_environment(
            tmp_path,
            image_rewrites=[{"from": "docker.io/", "to": "mirror.example.com/"}],
        )
        await env.start(force_build=False)
        assert _provider().created_specs[0].image == "mirror.example.com/example/task:1.0"

    @pytest.mark.asyncio
    async def test_start_applies_image_and_entrypoint_overrides(self, tmp_path):
        env = _make_environment(
            tmp_path,
            image_override="docker.io/example/replacement@sha256:1234",
            entrypoint=["sh", "/opt/entrypoint.sh", "tail", "-f", "/dev/null"],
        )
        await env.start(force_build=False)
        spec = _provider().created_specs[0]
        assert spec.image == "docker.io/example/replacement@sha256:1234"
        assert spec.entrypoint == ["sh", "/opt/entrypoint.sh", "tail", "-f", "/dev/null"]

    @pytest.mark.asyncio
    async def test_start_applies_task_env_and_workdir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUBRIC_MODEL_API_KEY", "test-key")  # pragma: allowlist secret
        task_config = TaskEnvironmentConfig(
            docker_image="docker.io/example/task:1.0",
            workdir="/app",
            env={
                "RUBRIC_MODEL_API_KEY": "${RUBRIC_MODEL_API_KEY}",
                "TASK_SETTING": "task-value",
            },
        )
        env = _make_environment(
            tmp_path,
            task_env_config=task_config,
            sandbox_env={"TASK_SETTING": "sandbox-value"},
        )

        await env.start(force_build=False)

        spec = _provider().created_specs[0]
        assert spec.workdir == "/app"
        assert spec.env == {
            "RUBRIC_MODEL_API_KEY": "test-key",  # pragma: allowlist secret
            "TASK_SETTING": "sandbox-value",
        }

    @pytest.mark.asyncio
    async def test_start_uploads_prebuilt_image_environment_dir(self, tmp_path):
        environment_dir = tmp_path / "task" / "environment"
        environment_dir.mkdir(parents=True)
        (environment_dir / "task.jsonl").write_text('{"task": 1}\n')
        env = _make_environment(
            tmp_path,
            task_env_config=TaskEnvironmentConfig(
                docker_image="docker.io/example/task:1.0",
                workdir="/app",
            ),
        )

        await env.start(force_build=False)

        provider = _provider()
        assert len(provider.uploads) == 1
        archive = next(iter(provider.uploads.values()))
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            task_file = tar.extractfile("./task.jsonl")
            assert task_file is not None
            assert task_file.read() == b'{"task": 1}\n'
        upload_commands = [call["command"] for call in provider.exec_calls]
        assert any("tar -xzf" in command and "-C /app" in command for command in upload_commands)

    @pytest.mark.asyncio
    async def test_separate_verifier_uploads_tests_then_accepts_policy_workspace(self, tmp_path):
        tests_dir = tmp_path / "task" / "steps" / "rollout" / "tests"
        tests_dir.mkdir(parents=True)
        (tests_dir / "judge_instruction.md").write_text("Judge the submission.\n")
        workspace = tmp_path / "policy-workspace"
        workspace.mkdir()
        (workspace / "REPORT.md").write_text("Policy report.\n")
        env = _make_environment(
            tmp_path,
            environment_dir=tests_dir,
            session_id="example-task__trial-1__verifier__rollout",
            sandbox_provider_options={
                "volumes": [
                    {
                        "name": "capsules",
                        "mount_path": "/app/data",
                        "sub_path": "CapsuleFolder-{task_id}",
                    }
                ]
            },
            task_env_config=TaskEnvironmentConfig(
                docker_image="docker.io/example/task:1.0",
                workdir="/app",
            ),
        )

        await env.start(force_build=False)
        await env.upload_dir(workspace, "/app")

        provider = _provider()
        assert provider.created_specs[0].metadata["harbor-role"] == "verifier"
        assert provider.created_specs[0].provider_options["volumes"] == [
            {
                "name": "capsules",
                "mount_path": "/app/data",
                "sub_path": "CapsuleFolder-example-task",
            }
        ]
        upload_commands = [call["command"] for call in provider.exec_calls]
        assert any("tar -xzf" in command and "-C /tests" in command for command in upload_commands)
        assert any("tar -xzf" in command and "-C /app" in command for command in upload_commands)

    @pytest.mark.asyncio
    async def test_separate_verifier_restores_sandbox_copy_after_workspace_reset(self, tmp_path):
        tests_dir = tmp_path / "task" / "steps" / "rollout" / "tests"
        tests_dir.mkdir(parents=True)
        env = _make_environment(
            tmp_path,
            environment_dir=tests_dir,
            session_id="example-task__trial-1__verifier__rollout",
            task_env_config=TaskEnvironmentConfig(
                docker_image="docker.io/example/task:1.0",
                workdir="/app",
            ),
            sandbox_path_copies=[
                {
                    "source": "/mounted/tasks/{environment_name}/environment/data",
                    "destination": "/app/data",
                }
            ],
        )
        await env.start(force_build=False)
        provider = _provider()
        provider.exec_calls.clear()

        await env.empty_dirs(["/app"], chmod=True)

        assert len(provider.exec_calls) == 2
        assert provider.exec_calls[0]["cwd"] == "/"
        assert "find /app -mindepth 1" in provider.exec_calls[0]["command"]
        assert provider.exec_calls[1] == {
            "command": (
                "test -d /mounted/tasks/example-task/environment/data && mkdir -p /app/data "
                "&& cp -a -- /mounted/tasks/example-task/environment/data/. /app/data/"
            ),
            "cwd": "/",
            "env": None,
            "timeout_s": 1200,
            "user": None,
        }

    @pytest.mark.asyncio
    async def test_workspace_reset_preserves_nested_sandbox_volume(self, tmp_path):
        env = _make_environment(
            tmp_path,
            session_id="example-task__trial-1__verifier__rollout",
            sandbox_provider_options={
                "volumes": [
                    {
                        "name": "problem-data",
                        "mountPath": "/app/data",
                        "readOnly": True,
                    }
                ]
            },
        )
        await env.start(force_build=False)
        provider = _provider()
        provider.exec_calls.clear()

        await env.empty_dirs(["/app"], chmod=True)

        command = provider.exec_calls[0]["command"]
        assert "! -path /app/data" in command
        assert "find /app -mindepth 1 -maxdepth 1" in command


class TestDirectSharedWorkspace:
    @staticmethod
    def _config() -> dict:
        return {
            "volume": {
                "name": "policy-workspace",
                "host": {"path": "/mnt/efs/data/shared/example-user/nemo-gym-harbor-artifacts/test-run/{context_id}"},
                "mountPath": "/app",
            },
            "handoff_timeout_s": 60,
            "cleanup_timeout_s": 60,
            "cleanup_ttl_s": 120,
        }

    @pytest.mark.asyncio
    async def test_handoff_is_zero_copy_and_verifier_mount_is_read_only(self, tmp_path):
        context_id = uuid4()
        source_env = _make_environment(
            tmp_path,
            shared_workspace=self._config(),
            sandbox_path_symlinks=[{"source": "/data", "destination": "/app/data"}],
            sandbox_provider_options={
                "volumes": [
                    {
                        "name": "problem-data",
                        "host": {"path": "/mnt/s3/data/train/{task_name}"},
                        "mountPath": "/data",
                        "readOnly": True,
                    }
                ]
            },
        )
        source_env.context_id = context_id
        await source_env.start(force_build=False)
        source_provider = _provider()
        source_provider.queue_exec_result(SandboxExecResult(stdout=f"{'a' * 64}\n", stderr=None, return_code=0))

        artifact_dir = tmp_path / "artifacts" / "app"
        await source_env.download_dir_with_exclusions(
            source_dir="/app",
            target_dir=artifact_dir,
            exclude=["data", ".opencode"],
        )
        marker_path = artifact_dir / ".nemo-gym-shared-workspace.json"
        before_stop = json.loads(marker_path.read_text())
        assert before_stop["source_environment_stopped"] is False
        assert before_stop["consumed"] is False
        assert before_stop["host_path"].endswith(f"/test-run/{context_id}")
        policy_volume, data_volume = source_provider.created_specs[0].provider_options["volumes"]
        assert policy_volume == {
            "name": "policy-workspace",
            "host": {"path": f"/mnt/efs/data/shared/example-user/nemo-gym-harbor-artifacts/test-run/{context_id}"},
            "mountPath": "/app",
            "readOnly": False,
        }
        assert data_volume["mountPath"] == "/data"
        assert data_volume["readOnly"] is True
        symlink_command = source_provider.exec_calls[1]["command"]
        assert "ln -s -- /data /app/data" in symlink_command
        handoff_command = source_provider.exec_calls[-1]["command"]
        assert "rm -rf -- /app/.opencode" in handoff_command
        assert "test -L /app/data" in handoff_command
        assert "rm -rf -- /app/data" not in handoff_command
        assert "--exclude=data" in handoff_command
        assert "cp -a" not in handoff_command
        assert "workspace.tar" not in handoff_command
        assert source_provider.uploads == {}
        assert source_provider.downloads == {}

        await source_env.stop(delete=True)
        after_stop = json.loads(marker_path.read_text())
        assert after_stop["source_environment_stopped"] is True
        assert len(FakeProvider.instances) == 1

        FakeProvider.instances.clear()
        tests_dir = tmp_path / "task" / "steps" / "rollout" / "tests"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test.sh").write_text("true\n")
        verifier_env = _make_environment(
            tmp_path,
            environment_dir=tests_dir,
            session_id="example-task__trial-1__verifier__rollout",
            shared_workspace=self._config(),
            sandbox_path_symlinks=[{"source": "/data", "destination": "/app/data"}],
            sandbox_provider_options={
                "volumes": [
                    {
                        "name": "problem-data",
                        "host": {"path": "/mnt/s3/data/train/{task_name}"},
                        "mountPath": "/data",
                        "readOnly": True,
                    }
                ]
            },
        )
        verifier_env.context_id = context_id
        await verifier_env.start(force_build=False)
        verifier_provider = _provider()
        verifier_symlink_command = verifier_provider.exec_calls[1]["command"]
        assert "test -L /app/data" in verifier_symlink_command
        assert "ln -s" not in verifier_symlink_command
        _, verifier_data_volume = verifier_provider.created_specs[0].provider_options["volumes"]
        assert verifier_data_volume["mountPath"] == "/data"
        assert verifier_data_volume["readOnly"] is True
        verifier_provider.exec_calls.clear()
        verifier_provider.uploads.clear()
        verifier_provider.queue_exec_result(SandboxExecResult(stdout=f"{'a' * 64}\n", stderr=None, return_code=0))

        await verifier_env.upload_dir(artifact_dir, "/app")

        verifier_volume = verifier_provider.created_specs[0].provider_options["volumes"][0]
        assert verifier_volume["host"]["path"].endswith(f"/test-run/{context_id}")
        assert verifier_volume["mountPath"] == "/app"
        assert verifier_volume["readOnly"] is True
        assert len(verifier_provider.exec_calls) == 1
        assert "sha256sum" in verifier_provider.exec_calls[0]["command"]
        assert "cp -a" not in verifier_provider.exec_calls[0]["command"]
        assert verifier_provider.uploads == {}
        assert verifier_provider.downloads == {}
        assert json.loads(marker_path.read_text())["consumed"] is True

        await verifier_env.stop(delete=True)

        assert verifier_provider.closed_handles == ["sbx-123"]
        assert len(FakeProvider.instances) == 2
        cleanup_provider = FakeProvider.instances[1]
        cleanup_volume = cleanup_provider.created_specs[0].provider_options["volumes"][0]
        assert cleanup_volume == {
            "name": "policy-workspace-cleanup",
            "host": {"path": "/mnt/efs/data/shared/example-user/nemo-gym-harbor-artifacts/test-run"},
            "mountPath": "/nemo-gym-workspace-cleanup",
            "readOnly": False,
        }
        assert cleanup_provider.created_specs[0].metadata["harbor-role"] == "workspace-cleanup"
        assert f"rm -rf -- /nemo-gym-workspace-cleanup/{context_id}" in cleanup_provider.exec_calls[0]["command"]
        assert cleanup_provider.closed_handles == ["sbx-123"]

    @pytest.mark.asyncio
    async def test_unconsumed_policy_workspace_is_cleaned_after_policy_stop(self, tmp_path):
        context_id = uuid4()
        env = _make_environment(tmp_path, shared_workspace=self._config())
        env.context_id = context_id
        await env.start(force_build=False)
        policy_provider = _provider()

        await env.stop(delete=True)

        assert policy_provider.closed_handles == ["sbx-123"]
        assert len(FakeProvider.instances) == 2
        cleanup_provider = FakeProvider.instances[1]
        assert f"rm -rf -- /nemo-gym-workspace-cleanup/{context_id}" in cleanup_provider.exec_calls[0]["command"]

    @pytest.mark.asyncio
    async def test_rejects_shared_workspace_before_policy_stop(self, tmp_path):
        context_id = uuid4()
        env = _make_environment(
            tmp_path,
            shared_workspace=self._config(),
        )
        env.context_id = context_id
        await env.start(force_build=False)
        _provider().queue_exec_result(SandboxExecResult(stdout=f"{'a' * 64}\n", stderr=None, return_code=0))
        artifact_dir = tmp_path / "artifacts" / "app"
        await env.download_dir(source_dir="/app", target_dir=artifact_dir)

        FakeProvider.instances.clear()
        verifier_env = _make_environment(
            tmp_path,
            session_id="example-task__trial-1__verifier__rollout",
            shared_workspace=self._config(),
        )
        verifier_env.context_id = context_id
        await verifier_env.start(force_build=False)

        with pytest.raises(RuntimeError, match="before source sandbox teardown"):
            await verifier_env.upload_dir(artifact_dir, "/app")

    @pytest.mark.asyncio
    async def test_workspace_validation_failure_reports_stdout_and_stderr(self, tmp_path):
        env = _make_environment(
            tmp_path,
            shared_workspace=self._config(),
        )
        await env.start(force_build=False)
        _provider().queue_exec_result(
            SandboxExecResult(
                stdout="unsupported workspace entry: ./socket\n",
                stderr="exit status 73",
                return_code=73,
            )
        )

        with pytest.raises(RuntimeError) as error:
            await env.download_dir_with_exclusions(
                source_dir="/app",
                target_dir=tmp_path / "artifacts" / "app",
                exclude=["data", ".opencode"],
            )

        assert "unsupported workspace entry: ./socket" in str(error.value)
        assert "exit status 73" in str(error.value)

    @pytest.mark.asyncio
    async def test_failed_policy_stop_does_not_authorize_shared_handoff(self, tmp_path):
        env = _make_environment(
            tmp_path,
            shared_workspace=self._config(),
        )
        await env.start(force_build=False)
        _provider().queue_exec_result(SandboxExecResult(stdout=f"{'a' * 64}\n", stderr=None, return_code=0))
        artifact_dir = tmp_path / "artifacts" / "app"
        await env.download_dir(source_dir="/app", target_dir=artifact_dir)
        marker_path = artifact_dir / ".nemo-gym-shared-workspace.json"
        provider = _provider()
        provider.close = AsyncMock(side_effect=TimeoutError("kill timed out"))

        with pytest.raises(TimeoutError, match="kill timed out"):
            await env.stop(delete=True)

        assert json.loads(marker_path.read_text())["source_environment_stopped"] is False

    @pytest.mark.asyncio
    async def test_policy_workspace_reset_does_not_restore_sandbox_copy(self, tmp_path):
        env = _make_environment(
            tmp_path,
            task_env_config=TaskEnvironmentConfig(
                docker_image="docker.io/example/task:1.0",
                workdir="/app",
            ),
            sandbox_path_copies=[{"source": "/mounted/data", "destination": "/app/data"}],
        )
        await env.start(force_build=False)
        provider = _provider()
        provider.exec_calls.clear()

        await env.empty_dirs(["/app"], chmod=True)

        assert len(provider.exec_calls) == 1
        assert provider.exec_calls[0]["cwd"] == "/"

    @pytest.mark.asyncio
    async def test_start_can_exclude_volume_backed_environment_data(self, tmp_path):
        environment_dir = tmp_path / "task" / "environment"
        (environment_dir / "data").mkdir(parents=True)
        (environment_dir / "task.jsonl").write_text('{"task": 1}\n')
        (environment_dir / "data" / "large.bin").write_bytes(b"large-data")
        env = _make_environment(
            tmp_path,
            task_env_config=TaskEnvironmentConfig(
                docker_image="docker.io/example/task:1.0",
                workdir="/app",
            ),
            environment_upload_excludes=["data"],
        )

        await env.start(force_build=False)

        archive = next(iter(_provider().uploads.values()))
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            names = {member.name for member in tar.getmembers()}
        assert "./task.jsonl" in names
        assert not any(name == "./data" or name.startswith("./data/") for name in names)

    def test_rejects_unsafe_environment_upload_excludes(self, tmp_path):
        with pytest.raises(ValueError, match="relative paths"):
            _make_environment(tmp_path, environment_upload_excludes=["../data"])

    @pytest.mark.asyncio
    async def test_stop_always_kills_sandbox(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        await env.stop(delete=False)

        provider = _provider()
        assert provider.closed_handles == ["sbx-123"]
        assert provider.provider_closed is True
        # Idempotent.
        await env.stop(delete=True)
        assert provider.closed_handles == ["sbx-123"]

    @pytest.mark.asyncio
    async def test_start_failure_on_log_dir_creation(self, tmp_path):
        env = _make_environment(tmp_path)
        # The provider instance is created inside start(); prime the failure on
        # the class so the first exec (the mkdir) fails.
        original_init = FakeProvider.__init__

        def _init_with_failure(self, **kwargs):
            original_init(self, **kwargs)
            self.queue_exec_result(SandboxExecResult(stdout=None, stderr="disk full", return_code=1))

        FakeProvider.__init__ = _init_with_failure
        try:
            with pytest.raises(RuntimeError, match="disk full"):
                await env.start(force_build=False)
        finally:
            FakeProvider.__init__ = original_init


class TestExec:
    @pytest.mark.asyncio
    async def test_exec_passthrough_and_result_mapping(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()
        provider.queue_exec_result(SandboxExecResult(stdout="out", stderr="err", return_code=7))

        result = await env.exec("echo hi", cwd="/app", env={"A": "1"}, timeout_sec=42, user=1000)
        assert (result.stdout, result.stderr, result.return_code) == ("out", "err", 7)
        call = provider.exec_calls[-1]
        assert call["command"] == "echo hi"
        assert call["cwd"] == "/app"
        assert call["env"] == {"A": "1"}
        assert call["timeout_s"] == 42
        assert call["user"] == 1000

    @pytest.mark.asyncio
    async def test_exec_uses_harbor_default_user(self, tmp_path):
        env = _make_environment(tmp_path)
        env.default_user = "agent"
        await env.start(force_build=False)

        await env.exec("id")

        assert _provider().exec_calls[-1]["user"] == "agent"

    @pytest.mark.asyncio
    async def test_exec_merges_task_and_per_command_env(self, tmp_path):
        task_config = TaskEnvironmentConfig(
            docker_image="docker.io/example/task:1.0",
            env={"TASK_SETTING": "task-value", "OVERRIDDEN": "task-value"},
        )
        env = _make_environment(tmp_path, task_env_config=task_config)
        await env.start(force_build=False)

        await env.exec("env", env={"OVERRIDDEN": "command-value"})

        assert _provider().exec_calls[-1]["env"] == {
            "TASK_SETTING": "task-value",
            "OVERRIDDEN": "command-value",
        }

    @pytest.mark.asyncio
    async def test_exec_wraps_commands_in_interactive_bash_by_default(self, tmp_path):
        env = _make_environment(tmp_path, exec_shell="bash -ic")
        await env.start(force_build=False)
        await env.exec("tmux -V")
        assert _provider().exec_calls[-1]["command"] == "bash -ic 'tmux -V'"

    @pytest.mark.asyncio
    async def test_exec_applies_default_timeout(self, tmp_path):
        env = _make_environment(tmp_path, default_exec_timeout_s=999)
        await env.start(force_build=False)
        await env.exec("true")
        assert _provider().exec_calls[-1]["timeout_s"] == 999

    @pytest.mark.asyncio
    async def test_exec_default_timeout_covers_verifier_budgets(self, tmp_path):
        # Harbor's verifier calls exec() without timeout_sec, so a small default
        # silently truncates verification and scores the task 0.
        env = _make_environment(tmp_path, exec_shell=None)
        await env.start(force_build=False)
        await env.exec("true")
        assert _provider().exec_calls[-1]["timeout_s"] >= 900

    @pytest.mark.asyncio
    async def test_exec_requires_started_sandbox(self, tmp_path):
        env = _make_environment(tmp_path)
        with pytest.raises(RuntimeError, match="not running"):
            await env.exec("true")

    @pytest.mark.asyncio
    async def test_exec_cpu_pin_wraps_outside_exec_shell(self, tmp_path):
        env = _make_environment(tmp_path, exec_shell="bash -ic", cpu_pin_enabled=True)
        await env.start(force_build=False)
        await env.exec("tmux -V")
        command = _provider().exec_calls[-1]["command"]
        # The pin must wrap the whole `bash -ic '...'` so children inherit it.
        assert command.startswith("__osb_w=4; ")
        assert command.endswith("$__osb_pin bash -ic 'tmux -V'")
        assert "taskset -c $__osb_s-$((__osb_s + __osb_w - 1))" in command
        # Fail-open branch present: unpinned when taskset/nproc can't cooperate.
        assert '__osb_pin=""' in command

    @pytest.mark.asyncio
    async def test_exec_cpu_pin_disabled_by_default(self, tmp_path):
        env = _make_environment(tmp_path, exec_shell=None)
        await env.start(force_build=False)
        await env.exec("true")
        assert _provider().exec_calls[-1]["command"] == "true"

    @pytest.mark.asyncio
    async def test_exec_cpu_pin_skips_unspecified_task_cpu(self, tmp_path):
        # Harbor 0.20 leaves an unspecified CPU count as None. Without a known
        # cgroup limit, there is no meaningful affinity width to apply.
        env = _make_environment(
            tmp_path,
            exec_shell=None,
            cpu_pin_enabled=True,
            task_env_config=TaskEnvironmentConfig(docker_image="docker.io/example/task:1.0"),
        )
        await env.start(force_build=False)
        await env.exec("true")
        command = _provider().exec_calls[-1]["command"]
        assert command == "true"

    def test_cpu_pin_prefix_is_valid_posix_sh(self):
        import shutil
        import subprocess

        if shutil.which("sh") is None:
            pytest.skip("sh not available")
        # Must run under plain POSIX sh; the huge width takes the fail-open
        # branch, which still has to run the command.
        from responses_api_agents.harbor_agent_general.custom_envs.nemo_gym_sandbox.environment import _cpu_pin_prefix

        script = f"{_cpu_pin_prefix(100000)} echo pinned-ok"
        proc = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "pinned-ok"


class TestFileTransfer:
    @pytest.mark.asyncio
    async def test_upload_file_creates_parent(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()

        source = tmp_path / "hello.txt"
        source.write_text("hello")
        await env.upload_file(source, "/opt/data/hello.txt")

        assert provider.uploads["/opt/data/hello.txt"] == b"hello"
        assert provider.exec_calls[-1]["command"] == "mkdir -p /opt/data"

    @pytest.mark.asyncio
    async def test_upload_dir_ships_contents_as_tarball(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()

        source = tmp_path / "tests"
        (source / "nested").mkdir(parents=True)
        (source / "test.sh").write_text("#!/bin/bash\necho ok\n")
        (source / "nested" / "data.txt").write_text("data")

        await env.upload_dir(source, "/tests")

        [(remote_tar, payload)] = [(path, data) for path, data in provider.uploads.items() if path.endswith(".tar.gz")]
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            names = {member.name for member in tar.getmembers() if member.isfile()}
        assert names == {"./test.sh", "./nested/data.txt"}

        extract_call = provider.exec_calls[-1]["command"]
        assert f"tar -xzf {remote_tar} -C /tests" in extract_call
        assert "mkdir -p /tests" in extract_call

    @pytest.mark.asyncio
    async def test_upload_dir_falls_back_to_per_file(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()

        source = tmp_path / "tests"
        source.mkdir()
        (source / "test.sh").write_text("echo ok")

        provider.queue_exec_result(SandboxExecResult(stdout=None, stderr="tar: not found", return_code=127))
        await env.upload_dir(source, "/tests")

        assert provider.uploads["/tests/test.sh"] == b"echo ok"

    @pytest.mark.asyncio
    async def test_download_file(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()
        provider.downloads["/logs/agent/log.txt"] = b"log-line"

        target = tmp_path / "out" / "log.txt"
        await env.download_file("/logs/agent/log.txt", target)
        assert target.read_bytes() == b"log-line"

    @pytest.mark.asyncio
    async def test_download_dir_extracts_tarball(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()

        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as tar:
            content = b"reward"
            info = tarfile.TarInfo("./reward.txt")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

        async def _exec_and_stash(handle, command, **kwargs):
            provider.exec_calls.append({"command": command, **kwargs})
            match = re.search(r"tar -czf (\S+)", command)
            if match is not None:
                remote_tar = match.group(1)
                provider.downloads[remote_tar] = payload.getvalue()
            return SandboxExecResult(stdout="", stderr=None, return_code=0)

        provider.exec = _exec_and_stash

        target = tmp_path / "verifier-out"
        await env.download_dir("/logs/verifier", target)
        assert (target / "reward.txt").read_bytes() == b"reward"
        archive_call = next(call for call in provider.exec_calls if "tar -czf" in call["command"])
        assert archive_call["cwd"] == "/"
        assert "cp -a -- /logs/verifier/." in archive_call["command"]

    @pytest.mark.asyncio
    async def test_download_dir_with_exclusions_does_not_snapshot_excluded_data(self, tmp_path):
        env = _make_environment(
            tmp_path,
            task_env_config=TaskEnvironmentConfig(
                docker_image="docker.io/example/task:1.0",
                workdir="/app",
            ),
        )
        await env.start(force_build=False)
        provider = _provider()

        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as tar:
            content = b"report"
            info = tarfile.TarInfo("./REPORT.md")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

        async def _exec_and_stash(handle, command, **kwargs):
            provider.exec_calls.append({"command": command, **kwargs})
            match = re.search(r"tar -czf (\S+)", command)
            if match is not None:
                provider.downloads[match.group(1)] = payload.getvalue()
            return SandboxExecResult(stdout="", stderr=None, return_code=0)

        provider.exec = _exec_and_stash
        target = tmp_path / "policy-workspace"

        await env.download_dir_with_exclusions(
            source_dir="/app",
            target_dir=target,
            exclude=["data"],
        )

        assert (target / "REPORT.md").read_bytes() == b"report"
        archive_call = next(call for call in provider.exec_calls if "tar -czf" in call["command"])
        assert archive_call["cwd"] == "/"
        assert "--exclude=data" in archive_call["command"]
        assert "cp -a" not in archive_call["command"]

    @pytest.mark.asyncio
    async def test_download_dir_with_exclusions_fallback_skips_excluded_data(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()
        provider.downloads["/app/REPORT.md"] = b"report"
        provider.queue_exec_result(SandboxExecResult(stdout=None, stderr="tar failed", return_code=1))
        provider.queue_exec_result(SandboxExecResult(stdout="", stderr=None, return_code=0))
        provider.queue_exec_result(
            SandboxExecResult(
                stdout="/app/REPORT.md\n/app/data/input.csv\n",
                stderr=None,
                return_code=0,
            )
        )

        target = tmp_path / "policy-workspace"
        await env.download_dir_with_exclusions(
            source_dir="/app",
            target_dir=target,
            exclude=["data"],
        )

        assert (target / "REPORT.md").read_bytes() == b"report"
        assert not (target / "data").exists()

    @pytest.mark.asyncio
    async def test_download_dir_falls_back_to_per_file(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()
        provider.downloads["/logs/verifier/reward.txt"] = b"1.0"

        provider.queue_exec_result(SandboxExecResult(stdout=None, stderr="tar: not found", return_code=127))
        # rm -f of the leftover tarball.
        provider.queue_exec_result(SandboxExecResult(stdout="", stderr=None, return_code=0))
        provider.queue_exec_result(SandboxExecResult(stdout="/logs/verifier/reward.txt\n", stderr=None, return_code=0))

        target = tmp_path / "verifier-out"
        await env.download_dir("/logs/verifier", target)
        assert (target / "reward.txt").read_bytes() == b"1.0"
