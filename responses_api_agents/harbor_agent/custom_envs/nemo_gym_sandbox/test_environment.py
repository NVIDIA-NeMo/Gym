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
import tarfile
from pathlib import Path
from typing import Optional

import pytest
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus, register_provider
from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import NemoGymSandboxEnvironment
from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.uploaded_environment import (
    UploadedNemoGymSandboxEnvironment,
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


def _make_environment(tmp_path: Path, *, task_env_config: Optional[TaskEnvironmentConfig] = None, **kwargs):
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
        environment_dir=tmp_path / "task" / "environment",
        environment_name="example-task",
        session_id="example-task__trial-1",
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
        config = TaskEnvironmentConfig(docker_image="example/task:1.0", allow_internet=False)
        with pytest.raises(ValueError, match="allow_internet"):
            _make_environment(tmp_path, task_env_config=config)

    def test_internet_isolation_opt_in(self, tmp_path):
        config = TaskEnvironmentConfig(docker_image="example/task:1.0", allow_internet=False)
        env = _make_environment(tmp_path, task_env_config=config, allow_unenforced_internet_isolation=True)
        assert env.can_disable_internet is True


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
        assert provider.exec_calls[0]["command"] == "mkdir -p /logs/agent /logs/verifier"

    @pytest.mark.asyncio
    async def test_start_applies_image_rewrites(self, tmp_path):
        env = _make_environment(
            tmp_path,
            image_rewrites=[{"from": "docker.io/", "to": "mirror.example.com/"}],
        )
        await env.start(force_build=False)
        assert _provider().created_specs[0].image == "mirror.example.com/example/task:1.0"

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
    async def test_exec_scopes_environment_to_command_and_maps_result(self, tmp_path):
        env = _make_environment(tmp_path)
        await env.start(force_build=False)
        provider = _provider()
        provider.queue_exec_result(SandboxExecResult(stdout="out", stderr="err", return_code=7))

        result = await env.exec(
            "echo hi",
            cwd="/app",
            env={"A": "value with space"},
            timeout_sec=42,
        )
        assert (result.stdout, result.stderr, result.return_code) == ("out", "err", 7)
        call = provider.exec_calls[-1]
        assert call["command"] == "env 'A=value with space' echo hi"
        assert call["cwd"] == "/app"
        assert call["env"] is None
        assert call["timeout_s"] == 42

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
        # Width comes from the task's cpu count (4 in _make_environment) and
        # the pin wraps the whole `bash -ic '...'` invocation so children
        # (e.g. the tmux server) inherit the affinity.
        assert command.startswith("__osb_w=4; ")
        assert command.endswith("$__osb_pin bash -ic 'tmux -V'")
        assert 'taskset -c $__osb_s-$((__osb_s + __osb_w - 1))' in command
        # Fail-open branch present: unpinned when taskset/nproc can't cooperate.
        assert '__osb_pin=""' in command

    @pytest.mark.asyncio
    async def test_exec_cpu_pin_disabled_by_default(self, tmp_path):
        env = _make_environment(tmp_path, exec_shell=None)
        await env.start(force_build=False)
        await env.exec("true")
        assert _provider().exec_calls[-1]["command"] == "true"

    @pytest.mark.asyncio
    async def test_exec_cpu_pin_width_tracks_task_cpu_default(self, tmp_path):
        # Harbor defaults EnvironmentConfig.cpus to 1, so a task without an
        # explicit cpu count pins with width 1 (matching its cgroup limit).
        env = _make_environment(
            tmp_path,
            exec_shell=None,
            cpu_pin_enabled=True,
            task_env_config=TaskEnvironmentConfig(docker_image="docker.io/example/task:1.0"),
        )
        await env.start(force_build=False)
        await env.exec("true")
        command = _provider().exec_calls[-1]["command"]
        assert command.startswith("__osb_w=1; ")
        assert command.endswith("$__osb_pin true")

    def test_cpu_pin_prefix_is_valid_posix_sh(self):
        import shutil
        import subprocess

        if shutil.which("sh") is None:
            pytest.skip("sh not available")
        # The prefix must parse and run under plain POSIX sh (execd does not
        # guarantee bash); with a huge width the fail-open branch must leave
        # the command unpinned but still run it.
        from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import _cpu_pin_prefix

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
            if command.startswith("tar -czf"):
                remote_tar = command.split()[2]
                provider.downloads[remote_tar] = payload.getvalue()
            return SandboxExecResult(stdout="", stderr=None, return_code=0)

        provider.exec = _exec_and_stash

        target = tmp_path / "verifier-out"
        await env.download_dir("/logs/verifier", target)
        assert (target / "reward.txt").read_bytes() == b"reward"

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


class TestUploadedEnvironment:
    @pytest.mark.asyncio
    async def test_start_uploads_task_environment_and_defaults_workdir(self, tmp_path):
        environment_dir = tmp_path / "task" / "environment"
        environment_dir.mkdir(parents=True)
        (environment_dir / "setup.sh").write_text("echo ready\n", encoding="utf-8")
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        env = UploadedNemoGymSandboxEnvironment(
            environment_dir=environment_dir,
            environment_name="uploaded-task",
            session_id="uploaded-task__trial-1",
            trial_paths=TrialPaths(trial_dir=trial_dir),
            task_env_config=TaskEnvironmentConfig(docker_image="docker.io/example/task:1.0"),
            sandbox_provider={PROVIDER_NAME: {}},
            exec_shell=None,
            upload_target_dir="/tmp/task-environment",
            workdir="/app",
        )

        await env.start(force_build=False)
        provider = _provider()
        assert provider.created_specs[0].workdir == "/"
        assert any(call["command"] == "mkdir -p /app /tmp/task-environment" for call in provider.exec_calls)
        assert any(
            "tar -xzf" in call["command"]
            and "-C /tmp/task-environment" in call["command"]
            for call in provider.exec_calls
        )

        await env.exec("pwd")
        assert provider.exec_calls[-1]["cwd"] == "/app"
