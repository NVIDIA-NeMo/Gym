# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from nemo_gym.sandbox.providers.apptainer import provider as apptainer_provider
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


FAKE_BINARY = "/usr/bin/apptainer"


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
class RunRecorder:
    """Stand-in for ApptainerProvider._run that records argv and returns canned output."""

    def __init__(self, responder: Callable[[list[str]], tuple[int, str, str]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responder = responder

    async def __call__(
        self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None, daemonize: bool = False
    ) -> tuple[int, str, str]:
        self.calls.append({"argv": list(argv), "timeout_s": timeout_s, "stdin": stdin, "daemonize": daemonize})
        return self._responder(list(argv))


def _contains_seq(haystack: list[str], needle: list[str]) -> bool:
    return any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _make_handle(
    staging: Path,
    *,
    name: str = "nemo-gym-x",
    mount: str = "/sandbox",
    env: dict[str, str] | None = None,
) -> SandboxHandle:
    inst = apptainer_provider._ApptainerInstance(
        name=name,
        staging_dir=staging,
        mount_point=mount,
        image="docker://img",
        env=env or {},
    )
    return SandboxHandle(sandbox_id=name, provider_name="apptainer", raw=inst)


@pytest.fixture
def fake_binary(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda: FAKE_BINARY)
    return FAKE_BINARY


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, responder: Callable[[list[str]], tuple[int, str, str]], **kwargs: Any
) -> tuple[Any, RunRecorder]:
    provider = apptainer_provider.ApptainerProvider(**kwargs)
    rec = RunRecorder(responder)
    monkeypatch.setattr(provider, "_run", rec)
    return provider, rec


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_require_apptainer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apptainer_provider.shutil, "which", lambda _name: "/opt/apptainer")
    assert apptainer_provider._require_apptainer() == "/opt/apptainer"

    monkeypatch.setattr(apptainer_provider.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="apptainer"):
        apptainer_provider._require_apptainer()


def test_coerce_config() -> None:
    coerce = apptainer_provider._coerce_config
    cls = apptainer_provider.ApptainerExecConfig

    assert coerce(None, cls) == cls()
    existing = cls(concurrency=4)
    assert coerce(existing, cls) is existing
    assert coerce({"concurrency": 7}, cls).concurrency == 7
    with pytest.raises(TypeError):
        coerce(123, cls)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="start_timeout_s"):
        apptainer_provider.ApptainerCreateConfig(start_timeout_s=0)
    with pytest.raises(ValueError, match="absolute"):
        apptainer_provider.ApptainerCreateConfig(mount_point="relative")
    with pytest.raises(ValueError, match="default_timeout_s"):
        apptainer_provider.ApptainerExecConfig(default_timeout_s=-1)
    with pytest.raises(ValueError, match="concurrency"):
        apptainer_provider.ApptainerExecConfig(concurrency=0)
    with pytest.raises(ValueError, match="timeout_s"):
        apptainer_provider.ApptainerProbeConfig(timeout_s=0)
    with pytest.raises(ValueError, match="deadline_s"):
        apptainer_provider.ApptainerProbeConfig(deadline_s=0)
    with pytest.raises(ValueError, match="stable_count"):
        apptainer_provider.ApptainerProbeConfig(stable_count=0)
    with pytest.raises(ValueError, match="stable_delay_s"):
        apptainer_provider.ApptainerProbeConfig(stable_delay_s=-1)
    # command=None disables the timeout_s validation gate.
    assert apptainer_provider.ApptainerProbeConfig(command=None, timeout_s=0).command is None


def test_resource_flags() -> None:
    flags = apptainer_provider._resource_flags(
        SandboxResources(cpu=2, memory_mib=1024, gpu=1, disk_gib=50, gpu_type="h100")
    )
    assert _contains_seq(flags, ["--cpus", "2"])
    assert _contains_seq(flags, ["--memory", "1024m"])
    assert "--nv" in flags
    # disk_gib and gpu_type have no flag.
    assert "50" not in flags and "h100" not in flags

    assert apptainer_provider._resource_flags(SandboxResources()) == []


def test_resolve_image() -> None:
    resolve = apptainer_provider._resolve_image
    assert resolve("ubuntu:22.04") == "docker://ubuntu:22.04"
    assert resolve("oras://registry.example/image:tag") == "oras://registry.example/image:tag"
    assert resolve("/tmp/image.sif") == "/tmp/image.sif"


def test_to_sandbox_status() -> None:
    to_status = apptainer_provider._to_sandbox_status
    assert to_status("running") is SandboxStatus.RUNNING
    assert to_status("active") is SandboxStatus.RUNNING
    assert to_status("starting") is SandboxStatus.STARTING
    assert to_status("stopped") is SandboxStatus.STOPPED
    assert to_status("failed") is SandboxStatus.ERROR
    assert to_status("nonsense") is SandboxStatus.UNKNOWN
    assert to_status(None) is SandboxStatus.UNKNOWN


def test_path_under_mount() -> None:
    under = apptainer_provider._path_under_mount
    assert under("/sandbox", "/sandbox/a/b.txt") == "a/b.txt"
    assert under("/sandbox", "/sandbox") == ""
    assert under("/sandbox/", "/sandbox/x") == "x"
    assert under("/sandbox", "/sandbox/../outside.txt") is None
    assert under("/sandbox", "/etc/passwd") is None


def test_is_runtime_failure() -> None:
    assert apptainer_provider._is_runtime_failure("FATAL: no instance found") is True
    assert apptainer_provider._is_runtime_failure("instance not found") is True
    assert apptainer_provider._is_runtime_failure("ls: cannot access") is False


def test_constructor_requires_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apptainer_provider.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError):
        apptainer_provider.ApptainerProvider()


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #
async def test_create_builds_argv_and_runs_probe(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "start" in argv:
            return (0, "", "")
        if "exec" in argv:
            return (0, apptainer_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(
        monkeypatch,
        responder,
        exec={"default_binds": ["/data:/data"], "extra_exec_args": ["--contain"]},
        create={"extra_start_args": ["--cleanenv"]},
    )

    spec = SandboxSpec(
        image="ubuntu:22.04",
        env={"FOO": "bar"},
        resources={"cpu": 2, "memory_mib": 1024, "gpu": 1},
        ttl_s=60,
    )

    with caplog.at_level("WARNING"):
        handle = await provider.create(spec)

    assert "ttl_s is not supported" in caplog.text
    assert handle.provider_name == "apptainer"
    assert handle.sandbox_id.startswith(apptainer_provider.INSTANCE_NAME_PREFIX)
    assert handle.raw.staging_dir == staging
    assert handle.raw.mount_point == "/sandbox"
    assert handle.raw.image == "docker://ubuntu:22.04"
    assert handle.raw.env == {"FOO": "bar"}

    start_argv = rec.calls[0]["argv"]
    assert start_argv[:3] == [FAKE_BINARY, "instance", "start"]
    assert _contains_seq(start_argv, ["--bind", f"{staging}:/sandbox"])
    assert _contains_seq(start_argv, ["--bind", "/data:/data"])
    assert _contains_seq(start_argv, ["--env", "FOO=bar"])
    assert _contains_seq(start_argv, ["--cpus", "2.0"])
    assert _contains_seq(start_argv, ["--memory", "1024m"])
    assert "--nv" in start_argv
    assert "--cleanenv" in start_argv
    assert start_argv[-2:] == ["docker://ubuntu:22.04", handle.sandbox_id]

    probe_argv = rec.calls[1]["argv"]
    assert "exec" in probe_argv
    assert f"instance://{handle.sandbox_id}" in probe_argv
    assert probe_argv[-1] == apptainer_provider.READY_PROBE_COMMAND
    assert rec.calls[1]["timeout_s"] == 30


@pytest.mark.parametrize("create_config", [{"apply_resource_limits": False}, {"extra_start_args": ["--fakeroot"]}])
async def test_create_skips_cgroup_resource_limits(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, create_config: dict[str, Any]
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, apptainer_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder, create=create_config)
    await provider.create(SandboxSpec(image="ubuntu:22.04", resources={"cpu": 2, "memory_mib": 1024, "gpu": 1}))

    start_argv = rec.calls[0]["argv"]
    assert "--cpus" not in start_argv
    assert "--memory" not in start_argv
    assert "--nv" in start_argv


async def test_create_extra_binds_from_provider_options(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, apptainer_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder, exec={"default_binds": ["/data:/data"]})

    spec = SandboxSpec(
        image="docker://img",
        provider_options={"binds": ["/host/a:/code/a", "/host/b:/code/b:ro"]},
    )
    await provider.create(spec)

    start_argv = rec.calls[0]["argv"]
    # staging + default_binds + the two per-sandbox binds are all present
    assert _contains_seq(start_argv, ["--bind", f"{staging}:/sandbox"])
    assert _contains_seq(start_argv, ["--bind", "/data:/data"])
    assert _contains_seq(start_argv, ["--bind", "/host/a:/code/a"])
    assert _contains_seq(start_argv, ["--bind", "/host/b:/code/b:ro"])


async def test_create_extra_binds_accepts_single_string(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, apptainer_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    await provider.create(SandboxSpec(image="docker://img", provider_options={"binds": "/host/x:/code/x"}))
    assert _contains_seq(rec.calls[0]["argv"], ["--bind", "/host/x:/code/x"])


async def test_create_extra_binds_invalid_type_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    with pytest.raises(apptainer_provider.ApptainerCreateError, match="must be a string or list"):
        await provider.create(SandboxSpec(image="docker://img", provider_options={"binds": 123}))


async def test_create_requires_image(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    with pytest.raises(apptainer_provider.ApptainerCreateError, match="image is required"):
        await provider.create(SandboxSpec(image=None))


async def test_create_start_failure_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "boom"))

    with pytest.raises(apptainer_provider.ApptainerCreateError, match="failed"):
        await provider.create(SandboxSpec(image="docker://img"))
    assert not staging.exists()


async def test_create_start_timeout_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec = _make_provider(monkeypatch, responder)
    with pytest.raises(apptainer_provider.ApptainerCreateError, match="timed out"):
        await provider.create(SandboxSpec(image="docker://img"))
    assert not staging.exists()


async def test_create_probe_failure_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "start" in argv:
            return (0, "", "")
        if "exec" in argv:
            return (1, "", "probe broke")
        return (0, "", "")  # instance stop during cleanup

    provider, rec = _make_provider(monkeypatch, responder)
    with pytest.raises(apptainer_provider.ApptainerCreateVerificationError):
        await provider.create(SandboxSpec(image="docker://img"))

    assert not staging.exists()
    assert any("stop" in call["argv"] for call in rec.calls)


# --------------------------------------------------------------------------- #
# exec
# --------------------------------------------------------------------------- #
async def test_exec_normal_with_cwd_and_env(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "hello", ""))
    handle = _make_handle(tmp_path)

    result = await provider.exec(handle, "echo hi", cwd="/work", env={"A": "b"})

    assert result.return_code == 0
    assert result.stdout == "hello"
    assert result.error_type is None

    argv = rec.calls[0]["argv"]
    assert argv[:2] == [FAKE_BINARY, "exec"]
    assert _contains_seq(argv, ["--pwd", "/work"])
    assert _contains_seq(argv, ["--env", "A=b"])
    assert argv[-4:] == ["instance://nemo-gym-x", "sh", "-c", "echo hi"]
    assert rec.calls[0]["timeout_s"] == 180  # default exec timeout


async def test_exec_reapplies_create_env_and_overrides_call_env(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(tmp_path, env={"A": "from-create", "B": "base"})

    await provider.exec(handle, "env", env={"A": "from-call"})

    argv = rec.calls[0]["argv"]
    assert _contains_seq(argv, ["--env", "A=from-call"])
    assert _contains_seq(argv, ["--env", "B=base"])
    assert "A=from-create" not in argv


async def test_exec_empty_streams_are_strings(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    result = await provider.exec(_make_handle(tmp_path), "true")

    assert result.return_code == 0
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize(
    "user,fakeroot_for_root,expect_fakeroot,expect_su",
    [
        (None, True, False, False),
        ("root", True, True, False),
        (0, True, True, False),
        ("root", False, False, False),
        ("alice", True, True, True),
    ],
)
async def test_exec_user_mapping(
    fake_binary: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    user: Any,
    fakeroot_for_root: bool,
    expect_fakeroot: bool,
    expect_su: bool,
) -> None:
    provider, rec = _make_provider(
        monkeypatch, lambda argv: (0, "", ""), exec={"fakeroot_for_root": fakeroot_for_root}
    )
    handle = _make_handle(tmp_path)

    await provider.exec(handle, "whoami", user=user)
    argv = rec.calls[0]["argv"]

    assert ("--fakeroot" in argv) is expect_fakeroot
    if expect_su:
        expected = f"su -s /bin/sh -c {shlex.quote('whoami')} {shlex.quote(str(user))}"
        assert argv[-1] == expected
    else:
        assert argv[-1] == "whoami"


async def test_exec_passes_stdin(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "ok", ""))
    await provider.exec(_make_handle(tmp_path), "cat", stdin=b"prompt-bytes")
    assert rec.calls[0]["stdin"] == b"prompt-bytes"


async def test_exec_timeout(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("too slow")

    provider, _rec = _make_provider(monkeypatch, responder)
    result = await provider.exec(_make_handle(tmp_path), "sleep 99", timeout_s=1)

    assert result.return_code == apptainer_provider.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "timeout"
    assert result.stdout is None


async def test_exec_runtime_failure(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "FATAL: no instance found"))
    result = await provider.exec(_make_handle(tmp_path), "echo hi")

    assert result.return_code == apptainer_provider.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "sandbox"
    assert "FATAL" in result.stderr


async def test_exec_command_failure_is_not_runtime_error(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (2, "", "ls: cannot access"))
    result = await provider.exec(_make_handle(tmp_path), "ls /nope")

    assert result.return_code == 2
    assert result.error_type is None


# --------------------------------------------------------------------------- #
# upload / download
# --------------------------------------------------------------------------- #
async def test_upload_fast_path(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(staging)

    src = tmp_path / "src.txt"
    src.write_bytes(b"payload")
    await provider.upload_file(handle, src, "/sandbox/sub/dest.txt")

    assert (staging / "sub" / "dest.txt").read_bytes() == b"payload"
    assert rec.calls == []  # fast path never shells out


async def test_upload_fallback(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(staging)

    captured: dict[str, Any] = {}

    async def fake_exec(h: SandboxHandle, command: str, *, user: Any = None, **_: Any) -> SandboxExecResult:
        captured["command"] = command
        captured["user"] = user
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(provider, "exec", fake_exec)

    src = tmp_path / "src.txt"
    src.write_bytes(b"payload")
    await provider.upload_file(handle, src, "/etc/app.conf")

    assert "cp" in captured["command"]
    assert "/etc/app.conf" in captured["command"]
    assert captured["user"] == "root"
    assert list(staging.iterdir()) == []  # temp staging file cleaned up


async def test_upload_fallback_error(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(staging)

    async def fake_exec(h: SandboxHandle, command: str, *, user: Any = None, **_: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="denied", return_code=1)

    monkeypatch.setattr(provider, "exec", fake_exec)

    src = tmp_path / "src.txt"
    src.write_bytes(b"payload")
    with pytest.raises(RuntimeError, match="upload"):
        await provider.upload_file(handle, src, "/etc/app.conf")
    assert list(staging.iterdir()) == []


async def test_download_fast_path(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "out").mkdir(parents=True)
    (staging / "out" / "r.txt").write_bytes(b"result")
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(staging)

    dest = tmp_path / "local.txt"
    await provider.download_file(handle, "/sandbox/out/r.txt", dest)

    assert dest.read_bytes() == b"result"
    assert rec.calls == []


async def test_download_fallback(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(staging)

    async def fake_exec(h: SandboxHandle, command: str, *, user: Any = None, **_: Any) -> SandboxExecResult:
        # Simulate the in-container `cp` by writing the host side of the staging file.
        container_tmp = shlex.split(command)[-1]
        (staging / Path(container_tmp).name).write_bytes(b"remote-bytes")
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(provider, "exec", fake_exec)

    dest = tmp_path / "local.txt"
    await provider.download_file(handle, "/var/log/app.log", dest)

    assert dest.read_bytes() == b"remote-bytes"
    assert list(staging.iterdir()) == []  # temp staging file cleaned up


async def test_download_fallback_error(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(staging)

    async def fake_exec(h: SandboxHandle, command: str, *, user: Any = None, **_: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="missing", return_code=1)

    monkeypatch.setattr(provider, "exec", fake_exec)

    with pytest.raises(RuntimeError, match="download"):
        await provider.download_file(handle, "/var/log/app.log", tmp_path / "local.txt")
    assert list(staging.iterdir()) == []


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
async def test_status_running(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = json.dumps({"instances": [{"instance": "nemo-gym-x"}]})
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, out, ""))
    assert await provider.status(_make_handle(tmp_path)) is SandboxStatus.RUNNING


async def test_status_stopped_when_absent(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = json.dumps({"instances": [{"instance": "other"}]})
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, out, ""))
    assert await provider.status(_make_handle(tmp_path)) is SandboxStatus.STOPPED


async def test_status_explicit_state(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = json.dumps({"instances": [{"instance": "nemo-gym-x", "state": "stopped"}]})
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, out, ""))
    assert await provider.status(_make_handle(tmp_path)) is SandboxStatus.STOPPED


async def test_status_unknown_paths(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handle = _make_handle(tmp_path)

    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "err"))
    assert await provider.status(handle) is SandboxStatus.UNKNOWN

    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "not-json", ""))
    assert await provider.status(handle) is SandboxStatus.UNKNOWN

    def timeout_responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec = _make_provider(monkeypatch, timeout_responder)
    assert await provider.status(handle) is SandboxStatus.UNKNOWN


# --------------------------------------------------------------------------- #
# close / aclose
# --------------------------------------------------------------------------- #
async def test_close_success(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    await provider.close(_make_handle(staging))

    assert not staging.exists()
    assert _contains_seq(rec.calls[0]["argv"], [FAKE_BINARY, "instance", "stop", "nemo-gym-x"])


async def test_close_missing_instance_is_success(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "no instance found"))
    await provider.close(_make_handle(staging))  # does not raise
    assert not staging.exists()


async def test_close_real_failure_raises_but_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "permission denied"))
    with pytest.raises(RuntimeError, match="stop failed"):
        await provider.close(_make_handle(staging))
    assert not staging.exists()


async def test_close_fatal_permission_denied_raises(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "FATAL: permission denied"))
    with pytest.raises(RuntimeError, match="stop failed"):
        await provider.close(_make_handle(staging))
    assert not staging.exists()


async def test_close_timeout_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec = _make_provider(monkeypatch, responder)
    with pytest.raises(TimeoutError):
        await provider.close(_make_handle(staging))
    assert not staging.exists()


async def test_close_staging_removal_failure_is_logged(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))

    def boom(path: Any, ignore_errors: bool = False) -> None:
        raise OSError("locked")

    monkeypatch.setattr(apptainer_provider.shutil, "rmtree", boom)
    with caplog.at_level("WARNING"):
        await provider.close(_make_handle(staging))  # does not raise
    assert "failed to remove staging dir" in caplog.text


async def test_aclose(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    assert await provider.aclose() is None


# --------------------------------------------------------------------------- #
# readiness probe
# --------------------------------------------------------------------------- #
async def test_verify_skipped_when_command_none(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""), probe={"command": None})

    async def boom(*_a: Any, **_k: Any) -> SandboxExecResult:
        raise AssertionError("exec should not be called when probe is disabled")

    monkeypatch.setattr(provider, "exec", boom)
    await provider._verify_created_handle(_make_handle(tmp_path))  # returns without exec


async def test_verify_polls_until_stable(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec = _make_provider(
        monkeypatch,
        lambda argv: (0, "", ""),
        probe={"deadline_s": 5, "stable_count": 2, "stable_delay_s": 0},
    )

    results = iter(
        [
            SandboxExecResult(stdout="", stderr="warming up", return_code=1),
            SandboxExecResult(stdout=apptainer_provider.READY_PROBE_EXPECTED, stderr="", return_code=0),
            SandboxExecResult(stdout=apptainer_provider.READY_PROBE_EXPECTED, stderr="", return_code=0),
        ]
    )

    async def fake_exec(*_a: Any, **_k: Any) -> SandboxExecResult:
        return next(results)

    monkeypatch.setattr(provider, "exec", fake_exec)
    await provider._verify_created_handle(_make_handle(tmp_path))  # 2 consecutive passes


async def test_verify_deadline_exceeded(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec = _make_provider(
        monkeypatch,
        lambda argv: (0, "", ""),
        probe={"deadline_s": 0.01, "stable_delay_s": 0.02},
    )

    async def always_fail(*_a: Any, **_k: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="nope", return_code=1)

    monkeypatch.setattr(provider, "exec", always_fail)
    with pytest.raises(apptainer_provider.ApptainerCreateVerificationError, match="within"):
        await provider._verify_created_handle(_make_handle(tmp_path))


# --------------------------------------------------------------------------- #
# _run against real lightweight binaries (exercises subprocess plumbing)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("echo") is None, reason="echo not available")
async def test_run_real_echo(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = apptainer_provider.ApptainerProvider()
    code, out, err = await provider._run([shutil.which("echo"), "hi"], timeout_s=10)
    assert code == 0
    assert out.strip() == "hi"
    assert err == ""


@pytest.mark.skipif(shutil.which("cat") is None, reason="cat not available")
async def test_run_real_stdin(fake_binary: str) -> None:
    provider = apptainer_provider.ApptainerProvider()
    code, out, _err = await provider._run([shutil.which("cat")], timeout_s=10, stdin=b"piped")
    assert code == 0
    assert out == "piped"


@pytest.mark.skipif(shutil.which("sleep") is None, reason="sleep not available")
async def test_run_real_timeout(fake_binary: str) -> None:
    provider = apptainer_provider.ApptainerProvider()
    with pytest.raises(TimeoutError):
        await provider._run([shutil.which("sleep"), "5"], timeout_s=0.1)


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not available")
async def test_run_daemonizing_returns_despite_lingering_child(fake_binary: str) -> None:
    """Regression: a backgrounded child inheriting stdout must not wedge the read.

    Mirrors ``apptainer instance start``, which forks a long-lived instance that
    keeps the child's stdout/stderr open. The pipe-based path would block on EOF
    until timeout; the daemonizing path waits only for the foreground process.
    """
    provider = apptainer_provider.ApptainerProvider()
    # Foreground prints and exits immediately; the backgrounded `sleep` holds the
    # inherited stdout fd open well past the (generous) timeout.
    argv = [shutil.which("sh"), "-c", "sleep 30 & printf started"]
    code, out, _err = await provider._run(argv, timeout_s=10, daemonize=True)
    assert code == 0
    assert out == "started"


async def test_create_uses_daemonize_for_instance_start(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, apptainer_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    await provider.create(SandboxSpec(image="docker://ubuntu:22.04"))

    start_call = next(c for c in rec.calls if "start" in c["argv"])
    assert start_call["daemonize"] is True
    # The readiness probe (exec) must NOT use the daemonizing path.
    exec_calls = [c for c in rec.calls if "exec" in c["argv"]]
    assert exec_calls and all(c["daemonize"] is False for c in exec_calls)
