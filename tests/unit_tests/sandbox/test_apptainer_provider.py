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
import os
import shlex
import shutil
import socket
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock

import anyio
import pytest

from nemo_gym.sandbox.providers.apptainer import provider as apptainer_provider
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from tests.unit_tests.sandbox.cli_testing import is_running, requires_sh, stand_in_cli


pytestmark = pytest.mark.sandbox


FAKE_BINARY = "/usr/bin/apptainer"


@pytest.mark.asyncio
async def test_reconnect_preserves_workdir_environment_and_transfers(fake_binary, tmp_path, monkeypatch):
    import tempfile

    from nemo_gym.sandbox.api import AsyncSandbox

    staging = Path(tempfile.mkdtemp(prefix="nemo-gym-apptainer-", dir=tmp_path))
    creator = apptainer_provider.ApptainerProvider()
    original = AsyncSandbox(creator, SandboxSpec(workdir="/testbed"))
    original._handle = _make_handle(staging, name="nemo-gym-" + "a" * 32, env={"OMP_NUM_THREADS": "2"})
    original._stopped = False
    descriptor = json.loads(json.dumps(await original.serialize()))
    receiver = apptainer_provider.ApptainerProvider()
    monkeypatch.setattr(receiver, "status", AsyncMock(return_value=SandboxStatus.RUNNING))
    received = await AsyncSandbox.connect(descriptor, provider=receiver)
    calls = []

    async def run(argv, **kwargs):
        calls.append(argv)
        assert shlex.split(_env_file_path(argv).read_text()) == ["OMP_NUM_THREADS=2"]
        return 0, "ok", ""

    monkeypatch.setattr(receiver, "_run", run)
    await received.exec("pwd")
    assert _contains_seq(calls[0], ["--pwd", "/testbed"])
    source = tmp_path / "source"
    source.write_text("payload")
    await received.upload(source, "/sandbox/transferred")
    await original.download("/sandbox/transferred", tmp_path / "download")
    assert (tmp_path / "download").read_text() == "payload"


@pytest.mark.asyncio
async def test_reconnect_rejects_bare_id_or_other_host(fake_binary, tmp_path):
    provider = apptainer_provider.ApptainerProvider()
    with pytest.raises(ValueError, match="full serialized descriptor"):
        await provider.connect({"sandbox_id": "name"})
    descriptor = await provider.serialize_handle(_make_handle(tmp_path))
    descriptor["hostname"] = socket.gethostname() + "-other"
    with pytest.raises(ValueError, match="same host and UID"):
        await provider.connect(descriptor)


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


def _env_file_path(argv: list[str]) -> Path:
    return Path(argv[argv.index("--env-file") + 1])


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
    monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: FAKE_BINARY)
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


def test_require_apptainer_uses_configured_binpath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / "apptainer"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    assert apptainer_provider._require_apptainer(str(tmp_path)) == str(binary)


def test_apptainer_subprocess_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")

    assert apptainer_provider._apptainer_subprocess_env("/workspace/apptainer/bin")["PATH"] == (
        "/workspace/apptainer/bin:/usr/bin"
    )


def test_coerce_config() -> None:
    coerce = apptainer_provider._coerce_config
    cls = apptainer_provider.ApptainerExecConfig

    assert coerce(None, cls) == cls()
    existing = cls(concurrency=4)
    assert coerce(existing, cls) is existing
    assert coerce({"concurrency": 7}, cls).concurrency == 7
    assert coerce({"timeout_grace_s": 0.5}, cls).timeout_grace_s == 0.5
    with pytest.raises(TypeError):
        coerce(123, cls)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="start_timeout_s"):
        apptainer_provider.ApptainerCreateConfig(start_timeout_s=0)
    with pytest.raises(ValueError, match="absolute"):
        apptainer_provider.ApptainerCreateConfig(mount_point="relative")
    with pytest.raises(ValueError, match="default_timeout_s"):
        apptainer_provider.ApptainerExecConfig(default_timeout_s=-1)
    with pytest.raises(ValueError, match="timeout_grace_s"):
        apptainer_provider.ApptainerExecConfig(timeout_grace_s=-1)
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


def test_serialize_env_file_quotes_special_characters() -> None:
    special = "spaces '$HOME' $(date) `id` \\\n+unicode=zażółć"
    assert (
        apptainer_provider._serialize_env_file({"EMPTY": "", "PLAIN": "value", "SPECIAL": special})
        == f"EMPTY=''\nPLAIN=value\nSPECIAL={shlex.quote(special)}\n".encode()
    )


@pytest.mark.parametrize(
    "env,error",
    [
        ({"NOT-VALID": "value"}, ValueError),
        ({1: "value"}, ValueError),
        ({"HOME": "/tmp/home"}, ValueError),
        ({"VALID": "before\x00after"}, ValueError),
        ({"VALID": 1}, TypeError),
        ({"VALID": "\ud800"}, ValueError),
    ],
)
def test_serialize_env_file_rejects_invalid_input(env: dict[Any, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        apptainer_provider._serialize_env_file(env)


def test_private_env_file_is_mode_0600_and_always_removed(tmp_path: Path) -> None:
    content = b"TOKEN='not-a-real-secret'\n"

    with apptainer_provider._private_env_file(tmp_path, content) as path:
        assert path is not None
        assert path.read_bytes() == content
        assert path.stat().st_mode & 0o777 == 0o600
    assert not path.exists()

    with pytest.raises(RuntimeError):
        with apptainer_provider._private_env_file(tmp_path, content) as failed_path:
            assert failed_path is not None
            raise RuntimeError("failure")
    assert not failed_path.exists()


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
    assert apptainer_provider._is_runtime_failure("FATAL   : container creation failed") is True
    assert apptainer_provider._is_runtime_failure("some output\nFATAL:   exec failed") is True
    # The user's own stderr must not be classified as an apptainer failure.
    assert apptainer_provider._is_runtime_failure("fatal: not a git repository (or any parent)") is False
    assert apptainer_provider._is_runtime_failure("Error: file /testbed/foo.py does not exist") is False
    assert apptainer_provider._is_runtime_failure("ERROR: tests/test_x.py::test_y - AssertionError", 1) is False
    # "ERROR:" lines only count together with apptainer's own exit code.
    assert apptainer_provider._is_runtime_failure("ERROR  : could not open image", 255) is True


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

    env_files: list[Path] = []

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "--env-file" in argv:
            env_file = _env_file_path(argv)
            env_files.append(env_file)
            assert env_file.read_bytes() == b"FOO=bar\n"
            assert env_file.stat().st_mode & 0o777 == 0o600
        assert "FOO=bar" not in argv
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
    assert _contains_seq(start_argv, ["--no-eval", "--env-file", str(env_files[0])])
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
    assert len(env_files) == 2
    assert all(not path.exists() for path in env_files)


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
    env_file: Path | None = None

    def responder(argv: list[str]) -> tuple[int, str, str]:
        nonlocal env_file
        env_file = _env_file_path(argv)
        assert env_file.exists()
        assert "not-a-real-secret" not in argv
        return (1, "", "boom")

    provider, _rec = _make_provider(monkeypatch, responder)

    with pytest.raises(apptainer_provider.ApptainerCreateError, match="failed"):
        await provider.create(SandboxSpec(image="docker://img", env={"TOKEN": "not-a-real-secret"}))
    assert env_file is not None and not env_file.exists()
    assert not staging.exists()


async def test_create_start_timeout_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    env_file: Path | None = None

    def responder(argv: list[str]) -> tuple[int, str, str]:
        nonlocal env_file
        env_file = _env_file_path(argv)
        assert env_file.exists()
        raise TimeoutError(f"slow: {argv}")

    provider, _rec = _make_provider(monkeypatch, responder)
    with pytest.raises(apptainer_provider.ApptainerCreateError, match="timed out") as exc_info:
        await provider.create(SandboxSpec(image="docker://img", env={"TOKEN": "not-a-real-secret"}))
    assert "not-a-real-secret" not in str(exc_info.value)
    assert env_file is not None and not env_file.exists()
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
    env_file: Path | None = None

    def responder(argv: list[str]) -> tuple[int, str, str]:
        nonlocal env_file
        env_file = _env_file_path(argv)
        assert env_file.read_bytes() == b"A=b\n"
        assert env_file.stat().st_mode & 0o777 == 0o600
        assert "A=b" not in argv
        return (0, "hello", "")

    provider, rec = _make_provider(monkeypatch, responder)
    handle = _make_handle(tmp_path)

    result = await provider.exec(handle, "echo hi", cwd="/work", env={"A": "b"})

    assert result.return_code == 0
    assert result.stdout == "hello"
    assert result.error_type is None

    argv = rec.calls[0]["argv"]
    assert argv[:2] == [FAKE_BINARY, "exec"]
    assert _contains_seq(argv, ["--pwd", "/work"])
    assert env_file is not None
    assert _contains_seq(argv, ["--no-eval", "--env-file", str(env_file)])
    assert argv[-4:] == ["instance://nemo-gym-x", "sh", "-c", "echo hi"]
    assert rec.calls[0]["timeout_s"] == 180  # default exec timeout
    assert not env_file.exists()


async def test_exec_reapplies_create_env_and_overrides_call_env(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file: Path | None = None

    def responder(argv: list[str]) -> tuple[int, str, str]:
        nonlocal env_file
        env_file = _env_file_path(argv)
        assert env_file.read_bytes() == b"A=from-call\nB=base\n"
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    handle = _make_handle(tmp_path, env={"A": "from-create", "B": "base"})

    await provider.exec(handle, "env", env={"A": "from-call"})

    argv = rec.calls[0]["argv"]
    assert "A=from-call" not in argv
    assert "B=base" not in argv
    assert "A=from-create" not in argv
    assert env_file is not None and not env_file.exists()


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
    env_file: Path | None = None

    def responder(argv: list[str]) -> tuple[int, str, str]:
        nonlocal env_file
        env_file = _env_file_path(argv)
        assert env_file.exists()
        raise TimeoutError(f"too slow: {argv}")

    provider, _rec = _make_provider(monkeypatch, responder)
    result = await provider.exec(
        _make_handle(tmp_path, env={"TOKEN": "not-a-real-secret"}),
        "sleep 99",
        timeout_s=1,
    )

    assert result.return_code == apptainer_provider.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "timeout"
    assert result.stdout is None
    assert "not-a-real-secret" not in (result.stderr or "")
    assert env_file is not None and not env_file.exists()


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


@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: not a git repository (or any of the parent directories): .git",
        "FileNotFoundError: path /testbed/data.csv does not exist",
    ],
)
async def test_exec_user_stderr_is_not_runtime_error(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stderr: str
) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (128, "", stderr))
    result = await provider.exec(_make_handle(tmp_path), "git log")

    assert result.return_code == 128
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


async def test_download_copies_a_large_nested_file_in_chunks(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(apptainer_provider, "STAGED_COPY_CHUNK_BYTES", 1000)
    staging = tmp_path / "staging"
    (staging / "a" / "b").mkdir(parents=True)
    payload = bytes(range(256)) * 41  # not a multiple of the chunk size, and not UTF-8
    (staging / "a" / "b" / "data.bin").write_bytes(payload)
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))

    dest = tmp_path / "new" / "dir" / "data.bin"
    await provider.download_file(_make_handle(staging), "/sandbox/a/./b/data.bin", dest)

    assert dest.read_bytes() == payload
    assert rec.calls == []


async def test_download_missing_file_still_raises_file_not_found(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    with pytest.raises(FileNotFoundError):
        await provider.download_file(_make_handle(staging), "/sandbox/absent.txt", tmp_path / "local.txt")
    assert not (tmp_path / "local.txt").exists()


@pytest.mark.parametrize("kind", ["file-link", "dir-link", "inside-link", "dangling-link", "directory", "mount"])
async def test_download_refuses_links_and_directories(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    # The sandbox can write the staging dir: a link there must not make the host read another host file.
    secret_dir = tmp_path / "host"
    secret_dir.mkdir()
    (secret_dir / "secret.txt").write_text("host secret")
    staging = tmp_path / "staging"
    (staging / "real").mkdir(parents=True)
    (staging / "real" / "ok.txt").write_text("fine")
    source = {
        "file-link": "/sandbox/link.txt",
        "dir-link": "/sandbox/linked/secret.txt",
        "inside-link": "/sandbox/inside/ok.txt",
        "dangling-link": "/sandbox/dangling.txt",
        "directory": "/sandbox/real",
        "mount": "/sandbox",
    }[kind]
    (staging / "link.txt").symlink_to(secret_dir / "secret.txt")
    (staging / "linked").symlink_to(secret_dir, target_is_directory=True)
    (staging / "inside").symlink_to(staging / "real", target_is_directory=True)
    (staging / "dangling.txt").symlink_to(tmp_path / "nowhere")
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))

    dest = tmp_path / "local.txt"
    with pytest.raises(RuntimeError, match="apptainer download from .* failed"):
        await provider.download_file(_make_handle(staging), source, dest)

    assert not dest.exists()
    assert rec.calls == []
    assert (secret_dir / "secret.txt").read_text() == "host secret"


def test_copy_staged_file_refuses_a_path_that_climbs_out(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (tmp_path / "outside.txt").write_text("outside")
    with pytest.raises(RuntimeError, match="not a file inside the sandbox mount"):
        apptainer_provider._copy_staged_file(staging, "../outside.txt", tmp_path / "local.txt", "/sandbox/x")
    assert not (tmp_path / "local.txt").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs not available")
async def test_download_of_a_fifo_fails_promptly(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    os.mkfifo(staging / "pipe")
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))

    # Nothing ever writes to the FIFO: a blocking open would never return.
    with anyio.fail_after(5):
        with pytest.raises(RuntimeError, match="not a regular file"):
            await provider.download_file(_make_handle(staging), "/sandbox/pipe", tmp_path / "local.txt")
    assert not (tmp_path / "local.txt").exists()


async def test_download_fallback_refuses_a_link_left_in_staging(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "secret.txt").write_text("host secret")
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))

    async def fake_exec(h: SandboxHandle, command: str, *, user: Any = None, **_: Any) -> SandboxExecResult:
        # The in-container copy leaves a link, not a file, at the staged name.
        (staging / Path(shlex.split(command)[-1]).name).symlink_to(tmp_path / "secret.txt")
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(provider, "exec", fake_exec)
    with pytest.raises(RuntimeError, match="apptainer download from '/etc/shadow' failed"):
        await provider.download_file(_make_handle(staging), "/etc/shadow", tmp_path / "local.txt")
    assert not (tmp_path / "local.txt").exists()
    assert list(staging.iterdir()) == []


async def test_download_does_not_block_the_event_loop(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading
    import time

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "slow.bin").write_bytes(b"slow")
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    copy_threads: list[int] = []
    copyfileobj = shutil.copyfileobj

    def slow_copy(*args: Any) -> None:
        copy_threads.append(threading.get_ident())
        time.sleep(0.4)
        copyfileobj(*args)

    monkeypatch.setattr(apptainer_provider.shutil, "copyfileobj", slow_copy)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await anyio.sleep(0.01)
            ticks += 1

    dest = tmp_path / "local.bin"
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(ticker)
        await provider.download_file(_make_handle(staging), "/sandbox/slow.bin", dest)
        ticks_during_copy = ticks
        tasks.cancel_scope.cancel()

    assert dest.read_bytes() == b"slow"
    assert copy_threads and copy_threads[0] != threading.get_ident()
    assert ticks_during_copy >= 10


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


def _direct_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any) -> tuple[Any, Path]:
    """A direct-mode provider over a fake CLI that delegates to the requested shell, and its staging dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_apptainer = bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == exec ]]
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bind|--env-file|--cpus|--memory|--home|--no-mount|--env) shift 2 ;;
        --no-eval|--cleanenv|--writable-tmpfs|--pid|--contain|--containall|--nv) shift ;;
        *) shift; exec "$@" ;;
    esac
done
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))
    provider = apptainer_provider.ApptainerProvider(
        bin_path=str(bin_dir),
        create={"direct_exec": True, "mount_point": str(staging)},
        exec={"fakeroot_for_root": False},
        **kwargs,
    )
    return provider, staging


async def _wait_until_gone(pid: int) -> None:
    with anyio.fail_after(10):
        while is_running(pid):
            await anyio.sleep(0.02)


async def test_direct_exec_lifecycle_preserves_overlay_and_supports_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise direct mode with a fake CLI that delegates to the requested shell."""
    provider, staging = _direct_provider(monkeypatch, tmp_path)

    handle = await provider.create(SandboxSpec(image="unused.sif"))
    assert await provider.status(handle) is SandboxStatus.RUNNING
    assert handle.raw.direct_process is not None

    first = await provider.exec(handle, "mkdir -p work && printf persistent > work/value", cwd=str(staging))
    assert first.return_code == 0
    second = await provider.exec(handle, "cat work/value", cwd=str(staging))
    assert second.return_code == 0
    assert second.stdout == "persistent"

    with_env = await provider.exec(handle, 'printf %s "$DIRECT_VALUE"', env={"DIRECT_VALUE": "hello world"})
    assert with_env.stdout == "hello world"
    with_stdin = await provider.exec(handle, "cat", stdin=b"stdin payload")
    assert with_stdin.stdout == "stdin payload"
    failed = await provider.exec(handle, "printf expected-error >&2; exit 7")
    assert failed.return_code == 7
    assert failed.stderr == "expected-error"

    # A command is one quoted argument: none of these can swallow the status line or read the next command.
    commented = await provider.exec(handle, "printf hi # trailing comment", timeout_s=5)
    assert (commented.return_code, commented.stdout) == (0, "hi")
    heredoc = await provider.exec(handle, "cat <<'EOF'\nhello\nEOF", timeout_s=5)
    assert (heredoc.return_code, heredoc.stdout) == (0, "hello\n")
    unbalanced = await provider.exec(handle, 'printf "x', timeout_s=5)
    assert unbalanced.error_type is None and unbalanced.return_code != 0
    no_stdin = await provider.exec(handle, "cat", timeout_s=5)
    assert (no_stdin.return_code, no_stdin.stdout) == (0, "")
    after = await provider.exec(handle, "printf still-running", timeout_s=5)
    assert (after.return_code, after.stdout) == (0, "still-running")

    # Downloads in direct mode: the file under the mount, and the copy from an arbitrary path.
    await provider.exec(handle, "printf under-mount > result.txt", cwd=str(staging), timeout_s=5)
    await provider.download_file(handle, f"{staging}/result.txt", tmp_path / "under-mount.txt")
    assert (tmp_path / "under-mount.txt").read_text() == "under-mount"
    (tmp_path / "elsewhere.txt").write_text("elsewhere")
    await provider.download_file(handle, str(tmp_path / "elsewhere.txt"), tmp_path / "copied.txt")
    assert (tmp_path / "copied.txt").read_text() == "elsewhere"

    await provider.close(handle)
    assert await provider.status(handle) is SandboxStatus.STOPPED
    assert not staging.exists()


async def test_direct_exec_output_is_exact_and_has_no_line_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _staging = _direct_provider(monkeypatch, tmp_path)
    handle = await provider.create(SandboxSpec(image="unused.sif"))
    try:
        # One line far past the stream reader's 64 KiB limit, with no newline at all.
        size = 1024 * 1024
        big = await provider.exec(handle, f"head -c {size} /dev/zero | tr '\\0' x", timeout_s=60)
        assert (big.return_code, big.error_type, big.stderr) == (0, None, "")
        assert big.stdout == "x" * size
        # The long command was drained: the next one sees only its own output.
        after = await provider.exec(handle, "printf next", timeout_s=10)
        assert (after.return_code, after.stdout) == (0, "next")

        for command, expected in [
            ("true", ""),
            ("printf x", "x"),
            ("printf 'x\\n'", "x\n"),
            ("printf 'x\\n\\n'", "x\n\n"),
            ("printf '\\n\\nx'", "\n\nx"),
            ("printf 'a\\377\\376b'", "a\ufffd\ufffdb"),
        ]:
            result = await provider.exec(handle, command, timeout_s=10)
            assert (result.return_code, result.stdout, result.stderr) == (0, expected, ""), command

        mixed = await provider.exec(handle, "printf out; printf 'err\\n' >&2; exit 3", timeout_s=10)
        assert (mixed.return_code, mixed.stdout, mixed.stderr, mixed.error_type) == (3, "out", "err\n", None)

        # Supplied stdin feeds the whole compound command, not only its last part.
        reads = await provider.exec(
            handle, 'read first; read second; printf "%s|%s" "$first" "$second"', stdin=b"one\ntwo\n", timeout_s=10
        )
        assert (reads.return_code, reads.stdout) == (0, "one|two")
    finally:
        await provider.close(handle)


async def test_direct_exec_finds_a_status_frame_split_across_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With 7-byte reads the frame always straddles several reads, wherever the output ends.
    monkeypatch.setattr(apptainer_provider, "DIRECT_EXEC_READ_CHUNK_BYTES", 7)
    provider, _staging = _direct_provider(monkeypatch, tmp_path)
    handle = await provider.create(SandboxSpec(image="unused.sif"))
    try:
        for length in range(0, 9):
            result = await provider.exec(handle, f"printf %s {'y' * length}; exit {length}", timeout_s=10)
            assert (result.return_code, result.stdout) == (length, "y" * length)
        newlines = await provider.exec(handle, "printf 'line one\\n\\nline three\\n\\n'", timeout_s=10)
        assert (newlines.return_code, newlines.stdout) == (0, "line one\n\nline three\n\n")
    finally:
        await provider.close(handle)


async def test_direct_exec_handle_cannot_be_serialized_and_stays_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nemo_gym.sandbox.api import AsyncSandbox

    provider, staging = _direct_provider(monkeypatch, tmp_path)
    handle = await provider.create(SandboxSpec(image="unused.sif"))
    with pytest.raises(NotImplementedError, match="direct-exec sandboxes cannot be serialized"):
        await provider.serialize_handle(handle)
    sandbox = AsyncSandbox(provider, SandboxSpec())
    sandbox._handle = handle
    sandbox._stopped = False
    with pytest.raises(NotImplementedError, match="direct-exec sandboxes cannot be serialized"):
        await sandbox.serialize()

    # The refusal leaves the sandbox usable and closeable.
    after = await provider.exec(handle, "printf still-usable", timeout_s=10)
    assert (after.return_code, after.stdout) == (0, "still-usable")
    assert await provider.status(handle) is SandboxStatus.RUNNING
    await provider.close(handle)
    assert await provider.status(handle) is SandboxStatus.STOPPED
    assert not staging.exists()


async def test_a_cancelled_direct_exec_stops_the_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, staging = _direct_provider(monkeypatch, tmp_path)
    handle = await provider.create(SandboxSpec(image="unused.sif"))
    shell = handle.raw.direct_process
    pid_file = staging / "command.pid"

    # An ordinary failing command leaves the sandbox reusable.
    failed = await provider.exec(handle, "exit 9", timeout_s=10)
    assert failed.return_code == 9
    assert await provider.status(handle) is SandboxStatus.RUNNING

    async def run_hanging_command() -> None:
        await provider.exec(handle, f"echo $$ > {shlex.quote(str(pid_file))}; exec sleep 600", timeout_s=None)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(run_hanging_command)
        with anyio.fail_after(10):
            while not (pid_file.exists() and pid_file.read_text().strip()):
                await anyio.sleep(0.02)
        tasks.cancel_scope.cancel()

    # The cancelled command and the shell it ran in are gone, not left running behind the released lock.
    assert shell.returncode is not None
    await _wait_until_gone(int(pid_file.read_text()))
    assert not handle.raw.direct_exec_lock.locked()
    assert await provider.status(handle) is SandboxStatus.STOPPED

    # A later exec reports the stopped sandbox at once instead of waiting behind the abandoned command.
    with anyio.fail_after(5):
        after = await provider.exec(handle, "printf next", timeout_s=60)
    assert (after.return_code, after.error_type) == (apptainer_provider.SANDBOX_RUNTIME_RETURN_CODE, "sandbox")
    assert "not running" in after.stderr

    await provider.close(handle)
    assert not staging.exists()


async def test_a_direct_exec_timeout_stops_the_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A documented limit of direct mode: the provider cannot stop one command, so a timeout stops the shell.
    provider, staging = _direct_provider(monkeypatch, tmp_path)
    handle = await provider.create(SandboxSpec(image="unused.sif"))
    timed_out = await provider.exec(handle, "exec sleep 600", timeout_s=0.3)
    assert (timed_out.error_type, timed_out.stdout) == ("timeout", None)
    assert handle.raw.direct_process.returncode is not None
    assert await provider.status(handle) is SandboxStatus.STOPPED
    after = await provider.exec(handle, "printf next", timeout_s=10)
    assert after.error_type == "sandbox"
    await provider.close(handle)
    assert not staging.exists()


async def test_direct_exec_reports_a_shell_that_exits_mid_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _staging = _direct_provider(monkeypatch, tmp_path)
    handle = await provider.create(SandboxSpec(image="unused.sif"))
    try:
        shell_pid = handle.raw.direct_process.pid
        result = await provider.exec(handle, f"printf partial; kill -KILL {shell_pid}", timeout_s=10)
        assert (result.error_type, result.stdout) == ("sandbox", "partial")
        assert "exited during command" in result.stderr
    finally:
        await provider.close(handle)


async def test_cancelling_direct_create_during_its_probe_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_file = tmp_path / "probe.pid"
    provider, staging = _direct_provider(
        monkeypatch,
        tmp_path,
        probe={"command": f"echo $$ > {shlex.quote(str(pid_file))}; exec sleep 600", "timeout_s": 600},
    )
    started: list[Any] = []
    start_direct = provider._start_direct

    async def recording_start(argv: list[str], stderr: Any) -> Any:
        started.append((await start_direct(argv, stderr), stderr))
        return started[-1][0]

    monkeypatch.setattr(provider, "_start_direct", recording_start)

    async def create() -> None:
        await provider.create(SandboxSpec(image="unused.sif"))

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(create)
        with anyio.fail_after(10):
            while not (pid_file.exists() and pid_file.read_text().strip()):
                await anyio.sleep(0.02)
        tasks.cancel_scope.cancel()

    (shell, stderr_file) = started[0]
    assert shell.returncode is not None
    await _wait_until_gone(int(pid_file.read_text()))
    assert stderr_file.closed
    assert not staging.exists()


async def test_cancelling_direct_create_while_the_shell_starts_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, staging = _direct_provider(monkeypatch, tmp_path)
    stderr_files: list[Any] = []

    async def hanging_start(argv: list[str], stderr: Any) -> Any:
        stderr_files.append(stderr)
        await anyio.sleep_forever()

    monkeypatch.setattr(provider, "_start_direct", hanging_start)
    with anyio.move_on_after(0.2) as scope:
        await provider.create(SandboxSpec(image="unused.sif"))
    assert scope.cancelled_caught
    assert stderr_files[0].closed
    assert not staging.exists()


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
    provider = apptainer_provider.ApptainerProvider(exec={"timeout_grace_s": 0})
    with pytest.raises(TimeoutError):
        await provider._run([shutil.which("sleep"), "5"], timeout_s=0.1)


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not available")
async def test_run_daemonizing_returns_despite_lingering_child(fake_binary: str, tmp_path: Path) -> None:
    """Regression: a backgrounded child inheriting stdout must not wedge the read.

    Mirrors ``apptainer instance start``, which forks a long-lived instance that
    keeps the child's stdout/stderr open. The pipe-based path would block on EOF
    until timeout; the daemonizing path waits only for the foreground process.
    """
    provider = apptainer_provider.ApptainerProvider()
    # Foreground prints and exits immediately.
    # The backgrounded loop holds the inherited stdout fd open until the test releases it, well past the timeout.
    release = tmp_path / "release"
    lingering = f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.05; done"
    argv = [shutil.which("sh"), "-c", f"{lingering} & printf started"]
    try:
        code, out, _err = await provider._run(argv, timeout_s=10, daemonize=True)
    finally:
        release.touch()
    assert code == 0
    assert out == "started"


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not available")
async def test_run_returns_when_the_command_exits_despite_a_lingering_child(fake_binary: str, tmp_path: Path) -> None:
    """Regression: an exec that leaves a process holding its stdout (`tmux new-session -d`) returns when it exits.

    Reading pipes to EOF waited for the lingering process: Terminus 2's tmux start never returned under uvloop.
    """
    provider = apptainer_provider.ApptainerProvider()
    release = tmp_path / "release"
    lingering = f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.05; done"
    argv = [shutil.which("sh"), "-c", f"{lingering} & printf started; printf warn >&2"]
    try:
        code, out, err = await provider._run(argv, timeout_s=5)
    finally:
        release.touch()
    assert (code, out, err) == (0, "started", "warn")


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


@pytest.mark.parametrize("workdir_arg", [None, ["--workdir", "/custom"], ["--workdir=/custom"], ["-W", "/custom"]])
async def test_containment_uses_private_disk_backed_workdir(fake_binary, monkeypatch, tmp_path, workdir_arg):
    staging = tmp_path / "staging"
    monkeypatch.setattr(apptainer_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))
    provider, rec = _make_provider(
        monkeypatch,
        lambda argv: (0, "", ""),
        create={"extra_start_args": ["--containall", *(workdir_arg or [])]},
        probe={"command": None},
    )
    handle = await provider.create(SandboxSpec(image="ubuntu:22.04"))
    try:
        await provider.exec(handle, "true")
        argv = rec.calls[0]["argv"]
        if workdir_arg:
            assert _contains_seq(argv, workdir_arg)
            assert not (staging / ".apptainer-workdir").exists()
        else:
            private = staging / ".apptainer-workdir"
            assert _contains_seq(argv, ["--workdir", str(private)])
            assert private.stat().st_mode & 0o777 == 0o700
    finally:
        await provider.close(handle)
    assert not staging.exists()


@requires_sh
@pytest.mark.parametrize("daemonize", [False, True])
async def test_a_cancelled_apptainer_command_is_killed(fake_binary: str, tmp_path: Path, daemonize: bool) -> None:
    # A cancellation is not the command's own timeout: no SIGTERM grace, the group is killed at once.
    cli = stand_in_cli(tmp_path)
    provider = apptainer_provider.ApptainerProvider(exec={"timeout_grace_s": 30})
    with anyio.move_on_after(0.5):
        await provider._run([str(cli.path)], timeout_s=None, daemonize=daemonize)
    assert not is_running(cli.last_pid())
    cli.hang.unlink()
    assert (await provider._run([str(cli.path)], timeout_s=10, daemonize=daemonize))[0] == 0
