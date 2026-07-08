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

import io
import shlex
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.enroot import provider as enroot_provider


pytestmark = pytest.mark.sandbox


FAKE_BINARY = "/usr/bin/enroot"
FAKE_PID = 987654


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
class RunRecorder:
    """Stand-in for EnrootProvider._run that records argv and returns canned output."""

    def __init__(self, responder: Callable[[list[str]], tuple[int, str, str]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responder = responder

    async def __call__(self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None):
        self.calls.append({"argv": list(argv), "timeout_s": timeout_s, "stdin": stdin})
        return self._responder(list(argv))


class FakeProc:
    """Stand-in for the detached ``enroot start`` subprocess."""

    def __init__(self, pid: int = FAKE_PID, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode

    async def wait(self) -> int:
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode


class StartRecorder:
    """Stand-in for EnrootProvider._start_detached."""

    def __init__(self, proc: FakeProc) -> None:
        self.calls: list[list[str]] = []
        self._proc = proc

    async def __call__(self, argv: list[str]):
        self.calls.append(list(argv))
        return self._proc, io.BytesIO(), io.BytesIO()


def _contains_seq(haystack: list[str], needle: list[str]) -> bool:
    return any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _list_output(name: str, pid: int | None = FAKE_PID) -> str:
    """Build a fake ``enroot list -f`` table (header + one data row)."""
    header = "NAME                    PID  COMM  STATE  STARTED  TIME  MNTNS  USERNS  COMMAND"
    row = f"{name}  {pid}  sh  Ss  now  0  1  1  sleep" if pid is not None else name
    return f"{header}\n{row}\n"


def _make_handle(
    staging: Path,
    *,
    name: str = "nemo-gym-x",
    mount: str = "/sandbox",
    env: dict[str, str] | None = None,
    container_pid: int | None = FAKE_PID,
    start_pgid: int | None = None,
) -> SandboxHandle:
    inst = enroot_provider._EnrootInstance(
        name=name,
        sqsh_path=staging / "img.sqsh",
        staging_dir=staging,
        mount_point=mount,
        image="ubuntu:22.04",
        env=env or {},
        container_pid=container_pid,
        start_pgid=start_pgid,
        proc=None,
    )
    return SandboxHandle(sandbox_id=name, provider_name="enroot", raw=inst)


@pytest.fixture
def fake_binary(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(enroot_provider, "_require_enroot", lambda: FAKE_BINARY)
    return FAKE_BINARY


@pytest.fixture
def no_killpg(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Prevent tests from sending real signals to real process groups."""
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(enroot_provider.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    return calls


def _make_provider(
    monkeypatch: pytest.MonkeyPatch,
    responder: Callable[[list[str]], tuple[int, str, str]],
    tmp_path: Path,
    *,
    start_proc: FakeProc | None = None,
    **kwargs: Any,
) -> tuple[Any, RunRecorder, StartRecorder]:
    create_kwargs = dict(kwargs.pop("create", {}) or {})
    create_kwargs.setdefault("base_dir", str(tmp_path / "enroot_home"))
    provider = enroot_provider.EnrootProvider(create=create_kwargs, **kwargs)
    rec = RunRecorder(responder)
    monkeypatch.setattr(provider, "_run", rec)
    start_rec = StartRecorder(start_proc or FakeProc())
    monkeypatch.setattr(provider, "_start_detached", start_rec)
    return provider, rec, start_rec


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_require_enroot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(enroot_provider.shutil, "which", lambda _name: "/opt/enroot")
    assert enroot_provider._require_enroot() == "/opt/enroot"

    monkeypatch.setattr(enroot_provider.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="enroot"):
        enroot_provider._require_enroot()


def test_coerce_config() -> None:
    coerce = enroot_provider._coerce_config
    cls = enroot_provider.EnrootExecConfig

    assert coerce(None, cls) == cls()
    existing = cls(concurrency=4)
    assert coerce(existing, cls) is existing
    assert coerce({"concurrency": 7}, cls).concurrency == 7
    with pytest.raises(TypeError):
        coerce(123, cls)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="start_timeout_s"):
        enroot_provider.EnrootCreateConfig(start_timeout_s=0)
    with pytest.raises(ValueError, match="start_poll_s"):
        enroot_provider.EnrootCreateConfig(start_poll_s=0)
    with pytest.raises(ValueError, match="absolute"):
        enroot_provider.EnrootCreateConfig(mount_point="relative")
    with pytest.raises(ValueError, match="default_timeout_s"):
        enroot_provider.EnrootExecConfig(default_timeout_s=-1)
    with pytest.raises(ValueError, match="concurrency"):
        enroot_provider.EnrootExecConfig(concurrency=0)
    with pytest.raises(ValueError, match="timeout_s"):
        enroot_provider.EnrootProbeConfig(timeout_s=0)
    with pytest.raises(ValueError, match="deadline_s"):
        enroot_provider.EnrootProbeConfig(deadline_s=0)
    with pytest.raises(ValueError, match="stable_count"):
        enroot_provider.EnrootProbeConfig(stable_count=0)
    with pytest.raises(ValueError, match="stable_delay_s"):
        enroot_provider.EnrootProbeConfig(stable_delay_s=-1)
    assert enroot_provider.EnrootProbeConfig(command=None, timeout_s=0).command is None


def test_translate_docker_uri() -> None:
    translate = enroot_provider._translate_docker_uri
    assert translate("ubuntu:22.04") == "docker://ubuntu:22.04"
    assert translate("library/ubuntu:22.04") == "docker://library/ubuntu:22.04"
    assert translate("nvcr.io/nvidia/pytorch:24.01") == "docker://nvcr.io#nvidia/pytorch:24.01"
    assert translate("localhost:5000/img:tag") == "docker://localhost:5000#img:tag"
    # Docker Hub canonical hosts are dropped (real API host is registry-1.docker.io).
    assert translate("docker.io/swebench/foo:latest") == "docker://swebench/foo:latest"
    assert translate("index.docker.io/library/ubuntu:22.04") == "docker://library/ubuntu:22.04"
    assert translate("registry-1.docker.io/swebench/foo:latest") == "docker://swebench/foo:latest"


def test_resolve_image(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    resolve = provider._resolve_image

    # A .sqsh reference (even if not existing) is treated as a local path.
    assert resolve("/imgs/foo.sqsh") == (None, Path("/imgs/foo.sqsh"))
    # An explicit scheme is passed through as an import URI.
    assert resolve("docker://ubuntu:22.04") == ("docker://ubuntu:22.04", None)
    # A bare Docker reference is translated.
    assert resolve("ubuntu:22.04") == ("docker://ubuntu:22.04", None)
    # An existing on-disk path is used directly.
    existing = tmp_path / "rootfs.img"
    existing.write_bytes(b"x")
    assert resolve(str(existing)) == (None, existing)


def test_resource_gpu_env() -> None:
    from nemo_gym.sandbox.providers.base import SandboxResources

    assert enroot_provider._resource_gpu_env(SandboxResources()) == {}
    assert enroot_provider._resource_gpu_env(SandboxResources(gpu=0)) == {}
    assert enroot_provider._resource_gpu_env(SandboxResources(gpu=2)) == {"NVIDIA_VISIBLE_DEVICES": "0,1"}


def test_path_under_mount() -> None:
    under = enroot_provider._path_under_mount
    assert under("/sandbox", "/sandbox/a/b.txt") == "a/b.txt"
    assert under("/sandbox", "/sandbox") == ""
    assert under("/sandbox/", "/sandbox/x") == "x"
    assert under("/sandbox", "/sandbox/../outside.txt") is None
    assert under("/sandbox", "/etc/passwd") is None


def test_is_runtime_failure() -> None:
    assert enroot_provider._is_runtime_failure("[ERROR] no such process") is True
    assert enroot_provider._is_runtime_failure("nsenter: failed") is True
    assert enroot_provider._is_runtime_failure("ls: cannot access") is False


def test_coerce_mounts() -> None:
    coerce = enroot_provider._coerce_mounts
    assert coerce(None) == []
    assert coerce("/a:/b") == ["/a:/b"]
    assert coerce(["/a:/b", "/c:/d:ro"]) == ["/a:/b", "/c:/d:ro"]
    with pytest.raises(enroot_provider.EnrootCreateError, match="must be a string or list"):
        coerce(123)


def test_constructor_requires_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(enroot_provider.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError):
        enroot_provider.EnrootProvider(create={"base_dir": str(tmp_path)})


def test_constructor_pins_enroot_env(fake_binary: str, tmp_path: Path) -> None:
    provider = enroot_provider.EnrootProvider(create={"base_dir": str(tmp_path / "home")})
    env = provider._enroot_env
    assert env["ENROOT_DATA_PATH"] == str(tmp_path / "home" / "data")
    assert env["ENROOT_CACHE_PATH"] == str(tmp_path / "home" / "cache")
    assert env["ENROOT_RUNTIME_PATH"] == str(tmp_path / "home" / "runtime")
    # Directories are created eagerly.
    assert (tmp_path / "home" / "data").is_dir()
    assert provider._sqsh_cache_dir.is_dir()


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #
async def test_create_builds_argv_and_runs_probe(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(enroot_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "list" in argv:
            return (0, _list_output(_CREATED_NAME[0], pid=FAKE_PID), "")
        if "exec" in argv:
            return (0, enroot_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")  # create rootfs

    provider, rec, start_rec = _make_provider(
        monkeypatch, responder, tmp_path, exec={"default_mounts": ["/data:/data"]}
    )

    # Bypass image import; return a canned squashfs path and capture the container name.
    async def fake_ensure(image: str) -> Path:
        return tmp_path / "img.sqsh"

    monkeypatch.setattr(provider, "_ensure_image", fake_ensure)

    spec = SandboxSpec(
        image="ubuntu:22.04",
        env={"FOO": "bar"},
        resources={"cpu": 2, "memory_mib": 1024, "gpu": 1, "disk_gib": 50},
        ttl_s=60,
        provider_options={"mounts": ["/host/a:/code/a"]},
    )

    # The `list` responder needs the real generated name; capture it via mkdtemp+uuid.
    with caplog.at_level("WARNING"):
        handle = await provider.create(spec)

    assert "ttl_s is not supported" in caplog.text
    assert "not enforced by standalone enroot" in caplog.text
    assert handle.provider_name == "enroot"
    assert handle.sandbox_id.startswith(enroot_provider.CONTAINER_NAME_PREFIX)
    assert handle.raw.container_pid == FAKE_PID
    assert handle.raw.start_pgid == FAKE_PID
    assert handle.raw.env == {"FOO": "bar"}

    # create rootfs
    create_argv = next(c["argv"] for c in rec.calls if "create" in c["argv"])
    assert create_argv[:4] == [FAKE_BINARY, "create", "-n", handle.sandbox_id]

    # detached start argv (captured by StartRecorder)
    start_argv = start_rec.calls[0]
    assert start_argv[:2] == [FAKE_BINARY, "start"]
    assert "--rw" in start_argv
    assert _contains_seq(start_argv, ["-m", f"{staging}:/sandbox"])
    assert _contains_seq(start_argv, ["-m", "/data:/data"])
    assert _contains_seq(start_argv, ["-m", "/host/a:/code/a"])
    assert _contains_seq(start_argv, ["-e", "FOO=bar"])
    assert _contains_seq(start_argv, ["-e", "NVIDIA_VISIBLE_DEVICES=0"])
    assert start_argv[-4:] == [handle.sandbox_id, "sh", "-c", enroot_provider.DEFAULT_INIT_COMMAND]


# The generated container name is random; the create test above needs the `list`
# responder to echo whatever name create picked. We capture it by patching uuid.
_CREATED_NAME = ["nemo-gym-testfixedname"]


@pytest.fixture(autouse=True)
def _fixed_container_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FixedHex:
        hex = "testfixedname"

    monkeypatch.setattr(enroot_provider.uuid, "uuid4", lambda: _FixedHex())


async def test_create_requires_image(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    with pytest.raises(enroot_provider.EnrootCreateError, match="image is required"):
        await provider.create(SandboxSpec(image=None))


async def test_create_rootfs_failure_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (1, "", "boom"), tmp_path)

    async def fake_ensure(image: str) -> Path:
        return tmp_path / "img.sqsh"

    monkeypatch.setattr(provider, "_ensure_image", fake_ensure)
    with pytest.raises(enroot_provider.EnrootCreateError, match="create failed"):
        await provider.create(SandboxSpec(image="ubuntu:22.04"))


async def test_create_start_early_exit_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_killpg: list
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(enroot_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        return (0, "", "")  # create + remove succeed; list never finds a pid

    provider, rec, _sr = _make_provider(monkeypatch, responder, tmp_path, start_proc=FakeProc(returncode=1))

    async def fake_ensure(image: str) -> Path:
        return tmp_path / "img.sqsh"

    monkeypatch.setattr(provider, "_ensure_image", fake_ensure)
    with pytest.raises(enroot_provider.EnrootCreateError, match="exited early"):
        await provider.create(SandboxSpec(image="ubuntu:22.04"))
    assert not staging.exists()
    assert any("remove" in c["argv"] for c in rec.calls)


async def test_create_start_timeout_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_killpg: list
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(enroot_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "list" in argv:
            return (0, "NAME PID\n", "")  # container never appears
        return (0, "", "")

    provider, _rec, _sr = _make_provider(
        monkeypatch, responder, tmp_path, create={"start_timeout_s": 0.05, "start_poll_s": 0.01}
    )

    async def fake_ensure(image: str) -> Path:
        return tmp_path / "img.sqsh"

    monkeypatch.setattr(provider, "_ensure_image", fake_ensure)
    with pytest.raises(enroot_provider.EnrootCreateError, match="did not start"):
        await provider.create(SandboxSpec(image="ubuntu:22.04"))
    assert not staging.exists()


async def test_create_probe_failure_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_killpg: list
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(enroot_provider.tempfile, "mkdtemp", lambda prefix: str(staging.mkdir() or staging))

    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "list" in argv:
            return (0, _list_output(_CREATED_NAME[0]), "")
        if "exec" in argv:
            return (1, "", "probe broke")
        return (0, "", "")

    provider, rec, _sr = _make_provider(monkeypatch, responder, tmp_path)

    async def fake_ensure(image: str) -> Path:
        return tmp_path / "img.sqsh"

    monkeypatch.setattr(provider, "_ensure_image", fake_ensure)
    with pytest.raises(enroot_provider.EnrootCreateVerificationError):
        await provider.create(SandboxSpec(image="ubuntu:22.04"))
    assert not staging.exists()
    assert any("remove" in c["argv"] for c in rec.calls)


# --------------------------------------------------------------------------- #
# _ensure_image (import + cache)
# --------------------------------------------------------------------------- #
async def test_ensure_image_local_sqsh_used_directly(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sqsh = tmp_path / "local.sqsh"
    sqsh.write_bytes(b"squashfs")
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    assert await provider._ensure_image(str(sqsh)) == sqsh
    assert rec.calls == []  # no import


async def test_ensure_image_missing_sqsh_raises(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    with pytest.raises(enroot_provider.EnrootCreateError, match="does not exist"):
        await provider._ensure_image(str(tmp_path / "nope.sqsh"))


async def test_ensure_image_imports_and_caches(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        # Simulate a successful import by writing the -o target file.
        out_idx = argv.index("-o") + 1
        Path(argv[out_idx]).write_bytes(b"imported")
        assert argv[-1] == "docker://ubuntu:22.04"
        return (0, "", "")

    provider, rec, _sr = _make_provider(monkeypatch, responder, tmp_path)
    first = await provider._ensure_image("ubuntu:22.04")
    assert first.exists() and first.read_bytes() == b"imported"
    assert len([c for c in rec.calls if "import" in c["argv"]]) == 1

    # A second call hits the cache — no new import.
    second = await provider._ensure_image("ubuntu:22.04")
    assert second == first
    assert len([c for c in rec.calls if "import" in c["argv"]]) == 1


async def test_ensure_image_import_failure_raises(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (1, "", "pull denied"), tmp_path)
    with pytest.raises(enroot_provider.EnrootCreateError, match="import failed"):
        await provider._ensure_image("ubuntu:22.04")


async def test_ensure_image_import_timeout_raises(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec, _sr = _make_provider(monkeypatch, responder, tmp_path)
    with pytest.raises(enroot_provider.EnrootCreateError, match="timed out"):
        await provider._ensure_image("ubuntu:22.04")


# --------------------------------------------------------------------------- #
# exec
# --------------------------------------------------------------------------- #
async def test_exec_normal_with_cwd_and_env(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "hello", ""), tmp_path)
    handle = _make_handle(tmp_path)

    result = await provider.exec(handle, "echo hi", cwd="/work", env={"A": "b"})

    assert result.return_code == 0
    assert result.stdout == "hello"
    assert result.error_type is None

    argv = rec.calls[0]["argv"]
    assert argv[:2] == [FAKE_BINARY, "exec"]
    assert _contains_seq(argv, ["-e", "A=b"])
    assert str(FAKE_PID) in argv
    assert argv[-3:] == ["sh", "-c", "cd /work && echo hi"]
    assert rec.calls[0]["timeout_s"] == 180


async def test_exec_reapplies_create_env_and_overrides_call_env(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    handle = _make_handle(tmp_path, env={"A": "from-create", "B": "base"})

    await provider.exec(handle, "env", env={"A": "from-call"})

    argv = rec.calls[0]["argv"]
    assert _contains_seq(argv, ["-e", "A=from-call"])
    assert _contains_seq(argv, ["-e", "B=base"])
    assert "A=from-create" not in argv


async def test_exec_no_pid_is_sandbox_error(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    handle = _make_handle(tmp_path, container_pid=None)
    result = await provider.exec(handle, "echo hi")
    assert result.return_code == enroot_provider.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "sandbox"
    assert rec.calls == []  # never shelled out


@pytest.mark.parametrize(
    "user,expect_su",
    [(None, False), ("root", False), (0, False), ("alice", True)],
)
async def test_exec_user_mapping(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user: Any, expect_su: bool
) -> None:
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    await provider.exec(_make_handle(tmp_path), "whoami", user=user)
    argv = rec.calls[0]["argv"]
    if expect_su:
        assert argv[-1] == f"su -s /bin/sh -c {shlex.quote('whoami')} {shlex.quote(str(user))}"
    else:
        assert argv[-1] == "whoami"


async def test_exec_passes_stdin(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "ok", ""), tmp_path)
    await provider.exec(_make_handle(tmp_path), "cat", stdin=b"prompt-bytes")
    assert rec.calls[0]["stdin"] == b"prompt-bytes"


async def test_exec_timeout(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("too slow")

    provider, _rec, _sr = _make_provider(monkeypatch, responder, tmp_path)
    result = await provider.exec(_make_handle(tmp_path), "sleep 99", timeout_s=1)
    assert result.return_code == enroot_provider.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "timeout"
    assert result.stdout is None


async def test_exec_runtime_failure(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (1, "", "[ERROR] no such process"), tmp_path)
    result = await provider.exec(_make_handle(tmp_path), "echo hi")
    assert result.return_code == enroot_provider.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "sandbox"


async def test_exec_command_failure_is_not_runtime_error(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (2, "", "ls: cannot access"), tmp_path)
    result = await provider.exec(_make_handle(tmp_path), "ls /nope")
    assert result.return_code == 2
    assert result.error_type is None


# --------------------------------------------------------------------------- #
# upload / download
# --------------------------------------------------------------------------- #
async def test_upload_fast_path(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    handle = _make_handle(staging)

    src = tmp_path / "src.txt"
    src.write_bytes(b"payload")
    await provider.upload_file(handle, src, "/sandbox/sub/dest.txt")

    assert (staging / "sub" / "dest.txt").read_bytes() == b"payload"
    assert rec.calls == []


async def test_upload_fallback(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    handle = _make_handle(staging)

    captured: dict[str, Any] = {}

    async def fake_exec(h: SandboxHandle, command: str, **_: Any) -> SandboxExecResult:
        captured["command"] = command
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(provider, "exec", fake_exec)
    src = tmp_path / "src.txt"
    src.write_bytes(b"payload")
    await provider.upload_file(handle, src, "/etc/app.conf")

    assert "cp" in captured["command"]
    assert "/etc/app.conf" in captured["command"]
    assert list(staging.iterdir()) == []


async def test_upload_fallback_error(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    handle = _make_handle(staging)

    async def fake_exec(h: SandboxHandle, command: str, **_: Any) -> SandboxExecResult:
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
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    handle = _make_handle(staging)

    dest = tmp_path / "local.txt"
    await provider.download_file(handle, "/sandbox/out/r.txt", dest)
    assert dest.read_bytes() == b"result"
    assert rec.calls == []


async def test_download_fallback(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    handle = _make_handle(staging)

    async def fake_exec(h: SandboxHandle, command: str, **_: Any) -> SandboxExecResult:
        container_tmp = shlex.split(command)[-1]
        (staging / Path(container_tmp).name).write_bytes(b"remote-bytes")
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(provider, "exec", fake_exec)
    dest = tmp_path / "local.txt"
    await provider.download_file(handle, "/var/log/app.log", dest)
    assert dest.read_bytes() == b"remote-bytes"
    assert list(staging.iterdir()) == []


async def test_download_fallback_error(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    handle = _make_handle(staging)

    async def fake_exec(h: SandboxHandle, command: str, **_: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="missing", return_code=1)

    monkeypatch.setattr(provider, "exec", fake_exec)
    with pytest.raises(RuntimeError, match="download"):
        await provider.download_file(handle, "/var/log/app.log", tmp_path / "local.txt")
    assert list(staging.iterdir()) == []


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
async def test_status_running(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, _list_output("nemo-gym-x"), ""), tmp_path)
    assert await provider.status(_make_handle(tmp_path)) is SandboxStatus.RUNNING


async def test_status_stopped_when_no_pid(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(
        monkeypatch, lambda argv: (0, _list_output("nemo-gym-x", pid=None), ""), tmp_path
    )
    assert await provider.status(_make_handle(tmp_path)) is SandboxStatus.STOPPED


async def test_status_stopped_when_absent(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, _list_output("other"), ""), tmp_path)
    assert await provider.status(_make_handle(tmp_path)) is SandboxStatus.STOPPED


async def test_status_unknown_paths(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handle = _make_handle(tmp_path)
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (1, "", "err"), tmp_path)
    assert await provider.status(handle) is SandboxStatus.UNKNOWN

    def timeout_responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec, _sr = _make_provider(monkeypatch, timeout_responder, tmp_path)
    assert await provider.status(handle) is SandboxStatus.UNKNOWN


# --------------------------------------------------------------------------- #
# close / aclose
# --------------------------------------------------------------------------- #
async def test_close_success(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_killpg: list
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    await provider.close(_make_handle(staging, start_pgid=12345))

    assert not staging.exists()
    assert _contains_seq(rec.calls[0]["argv"], [FAKE_BINARY, "remove", "-f", "nemo-gym-x"])
    assert (12345, enroot_provider.signal.SIGTERM) in no_killpg


async def test_close_missing_container_is_success(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (1, "", "does not exist"), tmp_path)
    await provider.close(_make_handle(staging))
    assert not staging.exists()


async def test_close_real_failure_raises_but_cleans_up(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (1, "", "permission denied"), tmp_path)
    with pytest.raises(RuntimeError, match="remove failed"):
        await provider.close(_make_handle(staging))
    assert not staging.exists()


async def test_close_timeout_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec, _sr = _make_provider(monkeypatch, responder, tmp_path)
    with pytest.raises(TimeoutError):
        await provider.close(_make_handle(staging))
    assert not staging.exists()


async def test_close_staging_removal_failure_is_logged(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)

    def boom(path: Any, ignore_errors: bool = False) -> None:
        raise OSError("locked")

    monkeypatch.setattr(enroot_provider.shutil, "rmtree", boom)
    with caplog.at_level("WARNING"):
        await provider.close(_make_handle(staging))
    assert "failed to remove staging dir" in caplog.text


async def test_aclose(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)
    assert await provider.aclose() is None


# --------------------------------------------------------------------------- #
# readiness probe
# --------------------------------------------------------------------------- #
async def test_verify_skipped_when_command_none(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path, probe={"command": None})

    async def boom(*_a: Any, **_k: Any) -> SandboxExecResult:
        raise AssertionError("exec should not be called when probe is disabled")

    monkeypatch.setattr(provider, "exec", boom)
    await provider._verify_created_handle(_make_handle(tmp_path))


async def test_verify_polls_until_stable(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(
        monkeypatch,
        lambda argv: (0, "", ""),
        tmp_path,
        probe={"deadline_s": 5, "stable_count": 2, "stable_delay_s": 0},
    )

    results = iter(
        [
            SandboxExecResult(stdout="", stderr="warming up", return_code=1),
            SandboxExecResult(stdout=enroot_provider.READY_PROBE_EXPECTED, stderr="", return_code=0),
            SandboxExecResult(stdout=enroot_provider.READY_PROBE_EXPECTED, stderr="", return_code=0),
        ]
    )

    async def fake_exec(*_a: Any, **_k: Any) -> SandboxExecResult:
        return next(results)

    monkeypatch.setattr(provider, "exec", fake_exec)
    await provider._verify_created_handle(_make_handle(tmp_path))


async def test_verify_deadline_exceeded(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(
        monkeypatch, lambda argv: (0, "", ""), tmp_path, probe={"deadline_s": 0.01, "stable_delay_s": 0.02}
    )

    async def always_fail(*_a: Any, **_k: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="nope", return_code=1)

    monkeypatch.setattr(provider, "exec", always_fail)
    with pytest.raises(enroot_provider.EnrootCreateVerificationError, match="within"):
        await provider._verify_created_handle(_make_handle(tmp_path))


async def test_verify_single_attempt_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec, _sr = _make_provider(monkeypatch, lambda argv: (0, "", ""), tmp_path)

    async def always_fail(*_a: Any, **_k: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="nope", return_code=1)

    monkeypatch.setattr(provider, "exec", always_fail)
    with pytest.raises(enroot_provider.EnrootCreateVerificationError, match="failed readiness probe"):
        await provider._verify_created_handle(_make_handle(tmp_path))


# --------------------------------------------------------------------------- #
# _run / _start_detached against real lightweight binaries
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("echo") is None, reason="echo not available")
async def test_run_real_echo(fake_binary: str, tmp_path: Path) -> None:
    provider = enroot_provider.EnrootProvider(create={"base_dir": str(tmp_path / "home")})
    code, out, err = await provider._run([shutil.which("echo"), "hi"], timeout_s=10)
    assert code == 0
    assert out.strip() == "hi"
    assert err == ""


@pytest.mark.skipif(shutil.which("cat") is None, reason="cat not available")
async def test_run_real_stdin(fake_binary: str, tmp_path: Path) -> None:
    provider = enroot_provider.EnrootProvider(create={"base_dir": str(tmp_path / "home")})
    code, out, _err = await provider._run([shutil.which("cat")], timeout_s=10, stdin=b"piped")
    assert code == 0
    assert out == "piped"


@pytest.mark.skipif(shutil.which("sleep") is None, reason="sleep not available")
async def test_run_real_timeout(fake_binary: str, tmp_path: Path) -> None:
    provider = enroot_provider.EnrootProvider(create={"base_dir": str(tmp_path / "home")})
    with pytest.raises(TimeoutError):
        await provider._run([shutil.which("sleep"), "5"], timeout_s=0.1)


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not available")
async def test_start_detached_does_not_await_exit(fake_binary: str, tmp_path: Path) -> None:
    """The detached start must return immediately even though the child lives on."""
    provider = enroot_provider.EnrootProvider(create={"base_dir": str(tmp_path / "home")})
    argv = [shutil.which("sh"), "-c", "sleep 30 & printf started; wait"]
    proc, out_f, err_f = await provider._start_detached(argv)
    try:
        assert proc.pid > 0
        assert proc.returncode is None  # still running (did not block on the sleep)
    finally:
        enroot_provider.os.killpg(proc.pid, enroot_provider.signal.SIGKILL)
        await proc.wait()
        out_f.close()
        err_f.close()
