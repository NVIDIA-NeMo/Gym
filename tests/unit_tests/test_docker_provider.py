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

import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.docker import provider as docker_provider


pytestmark = pytest.mark.sandbox


FAKE_BINARY = "/usr/bin/docker"


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
class RunRecorder:
    """Stand-in for DockerProvider._run that records argv and returns canned output."""

    def __init__(self, responder: Callable[[list[str]], tuple[int, str, str]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responder = responder

    async def __call__(
        self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None
    ) -> tuple[int, str, str]:
        self.calls.append({"argv": list(argv), "timeout_s": timeout_s, "stdin": stdin})
        return self._responder(list(argv))


def _contains_seq(haystack: list[str], needle: list[str]) -> bool:
    return any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _make_handle(
    *, name: str = "nemo-gym-x", image: str = "img", shell: str = "sh", env: dict[str, str] | None = None
) -> SandboxHandle:
    inst = docker_provider._DockerContainer(name=name, image=image, shell=shell, env=env or {})
    return SandboxHandle(sandbox_id=name, provider_name="docker", raw=inst)


@pytest.fixture
def fake_binary(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: FAKE_BINARY)
    return FAKE_BINARY


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, responder: Callable[[list[str]], tuple[int, str, str]], **kwargs: Any
) -> tuple[Any, RunRecorder]:
    # Pin the exec shell by default so create tests skip the auto-detect probe; pass exec={} to opt in.
    kwargs.setdefault("exec", {"exec_shell": "sh"})
    provider = docker_provider.DockerProvider(**kwargs)
    rec = RunRecorder(responder)
    monkeypatch.setattr(provider, "_run", rec)
    return provider, rec


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_require_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_provider.shutil, "which", lambda _name: "/opt/docker")
    assert docker_provider._require_docker() == "/opt/docker"

    monkeypatch.setattr(docker_provider.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="docker"):
        docker_provider._require_docker()


def test_coerce_config() -> None:
    coerce = docker_provider._coerce_config
    cls = docker_provider.DockerExecConfig

    assert coerce(None, cls) == cls()
    existing = cls(concurrency=4)
    assert coerce(existing, cls) is existing
    assert coerce({"concurrency": 7}, cls).concurrency == 7
    with pytest.raises(TypeError):
        coerce(123, cls)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="start_timeout_s"):
        docker_provider.DockerCreateConfig(start_timeout_s=0)
    with pytest.raises(ValueError, match="pids_limit"):
        docker_provider.DockerCreateConfig(pids_limit=0)
    with pytest.raises(ValueError, match="default_timeout_s"):
        docker_provider.DockerExecConfig(default_timeout_s=-1)
    with pytest.raises(ValueError, match="concurrency"):
        docker_provider.DockerExecConfig(concurrency=0)
    with pytest.raises(ValueError, match="exec_shell"):
        docker_provider.DockerExecConfig(exec_shell="")
    with pytest.raises(ValueError, match="timeout_s"):
        docker_provider.DockerProbeConfig(timeout_s=0)
    with pytest.raises(ValueError, match="deadline_s"):
        docker_provider.DockerProbeConfig(deadline_s=0)
    with pytest.raises(ValueError, match="stable_count"):
        docker_provider.DockerProbeConfig(stable_count=0)
    with pytest.raises(ValueError, match="stable_delay_s"):
        docker_provider.DockerProbeConfig(stable_delay_s=-1)
    # command=None disables the timeout_s validation gate.
    assert docker_provider.DockerProbeConfig(command=None, timeout_s=0).command is None


def test_resource_flags() -> None:
    res = SandboxResources(cpu=2, memory_mib=1024, gpu=1, disk_gib=50, gpu_type="h100")
    flags = docker_provider._resource_limit_flags(res) + docker_provider._resource_passthrough_flags(res)
    assert _contains_seq(flags, ["--cpus", "2"])
    assert _contains_seq(flags, ["--memory", "1024m"])
    assert _contains_seq(flags, ["--memory-swap", "1024m"])  # hard cap: swap == memory
    assert _contains_seq(flags, ["--gpus", "1"])
    # disk_gib and gpu_type have no flag.
    assert "50" not in flags and "h100" not in flags

    empty = SandboxResources()
    assert docker_provider._resource_limit_flags(empty) + docker_provider._resource_passthrough_flags(empty) == []


def test_normalize_image() -> None:
    normalize = docker_provider._normalize_image
    assert normalize("ubuntu:22.04") == "ubuntu:22.04"
    assert normalize("docker://ubuntu:22.04") == "ubuntu:22.04"
    assert normalize("ghcr.io/org/img:tag") == "ghcr.io/org/img:tag"


def test_to_sandbox_status() -> None:
    to_status = docker_provider._to_sandbox_status
    assert to_status("running") is SandboxStatus.RUNNING
    assert to_status("paused") is SandboxStatus.RUNNING
    assert to_status("created") is SandboxStatus.STARTING
    assert to_status("restarting") is SandboxStatus.STARTING
    assert to_status("exited") is SandboxStatus.STOPPED
    assert to_status("dead") is SandboxStatus.STOPPED
    assert to_status("nonsense") is SandboxStatus.UNKNOWN
    assert to_status(None) is SandboxStatus.UNKNOWN


def test_is_runtime_failure() -> None:
    assert docker_provider._is_runtime_failure("Error response from daemon: No such container: x") is True
    assert docker_provider._is_runtime_failure("Cannot connect to the Docker daemon") is True
    assert docker_provider._is_runtime_failure("ls: cannot access") is False


def test_is_missing_container() -> None:
    assert docker_provider._is_missing_container("No such container: x") is True
    assert docker_provider._is_missing_container("Error: No such object: y") is True
    assert docker_provider._is_missing_container("permission denied") is False


def test_coerce_str_list() -> None:
    f = docker_provider._coerce_str_list
    assert f(None, "volumes") == []
    assert f("/h:/c", "volumes") == ["/h:/c"]
    assert f(["/h:/c", "/h2:/c2:ro"], "volumes") == ["/h:/c", "/h2:/c2:ro"]
    with pytest.raises(docker_provider.DockerCreateError, match="run_args"):
        f(123, "run_args")


def test_redact_argv() -> None:
    argv = [FAKE_BINARY, "run", "--env", "SECRET=abc123", "--name", "x", "--env", "PLAIN", "img"]
    red = docker_provider._redact_argv(argv)
    assert "SECRET=abc123" not in red
    assert "SECRET=***" in red
    assert "PLAIN" in red  # no '=' -> left untouched
    assert red[:2] == [FAKE_BINARY, "run"]


def test_constructor_requires_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_provider.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError):
        docker_provider.DockerProvider()


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #
async def test_create_builds_argv_and_runs_probe(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "run" in argv:
            return (0, "cid", "")
        if "exec" in argv:
            return (0, docker_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    spec = SandboxSpec(
        image="ubuntu:22.04",
        workdir="/sandbox",
        env={"FOO": "bar"},
        resources={"cpu": 2, "memory_mib": 1024, "gpu": 1},
    )
    handle = await provider.create(spec)

    assert handle.provider_name == "docker"
    assert handle.sandbox_id.startswith(docker_provider.CONTAINER_NAME_PREFIX)
    assert handle.raw.image == "ubuntu:22.04"
    assert handle.raw.env == {"FOO": "bar"}

    run_argv = rec.calls[0]["argv"]
    assert run_argv[:4] == [FAKE_BINARY, "run", "-d", "--name"]
    assert run_argv[4] == handle.sandbox_id
    assert run_argv[5:7] == ["--label", f"{docker_provider.SANDBOX_LABEL}=1"]
    assert "--init" in run_argv
    assert "--rm" not in run_argv  # no ttl_s -> no self-remove
    assert _contains_seq(run_argv, ["-w", "/sandbox"])
    assert _contains_seq(run_argv, ["--env", "FOO=bar"])
    assert _contains_seq(run_argv, ["--cpus", "2.0"])
    assert _contains_seq(run_argv, ["--memory", "1024m"])
    assert _contains_seq(run_argv, ["--gpus", "1"])
    assert run_argv[-5:] == ["--entrypoint", "/bin/sh", "ubuntu:22.04", "-c", docker_provider.DEFAULT_KEEPALIVE_CMD]

    probe_argv = rec.calls[1]["argv"]
    assert "exec" in probe_argv
    assert handle.sandbox_id in probe_argv
    assert probe_argv[-3:] == ["sh", "-c", docker_provider.READY_PROBE_COMMAND]
    assert rec.calls[1]["timeout_s"] == 30


async def test_create_strips_docker_prefix(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, docker_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    handle = await provider.create(SandboxSpec(image="docker://ubuntu:22.04"))
    assert handle.raw.image == "ubuntu:22.04"
    assert rec.calls[0]["argv"][-3] == "ubuntu:22.04"


async def test_create_skips_cgroup_resource_limits(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, docker_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder, create={"apply_resource_limits": False})
    await provider.create(SandboxSpec(image="ubuntu:22.04", resources={"cpu": 2, "memory_mib": 1024, "gpu": 1}))

    run_argv = rec.calls[0]["argv"]
    assert "--cpus" not in run_argv
    assert "--memory" not in run_argv
    assert _contains_seq(run_argv, ["--gpus", "1"])  # passthrough is always applied


async def test_create_security_flags(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, docker_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(
        monkeypatch,
        responder,
        create={
            "network": "none",
            "read_only": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "pids_limit": 512,
            "extra_run_args": ["--dns", "1.1.1.1"],
        },
    )
    await provider.create(SandboxSpec(image="ubuntu:22.04"))

    run_argv = rec.calls[0]["argv"]
    assert _contains_seq(run_argv, ["--network", "none"])
    assert "--read-only" in run_argv
    assert _contains_seq(run_argv, ["--cap-drop", "ALL"])
    assert _contains_seq(run_argv, ["--security-opt", "no-new-privileges"])
    assert _contains_seq(run_argv, ["--pids-limit", "512"])
    assert _contains_seq(run_argv, ["--dns", "1.1.1.1"])


async def test_create_no_init_when_disabled(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, docker_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder, create={"use_init": False})
    await provider.create(SandboxSpec(image="ubuntu:22.04"))
    assert "--init" not in rec.calls[0]["argv"]


async def test_create_uses_spec_entrypoint(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, docker_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    await provider.create(SandboxSpec(image="ubuntu:22.04", entrypoint=["/bin/bash", "-lc", "serve"]))

    run_argv = rec.calls[0]["argv"]
    assert run_argv[-5:] == ["--entrypoint", "/bin/bash", "ubuntu:22.04", "-lc", "serve"]
    assert docker_provider.DEFAULT_KEEPALIVE_CMD not in run_argv


async def test_create_requires_image(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    with pytest.raises(docker_provider.DockerCreateError, match="image is required"):
        await provider.create(SandboxSpec(image=None))


async def test_create_run_failure_cleans_up(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "run" in argv:
            return (1, "", "boom")
        return (0, "", "")  # force-remove during cleanup

    provider, rec = _make_provider(monkeypatch, responder)
    with pytest.raises(docker_provider.DockerCreateError, match="failed"):
        await provider.create(SandboxSpec(image="ubuntu:22.04"))
    assert any(c["argv"][:3] == [FAKE_BINARY, "rm", "-f"] for c in rec.calls)


async def test_create_run_timeout_cleans_up(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "run" in argv:
            raise TimeoutError("slow")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    with pytest.raises(docker_provider.DockerCreateError, match="timed out"):
        await provider.create(SandboxSpec(image="ubuntu:22.04"))
    assert any(c["argv"][:3] == [FAKE_BINARY, "rm", "-f"] for c in rec.calls)


async def test_create_probe_failure_cleans_up(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "run" in argv:
            return (0, "", "")
        if "exec" in argv:
            return (1, "", "probe broke")
        return (0, "", "")  # force-remove during cleanup

    provider, rec = _make_provider(monkeypatch, responder)
    with pytest.raises(docker_provider.DockerCreateVerificationError):
        await provider.create(SandboxSpec(image="ubuntu:22.04"))
    assert any(c["argv"][:3] == [FAKE_BINARY, "rm", "-f"] for c in rec.calls)


async def test_resolve_shell_configured_skips_probe(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""), exec={"exec_shell": "/bin/zsh"})
    assert await provider._resolve_shell("c") == "/bin/zsh"
    assert rec.calls == []  # configured -> no probe


async def test_resolve_shell_autodetects_bash(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        return (0, "/bin/bash", "") if "command -v bash" in argv else (0, "", "")

    provider, _rec = _make_provider(monkeypatch, responder, exec={})
    assert await provider._resolve_shell("c") == "bash"


async def test_resolve_shell_falls_back_to_sh(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "not found"), exec={})
    assert await provider._resolve_shell("c") == "sh"


async def test_resolve_shell_suppresses_probe_error(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec = _make_provider(monkeypatch, boom, exec={})
    assert await provider._resolve_shell("c") == "sh"


async def test_create_autodetects_bash_shell(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "run" in argv:
            return (0, "cid", "")
        if "command -v bash" in argv:
            return (0, "/bin/bash", "")
        if "exec" in argv:
            return (0, docker_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, _rec = _make_provider(monkeypatch, responder, exec={})
    handle = await provider.create(SandboxSpec(image="ubuntu:22.04"))
    assert handle.raw.shell == "bash"


async def test_create_provider_options_volumes_and_run_args(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "exec" in argv:
            return (0, docker_provider.READY_PROBE_EXPECTED, "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    await provider.create(
        SandboxSpec(
            image="ubuntu:22.04",
            provider_options={"volumes": ["/h:/c", "/h2:/c2:ro"], "run_args": ["--dns", "8.8.8.8"]},
        )
    )
    run_argv = rec.calls[0]["argv"]
    assert _contains_seq(run_argv, ["-v", "/h:/c"])
    assert _contains_seq(run_argv, ["-v", "/h2:/c2:ro"])
    assert _contains_seq(run_argv, ["--dns", "8.8.8.8"])


async def test_create_provider_options_bad_type_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    with pytest.raises(docker_provider.DockerCreateError, match="volumes"):
        await provider.create(SandboxSpec(image="ubuntu:22.04", provider_options={"volumes": 123}))


async def test_create_ttl_s_bounds_keepalive_and_self_removes(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        return (0, docker_provider.READY_PROBE_EXPECTED, "") if "exec" in argv else (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    await provider.create(SandboxSpec(image="ubuntu:22.04", ttl_s=90))
    run_argv = rec.calls[0]["argv"]
    assert "--rm" in run_argv
    assert run_argv[-5:] == ["--entrypoint", "/bin/sh", "ubuntu:22.04", "-c", "sleep 90"]


async def test_create_ttl_s_ignored_with_entrypoint(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        return (0, docker_provider.READY_PROBE_EXPECTED, "") if "exec" in argv else (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    with caplog.at_level("WARNING"):
        await provider.create(SandboxSpec(image="ubuntu:22.04", ttl_s=90, entrypoint=["/bin/bash", "-lc", "serve"]))
    assert "ttl_s is not enforced" in caplog.text
    run_argv = rec.calls[0]["argv"]
    assert "--rm" not in run_argv
    assert run_argv[-5:] == ["--entrypoint", "/bin/bash", "ubuntu:22.04", "-lc", "serve"]


# --------------------------------------------------------------------------- #
# exec
# --------------------------------------------------------------------------- #
async def test_exec_normal_with_cwd_and_env(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "hello", ""))
    result = await provider.exec(_make_handle(), "echo hi", cwd="/work", env={"A": "b"})

    assert result.return_code == 0
    assert result.stdout == "hello"
    assert result.error_type is None

    argv = rec.calls[0]["argv"]
    assert argv[:2] == [FAKE_BINARY, "exec"]
    assert _contains_seq(argv, ["-w", "/work"])
    assert _contains_seq(argv, ["--env", "A=b"])
    assert argv[-4:] == ["nemo-gym-x", "sh", "-c", "echo hi"]
    assert rec.calls[0]["timeout_s"] == 180


async def test_exec_reapplies_create_env_and_overrides_call_env(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(env={"A": "from-create", "B": "base"})

    await provider.exec(handle, "env", env={"A": "from-call"})

    argv = rec.calls[0]["argv"]
    assert _contains_seq(argv, ["--env", "A=from-call"])
    assert _contains_seq(argv, ["--env", "B=base"])
    assert "A=from-create" not in argv


async def test_exec_empty_streams_are_strings(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    result = await provider.exec(_make_handle(), "true")
    assert result.return_code == 0
    assert result.stdout == ""
    assert result.stderr == ""


async def test_exec_uses_handle_shell(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # exec runs under the shell resolved at create (bash for conda images that use `source`).
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    await provider.exec(_make_handle(shell="bash"), "source x && echo hi")
    assert rec.calls[0]["argv"][-4:] == ["nemo-gym-x", "bash", "-c", "source x && echo hi"]


@pytest.mark.parametrize(
    "user,expect_user_flag",
    [(None, None), ("root", "0"), (0, "0"), ("alice", "alice"), (1000, "1000")],
)
async def test_exec_user_mapping(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, user: Any, expect_user_flag: str | None
) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    await provider.exec(_make_handle(), "whoami", user=user)
    argv = rec.calls[0]["argv"]

    if expect_user_flag is None:
        assert "--user" not in argv
    else:
        assert _contains_seq(argv, ["--user", expect_user_flag])


async def test_exec_passes_stdin(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "ok", ""))
    await provider.exec(_make_handle(), "cat", stdin=b"prompt-bytes")
    assert rec.calls[0]["stdin"] == b"prompt-bytes"
    assert "-i" in rec.calls[0]["argv"]


async def test_exec_timeout(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("too slow")

    provider, _rec = _make_provider(monkeypatch, responder)
    result = await provider.exec(_make_handle(), "sleep 99", timeout_s=1)

    assert result.return_code == docker_provider.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "timeout"
    assert result.stdout is None


async def test_exec_runtime_failure(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "Error: No such container: nemo-gym-x"))
    result = await provider.exec(_make_handle(), "echo hi")

    assert result.return_code == docker_provider.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "sandbox"
    assert "No such container" in result.stderr


async def test_exec_command_failure_is_not_runtime_error(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (2, "", "ls: cannot access"))
    result = await provider.exec(_make_handle(), "ls /nope")
    assert result.return_code == 2
    assert result.error_type is None


# --------------------------------------------------------------------------- #
# upload / download
# --------------------------------------------------------------------------- #
async def test_upload_builds_mkdir_and_cp(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    src = tmp_path / "src.txt"
    src.write_bytes(b"payload")

    await provider.upload_file(_make_handle(), src, "/etc/app.conf")

    exec_calls = [c for c in rec.calls if "exec" in c["argv"]]
    cp_calls = [c for c in rec.calls if "cp" in c["argv"]]
    assert exec_calls and "mkdir -p /etc" in exec_calls[0]["argv"][-1]
    assert _contains_seq(exec_calls[0]["argv"], ["--user", "0"])  # mkdir runs as root
    assert cp_calls[0]["argv"] == [FAKE_BINARY, "cp", str(src), "nemo-gym-x:/etc/app.conf"]


async def test_upload_no_parent_skips_mkdir(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    src = tmp_path / "src.txt"
    src.write_bytes(b"payload")

    await provider.upload_file(_make_handle(), src, "out.txt")  # relative -> no parent dir

    assert not any("exec" in c["argv"] for c in rec.calls)
    assert rec.calls[0]["argv"] == [FAKE_BINARY, "cp", str(src), "nemo-gym-x:out.txt"]


async def test_upload_mkdir_failure_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "denied") if "exec" in argv else (0, "", ""))
    src = tmp_path / "src.txt"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="mkdir parent"):
        await provider.upload_file(_make_handle(), src, "/etc/app.conf")


async def test_upload_cp_failure_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", "") if "exec" in argv else (1, "", "boom"))
    src = tmp_path / "src.txt"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="docker cp upload"):
        await provider.upload_file(_make_handle(), src, "/etc/app.conf")


async def test_download_builds_cp(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    dest = tmp_path / "nested" / "local.txt"

    await provider.download_file(_make_handle(), "/var/log/app.log", dest)

    assert dest.parent.exists()  # local parent created
    assert rec.calls[0]["argv"] == [FAKE_BINARY, "cp", "nemo-gym-x:/var/log/app.log", str(dest)]


async def test_download_cp_failure_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "missing"))
    with pytest.raises(RuntimeError, match="docker cp download"):
        await provider.download_file(_make_handle(), "/var/log/app.log", tmp_path / "local.txt")


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state,expected",
    [
        ("running", SandboxStatus.RUNNING),
        ("created", SandboxStatus.STARTING),
        ("exited", SandboxStatus.STOPPED),
        ("weird", SandboxStatus.UNKNOWN),
    ],
)
async def test_status_maps_state(
    fake_binary: str, monkeypatch: pytest.MonkeyPatch, state: str, expected: SandboxStatus
) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, f"{state}\n", ""))
    assert await provider.status(_make_handle()) is expected


async def test_status_missing_container_is_stopped(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "Error: No such object: nemo-gym-x"))
    assert await provider.status(_make_handle()) is SandboxStatus.STOPPED


async def test_status_other_error_is_unknown(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "some daemon hiccup"))
    assert await provider.status(_make_handle()) is SandboxStatus.UNKNOWN


async def test_status_timeout_is_unknown(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec = _make_provider(monkeypatch, responder)
    assert await provider.status(_make_handle()) is SandboxStatus.UNKNOWN


# --------------------------------------------------------------------------- #
# close / aclose
# --------------------------------------------------------------------------- #
async def test_close_success(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    await provider.close(_make_handle())
    assert rec.calls[0]["argv"] == [FAKE_BINARY, "rm", "-f", "nemo-gym-x"]


async def test_close_missing_container_is_success(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "No such container: nemo-gym-x"))
    await provider.close(_make_handle())  # does not raise


async def test_close_real_failure_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (1, "", "permission denied"))
    with pytest.raises(RuntimeError, match="docker rm -f failed"):
        await provider.close(_make_handle())


async def test_close_timeout_raises(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        raise TimeoutError("slow")

    provider, _rec = _make_provider(monkeypatch, responder)
    with pytest.raises(TimeoutError):
        await provider.close(_make_handle())


async def test_aclose(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    assert await provider.aclose() is None


# --------------------------------------------------------------------------- #
# readiness probe
# --------------------------------------------------------------------------- #
async def test_verify_skipped_when_command_none(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""), probe={"command": None})

    async def boom(*_a: Any, **_k: Any) -> SandboxExecResult:
        raise AssertionError("exec should not be called when probe is disabled")

    monkeypatch.setattr(provider, "exec", boom)
    await provider._verify_created_handle(_make_handle())


async def test_verify_single_attempt_failure(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))  # probe.deadline_s is None by default

    async def fail(*_a: Any, **_k: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="nope", return_code=1)

    monkeypatch.setattr(provider, "exec", fail)
    with pytest.raises(docker_provider.DockerCreateVerificationError, match="failed readiness probe"):
        await provider._verify_created_handle(_make_handle())


async def test_verify_polls_until_stable(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(
        monkeypatch, lambda argv: (0, "", ""), probe={"deadline_s": 5, "stable_count": 2, "stable_delay_s": 0}
    )
    results = iter(
        [
            SandboxExecResult(stdout="", stderr="warming up", return_code=1),
            SandboxExecResult(stdout=docker_provider.READY_PROBE_EXPECTED, stderr="", return_code=0),
            SandboxExecResult(stdout=docker_provider.READY_PROBE_EXPECTED, stderr="", return_code=0),
        ]
    )

    async def fake_exec(*_a: Any, **_k: Any) -> SandboxExecResult:
        return next(results)

    monkeypatch.setattr(provider, "exec", fake_exec)
    await provider._verify_created_handle(_make_handle())


async def test_verify_deadline_exceeded(fake_binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(
        monkeypatch, lambda argv: (0, "", ""), probe={"deadline_s": 0.01, "stable_delay_s": 0.02}
    )

    async def always_fail(*_a: Any, **_k: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="nope", return_code=1)

    monkeypatch.setattr(provider, "exec", always_fail)
    with pytest.raises(docker_provider.DockerCreateVerificationError, match="within"):
        await provider._verify_created_handle(_make_handle())


# --------------------------------------------------------------------------- #
# _run against real lightweight binaries (exercises subprocess plumbing)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("echo") is None, reason="echo not available")
async def test_run_real_echo(fake_binary: str) -> None:
    provider = docker_provider.DockerProvider()
    code, out, err = await provider._run([shutil.which("echo"), "hi"], timeout_s=10)
    assert code == 0
    assert out.strip() == "hi"
    assert err == ""


@pytest.mark.skipif(shutil.which("cat") is None, reason="cat not available")
async def test_run_real_stdin(fake_binary: str) -> None:
    provider = docker_provider.DockerProvider()
    code, out, _err = await provider._run([shutil.which("cat")], timeout_s=10, stdin=b"piped")
    assert code == 0
    assert out == "piped"


@pytest.mark.skipif(shutil.which("sleep") is None, reason="sleep not available")
async def test_run_real_timeout(fake_binary: str) -> None:
    provider = docker_provider.DockerProvider()
    with pytest.raises(TimeoutError):
        await provider._run([shutil.which("sleep"), "5"], timeout_s=0.1)
