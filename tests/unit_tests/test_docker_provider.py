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

"""Unit tests for the local Docker ``SandboxProvider`` (CLI mocked, no docker required)."""

import asyncio
from pathlib import Path
from typing import Any, Callable

import pytest

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.docker.provider import DockerSandboxProvider


class RunRecorder:
    """Stand-in for ``DockerSandboxProvider._run`` that records argv and returns canned output.

    The responder maps the captured ``docker`` args to a ``(rc, stdout, stderr)`` tuple, and may
    raise (e.g. ``TimeoutError``) to simulate a CLI failure.
    """

    def __init__(self, responder: Callable[[list[str]], tuple[int, str, str]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responder = responder

    async def __call__(self, *args: str, timeout_s: float | None = None) -> tuple[int, str, str]:
        self.calls.append({"args": list(args), "timeout_s": timeout_s})
        return self._responder(list(args))


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, responder: Callable[[list[str]], tuple[int, str, str]], **kwargs: Any
) -> tuple[DockerSandboxProvider, RunRecorder]:
    provider = DockerSandboxProvider(**kwargs)
    rec = RunRecorder(responder)
    monkeypatch.setattr(provider, "_run", rec)
    return provider, rec


def _ran(rec: RunRecorder, *prefix: str) -> bool:
    """True if any recorded call's args start with ``prefix`` (e.g. ``"rm", "-f"``)."""
    return any(call["args"][: len(prefix)] == list(prefix) for call in rec.calls)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_concurrency_must_be_positive() -> None:
    """A non-positive concurrency is rejected up front."""
    with pytest.raises(ValueError):
        DockerSandboxProvider(concurrency=0)


def test_concurrency_bounds_the_semaphore() -> None:
    """The provider's shared semaphore is sized to the configured concurrency."""
    assert DockerSandboxProvider(concurrency=4)._semaphore._value == 4


def test_missing_docker_binary_raises_clear_error() -> None:
    """A missing docker binary surfaces a clear SandboxCreateError, not a bare FileNotFoundError."""
    provider = DockerSandboxProvider(docker_bin="nemo-gym-no-such-docker-bin")
    with pytest.raises(SandboxCreateError, match="not found on PATH"):
        asyncio.run(provider.create(SandboxSpec(image="img:tag")))


# --------------------------------------------------------------------------- #
# create()
# --------------------------------------------------------------------------- #
def test_create_returns_handle_with_last_line_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """create() uses the LAST stdout line as the container id and pre-assigns a unique name."""
    provider, rec = _make_provider(
        monkeypatch, lambda args: (0, "WARNING: noise\ncontainer-abc\n", ""), network="host"
    )
    handle = asyncio.run(provider.create(SandboxSpec(image="img:tag", workdir="/testbed", env={"A": "1"})))
    assert handle.sandbox_id == "container-abc"
    run_args = rec.calls[0]["args"]
    assert run_args[:3] == ["run", "-d", "--init"]
    assert "--name" in run_args and run_args[run_args.index("--name") + 1].startswith("nemo-gym-")
    assert ["--network", "host"] == run_args[run_args.index("--network") : run_args.index("--network") + 2]
    assert "img:tag" in run_args


def test_create_requires_image() -> None:
    """A spec without an image is rejected before any docker call."""
    with pytest.raises(SandboxCreateError):
        asyncio.run(DockerSandboxProvider().create(SandboxSpec(image=None)))


def test_create_empty_stdout_guard_and_reap(monkeypatch: pytest.MonkeyPatch) -> None:
    """rc 0 with empty stdout raises (no IndexError) and reaps the pre-assigned name."""
    provider, rec = _make_provider(monkeypatch, lambda args: (0, "   \n", "") if args[0] == "run" else (0, "", ""))
    with pytest.raises(SandboxCreateError, match="did not return a container id"):
        asyncio.run(provider.create(SandboxSpec(image="img:tag")))
    assert _ran(rec, "rm", "-f")


def test_create_nonzero_rc_reaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero ``docker run`` reaps the orphan and raises with the stderr."""
    provider, rec = _make_provider(monkeypatch, lambda args: (125, "", "boom") if args[0] == "run" else (0, "", ""))
    with pytest.raises(SandboxCreateError, match="boom"):
        asyncio.run(provider.create(SandboxSpec(image="img:tag")))
    assert _ran(rec, "rm", "-f")


def test_create_timeout_reaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out ``docker run`` reaps the (possibly daemon-started) orphan by name."""

    def responder(args: list[str]) -> tuple[int, str, str]:
        if args[0] == "run":
            raise asyncio.TimeoutError
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    with pytest.raises(SandboxCreateError, match="timed out"):
        asyncio.run(provider.create(SandboxSpec(image="img:tag")))
    assert _ran(rec, "rm", "-f")


def test_create_applies_resource_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resource requests become ``--memory``/``--cpus``/``--gpus`` run args."""
    provider, rec = _make_provider(monkeypatch, lambda args: (0, "cid\n", ""))
    spec = SandboxSpec(image="img:tag", resources=SandboxResources(cpu=2, memory_mib=512, gpu=1))
    asyncio.run(provider.create(spec))
    run_args = rec.calls[0]["args"]
    assert "--memory=512m" in run_args
    assert "--cpus=2" in run_args
    assert "--gpus=all" in run_args


# --------------------------------------------------------------------------- #
# exec()
# --------------------------------------------------------------------------- #
def test_exec_classifies_docker_daemon_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """rc 125 with no stdout is a docker-daemon (``sandbox``) failure."""
    provider, _ = _make_provider(monkeypatch, lambda args: (125, "", "no such container"))
    res = asyncio.run(provider.exec(_handle(), "echo hi"))
    assert res.return_code == 125
    assert res.error_type == "sandbox"


def test_exec_126_127_are_user_errors_not_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """rc 126 (not executable) / 127 (command not found) are user-command errors, not infra."""
    for rc in (126, 127):
        provider, _ = _make_provider(monkeypatch, lambda args, rc=rc: (rc, "", "no"))
        res = asyncio.run(provider.exec(_handle(), "nope"))
        assert res.return_code == rc and res.error_type is None


def test_exec_uses_provider_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """exec() falls back to ``default_exec_timeout_s`` when the caller passes no timeout."""
    provider, rec = _make_provider(monkeypatch, lambda args: (0, "ok", ""), default_exec_timeout_s=99)
    asyncio.run(provider.exec(_handle(), "true"))
    assert rec.calls[-1]["timeout_s"] == 99


def test_exec_success_has_no_error_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful exec carries stdout and no error type."""
    provider, _ = _make_provider(monkeypatch, lambda args: (0, "ok", ""))
    res = asyncio.run(provider.exec(_handle(), "true"))
    assert res.return_code == 0 and res.stdout == "ok" and res.error_type is None


def test_exec_timeout_returns_124(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out exec returns rc 124 + ``timeout`` error type rather than raising."""

    def responder(args: list[str]) -> tuple[int, str, str]:
        raise asyncio.TimeoutError

    provider, _ = _make_provider(monkeypatch, responder)
    res = asyncio.run(provider.exec(_handle(), "sleep 1", timeout_s=0.01))
    assert res.return_code == 124 and res.error_type == "timeout"


# --------------------------------------------------------------------------- #
# status / close / file transfer
# --------------------------------------------------------------------------- #
def test_status_running_and_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """status() maps docker inspect output to RUNNING/STOPPED/UNKNOWN."""
    provider, _ = _make_provider(monkeypatch, lambda args: (0, "true\n", ""))
    assert asyncio.run(provider.status(_handle())) is SandboxStatus.RUNNING
    provider2, _ = _make_provider(monkeypatch, lambda args: (0, "false\n", ""))
    assert asyncio.run(provider2.status(_handle())) is SandboxStatus.STOPPED
    provider3, _ = _make_provider(monkeypatch, lambda args: (1, "", "gone"))
    assert asyncio.run(provider3.status(_handle())) is SandboxStatus.UNKNOWN


def test_close_force_removes(monkeypatch: pytest.MonkeyPatch) -> None:
    """close() force-removes the container by id."""
    provider, rec = _make_provider(monkeypatch, lambda args: (0, "", ""))
    asyncio.run(provider.close(_handle()))
    assert _ran(rec, "rm", "-f", "cid")


def test_upload_failure_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed ``docker cp`` upload raises a clear RuntimeError."""
    provider, _ = _make_provider(monkeypatch, lambda args: (0, "", "") if args[0] == "exec" else (1, "", "nope"))
    src = tmp_path / "f.txt"
    src.write_text("x")
    with pytest.raises(RuntimeError, match="upload failed"):
        asyncio.run(provider.upload_file(_handle(), src, "/dst/f.txt"))


def test_reap_orphan_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """_reap_orphan never raises, even when the ``docker rm`` itself fails/raises."""

    def responder(args: list[str]) -> tuple[int, str, str]:
        raise RuntimeError("docker daemon down")

    provider, _ = _make_provider(monkeypatch, responder)
    asyncio.run(provider._reap_orphan("nemo-gym-x"))  # must not raise


def _handle():
    """A minimal docker SandboxHandle for exec/status/close tests."""
    from nemo_gym.sandbox.providers.base import SandboxHandle

    return SandboxHandle(sandbox_id="cid", provider_name="docker", raw={"workdir": "/testbed"})
