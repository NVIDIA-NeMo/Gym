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
"""Unit tests for the OpenSandbox provider's sandbox identity guard (RL-1469).

The server may route requests for a dead sandbox to a live one that reused its
pod IP. The provider defends itself by making every command prove, inside the
pod that receives it, that the pod is the requested sandbox (``<sandbox_id>-0``
is the pod hostname), by moving file transfer onto that guarded path, and by
surfacing "sandbox gone" as a typed, non-retryable error.
"""

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxEndedError, SandboxHandle, SandboxMisrouteError
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider
from nemo_gym.sandbox.providers.opensandbox.provider import (
    MISROUTE_EXIT_CODE,
    MISROUTE_MARKER,
    OpenSandboxProvider,
)


class FakeRunCommandOpts:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeLog:
    def __init__(self, text: str) -> None:
        self.text = text


def _execution(stdout: str = "", stderr: str = "", exit_code: int | None = 0) -> SimpleNamespace:
    return SimpleNamespace(
        logs=SimpleNamespace(
            stdout=[FakeLog(stdout)] if stdout else [],
            stderr=[FakeLog(stderr)] if stderr else [],
        ),
        error=None,
        exit_code=exit_code,
    )


class FakeCommands:
    """Records commands; ``hostname`` decides whether the guard inside them passes."""

    def __init__(self, hostname: str) -> None:
        self.hostname = hostname
        self.calls: list[str] = []
        self.responses: list[SimpleNamespace] = []
        self.exceptions: list[BaseException] = []

    async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
        self.calls.append(command)
        if self.exceptions:
            raise self.exceptions.pop(0)
        if (
            f'"{self.hostname}"' not in command
            and f"'{self.hostname}'" not in command
            and self.hostname not in command
        ):
            return _execution(
                stderr=f"{MISROUTE_MARKER} expected={self.hostname} actual=other-pod-0",
                exit_code=MISROUTE_EXIT_CODE,
            )
        if self.responses:
            return self.responses.pop(0)
        return _execution(stdout="ok")


class FakeFiles:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str | bytes]] = []
        self.reads: list[str] = []

    async def write_file(self, target_path: str, data: str | bytes) -> None:
        self.writes.append((target_path, data))

    async def read_bytes(self, source_path: str) -> bytes:
        self.reads.append(source_path)
        return b"payload-from-sandbox"


class FakeRaw:
    def __init__(self, hostname: str) -> None:
        self.commands = FakeCommands(hostname)
        self.files = FakeFiles()


def _provider(**operations: Any) -> OpenSandboxProvider:
    return OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
        operations={"retries": 3, "retry_delay_s": 0.0, "retry_max_delay_s": 0.0, **operations},
    )


def _handle(sandbox_id: str, hostname: str) -> tuple[SandboxHandle, FakeRaw]:
    raw = FakeRaw(hostname)
    return SandboxHandle(sandbox_id=sandbox_id, provider_name="opensandbox", raw=raw), raw


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )


async def test_exec_guards_command_with_the_handle_pod_hostname(fake_sdk: None) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")

    result = await provider.exec(handle, "echo hi", timeout_s=5)

    assert result.return_code == 0
    assert len(raw.commands.calls) == 1
    guarded = raw.commands.calls[0]
    assert "sb-1-0" in guarded
    assert MISROUTE_MARKER in guarded
    assert guarded.endswith("echo hi")


async def test_exec_raises_misroute_when_the_pod_is_not_the_requested_sandbox(fake_sdk: None) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")
    raw.commands.hostname = "victim-0"

    with pytest.raises(SandboxMisrouteError) as excinfo:
        await provider.exec(handle, "echo hi", timeout_s=5)

    assert "sb-1" in str(excinfo.value)
    assert isinstance(excinfo.value, SandboxEndedError)
    assert len(raw.commands.calls) == 1, "a misroute must not be retried"


async def test_exec_exit_199_without_marker_is_an_ordinary_result(fake_sdk: None) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")
    raw.commands.responses.append(_execution(stderr="boom", exit_code=MISROUTE_EXIT_CODE))

    result = await provider.exec(handle, "exit 199", timeout_s=5)

    assert result.return_code == MISROUTE_EXIT_CODE
    assert result.stderr == "boom"


async def test_exec_guard_precedes_user_switch(fake_sdk: None) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")

    await provider.exec(handle, "whoami", timeout_s=5, user="agent")

    guarded = raw.commands.calls[0]
    assert guarded.index("sb-1-0") < guarded.index("su -s /bin/sh -c")
    assert guarded.endswith("su -s /bin/sh -c whoami agent")


async def test_upload_goes_through_guarded_exec_not_the_raw_file_api(fake_sdk: None, tmp_path: Path) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")
    source = tmp_path / "test.sh"
    source.write_bytes(b"echo grader\n")

    await provider.upload_file(handle, source, "/tests/test.sh")

    assert raw.files.writes == []
    assert raw.commands.calls, "upload must be executed as guarded commands"
    joined = "\n".join(raw.commands.calls)
    assert base64.b64encode(b"echo grader\n").decode() in joined
    assert "/tests/test.sh" in joined
    assert all("sb-1-0" in command for command in raw.commands.calls)


async def test_upload_is_chunked_for_large_files(fake_sdk: None, tmp_path: Path) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")
    payload = bytes(range(256)) * 1024  # 256 KiB, above one exec argument
    source = tmp_path / "blob.bin"
    source.write_bytes(payload)

    await provider.upload_file(handle, source, "/tmp/blob.bin")

    assert len(raw.commands.calls) >= 3
    assert all(len(command) < 131072 for command in raw.commands.calls), "each exec must fit one shell argument"


async def test_upload_to_a_misrouted_sandbox_is_refused_before_any_byte_lands(fake_sdk: None, tmp_path: Path) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")
    raw.commands.hostname = "victim-0"
    source = tmp_path / "test.sh"
    source.write_bytes(b"echo grader\n")

    with pytest.raises(SandboxMisrouteError):
        await provider.upload_file(handle, source, "/tests/test.sh")

    assert raw.files.writes == []
    assert len(raw.commands.calls) == 1, "upload must stop at the first refusal"


async def test_download_copies_under_guard_then_reads_the_private_staging_path(fake_sdk: None, tmp_path: Path) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")
    target = tmp_path / "reward.txt"

    await provider.download_file(handle, "/logs/verifier/reward.txt", target)

    assert target.read_bytes() == b"payload-from-sandbox"
    assert len(raw.files.reads) == 1
    staging_path = raw.files.reads[0]
    assert staging_path != "/logs/verifier/reward.txt"
    assert "sb-1" in staging_path
    copy_command = raw.commands.calls[0]
    assert "sb-1-0" in copy_command
    assert "/logs/verifier/reward.txt" in copy_command
    assert staging_path in copy_command


async def test_download_from_a_misrouted_sandbox_never_reads(fake_sdk: None, tmp_path: Path) -> None:
    provider = _provider()
    handle, raw = _handle("sb-1", "sb-1-0")
    raw.commands.hostname = "victim-0"

    with pytest.raises(SandboxMisrouteError):
        await provider.download_file(handle, "/logs/verifier/reward.txt", tmp_path / "reward.txt")

    assert raw.files.reads == []
    assert not (tmp_path / "reward.txt").exists()


async def test_sandbox_not_found_is_a_typed_error_and_not_retried(fake_sdk: None) -> None:
    provider = _provider(retries=3)
    handle, raw = _handle("sb-1", "sb-1-0")

    class FakeApiError(Exception):
        status_code = 404

    raw.commands.exceptions.append(FakeApiError("sandbox not found: SANDBOX_NOT_FOUND"))

    with pytest.raises(SandboxEndedError):
        await provider.exec(handle, "true", timeout_s=5)

    assert len(raw.commands.calls) == 1


async def test_identity_check_can_be_disabled(fake_sdk: None, tmp_path: Path) -> None:
    provider = _provider(identity_check=False)
    handle, raw = _handle("sb-1", "sb-1-0")
    source = tmp_path / "test.sh"
    source.write_bytes(b"echo grader\n")

    await provider.exec(handle, "echo hi", timeout_s=5)
    await provider.upload_file(handle, source, "/tests/test.sh")

    assert raw.commands.calls == ["echo hi"]
    assert raw.files.writes == [("/tests/test.sh", b"echo grader\n")]


class FakeConnectionConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeConnectedSandbox:
    """Stands in for the SDK ``Sandbox`` returned by ``Sandbox.connect``."""

    hostname = "sb-1-0"

    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id
        self.commands = FakeCommands(type(self).hostname)
        self.files = FakeFiles()

    @classmethod
    async def connect(cls, sandbox_id: str, **_kwargs: Any) -> "FakeConnectedSandbox":
        return cls(sandbox_id)


async def test_connect_verifies_the_reattached_sandbox_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeConnectedSandbox, FakeConnectionConfig, FakeRunCommandOpts, object, object),
    )
    provider = _provider()

    FakeConnectedSandbox.hostname = "sb-1-0"
    handle = await provider.connect({"sandbox_id": "sb-1"})
    assert handle.sandbox_id == "sb-1"
    assert len(handle.raw.commands.calls) == 1, "connect must run exactly one identity probe"

    FakeConnectedSandbox.hostname = "victim-0"
    with pytest.raises(SandboxMisrouteError):
        await provider.connect({"sandbox_id": "sb-1"})
