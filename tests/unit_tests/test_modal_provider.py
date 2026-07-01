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

import asyncio
import traceback
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxResources, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.modal import provider as modal_provider
from nemo_gym.sandbox.providers.registry import create_provider, get_provider_class, list_providers


pytestmark = pytest.mark.sandbox


StreamChunk = str | bytes
StreamValue = StreamChunk | list[StreamChunk]


class FakeStream:
    def __init__(self, value: StreamValue) -> None:
        chunks = value if isinstance(value, list) else [value]
        self.chunks = [chunk.encode("utf-8") if isinstance(chunk, str) else chunk for chunk in chunks]

    async def read(self) -> bytes:
        raise AssertionError("Modal provider must stream output instead of calling read()")

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk


class HangingCloseStream:
    def __init__(
        self,
        chunk: bytes | None,
        *,
        read_started: asyncio.Event,
        close_started: asyncio.Event,
        release_close: asyncio.Event,
    ) -> None:
        self.chunk = chunk
        self.read_started = read_started
        self.close_started = close_started
        self.release_close = release_close
        self.emitted = False

    def __aiter__(self) -> "HangingCloseStream":
        return self

    async def __anext__(self) -> bytes:
        if self.emitted:
            raise StopAsyncIteration
        self.read_started.set()
        if self.chunk is None:
            await asyncio.Event().wait()
            raise StopAsyncIteration
        self.emitted = True
        return self.chunk

    async def aclose(self) -> None:
        self.close_started.set()
        await self.release_close.wait()


class FakeProcess:
    def __init__(
        self,
        stdout: StreamValue = "modal-sandbox-ready",
        stderr: StreamValue = "",
        return_code: int = 0,
    ) -> None:
        self.stdout = FakeStream(stdout)
        self.stderr = FakeStream(stderr)
        self.return_code = return_code

    async def wait(self) -> int:
        return self.return_code


class FakeFilesystem:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def make_directory(self, path: str, *, create_parents: bool) -> None:
        self.calls.append(("mkdir", path, create_parents))

    async def copy_from_local(self, source: Path, target: str) -> None:
        self.calls.append(("upload", source, target))

    async def copy_to_local(self, source: str, target: Path) -> None:
        self.calls.append(("download", source, target))
        target.write_text("downloaded", encoding="utf-8")


class FakeSandboxInstance:
    def __init__(self) -> None:
        self.object_id = "sb-test"
        self.filesystem = FakeFilesystem()
        self.exec_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.processes: list[FakeProcess] = []
        self.poll_result: int | None = None
        self.terminated = False

    async def exec(self, *args: str, **kwargs: Any) -> FakeProcess:
        self.exec_calls.append((args, kwargs))
        return self.processes.pop(0) if self.processes else FakeProcess()

    async def poll(self) -> int | None:
        return self.poll_result

    async def terminate(self, *, wait: bool) -> int | None:
        assert wait is True
        self.terminated = True
        return 0


class FakeApp:
    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    async def lookup(cls, name: str, **kwargs: Any) -> str:
        cls.calls.append((name, kwargs))
        return f"app:{name}"


class FakeImageInstance:
    def __init__(self, tag: str, registry_secret: Any) -> None:
        self.tag = tag
        self.registry_secret = registry_secret
        self.setup_calls: list[tuple[str, dict[str, Any]]] = []

    def run_commands(self, command: str, **kwargs: Any) -> "FakeImageInstance":
        self.setup_calls.append((command, kwargs))
        return self


class FakeImage:
    last: FakeImageInstance | None = None

    @classmethod
    def from_registry(cls, tag: str, *, secret: Any) -> FakeImageInstance:
        cls.last = FakeImageInstance(tag, secret)
        return cls.last


class FakeSecret:
    calls: list[tuple[str, dict[str, Any]]] = []
    dict_calls: list[dict[str, str]] = []

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> tuple[str, str]:
        cls.calls.append((name, kwargs))
        return ("secret", name)

    @classmethod
    def from_dict(cls, values: dict[str, str]) -> tuple[str, tuple[str, ...]]:
        cls.dict_calls.append(values)
        return ("ephemeral-secret", tuple(sorted(values)))


class FakeSandbox:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    instance = FakeSandboxInstance()

    @classmethod
    async def create(cls, *args: str, **kwargs: Any) -> FakeSandboxInstance:
        cls.calls.append((args, kwargs))
        return cls.instance


@pytest.fixture(autouse=True)
def fake_modal_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeApp.calls = []
    FakeImage.last = None
    FakeSecret.calls = []
    FakeSecret.dict_calls = []
    FakeSandbox.calls = []
    FakeSandbox.instance = FakeSandboxInstance()
    monkeypatch.setattr(
        modal_provider,
        "_require_modal_sdk",
        lambda: (FakeApp, FakeImage, FakeSandbox, FakeSecret),
    )


def test_configs_and_provider_options_validation() -> None:
    assert modal_provider.ModalConnectionConfig().app_name == "nemo-gym-sandbox"
    assert modal_provider.ModalCreateConfig().default_timeout_s == 21600
    assert modal_provider.ModalCreateConfig().cleanup_timeout_s == 30
    assert modal_provider.ModalCreateConfig().exec_stdout_limit_bytes == 16 * 1024 * 1024
    assert modal_provider.ModalCreateConfig().exec_stderr_limit_bytes == 16 * 1024 * 1024
    assert modal_provider.ModalCreateConfig().exec_combined_output_limit_bytes == 32 * 1024 * 1024
    assert modal_provider.ModalProbeConfig().command == "printf modal-sandbox-ready"
    with pytest.raises(ValueError, match="app_name"):
        modal_provider.ModalConnectionConfig(app_name="")
    with pytest.raises(ValueError, match="environment_name"):
        modal_provider.ModalConnectionConfig(environment_name="")
    with pytest.raises(TypeError, match="create_if_missing"):
        modal_provider.ModalConnectionConfig(create_if_missing=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default_timeout_s"):
        modal_provider.ModalCreateConfig(default_timeout_s=0)
    with pytest.raises(ValueError, match="cleanup_timeout_s"):
        modal_provider.ModalCreateConfig(cleanup_timeout_s=0)
    for field in (
        "exec_stdout_limit_bytes",
        "exec_stderr_limit_bytes",
        "exec_combined_output_limit_bytes",
    ):
        with pytest.raises(ValueError, match=field):
            modal_provider.ModalCreateConfig(**{field: 0})
        with pytest.raises(ValueError, match=field):
            modal_provider.ModalCreateConfig(**{field: True})
    with pytest.raises(ValueError, match="probe command"):
        modal_provider.ModalProbeConfig(command="")
    with pytest.raises(ValueError, match="probe.timeout_s"):
        modal_provider.ModalProbeConfig(timeout_s=float("inf"))
    with pytest.raises(TypeError, match="must be a mapping"):
        modal_provider.ModalProviderOptions.from_mapping([])  # type: ignore[arg-type]
    assert modal_provider.ModalProviderOptions.from_mapping(None) == modal_provider.ModalProviderOptions()
    with pytest.raises(ValueError, match="Unknown Modal provider option"):
        modal_provider.ModalProviderOptions.from_mapping({"unknown": True})
    with pytest.raises(ValueError, match="only one"):
        modal_provider.ModalProviderOptions.from_mapping(
            {"network_allowlist": ["example.com"], "outbound_domain_allowlist": ["other.example.com"]}
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        modal_provider.ModalProviderOptions.from_mapping({"block_network": True, "network_allowlist": ["example.com"]})
    with pytest.raises(ValueError, match="cannot be combined"):
        modal_provider.ModalProviderOptions.from_mapping(
            {"block_network": True, "inbound_cidr_allowlist": ["10.0.0.0/8"]}
        )
    with pytest.raises(TypeError, match="block_network"):
        modal_provider.ModalProviderOptions.from_mapping({"block_network": "true"})
    with pytest.raises(TypeError, match="secret_names"):
        modal_provider.ModalProviderOptions.from_mapping({"secret_names": "not-a-list"})  # pragma: allowlist secret
    with pytest.raises(TypeError, match="image_setup_steps"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": "not-a-list"})
    with pytest.raises(TypeError, match="contain only mappings"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": ["not-a-mapping"]})
    with pytest.raises(ValueError, match="Unknown Modal image setup"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": [{"run": "true", "extra": "bad"}]})
    with pytest.raises(ValueError, match="numeric uids"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": [{"run": "true", "user": 1000}]})
    with pytest.raises(ValueError, match="must not be empty"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": [{"run": []}]})
    with pytest.raises(TypeError, match="username or uid"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": [{"run": "true", "user": object()}]})
    with pytest.raises(ValueError, match="non-empty"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": [{"run": "true", "user": ""}]})
    with pytest.raises(ValueError, match="absolute executable"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": [{"run": "true", "shell": "sh"}]})
    with pytest.raises(TypeError, match="must map"):
        modal_provider.ModalProviderOptions.from_mapping({"image_setup_steps": [{"run": "true", "env": {"VALUE": 1}}]})
    with pytest.raises(TypeError, match="non-empty string"):
        modal_provider.ModalProviderOptions.from_mapping({"name": 1})
    with pytest.raises(TypeError, match="non-empty strings"):
        modal_provider.ModalProviderOptions.from_mapping({"secret_names": [""]})
    with pytest.raises(ValueError, match="cpu_limit"):
        modal_provider.ModalProviderOptions.from_mapping({"cpu_limit": 0})
    with pytest.raises(TypeError, match="memory_limit_mib"):
        modal_provider.ModalProviderOptions.from_mapping({"memory_limit_mib": 1.5})
    with pytest.raises(ValueError, match="at least one"):
        modal_provider._gpu_config(SandboxResources(gpu=0))

    config = modal_provider.ModalConnectionConfig(app_name="custom")
    assert modal_provider._coerce_config(config, modal_provider.ModalConnectionConfig) is config
    with pytest.raises(TypeError, match="ModalConnectionConfig"):
        modal_provider._coerce_config(1, modal_provider.ModalConnectionConfig)
    assert modal_provider._sandbox_id(type("LegacySandbox", (), {"id": "legacy-id"})()) == "legacy-id"
    with pytest.raises(modal_provider.ModalCreateError, match="without an object id"):
        modal_provider._sandbox_id(object())

    options = modal_provider.ModalProviderOptions.from_mapping(
        {
            "network_allowlist": ["api.example.com"],
            "region": ["us-east", "us-west"],
            "image_setup_steps": [
                {
                    "run": ["install one", "install two"],
                    "user": "agent",
                    "env": {"MODE": "build"},
                    "secret_names": ["build-secret"],
                }
            ],
        }
    )
    assert options.outbound_domain_allowlist == ("api.example.com",)
    assert options.region == ("us-east", "us-west")
    assert options.image_setup_steps[0].run == ("install one", "install two")


def test_modal_is_a_builtin_provider() -> None:
    assert "modal" in list_providers()
    assert get_provider_class("modal") is modal_provider.ModalProvider
    assert isinstance(create_provider({"modal": {"probe": {"command": None}}}), modal_provider.ModalProvider)


def test_preflight_requires_complete_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modal_provider, "_modal_authentication_configured", lambda: False)
    with pytest.raises(modal_provider.ModalCreateError, match="MODAL_TOKEN_ID"):
        modal_provider.ModalProvider.preflight()
    monkeypatch.setattr(modal_provider, "_modal_authentication_configured", lambda: True)
    modal_provider.ModalProvider.preflight()


def test_installed_modal_sdk_import_and_auth_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modal")
    monkeypatch.undo()
    app_cls, image_cls, sandbox_cls, secret_cls = modal_provider._require_modal_sdk()
    assert all(value is not None for value in (app_cls, image_cls, sandbox_cls, secret_cls))
    assert isinstance(modal_provider._modal_authentication_configured(), bool)


@pytest.mark.asyncio
async def test_create_maps_spec_and_builds_image() -> None:
    provider = modal_provider.ModalProvider(
        connection={"app_name": "gym-evals", "environment_name": "staging"},
        create={"default_timeout_s": 100, "ready_timeout_s": 50, "idle_timeout_s": 20},
    )
    spec = SandboxSpec(
        image="registry.example.com/task:sha",
        ttl_s=99.2,
        ready_timeout_s=12.1,
        workdir="/repo",
        env={"MODE": "eval"},
        metadata={"benchmark": "deep-swe"},
        resources=SandboxResources(cpu=3.5, memory_mib=8192, gpu=2, gpu_type="H100"),
        entrypoint=["sleep", "infinity"],
        provider_options={
            "name": "deep-swe-task",
            "registry_secret_name": "registry-creds",  # pragma: allowlist secret
            "secret_names": ["runtime-secret"],
            "network_allowlist": ["api.example.com"],
            "inbound_cidr_allowlist": ["10.0.0.0/8"],
            "cloud": "aws",
            "region": "us-east",
            "idle_timeout_s": 17.1,
            "cpu_limit": 3.5,
            "memory_limit_mib": 8192,
            "image_setup_steps": [
                {
                    "run": "npm install -g agent",
                    "user": "agent",
                    "env": {"NPM_CONFIG_LOGLEVEL": "error"},
                    "secret_names": ["npm-secret"],
                },
                {"run": "prepare root", "user": 0},
            ],
        },
    )

    handle = await provider.create(spec)

    assert handle.sandbox_id == "sb-test"
    assert handle.provider_name == "modal"
    assert FakeApp.calls == [("gym-evals", {"create_if_missing": True, "environment_name": "staging"})]
    assert FakeSecret.calls == [
        ("registry-creds", {"environment_name": "staging"}),
        ("npm-secret", {"environment_name": "staging"}),
        ("runtime-secret", {"environment_name": "staging"}),
    ]
    assert FakeSecret.dict_calls == [{"MODE": "eval"}]
    assert FakeImage.last is not None
    assert FakeImage.last.registry_secret == ("secret", "registry-creds")
    assert FakeImage.last.setup_calls == [
        (
            "su -s /bin/sh -c 'npm install -g agent' agent",
            {"env": {"NPM_CONFIG_LOGLEVEL": "error"}, "secrets": [("secret", "npm-secret")]},
        ),
        ("/bin/sh -c 'prepare root'", {"env": None, "secrets": []}),
    ]
    entrypoint, kwargs = FakeSandbox.calls[0]
    assert entrypoint == ("sleep", "infinity")
    assert kwargs == {
        "app": "app:gym-evals",
        "image": FakeImage.last,
        "secrets": [
            ("secret", "runtime-secret"),
            ("ephemeral-secret", ("MODE",)),
        ],
        "tags": {"benchmark": "deep-swe"},
        "timeout": 100,
        "workdir": "/repo",
        "block_network": False,
        "outbound_cidr_allowlist": None,
        "outbound_domain_allowlist": ("api.example.com",),
        "inbound_cidr_allowlist": ("10.0.0.0/8",),
        "name": "deep-swe-task",
        "idle_timeout": 18,
        "cpu": (3.5, 3.5),
        "memory": (8192, 8192),
        "gpu": "H100:2",
        "cloud": "aws",
        "region": "us-east",
    }
    assert FakeSandbox.instance.exec_calls[0] == (
        ("/bin/sh", "-c", "printf modal-sandbox-ready"),
        {"workdir": None, "secrets": [], "text": False, "timeout": 60},
    )


@pytest.mark.asyncio
async def test_exec_files_status_and_close(tmp_path: Path) -> None:
    provider = modal_provider.ModalProvider(probe={"command": None})
    raw = FakeSandboxInstance()
    raw.processes = [FakeProcess(["o", "ut"], ["e", "rr"], 3), FakeProcess("", "timeout", -1)]
    handle = SandboxHandle(sandbox_id="sb-test", provider_name="modal", raw=raw)

    result = await provider.exec(handle, "echo hello", cwd="/repo", env={"A": "B"}, user="agent")
    assert result == modal_provider.SandboxExecResult("out", "err", 3)
    assert raw.exec_calls[0] == (
        ("su", "-s", "/bin/sh", "-c", "echo hello", "agent"),
        {
            "workdir": "/repo",
            "secrets": [("ephemeral-secret", ("A",))],
            "text": False,
        },
    )
    assert FakeSecret.dict_calls == [{"A": "B"}]
    assert "env" not in raw.exec_calls[0][1]
    result = await provider.exec(handle, "slow", timeout_s=0.1, user=0)
    assert result.return_code == -1
    assert result.error_type == "timeout"
    assert raw.exec_calls[1][1]["timeout"] == 1
    with pytest.raises(ValueError, match="username"):
        await provider.exec(handle, "id", user=1000)

    source = tmp_path / "source.txt"
    source.write_text("upload", encoding="utf-8")
    await provider.upload_file(handle, source, "/nested/source.txt")
    target = tmp_path / "target" / "download.txt"
    await provider.download_file(handle, "/remote/download.txt", target)
    assert target.read_text(encoding="utf-8") == "downloaded"
    assert raw.filesystem.calls == [
        ("mkdir", "/nested", True),
        ("upload", source, "/nested/source.txt"),
        ("download", "/remote/download.txt", target),
    ]
    with pytest.raises(ValueError, match="absolute"):
        await provider.upload_file(handle, source, "relative.txt")
    with pytest.raises(FileNotFoundError):
        await provider.upload_file(handle, tmp_path / "missing.txt", "/missing.txt")

    assert await provider.status(handle) == SandboxStatus.RUNNING
    raw.poll_result = 0
    assert await provider.status(handle) == SandboxStatus.STOPPED
    await provider.close(handle)
    assert raw.terminated is True
    await provider.aclose()


@pytest.mark.asyncio
async def test_bounded_download_streams_atomically_and_stops_on_overflow(tmp_path: Path) -> None:
    provider = modal_provider.ModalProvider(probe={"command": None})
    raw = FakeSandboxInstance()
    handle = SandboxHandle(sandbox_id="sb-download", provider_name="modal", raw=raw)
    target = tmp_path / "nested" / "download.bin"
    target.parent.mkdir()
    target.write_bytes(b"original")

    raw.processes = [FakeProcess([b"abc", b"def"], b"", 0)]
    await provider.download_file(handle, "/remote/data.bin", target, max_bytes=6)

    assert target.read_bytes() == b"abcdef"
    assert raw.exec_calls == [
        (("/bin/sh", "-c", "exec cat -- /remote/data.bin"), {"text": False}),
    ]
    assert not list(target.parent.glob(".nemo-gym-download-*"))
    assert raw.terminated is False

    raw.processes = [FakeProcess([b"1234", b"5678"], b"", 0)]
    target.write_bytes(b"preserved")
    with pytest.raises(modal_provider.SandboxDownloadLimitExceeded, match="max_bytes=7"):
        await provider.download_file(handle, "/remote/too-large.bin", target, max_bytes=7)

    assert target.read_bytes() == b"preserved"
    assert not list(target.parent.glob(".nemo-gym-download-*"))
    assert raw.terminated is True

    with pytest.raises(ValueError, match="positive integer"):
        await provider.download_file(handle, "/remote/data.bin", target, max_bytes=0)


@pytest.mark.asyncio
async def test_bounded_download_preserves_target_when_remote_read_fails(tmp_path: Path) -> None:
    provider = modal_provider.ModalProvider(probe={"command": None})
    raw = FakeSandboxInstance()
    raw.processes = [FakeProcess(b"partial", b"cat: read error", 1)]
    handle = SandboxHandle(sandbox_id="sb-download-failed", provider_name="modal", raw=raw)
    target = tmp_path / "download.bin"
    target.write_bytes(b"preserved")

    with pytest.raises(RuntimeError, match="return_code=1"):
        await provider.download_file(handle, "/remote/data.bin", target, max_bytes=1024)

    assert target.read_bytes() == b"preserved"
    assert not list(tmp_path.glob(".nemo-gym-download-*"))
    assert raw.terminated is False


@pytest.mark.asyncio
async def test_cancelled_bounded_download_terminates_sandbox_and_preserves_target(tmp_path: Path) -> None:
    read_started = asyncio.Event()

    class BlockingStream:
        def __aiter__(self) -> AsyncIterator[bytes]:
            return self._iterate()

        async def _iterate(self) -> AsyncIterator[bytes]:
            read_started.set()
            await asyncio.Event().wait()
            yield b""

    class BlockingProcess:
        stdout = BlockingStream()
        stderr = FakeStream(b"")

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0

    class RunningSandbox(FakeSandboxInstance):
        async def exec(self, *args: str, **kwargs: Any) -> BlockingProcess:
            self.exec_calls.append((args, kwargs))
            return BlockingProcess()

    provider = modal_provider.ModalProvider(probe={"command": None})
    raw = RunningSandbox()
    handle = SandboxHandle(sandbox_id="sb-download-cancelled", provider_name="modal", raw=raw)
    target = tmp_path / "download.bin"
    target.write_bytes(b"preserved")
    download = asyncio.create_task(provider.download_file(handle, "/remote/data.bin", target, max_bytes=1024))
    await read_started.wait()

    download.cancel()
    with pytest.raises(asyncio.CancelledError):
        await download

    assert target.read_bytes() == b"preserved"
    assert not list(tmp_path.glob(".nemo-gym-download-*"))
    assert raw.terminated is True
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_exec_preserves_unicode_across_stream_chunks() -> None:
    provider = modal_provider.ModalProvider(
        create={
            "exec_stdout_limit_bytes": 32,
            "exec_stderr_limit_bytes": 32,
            "exec_combined_output_limit_bytes": 64,
        },
        probe={"command": None},
    )
    raw = FakeSandboxInstance()
    raw.processes = [
        FakeProcess(
            [b"\xce", b"\xb1\xce\xb2\xf0\x9f", b"\x99\x82"],
            [b"\xe9\x9b", b"\xaa\xe3", b"\x81\xa0"],
            0,
        )
    ]
    handle = SandboxHandle(sandbox_id="sb-unicode", provider_name="modal", raw=raw)

    result = await provider.exec(handle, "unicode")

    assert result == modal_provider.SandboxExecResult("αβ🙂", "雪だ", 0)
    assert raw.terminated is False


@pytest.mark.asyncio
async def test_exec_ignores_invalid_binary_output_without_exceeding_byte_limits() -> None:
    provider = modal_provider.ModalProvider(
        create={
            "exec_stdout_limit_bytes": 3,
            "exec_stderr_limit_bytes": 3,
            "exec_combined_output_limit_bytes": 6,
        },
        probe={"command": None},
    )
    raw = FakeSandboxInstance()
    raw.processes = [FakeProcess(b"a\xffb", b"\xfeok", 0)]
    handle = SandboxHandle(sandbox_id="sb-invalid", provider_name="modal", raw=raw)

    result = await provider.exec(handle, "binary-output")

    assert result == modal_provider.SandboxExecResult("ab", "ok", 0)
    assert len((result.stdout or "").encode("utf-8")) <= 3
    assert len((result.stderr or "").encode("utf-8")) <= 3
    assert raw.terminated is False


@pytest.mark.asyncio
async def test_exec_accepts_output_at_exact_raw_byte_boundaries() -> None:
    provider = modal_provider.ModalProvider(
        create={
            "exec_stdout_limit_bytes": 4,
            "exec_stderr_limit_bytes": 3,
            "exec_combined_output_limit_bytes": 7,
        },
        probe={"command": None},
    )
    raw = FakeSandboxInstance()
    raw.processes = [FakeProcess([b"\xf0\x9f", b"\x99\x82"], b"abc", 0)]
    handle = SandboxHandle(sandbox_id="sb-exact-limit", provider_name="modal", raw=raw)

    result = await provider.exec(handle, "exact-output")

    assert result == modal_provider.SandboxExecResult("🙂", "abc", 0)
    assert raw.terminated is False


@pytest.mark.asyncio
async def test_exec_bounds_an_oversized_single_binary_chunk() -> None:
    provider = modal_provider.ModalProvider(
        create={
            "exec_stdout_limit_bytes": 5,
            "exec_stderr_limit_bytes": 5,
            "exec_combined_output_limit_bytes": 10,
        },
        probe={"command": None},
    )
    raw = FakeSandboxInstance()
    raw.processes = [FakeProcess(b"x" * 1_000_000, b"", 0)]
    handle = SandboxHandle(sandbox_id="sb-single-chunk", provider_name="modal", raw=raw)

    result = await provider.exec(handle, "oversized-chunk")

    assert result == modal_provider.SandboxExecResult("xxxxx", "", -1, "output_limit_stdout")
    assert raw.terminated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "stderr", "expected_error", "expected_stdout", "expected_stderr"),
    [
        (["éé", "🙂"], "ok", "output_limit_stdout", "🙂", "ok"),
        ("ok", ["12", "3456"], "output_limit_stderr", "ok", "23456"),
    ],
)
async def test_exec_enforces_per_stream_byte_limits(
    stdout: StreamValue,
    stderr: StreamValue,
    expected_error: str,
    expected_stdout: str,
    expected_stderr: str,
) -> None:
    provider = modal_provider.ModalProvider(
        create={
            "exec_stdout_limit_bytes": 5,
            "exec_stderr_limit_bytes": 5,
            "exec_combined_output_limit_bytes": 16,
        },
        probe={"command": None},
    )
    raw = FakeSandboxInstance()
    raw.processes = [FakeProcess(stdout, stderr, 0)]
    handle = SandboxHandle(sandbox_id="sb-limit", provider_name="modal", raw=raw)

    result = await provider.exec(handle, "large-output")

    assert result.return_code == -1
    assert result.error_type == expected_error
    assert result.stdout == expected_stdout
    assert result.stderr == expected_stderr
    assert raw.terminated is True


@pytest.mark.asyncio
async def test_exec_enforces_combined_output_limit() -> None:
    provider = modal_provider.ModalProvider(
        create={
            "exec_stdout_limit_bytes": 16,
            "exec_stderr_limit_bytes": 16,
            "exec_combined_output_limit_bytes": 7,
        },
        probe={"command": None},
    )
    raw = FakeSandboxInstance()
    raw.processes = [FakeProcess(["abc", "def"], ["12", "34"], 0)]
    handle = SandboxHandle(sandbox_id="sb-combined", provider_name="modal", raw=raw)

    result = await provider.exec(handle, "combined-output")

    assert result.return_code == -1
    assert result.error_type == "output_limit_combined"
    assert len((result.stdout or "").encode()) + len((result.stderr or "").encode()) <= 7
    assert raw.terminated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_result", "expected_suffix"),
    [(False, "_cleanup_timeout"), (modal_provider.ModalCleanupError("failed"), "_cleanup_error")],
)
async def test_exec_output_limit_reports_cleanup_failure(
    cleanup_result: bool | Exception,
    expected_suffix: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = modal_provider.ModalProvider(
        create={
            "exec_stdout_limit_bytes": 3,
            "exec_stderr_limit_bytes": 3,
            "exec_combined_output_limit_bytes": 6,
        },
        probe={"command": None},
    )
    raw = FakeSandboxInstance()
    raw.processes = [FakeProcess("overflow", "", 0)]
    handle = SandboxHandle(sandbox_id="sb-cleanup", provider_name="modal", raw=raw)

    async def cleanup(
        _sandbox: Any,
        _tasks: tuple[asyncio.Task[Any], ...],
        _iterators: tuple[AsyncIterator[Any], ...],
    ) -> tuple[bool | BaseException, bool | BaseException]:
        return cleanup_result, True

    monkeypatch.setattr(provider, "_clean_up_exec", cleanup)
    result = await provider.exec(handle, "cleanup-failure")

    assert result.error_type == "output_limit_stdout" + expected_suffix
    assert result.return_code == -1
    assert len((result.stdout or "").encode()) <= 3
    assert len((result.stderr or "").encode()) <= 3


@pytest.mark.asyncio
async def test_exec_output_limit_terminates_before_hanging_iterator_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_started = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    real_seconds = modal_provider._seconds

    def short_cleanup_timeout(value: int | float, *, field_name: str) -> int | float:
        if field_name == "cleanup_timeout_s":
            return 0.01
        return real_seconds(value, field_name=field_name)

    monkeypatch.setattr(modal_provider, "_seconds", short_cleanup_timeout)
    provider = modal_provider.ModalProvider(
        create={
            "cleanup_timeout_s": 1,
            "exec_stdout_limit_bytes": 3,
            "exec_stderr_limit_bytes": 3,
            "exec_combined_output_limit_bytes": 6,
        },
        probe={"command": None},
    )
    process = FakeProcess()
    process.stdout = HangingCloseStream(
        b"overflow",
        read_started=read_started,
        close_started=close_started,
        release_close=release_close,
    )
    raw = FakeSandboxInstance()
    raw.processes = [process]
    handle = SandboxHandle(sandbox_id="sb-hanging-close", provider_name="modal", raw=raw)

    result = await asyncio.wait_for(provider.exec(handle, "overflow"), timeout=0.5)

    assert read_started.is_set()
    assert close_started.is_set()
    assert raw.terminated is True
    assert result.error_type == "output_limit_stdout_cleanup_timeout"
    assert provider._late_create_cleanup_tasks

    release_close.set()
    await asyncio.wait_for(provider.aclose(), timeout=0.5)
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_cancelled_exec_propagates_before_hanging_iterator_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_started = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    real_seconds = modal_provider._seconds

    def short_cleanup_timeout(value: int | float, *, field_name: str) -> int | float:
        if field_name == "cleanup_timeout_s":
            return 0.01
        return real_seconds(value, field_name=field_name)

    monkeypatch.setattr(modal_provider, "_seconds", short_cleanup_timeout)
    provider = modal_provider.ModalProvider(create={"cleanup_timeout_s": 1}, probe={"command": None})
    process = FakeProcess()
    process.stdout = HangingCloseStream(
        None,
        read_started=read_started,
        close_started=close_started,
        release_close=release_close,
    )
    raw = FakeSandboxInstance()
    raw.processes = [process]
    handle = SandboxHandle(sandbox_id="sb-cancel-hanging-close", provider_name="modal", raw=raw)
    exec_call = asyncio.create_task(provider.exec(handle, "long-running"))
    await read_started.wait()

    exec_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(exec_call, timeout=0.5)

    assert close_started.is_set()
    assert raw.terminated is True
    assert provider._late_create_cleanup_tasks

    release_close.set()
    await asyncio.wait_for(provider.aclose(), timeout=0.5)
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_create_and_lifecycle_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = modal_provider.ModalProvider(probe={"command": None})
    with pytest.raises(modal_provider.ModalCreateError, match="spec.image"):
        await provider.create(SandboxSpec())
    with pytest.raises(modal_provider.ModalCreateError, match="disk"):
        await provider.create(SandboxSpec(image="image:tag", resources=SandboxResources(disk_gib=10)))
    with pytest.raises(modal_provider.ModalCreateError, match="creation failed"):
        await provider.create(
            SandboxSpec(
                image="image:tag",
                resources=SandboxResources(cpu=4),
                provider_options={"cpu_limit": 2},
            )
        )

    handle = await provider.create(
        SandboxSpec(
            image="image:tag",
            resources=SandboxResources(cpu=1, memory_mib=256),
        )
    )
    assert handle.sandbox_id == "sb-test"
    assert FakeSandbox.calls[-1][1]["cpu"] == 1
    assert FakeSandbox.calls[-1][1]["memory"] == 256
    with pytest.raises(modal_provider.ModalCreateError, match="creation failed"):
        await provider.create(
            SandboxSpec(
                image="image:tag",
                resources=SandboxResources(memory_mib=256),
                provider_options={"memory_limit_mib": 128},
            )
        )

    async def failed_create() -> Any:
        raise RuntimeError("late create failed")

    failed_task = asyncio.create_task(failed_create())
    await provider._terminate_late_create(failed_task)

    create_finished = asyncio.Event()

    class SlowSandbox(FakeSandbox):
        @classmethod
        async def create(cls, *args: str, **kwargs: Any) -> FakeSandboxInstance:
            del args, kwargs
            await asyncio.sleep(1.05)
            create_finished.set()
            return cls.instance

    monkeypatch.setattr(
        modal_provider,
        "_require_modal_sdk",
        lambda: (FakeApp, FakeImage, SlowSandbox, FakeSecret),
    )
    provider = modal_provider.ModalProvider(create={"ready_timeout_s": 1, "cleanup_timeout_s": 2})
    with pytest.raises(modal_provider.ModalCreateTimeoutError, match="timed out"):
        await provider.create(SandboxSpec(image="image:tag"))
    await asyncio.wait_for(create_finished.wait(), timeout=1)
    assert SlowSandbox.instance.terminated is True


@pytest.mark.asyncio
async def test_probe_failure_cleans_up_and_status_handles_gone() -> None:
    FakeSandbox.instance.processes = [FakeProcess("wrong", "failed", 1)]
    provider = modal_provider.ModalProvider()
    with pytest.raises(modal_provider.ModalCreateVerificationError, match="readiness probe"):
        await provider.create(SandboxSpec(image="image:tag"))
    assert FakeSandbox.instance.terminated is True

    class GoneSandbox(FakeSandboxInstance):
        async def poll(self) -> int | None:
            raise type("SandboxTerminatedError", (RuntimeError,), {})()

        async def terminate(self, *, wait: bool) -> int | None:
            del wait
            raise type("NotFoundError", (RuntimeError,), {})()

    handle = SandboxHandle(sandbox_id="gone", provider_name="modal", raw=GoneSandbox())
    assert await provider.status(handle) == SandboxStatus.STOPPED
    await provider.close(handle)

    class BrokenSandbox(GoneSandbox):
        async def poll(self) -> int | None:
            raise RuntimeError("broken")

        async def terminate(self, *, wait: bool) -> int | None:
            del wait
            raise RuntimeError("broken")

    handle = SandboxHandle(sandbox_id="broken", provider_name="modal", raw=BrokenSandbox())
    assert await provider.status(handle) == SandboxStatus.UNKNOWN
    with pytest.raises(modal_provider.ModalCleanupError, match="termination failed"):
        await provider.close(handle)


@pytest.mark.asyncio
async def test_create_failure_redacts_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingApp:
        @classmethod
        async def lookup(cls, name: str, **kwargs: Any) -> Any:
            del cls, name, kwargs
            raise RuntimeError("credential=do-not-report")

    monkeypatch.setattr(
        modal_provider,
        "_require_modal_sdk",
        lambda: (FailingApp, FakeImage, FakeSandbox, FakeSecret),
    )
    with pytest.raises(modal_provider.ModalCreateError) as exc_info:
        await modal_provider.ModalProvider().create(SandboxSpec(image="image:tag"))
    assert "do-not-report" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert "do-not-report" not in "".join(
        traceback.format_exception(type(exc_info.value), exc_info.value, exc_info.value.__traceback__)
    )


@pytest.mark.asyncio
async def test_cancelled_probe_terminates_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = modal_provider.ModalProvider()

    async def cancel_probe(_handle: SandboxHandle) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(provider, "_verify_created_handle", cancel_probe)
    with pytest.raises(asyncio.CancelledError):
        await provider.create(SandboxSpec(image="image:tag"))
    assert FakeSandbox.instance.terminated is True


@pytest.mark.asyncio
async def test_cancelled_create_terminates_late_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    create_started = asyncio.Event()
    release_create = asyncio.Event()

    class DelayedSandbox(FakeSandbox):
        @classmethod
        async def create(cls, *args: str, **kwargs: Any) -> FakeSandboxInstance:
            del args, kwargs
            create_started.set()
            await release_create.wait()
            return cls.instance

    monkeypatch.setattr(
        modal_provider,
        "_require_modal_sdk",
        lambda: (FakeApp, FakeImage, DelayedSandbox, FakeSecret),
    )
    provider = modal_provider.ModalProvider(
        create={"ready_timeout_s": 10, "cleanup_timeout_s": 1},
        probe={"command": None},
    )
    create_call = asyncio.create_task(provider.create(SandboxSpec(image="image:tag")))
    await create_started.wait()
    create_call.cancel()
    release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create_call
    assert DelayedSandbox.instance.terminated is True
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_cancellation_during_late_create_cleanup_is_preserved() -> None:
    release_create = asyncio.Event()

    async def delayed_create() -> FakeSandboxInstance:
        await release_create.wait()
        return FakeSandbox.instance

    provider = modal_provider.ModalProvider(probe={"command": None})
    create_task = asyncio.create_task(delayed_create())
    cleanup_call = asyncio.create_task(provider._clean_up_late_create(create_task))
    await asyncio.sleep(0)
    cleanup_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup_call

    release_create.set()
    await provider.aclose()
    assert FakeSandbox.instance.terminated is True
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_cancelled_probe_has_bounded_stalled_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_started = asyncio.Event()
    release_termination = asyncio.Event()

    class StalledTerminationSandbox(FakeSandboxInstance):
        async def terminate(self, *, wait: bool) -> int | None:
            assert wait is True
            await release_termination.wait()
            self.terminated = True
            return 0

    async def stalled_probe(_handle: SandboxHandle) -> None:
        probe_started.set()
        await asyncio.Event().wait()

    real_seconds = modal_provider._seconds

    def short_cleanup_timeout(value: int | float, *, field_name: str) -> int | float:
        if field_name == "cleanup_timeout_s":
            return 0.01
        return real_seconds(value, field_name=field_name)

    FakeSandbox.instance = StalledTerminationSandbox()
    monkeypatch.setattr(modal_provider, "_seconds", short_cleanup_timeout)
    provider = modal_provider.ModalProvider()
    monkeypatch.setattr(provider, "_verify_created_handle", stalled_probe)
    create_call = asyncio.create_task(provider.create(SandboxSpec(image="image:tag")))
    await probe_started.wait()
    create_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(create_call, timeout=0.5)
    assert provider._late_create_cleanup_tasks

    handle = SandboxHandle(sandbox_id="stalled", provider_name="modal", raw=StalledTerminationSandbox())
    with pytest.raises(TimeoutError, match="termination timed out"):
        await provider.close(handle)

    release_termination.set()
    await provider.aclose()
    assert FakeSandbox.instance.terminated is True
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_cancelled_close_keeps_termination_tracked() -> None:
    termination_started = asyncio.Event()
    release_termination = asyncio.Event()

    class DelayedTerminationSandbox(FakeSandboxInstance):
        async def terminate(self, *, wait: bool) -> int | None:
            assert wait is True
            termination_started.set()
            await release_termination.wait()
            self.terminated = True
            return 0

    raw = DelayedTerminationSandbox()
    handle = SandboxHandle(sandbox_id="delayed", provider_name="modal", raw=raw)
    provider = modal_provider.ModalProvider(probe={"command": None})
    close_call = asyncio.create_task(provider.close(handle))
    await termination_started.wait()
    close_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_call
    assert provider._late_create_cleanup_tasks

    release_termination.set()
    await provider.aclose()
    assert raw.terminated is True
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_cancelled_exec_terminates_enclosing_sandbox() -> None:
    read_started = asyncio.Event()

    class BlockingStream:
        def __aiter__(self) -> AsyncIterator[bytes]:
            return self._iterate()

        async def _iterate(self) -> AsyncIterator[bytes]:
            read_started.set()
            await asyncio.Event().wait()
            yield b""

    class BlockingProcess:
        stdout = BlockingStream()
        stderr = BlockingStream()

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0

    class RunningSandbox(FakeSandboxInstance):
        async def exec(self, *args: str, **kwargs: Any) -> BlockingProcess:
            self.exec_calls.append((args, kwargs))
            return BlockingProcess()

    raw = RunningSandbox()
    handle = SandboxHandle(sandbox_id="running", provider_name="modal", raw=raw)
    provider = modal_provider.ModalProvider(probe={"command": None})
    exec_call = asyncio.create_task(provider.exec(handle, "long-running"))
    await read_started.wait()
    exec_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await exec_call
    assert raw.terminated is True
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_cancelled_exec_start_terminates_enclosing_sandbox() -> None:
    exec_started = asyncio.Event()

    class StartingSandbox(FakeSandboxInstance):
        async def exec(self, *args: str, **kwargs: Any) -> FakeProcess:
            self.exec_calls.append((args, kwargs))
            exec_started.set()
            await asyncio.Event().wait()
            return FakeProcess()

    raw = StartingSandbox()
    handle = SandboxHandle(sandbox_id="starting", provider_name="modal", raw=raw)
    provider = modal_provider.ModalProvider(probe={"command": None})
    exec_call = asyncio.create_task(provider.exec(handle, "starting"))
    await exec_started.wait()
    exec_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await exec_call
    assert raw.terminated is True
    assert not provider._late_create_cleanup_tasks


@pytest.mark.asyncio
async def test_termination_retries_and_reports_sandbox_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSandbox(FakeSandboxInstance):
        attempts = 0

        async def terminate(self, *, wait: bool) -> int | None:
            assert wait is True
            self.attempts += 1
            raise RuntimeError("provider detail must not be retained")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(modal_provider.asyncio, "sleep", no_sleep)
    raw = FailingSandbox()
    handle = SandboxHandle(sandbox_id="sb-test", provider_name="modal", raw=raw)
    provider = modal_provider.ModalProvider(probe={"command": None})
    with pytest.raises(modal_provider.ModalCleanupError) as exc_info:
        await provider.close(handle)
    assert raw.attempts == 3
    assert "sb-test" in str(exc_info.value)
    assert "provider detail" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_delayed_cleanup_failure_is_reported_by_aclose(monkeypatch: pytest.MonkeyPatch) -> None:
    release_termination = asyncio.Event()

    class DelayedFailureSandbox(FakeSandboxInstance):
        async def terminate(self, *, wait: bool) -> int | None:
            assert wait is True
            await release_termination.wait()
            raise RuntimeError("provider detail must not be retained")

    real_seconds = modal_provider._seconds

    def short_cleanup_timeout(value: int | float, *, field_name: str) -> int | float:
        if field_name == "cleanup_timeout_s":
            return 0.01
        return real_seconds(value, field_name=field_name)

    monkeypatch.setattr(modal_provider, "_seconds", short_cleanup_timeout)
    raw = DelayedFailureSandbox()
    provider = modal_provider.ModalProvider(probe={"command": None})
    assert await provider._clean_up_sandbox(raw) is False
    release_termination.set()
    await asyncio.sleep(0.8)
    assert not provider._late_create_cleanup_tasks
    with pytest.raises(modal_provider.ModalCleanupError) as exc_info:
        await provider.aclose()
    assert "sb-test" in str(exc_info.value)
    assert "provider detail" not in str(exc_info.value)
