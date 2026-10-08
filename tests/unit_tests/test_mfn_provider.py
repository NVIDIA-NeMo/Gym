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

from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxPtySpec, SandboxResources, SandboxSpec
from nemo_gym.sandbox.providers.mfn import MFNProvider
from nemo_gym.sandbox.providers.mfn.protos import mfn_sandbox_pb2 as pb
from nemo_gym.sandbox.providers.registry import get_provider_class


class _Stream:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._values:
            raise StopAsyncIteration
        return self._values.pop(0)


class _FakeStub:
    def __init__(self) -> None:
        self.create_request = None
        self.exec_request = None
        self.upload = bytearray()
        self.upload_metadata = None
        self.shutdown_ids: list[str] = []

    async def Create(self, request, *, timeout):
        self.create_request = request
        assert timeout == 60
        return pb.CreateResult(sandbox_id="mfn-1", is_ready=True)

    async def Get(self, request, *, timeout):
        del timeout
        return pb.GetResponse(sandbox_id=request.sandbox_id, status=pb.Status(is_ready=True, phase="Running"))

    async def Shutdown(self, request, *, timeout):
        del timeout
        self.shutdown_ids.append(request.sandbox_id)
        return pb.ShutdownResult()

    def ExecStream(self, requests, *, timeout):
        del timeout

        async def responses():
            first = await anext(requests)
            self.exec_request = first.request
            with pytest.raises(StopAsyncIteration):
                await anext(requests)
            yield pb.ExecStreamResponse(output=pb.ExecOutput(stream=pb.ExecOutput.STDOUT, data=b"hello"))
            yield pb.ExecStreamResponse(output=pb.ExecOutput(stream=pb.ExecOutput.STDERR, data=b"warning"))
            yield pb.ExecStreamResponse(complete=pb.ExecComplete(exit_code=7))

        return responses()

    async def AddFile(self, requests, *, metadata, timeout):
        del timeout
        self.upload_metadata = metadata
        async for request in requests:
            self.upload.extend(request.content)
        return pb.AddFileResult()

    def ReadFile(self, request, *, timeout):
        del request, timeout
        return _Stream([pb.ReadFileResponse(content=b"ab"), pb.ReadFileResponse(content=b"cd")])

    async def GetHost(self, request, *, timeout):
        del request, timeout
        return pb.GetHostResponse(uri="mfn.example:8080")


@pytest.fixture
async def provider():
    value = MFNProvider(probe={"command": None}, create={"poll_initial_delay_s": 0})
    value._stub = _FakeStub()
    try:
        yield value
    finally:
        await value.aclose()


def test_mfn_is_registered():
    assert get_provider_class("mfn") is MFNProvider


async def test_create_supplies_mfn_required_resource_defaults(provider):
    await provider.create(SandboxSpec(image="image:tag"))
    resource = provider._stub.create_request.specs.resource_request
    assert resource.cpu_request == "1"
    assert resource.memory_limit == "1024Mi"


async def test_create_maps_gym_spec(provider):
    spec = SandboxSpec(
        image="docker://registry.example/image:tag",
        ttl_s=12.5,
        env={"A": "b"},
        metadata={"workload": "test"},
        ports=[8080],
        resources=SandboxResources(cpu=1.5, memory_mib=256, disk_gib=2, gpu=1, gpu_type="H100"),
    )

    handle = await provider.create(spec)
    request = provider._stub.create_request

    assert handle.sandbox_id == "mfn-1"
    assert request.idempotency_key
    assert request.specs.image == "registry.example/image:tag"
    assert request.specs.sandbox_ttl.seconds == 12
    assert request.specs.sandbox_ttl.nanos == 500_000_000
    assert request.specs.resource_request.cpu_request == "1.5"
    assert request.specs.resource_request.memory_limit == "256Mi"
    assert request.specs.resource_request.storage_limit == "2Gi"
    assert list(request.specs.resource_request.gpu.type_preferences) == ["H100"]
    assert request.attributes.attributes["workload"] == "test"
    assert request.ports[0].name == "tcp-8080"


async def test_create_allow_all_omits_network_policy(provider):
    await provider.create(SandboxSpec(image="image:tag", provider_options={"network_mode": "allow_all"}))
    assert not provider._stub.create_request.HasField("network_config")


async def test_exec_and_file_transfer(provider, tmp_path: Path):
    handle = SandboxHandle("mfn-1", "mfn", object())
    result = await provider.exec(handle, "exit 7", cwd="/work", env={"X": "1"})

    assert result.stdout == "hello"
    assert result.stderr == "warning"
    assert result.return_code == 7
    assert list(provider._stub.exec_request.command) == ["/bin/sh", "-c", "exit 7"]
    assert provider._stub.exec_request.cwd == "/work"

    source = tmp_path / "source.bin"
    source.write_bytes(b"\x00payload")
    await provider.upload_file(handle, source, "/tmp/target.bin")
    assert provider._stub.upload == b"\x00payload"
    assert ("sandbox-id", "mfn-1") in provider._stub.upload_metadata

    target = tmp_path / "nested" / "target.bin"
    await provider.download_file(handle, "/tmp/target.bin", target)
    assert target.read_bytes() == b"abcd"


async def test_status_endpoint_and_close(provider):
    handle = SandboxHandle("mfn-1", "mfn", object())

    assert (await provider.status(handle)).value == "running"
    assert (await provider.endpoint(handle, 8080)).endpoint == "http://mfn.example:8080"
    await provider.close(handle)
    assert provider._stub.shutdown_ids == ["mfn-1"]


async def test_pty_round_trip(provider):
    class PtyStub(_FakeStub):
        def ExecStream(self, requests):
            async def responses():
                first = await anext(requests)
                assert first.request.pty.rows == 24
                yield pb.ExecStreamResponse(output=pb.ExecOutput(data=b"prompt> "))
                second = await anext(requests)
                assert (second.resize.rows, second.resize.cols) == (40, 100)
                third = await anext(requests)
                yield pb.ExecStreamResponse(output=pb.ExecOutput(data=third.stdin.data))
                yield pb.ExecStreamResponse(complete=pb.ExecComplete(exit_code=0))

            return responses()

    provider._stub = PtyStub()
    handle = SandboxHandle("mfn-1", "mfn", object())
    session = await provider.create_pty(handle, SandboxPtySpec(rows=24, cols=80))

    assert await session.read() == b"prompt> "
    await session.resize(40, 100)
    await session.resize(40, 100)  # Repeating the current size sends no frame.
    await session.write(b"echo hi\n")
    assert await session.read() == b"echo hi\n"
    assert await session.wait_exit() == 0
    await session.close()


# --- configuration validation -------------------------------------------------------------------------------------

import asyncio  # noqa: E402

import grpc  # noqa: E402

from nemo_gym.sandbox.providers.base import SandboxPtyError, SandboxStatus  # noqa: E402
from nemo_gym.sandbox.providers.mfn import _provider as mfn  # noqa: E402
from nemo_gym.sandbox.providers.mfn.pty import MFNPtySession  # noqa: E402


@pytest.mark.parametrize(
    "factory",
    [
        lambda: mfn.MFNConnectionConfig(address=" "),
        lambda: mfn.MFNConnectionConfig(caller=""),
        lambda: mfn.MFNConnectionConfig(username=" "),
        lambda: mfn.MFNConnectionConfig(keepalive_time_s=0),
        lambda: mfn.MFNConnectionConfig(keepalive_timeout_s=-1),
        lambda: mfn.MFNConnectionConfig(max_receive_message_mib=0),
        lambda: mfn.MFNCreateConfig(request_timeout_s=0),
        lambda: mfn.MFNCreateConfig(poll_initial_delay_s=-1),
        lambda: mfn.MFNCreateConfig(retries=-1),
        lambda: mfn.MFNCreateConfig(retry_delay_s=-1),
        lambda: mfn.MFNResourceDefaults(cpu=0),
        lambda: mfn.MFNResourceDefaults(memory_mib=0),
        lambda: mfn.MFNResourceDefaults(disk_gib=0),
        lambda: mfn.MFNOperationConfig(default_exec_timeout_s=0),
        lambda: mfn.MFNOperationConfig(file_timeout_s=0),
        lambda: mfn.MFNOperationConfig(exec_shell=""),
        lambda: mfn.MFNOperationConfig(upload_chunk_bytes=0),
        lambda: mfn.MFNProbeConfig(command="true", timeout_s=0),
    ],
)
def test_invalid_config_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "options",
    [
        {"unknown": 1},
        {"network_mode": "bogus"},
        {"allowed_cidrs": ["10.0.0.0/8"]},
        {"network_mode": "allow_all", "blocked_domains": ["example.com"]},
        {"network_mode": "allow", "blocked_cidrs": ["10.0.0.0/8"]},
        {"network_mode": "block", "allowed_domains": ["example.com"]},
        {"network_mode": "block"},
    ],
)
def test_invalid_provider_options_are_rejected(options):
    with pytest.raises(ValueError):
        mfn.MFNProviderOptions.from_mapping(options)


def test_provider_options_default_to_empty():
    options = mfn.MFNProviderOptions.from_mapping(None)
    assert options.network_mode is None and options.allowed_cidrs == ()


@pytest.mark.parametrize(
    "spec",
    [
        SandboxSpec(image="a", entrypoint=["x"]),
        SandboxSpec(),
        SandboxSpec(image="a", provider_options={"snapshot_id": "s"}),
        SandboxSpec(image="a", resources=SandboxResources(gpu_type="H100")),
    ],
)
async def test_create_rejects_invalid_specs(provider, spec):
    with pytest.raises(ValueError):
        await provider.create(spec)


async def test_create_maps_snapshot_shard_and_network_rules(provider):
    options = {
        "snapshot_id": "snap-1",
        "shard_pin": "shard-a",
        "network_mode": "block",
        "blocked_cidrs": ["10.0.0.0/8"],
        "blocked_domains": ["example.com"],
        "network_name": "policy",
    }
    await provider.create(SandboxSpec(provider_options=options))
    request = provider._stub.create_request
    assert request.specs.snapshot_id == "snap-1"
    assert request.shard_pin == "shard-a"
    assert request.network_config.mode == pb.NetworkConfig.BLOCK
    assert list(request.network_config.blocked_cidrs) == ["10.0.0.0/8"]
    assert request.network_config.name == "policy"

    await provider.create(
        SandboxSpec(image="a", provider_options={"network_mode": "allow", "allowed_domains": ["example.com"]})
    )
    assert provider._stub.create_request.network_config.mode == pb.NetworkConfig.ALLOW


def test_quantity_and_error_helpers():
    assert mfn._quantity(2.0) == "2"
    assert mfn._quantity(0.5) == "0.5"
    assert mfn._rpc_code(RuntimeError("x")) is None
    assert mfn._rpc_detail(RuntimeError("boom")) == "boom"
    assert mfn._rpc_detail(_rpc_error(grpc.StatusCode.UNAVAILABLE, "gone")) == "gone"


def _rpc_error(code: grpc.StatusCode, details: str = "failure") -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(code, grpc.aio.Metadata(), grpc.aio.Metadata(), details=details)


# --- create: readiness, retries and cleanup -----------------------------------------------------------------------


class _CreateStub(_FakeStub):
    """Scripted Create/Get responses; each entry is a value to return or an exception to raise."""

    def __init__(self, creates: list[Any], gets: list[Any] | None = None) -> None:
        super().__init__()
        self.creates = creates
        self.gets = gets or []
        self.create_calls = 0

    async def Create(self, request, *, timeout):
        del timeout
        self.create_calls += 1
        value = self.creates.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def Get(self, request, *, timeout):
        del timeout
        value = self.gets.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _status(ready: bool, phase: str) -> Any:
    return pb.GetResponse(status=pb.Status(is_ready=ready, phase=phase))


def _fast_provider(stub: Any, **kwargs: Any) -> MFNProvider:
    value = MFNProvider(
        create={"poll_initial_delay_s": 0, "poll_interval_s": 0.001, "poll_max_interval_s": 0.001, "retry_delay_s": 0},
        probe={"command": None},
        **kwargs,
    )
    value._stub = stub
    return value


async def test_create_polls_until_ready_through_transient_errors():
    stub = _CreateStub(
        [pb.CreateResult(sandbox_id="s1", is_ready=False)],
        [_rpc_error(grpc.StatusCode.UNAVAILABLE), _status(False, "Pending"), _status(True, "Running")],
    )
    provider = _fast_provider(stub)
    try:
        assert (await provider.create(SandboxSpec(image="i"))).sandbox_id == "s1"
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    "gets, message",
    [
        ([_status(False, "Failed")], "Failed phase"),
        ([_rpc_error(grpc.StatusCode.NOT_FOUND, "pod deleted")], "readiness check failed"),
    ],
)
async def test_create_fails_and_cleans_up_when_not_ready(gets, message):
    stub = _CreateStub([pb.CreateResult(sandbox_id="s1", is_ready=False)], gets)
    provider = _fast_provider(stub)
    try:
        with pytest.raises(mfn.MFNCreateError, match=message):
            await provider.create(SandboxSpec(image="i"))
        assert stub.shutdown_ids == ["s1"]
    finally:
        await provider.aclose()


async def test_create_times_out_waiting_for_ready():
    stub = _CreateStub([pb.CreateResult(sandbox_id="s1", is_ready=False)], [_status(False, "Pending")] * 1000)
    provider = _fast_provider(stub)
    try:
        with pytest.raises(mfn.MFNCreateError, match="not ready within"):
            await provider.create(SandboxSpec(image="i", ready_timeout_s=0.05))
    finally:
        await provider.aclose()


async def test_create_retries_transient_errors_then_gives_up():
    ok = pb.CreateResult(sandbox_id="s1", is_ready=True)
    stub = _CreateStub([_rpc_error(grpc.StatusCode.UNAVAILABLE), ok])
    provider = _fast_provider(stub)
    try:
        assert (await provider.create(SandboxSpec(image="i"))).sandbox_id == "s1"
        assert stub.create_calls == 2

        stub.creates = [_rpc_error(grpc.StatusCode.UNAVAILABLE)] * 3
        with pytest.raises(mfn.MFNCreateError, match="Create failed"):
            await provider.create(SandboxSpec(image="i"))

        stub.creates = [_rpc_error(grpc.StatusCode.INVALID_ARGUMENT, "bad image")]
        with pytest.raises(mfn.MFNCreateError, match="bad image"):
            await provider.create(SandboxSpec(image="i"))
    finally:
        await provider.aclose()


async def test_create_probe_success_and_failure():
    class Probe(_FakeStub):
        def __init__(self, exit_code: int) -> None:
            super().__init__()
            self.exit_code = exit_code

        def ExecStream(self, requests, *, timeout):
            del timeout

            async def responses():
                self.exec_request = (await anext(requests)).request
                yield pb.ExecStreamResponse(output=pb.ExecOutput(stream=pb.ExecOutput.STDOUT, data=b"ready"))
                yield pb.ExecStreamResponse(complete=pb.ExecComplete(exit_code=self.exit_code))

            return responses()

    for exit_code, expected_stdout, ok in ((0, "ready", True), (0, "nope", False), (1, "ready", False)):
        provider = MFNProvider(
            create={"poll_initial_delay_s": 0}, probe={"command": "probe", "expected_stdout": expected_stdout}
        )
        provider._stub = Probe(exit_code)
        try:
            if ok:
                assert (await provider.create(SandboxSpec(image="i"))).sandbox_id == "mfn-1"
                assert provider._stub.exec_request.command[-1] == "probe"
            else:
                with pytest.raises(mfn.MFNCreateVerificationError):
                    await provider.create(SandboxSpec(image="i"))
                assert provider._stub.shutdown_ids == ["mfn-1"]
        finally:
            await provider.aclose()


async def test_create_cleanup_failure_does_not_mask_original_error():
    class BrokenShutdown(_CreateStub):
        async def Shutdown(self, request, *, timeout):
            raise _rpc_error(grpc.StatusCode.UNAVAILABLE, "no shutdown")

    stub = BrokenShutdown([pb.CreateResult(sandbox_id="s1", is_ready=False)], [_status(False, "Failed")])
    provider = _fast_provider(stub)
    try:
        with pytest.raises(mfn.MFNCreateError, match="Failed phase"):
            await provider.create(SandboxSpec(image="i"))
    finally:
        await provider.aclose()


# --- exec, status and lifecycle -----------------------------------------------------------------------------------


async def test_exec_maps_rpc_errors_and_non_root_user(provider):
    handle = SandboxHandle("mfn-1", "mfn", object())

    class Failing(_FakeStub):
        def __init__(self, code):
            super().__init__()
            self.code = code

        def ExecStream(self, requests, *, timeout):
            del requests, timeout

            async def responses():
                raise _rpc_error(self.code, "exec failed")
                yield  # pragma: no cover

            return responses()

    provider._stub = Failing(grpc.StatusCode.DEADLINE_EXCEEDED)
    result = await provider.exec(handle, "sleep 9", timeout_s=1)
    assert (result.return_code, result.error_type) == (125, "timeout")

    provider._stub = Failing(grpc.StatusCode.UNAVAILABLE)
    assert (await provider.exec(handle, "true")).error_type == "sandbox"

    provider._stub = _FakeStub()
    await provider.exec(handle, "id", user="alice")
    assert list(provider._stub.exec_request.command)[-1].startswith("su -s /bin/sh -c ")
    assert provider._stub.exec_request.command[-1].endswith("alice")


async def test_exec_without_complete_and_with_termination_detail(provider):
    handle = SandboxHandle("mfn-1", "mfn", object())

    class Scripted(_FakeStub):
        def __init__(self, frames):
            super().__init__()
            self.frames = frames

        def ExecStream(self, requests, *, timeout):
            del requests, timeout

            async def responses():
                for frame in self.frames:
                    yield frame

            return responses()

    provider._stub = Scripted([pb.ExecStreamResponse(output=pb.ExecOutput(stream=pb.ExecOutput.STDOUT, data=b"x"))])
    result = await provider.exec(handle, "true")
    assert (result.stdout, result.return_code, result.error_type) == ("x", 125, "sandbox")

    provider._stub = Scripted(
        [pb.ExecStreamResponse(complete=pb.ExecComplete(exit_code=1, error="oom", termination_detail="killed"))]
    )
    result = await provider.exec(handle, "true")
    assert result.stderr == "oom\nkilled"


async def test_download_failure_leaves_no_partial_file(provider, tmp_path: Path):
    class Broken(_FakeStub):
        def ReadFile(self, request, *, timeout):
            del request, timeout

            async def responses():
                yield pb.ReadFileResponse(content=b"partial")
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)

            return responses()

    provider._stub = Broken()
    with pytest.raises(grpc.aio.AioRpcError):
        await provider.download_file(SandboxHandle("mfn-1", "mfn", object()), "/x", tmp_path / "out.bin")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "response, expected",
    [
        (_status(True, "Running"), SandboxStatus.RUNNING),
        (_status(False, "Running"), SandboxStatus.STARTING),
        (_status(False, "Pending"), SandboxStatus.STARTING),
        (_status(False, "Succeeded"), SandboxStatus.STOPPED),
        (_status(False, "Failed"), SandboxStatus.ERROR),
        (_status(False, "Mystery"), SandboxStatus.UNKNOWN),
        (_rpc_error(grpc.StatusCode.NOT_FOUND), SandboxStatus.STOPPED),
        (_rpc_error(grpc.StatusCode.UNAVAILABLE), SandboxStatus.UNKNOWN),
    ],
)
async def test_status_mapping(provider, response, expected):
    stub = _CreateStub([], [response])
    provider._stub = stub
    assert await provider.status(SandboxHandle("mfn-1", "mfn", object())) == expected


async def test_connect_serialize_and_close_errors(provider):
    handle = SandboxHandle("mfn-1", "mfn", object())
    assert await provider.serialize_handle(handle, scope="x") == {"sandbox_id": "mfn-1"}
    assert (await provider.connect({"sandbox_id": "mfn-1"})).sandbox_id == "mfn-1"

    provider._stub = _CreateStub([], [_status(False, "Failed")])
    with pytest.raises(RuntimeError, match="Cannot connect"):
        await provider.connect({"sandbox_id": "mfn-2"})

    class Shutdown(_FakeStub):
        def __init__(self, code):
            super().__init__()
            self.code = code

        async def Shutdown(self, request, *, timeout):
            raise _rpc_error(self.code)

    provider._stub = Shutdown(grpc.StatusCode.NOT_FOUND)
    await provider.close(handle)  # an already-deleted sandbox is not an error
    provider._stub = Shutdown(grpc.StatusCode.UNAVAILABLE)
    with pytest.raises(RuntimeError, match="Shutdown failed"):
        await provider.close(handle)

    await provider.aclose()
    await provider.aclose()  # idempotent


async def test_endpoint_keeps_explicit_scheme(provider):
    class Host(_FakeStub):
        async def GetHost(self, request, *, timeout):
            return pb.GetHostResponse(uri="https://mfn.example")

    provider._stub = Host()
    assert (await provider.endpoint(SandboxHandle("mfn-1", "mfn", object()), 443)).endpoint == "https://mfn.example"


# --- PTY / pipe sessions ------------------------------------------------------------------------------------------


class _ScriptedStub:
    """ExecStream stub that replays frames, or raises, after reading the first request."""

    def __init__(self, frames: list[Any] | None = None, error: BaseException | None = None) -> None:
        self.frames = frames or []
        self.error = error
        self.first = None

    def ExecStream(self, requests):
        async def responses():
            self.first = await anext(requests)
            for frame in self.frames:
                yield frame
            if self.error is not None:
                raise self.error

        return responses()


def _output(data: bytes, stream: int = pb.ExecOutput.STDOUT) -> Any:
    return pb.ExecStreamResponse(output=pb.ExecOutput(stream=stream, data=data))


async def test_pipe_session_splits_streams_and_reports_details():
    stub = _ScriptedStub(
        [
            _output(b"out"),
            _output(b"err", pb.ExecOutput.STDERR),
            pb.ExecStreamResponse(complete=pb.ExecComplete(exit_code=3, error="bad", termination_detail="gone")),
        ]
    )
    spec = SandboxPtySpec(command="run", pty=False, user="alice", cwd="/w", env={"A": "1"})
    async with MFNPtySession(stub, "mfn-1", spec, shell="/bin/bash") as session:
        await session.start()
        assert session.mode == "pipe" and not session.closed
        assert [chunk async for chunk in session] == [b"out"]
        assert await session.read_stderr() == b"err"
        assert await session.read_stderr() == b"bad"
        assert await session.read_stderr(timeout_s=1) == b"gone"
        assert await session.wait_exit(timeout_s=1) == 3
        with pytest.raises(SandboxPtyError, match="pipe-mode"):
            await session.resize(10, 10)
        with pytest.raises(NotImplementedError):
            await session.send_signal("SIGINT")
        with pytest.raises(NotImplementedError):
            await session.run_detached("x")
    assert list(stub.first.request.command)[:2] == ["/bin/bash", "-c"]
    assert stub.first.request.command[2].startswith("su -s /bin/bash -c ")
    assert stub.first.request.cwd == "/w" and not stub.first.request.HasField("pty")


async def test_pty_session_defaults_to_shell_and_handles_sigint_and_closed_state():
    stub = _ScriptedStub([pb.ExecStreamResponse(complete=pb.ExecComplete(exit_code=0))])
    session = await MFNPtySession(stub, "mfn-1", SandboxPtySpec(), shell="/bin/sh").start()
    assert list(stub.first.request.command) == ["/bin/sh"]
    await session.send_signal("sigint")
    assert await session.wait_exit() == 0
    with pytest.raises(NotImplementedError, match="SIGTERM"):
        await session.send_signal("SIGTERM")
    await session.close()
    await session.close()  # idempotent
    with pytest.raises(SandboxPtyError, match="closed"):
        await session.write(b"x")
    with pytest.raises(SandboxPtyError, match="closed"):
        await session.resize(1, 1)


async def test_pty_session_reports_stream_failures():
    session = await MFNPtySession(
        _ScriptedStub(error=RuntimeError("stream broke")), "mfn-1", SandboxPtySpec(), shell="/bin/sh"
    ).start()
    with pytest.raises(SandboxPtyError, match="stream broke"):
        await session.wait_exit(timeout_s=1)
    await session.close()

    ended = await MFNPtySession(_ScriptedStub(), "mfn-1", SandboxPtySpec(), shell="/bin/sh").start()
    with pytest.raises(SandboxPtyError, match="without an exit result"):
        await ended.wait_exit(timeout_s=1)
    assert await ended.read() == b""
    await ended.close()


async def test_pty_close_cancels_a_live_stream_and_reads_time_out():
    class Hanging:
        cancelled = False

        def ExecStream(self, requests):
            async def responses():
                await anext(requests)
                await asyncio.Event().wait()
                yield  # pragma: no cover

            return _Cancellable(responses(), lambda: setattr(Hanging, "cancelled", True))

    session = await MFNPtySession(Hanging(), "mfn-1", SandboxPtySpec(), shell="/bin/sh").start()
    with pytest.raises(TimeoutError):
        await session.read(timeout_s=0.01)
    with pytest.raises(TimeoutError):
        await session.wait_exit(timeout_s=0.01)
    await session.close()
    assert Hanging.cancelled


class _Cancellable:
    def __init__(self, inner: Any, on_cancel: Any) -> None:
        self._inner = inner
        self.cancel = on_cancel

    def __aiter__(self):
        return self._inner.__aiter__()
