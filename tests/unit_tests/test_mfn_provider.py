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
                yield pb.ExecStreamResponse(output=pb.ExecOutput(data=second.stdin.data))
                yield pb.ExecStreamResponse(complete=pb.ExecComplete(exit_code=0))

            return responses()

    provider._stub = PtyStub()
    handle = SandboxHandle("mfn-1", "mfn", object())
    session = await provider.create_pty(handle, SandboxPtySpec(rows=24, cols=80))

    assert await session.read() == b"prompt> "
    await session.write(b"echo hi\n")
    assert await session.read() == b"echo hi\n"
    assert await session.wait_exit() == 0
    with pytest.raises(NotImplementedError):
        await session.resize(40, 100)
    await session.close()
