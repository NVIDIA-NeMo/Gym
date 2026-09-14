# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientSession, WSServerHandshakeError

from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec
from nemo_gym.sandbox.providers.e2b._network import E2BNetworkingConfig
from nemo_gym.sandbox.providers.e2b._tunnel import main
from nemo_gym.sandbox.providers.e2b.provider import E2BProvider


pytestmark = pytest.mark.sandbox


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_binary_tunnel_half_close_auth_and_port_allowlist(tmp_path):
    payload = bytes(range(256)) * 4096

    async def respond(reader, writer):
        received = await reader.read()
        writer.write(received[::-1])
        await writer.drain()
        writer.close()

    backend = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = backend.sockets[0].getsockname()[1]
    tunnel_port, local_port = free_port(), free_port()
    server_ready, client_ready = tmp_path / "server", tmp_path / "client"
    url = f"http://127.0.0.1:{tunnel_port}/"
    server = asyncio.create_task(
        main(
            {
                "token": "test-token",
                "ports": [port],
                "tunnel_port": tunnel_port,
                "ready_file": str(server_ready),
                "peers": [],
            }
        )
    )
    client = asyncio.create_task(
        main(
            {
                "ready_file": str(client_ready),
                "peers": [
                    {
                        "addresses": ["::1"],
                        "ports": [local_port, port],
                        "url": url,
                        "headers": {"X-Gym-Tunnel-Token": "test-token"},
                    }
                ],
            }
        )
    )
    try:
        async with asyncio.timeout(5):
            while not server_ready.exists() or not client_ready.exists():
                await asyncio.sleep(0.01)
        async with ClientSession() as session:
            with pytest.raises(WSServerHandshakeError) as denied:
                await session.ws_connect(url, params={"port": port})
            assert denied.value.status == 403
            with pytest.raises(WSServerHandshakeError) as denied:
                await session.ws_connect(
                    url, params={"port": local_port}, headers={"X-Gym-Tunnel-Token": "test-token"}
                )
            assert denied.value.status == 403
            with pytest.raises(WSServerHandshakeError) as malformed:
                await session.ws_connect(url, params={"port": "invalid"}, headers={"X-Gym-Tunnel-Token": "test-token"})
            assert malformed.value.status == 400
            async with session.ws_connect(
                url, params={"port": port}, headers={"X-Gym-Tunnel-Token": "test-token"}
            ) as ws:
                await ws.send_bytes(payload)
                await ws.send_str("eof")
                received = bytearray()
                async with asyncio.timeout(5):
                    async for message in ws:
                        if message.data == "eof":
                            break
                        received.extend(message.data)
                assert received == payload[::-1]
        reader, writer = await asyncio.open_connection("::1", port)
        writer.write(payload)
        await writer.drain()
        writer.write_eof()
        assert await asyncio.wait_for(reader.read(), 5) == payload[::-1]
        writer.close()
        # A refused/undeclared upstream closes the local socket instead of hanging.
        reader, writer = await asyncio.open_connection("::1", local_port)
        writer.write(b"request")
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), 5) == b""
        writer.close()
    finally:
        for task in [server, client]:
            task.cancel()
        await asyncio.gather(server, client, return_exceptions=True)
        backend.close()
        await backend.wait_closed()


@pytest.mark.asyncio
async def test_endpoint_routes_and_network_opt_in():
    provider = E2BProvider(
        endpoints={"url_template": "https://gateway.invalid/proxy", "headers": {"target": "{sandbox_id}:{port}"}}
    )
    handle = SandboxHandle("sandbox-one", "e2b", SimpleNamespace(get_host=lambda port: f"{port}-one.e2b.app"))
    endpoint = await provider.endpoint(handle, 8000)
    assert endpoint.endpoint == "https://gateway.invalid/proxy"
    assert endpoint.headers == {"target": "sandbox-one:8000"}
    assert (await E2BProvider().endpoint(handle, 8000)).endpoint == "https://8000-one.e2b.app"
    with pytest.raises(NotImplementedError, match="enabled"):
        provider.validate_networking()


@pytest.mark.asyncio
async def test_network_routes_and_hosts():
    provider = E2BProvider(networking={"enabled": True})
    handles = [
        SandboxHandle(str(i), "e2b", SimpleNamespace(get_host=lambda port: f"{port}-one.e2b.app")) for i in range(2)
    ]
    for index, handle in enumerate(handles):
        provider._register_network(handle, SandboxSpec(ports=(8000 + index,)))
    provider._run_tunnel = AsyncMock()
    provider.exec = AsyncMock(return_value=SandboxExecResult(stdout="", stderr="", return_code=0))
    address = await provider.network_address(handles[0])
    assert address == "198.18.0.1"
    config = provider._run_tunnel.call_args.args[1]
    assert config["ports"] == [8000]
    assert config["peers"][0]["addresses"] == ["198.18.0.2"]
    await provider.set_hosts(handles[0], {"api": "198.18.0.2"})
    assert "198.18.0.2 api" in provider.exec.call_args.args[1]
    with pytest.raises(ValueError, match="hostname"):
        await provider.set_hosts(handles[0], {"bad;name": address})


@pytest.mark.asyncio
async def test_runtime_requirements_are_explicit_and_checked():
    provider = E2BProvider(
        runtime_requirements={"resize_shared_memory": True, "capability_probes": {"SYS_PTRACE": "probe"}}
    )
    provider.exec = AsyncMock(return_value=SandboxExecResult(stdout="", stderr="", return_code=0))
    await provider.configure_runtime(None, cap_add=("SYS_PTRACE",), shm_size=67108864)
    assert provider.exec.call_args_list[0].args[1] == "probe"
    assert "remount,size=67108864" in provider.exec.call_args_list[1].args[1]
    with pytest.raises(NotImplementedError):
        provider.validate_runtime_requirements(cap_add=("UNKNOWN",), shm_size=None)
    with pytest.raises(ValueError):
        provider.validate_runtime_requirements(cap_add=(), shm_size=0)
    with pytest.raises(NotImplementedError):
        E2BProvider().validate_runtime_requirements(cap_add=(), shm_size=1)
    provider.exec.return_value = SandboxExecResult(stdout="", return_code=1, stderr="denied")
    with pytest.raises(RuntimeError, match="denied"):
        await provider.configure_runtime(None, cap_add=("SYS_PTRACE",), shm_size=None)


@pytest.mark.parametrize("port", [0, 65536, True])
def test_invalid_tunnel_ports(port):
    with pytest.raises(ValueError):
        E2BNetworkingConfig(tunnel_port=port)


@pytest.mark.asyncio
async def test_tunnel_startup_configuration_and_cleanup(monkeypatch):
    provider = E2BProvider(networking={"enabled": True})
    handle = SandboxHandle("one", "e2b", None)
    provider.write_file = AsyncMock()
    launched = asyncio.Event()

    async def execute(handle, command, **kwargs):
        if command.endswith("/config.json"):
            launched.set()
            await asyncio.Event().wait()
        if command.startswith("test -f"):
            await launched.wait()
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    provider.exec = AsyncMock(side_effect=execute)
    task = await provider._run_tunnel(handle, {"peers": []})
    assert not task.done()
    config = provider.write_file.call_args_list[0].args[2]
    assert '"ready_file"' in config
    assert provider.exec.call_args_list[0].args[1].startswith("mkdir -m 700")
    await provider.aclose()
    assert task.cancelled()
    assert not provider._network_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["mkdir", "process", "timeout"])
async def test_tunnel_startup_failure_is_reported_and_tasks_released(failure):
    provider = E2BProvider(networking={"enabled": True, "startup_timeout_s": 0.2 if failure == "process" else 0.02})
    provider.write_file = AsyncMock()

    async def execute(handle, command, **kwargs):
        if command.startswith("mkdir"):
            code = int(failure == "mkdir")
        elif command.startswith("test"):
            code = 1
        elif failure == "timeout":
            await asyncio.Event().wait()
        else:
            code = 1
        return SandboxExecResult(stdout="", stderr="failed", return_code=code)

    provider.exec = AsyncMock(side_effect=execute)
    try:
        with pytest.raises((RuntimeError, TimeoutError)):
            await provider._run_tunnel(SandboxHandle("one", "e2b", None), {"peers": []})
    finally:
        await provider.aclose()
    assert not provider._network_tasks


@pytest.mark.asyncio
async def test_setup_failures_and_forwarding():
    provider = E2BProvider(networking={"enabled": True, "setup_command": "setup"})
    handle = SandboxHandle("one", "e2b", SimpleNamespace(get_host=lambda port: f"{port}-one.e2b.app"))
    provider._register_network(handle, SandboxSpec(ports=(8000,)))
    provider.exec = AsyncMock(return_value=SandboxExecResult(stdout="", stderr="denied", return_code=1))
    with pytest.raises(RuntimeError, match="setup failed"):
        await provider.network_address(handle)
    with pytest.raises(RuntimeError, match="configure E2B hosts"):
        await provider.set_hosts(handle, {"api": "198.18.0.1"})
    provider.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    provider._run_tunnel = AsyncMock()
    await provider.network_address(handle)
    provider._network_tasks["one"] = []
    await provider.network_address(handle)
    provider._run_tunnel.assert_awaited_once()
    future = asyncio.get_running_loop().create_future()
    future.set_result(SandboxExecResult(stdout="", stderr="stopped", return_code=1))
    provider._run_tunnel.return_value = future
    with pytest.raises(RuntimeError, match="forwarding exited"):
        await provider.forward_ports(handle, "198.18.0.1", (8000,), ready_file="/tmp/ready")
    config = provider._run_tunnel.call_args.args[1]
    assert config["peers"][0]["addresses"] == ["127.0.0.1", "::1"]
    assert config["ready_file"] == "/tmp/ready"


@pytest.mark.parametrize(
    "config",
    [
        {"startup_timeout_s": 0},
        {"startup_timeout_s": float("nan")},
        {"address_cidr": "127.0.0.0/8"},
        {"address_cidr": "198.18.0.1/32"},
    ],
)
def test_invalid_network_configuration(config):
    with pytest.raises(ValueError):
        E2BNetworkingConfig(**config)


@pytest.mark.asyncio
async def test_guest_addresses_are_configured_before_readiness(tmp_path, monkeypatch):
    import sys
    from unittest.mock import MagicMock

    routes = MagicMock()
    routes.link_lookup.return_value = [1]
    factory = MagicMock()
    factory.return_value.__enter__.return_value = routes
    monkeypatch.setitem(sys.modules, "pyroute2", SimpleNamespace(IPRoute=factory))
    ready = tmp_path / "ready"
    task = asyncio.create_task(main({"local_addresses": ["198.18.0.1"], "peers": [], "ready_file": str(ready)}))
    try:
        async with asyncio.timeout(5):
            while not ready.exists():
                await asyncio.sleep(0.01)
        routes.addr.assert_called_once_with("add", index=1, address="198.18.0.1", prefixlen=32)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_address_pool_exhaustion_does_not_wrap_into_another_network():
    provider = E2BProvider(networking={"enabled": True, "address_cidr": "198.18.0.0/30"})
    for i in range(2):
        provider._register_network(SandboxHandle(str(i), "e2b", None), SandboxSpec())
    with pytest.raises(RuntimeError, match="pool exhausted"):
        provider._register_network(SandboxHandle("overflow", "e2b", None), SandboxSpec())
    assert len(provider._network_members) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [[], [("unexpected", "data")]])
async def test_bridge_rejects_premature_or_invalid_websocket_messages(messages):
    from nemo_gym.sandbox.providers.e2b._tunnel import bridge

    class Socket:
        async def __aiter__(self):
            for kind, data in messages:
                yield SimpleNamespace(type=kind, data=data)

        close = AsyncMock()

    from unittest.mock import MagicMock

    async def wait_for_data(size):
        await asyncio.Event().wait()

    reader = SimpleNamespace(read=AsyncMock(side_effect=wait_for_data))
    writer = MagicMock()
    with pytest.raises(ConnectionError, match="TCP tunnel closed"):
        await bridge(reader, writer, Socket())
    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_shell_override_uses_installed_sdk_process_request():
    pytest.importorskip("e2b")
    from unittest.mock import MagicMock

    from e2b.envd.process import process_pb

    original = MagicMock()
    commands = SimpleNamespace(_rpc=original)
    provider = E2BProvider(exec={"shell": "/bin/sh"})
    provider._configure_shell(SimpleNamespace(commands=commands))
    request = process_pb.StartRequest(process=process_pb.ProcessConfig(cmd="/bin/bash", args=["-l", "-c", "echo ok"]))
    commands._rpc.start(request, timeout=10)
    assert request.process.cmd == "/bin/sh"
    assert list(request.process.args) == ["-c", "echo ok"]
    original.start.assert_called_once_with(request, timeout=10)
    commands._rpc.connect("pid")
    original.connect.assert_called_once_with("pid")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mkdir", "write", "assemble"])
async def test_chunked_upload_preserves_bytes_and_cleans_up(tmp_path, failure):
    from unittest.mock import MagicMock

    source = tmp_path / "source"
    payload = bytes(range(256)) * 3
    source.write_bytes(payload)
    provider = E2BProvider(operations={"upload_chunk_size_bytes": 100})
    writes = []
    commands = []

    async def execute(handle, command):
        commands.append(command)
        code = int(
            (failure == "mkdir" and command.startswith("mkdir"))
            or (failure == "assemble" and command.startswith("cat"))
        )
        return SandboxExecResult(stdout="", stderr="failure", return_code=code)

    async def write(handle, path, data):
        if failure == "write":
            raise RuntimeError("write failure")
        writes.append((path, data))

    provider.exec = AsyncMock(side_effect=execute)
    provider.write_file = AsyncMock(side_effect=write)
    handle = MagicMock()
    if failure:
        with pytest.raises(RuntimeError):
            await provider.upload_file(handle, source, "/tmp/target with spaces")
    else:
        await provider.upload_file(handle, source, "/tmp/target with spaces")
        assert b"".join(data for _, data in sorted(writes)) == payload
        assert all(len(data) <= 100 for _, data in writes)
        assert "'/tmp/target with spaces'" in commands[-2]
    if failure != "mkdir":
        assert commands[-1].startswith("rm -rf /tmp/gym-upload-")


@pytest.mark.parametrize(
    "config",
    [
        {"exec": {"shell": "sh"}},
        {"operations": {"upload_chunk_size_bytes": 0}},
        {"operations": {"upload_chunk_size_bytes": True}},
    ],
)
def test_invalid_shell_and_upload_configuration(config):
    with pytest.raises(ValueError):
        E2BProvider(**config)


def test_rejects_peer_ports_that_would_conflict_with_wildcard_service_bind():
    provider = E2BProvider(networking={"enabled": True})
    provider._register_network(SandboxHandle("api", "e2b", None), SandboxSpec(ports=(8000,)))
    with pytest.raises(NotImplementedError, match="distinct declared TCP ports"):
        provider._register_network(SandboxHandle("other", "e2b", None), SandboxSpec(ports=(8000,)))
    assert list(provider._network_members) == ["api"]
