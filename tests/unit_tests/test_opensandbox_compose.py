# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider


pytestmark = pytest.mark.sandbox


def test_capabilities_require_operator_configuration():
    provider = OpenSandboxProvider()
    with pytest.raises(NotImplementedError, match="network"):
        provider.validate_networking()
    with pytest.raises(NotImplementedError, match="storage"):
        provider.shared_volume_options("project/data", "/data")
    with pytest.raises(NotImplementedError, match="storage"):
        provider.shared_volume_metadata()
    OpenSandboxProvider(networking={"enabled": True}).validate_networking()


def test_shared_volume_bootstrap_mounts_root_without_subpath():
    provider = OpenSandboxProvider(shared_storage={"host_path": "/shared"})
    volume = provider.shared_volume_options(None, "/bootstrap")["volumes"][0]
    assert volume["host"] == {"path": "/shared"}
    assert "subPath" not in volume
    assert volume["mountPath"] == "/bootstrap"
    assert volume["readOnly"] is False
    from opensandbox.models.sandboxes import Volume

    assert Volume(**volume).sub_path is None


def test_shared_volume_preserves_base_subpath_and_metadata():
    provider = OpenSandboxProvider(shared_storage={"host_path": "/shared", "metadata": {"pool": "test"}})
    volume = provider.shared_volume_options("project/data", "/data", read_only=True)["volumes"][0]
    assert volume["host"] == {"path": "/shared"}
    assert volume["subPath"] == "project/data"
    assert volume["mountPath"] == "/data"
    assert volume["readOnly"] is True
    assert volume["name"] == provider.shared_volume_options("project/data", "/data")["volumes"][0]["name"]
    assert provider.shared_volume_metadata() == {"pool": "test"}
    from opensandbox.models.sandboxes import Volume

    assert Volume(**volume).sub_path == "project/data"


@pytest.mark.parametrize(
    "source", ["", "/absolute", "../escape", "project/../escape", "project\\escape", ".", "./", "././"]
)
def test_shared_volume_rejects_unsafe_source(source):
    provider = OpenSandboxProvider(shared_storage={"host_path": "/shared"})
    with pytest.raises(ValueError):
        provider.shared_volume_options(source, "/data")


@pytest.mark.parametrize("target", ["relative", "", "/data/../etc"])
def test_shared_volume_requires_absolute_target(target):
    provider = OpenSandboxProvider(shared_storage={"host_path": "/shared"})
    with pytest.raises(ValueError):
        provider.shared_volume_options("project/data", target)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,headers,expected",
    [
        ("10.2.3.4:44772", {}, "10.2.3.4"),
        ("http://[fd00::1]:44772", {}, "fd00::1"),
        ("proxy.test:44772", {}, None),
        ("10.2.3.4:32000", {}, None),
        ("10.2.3.4:44772/proxy", {}, None),
        ("10.2.3.4:44772", {"X-Route": "sandbox"}, None),
    ],
)
async def test_network_address_requires_direct_unmapped_ip(endpoint, headers, expected):
    sdk_service = SimpleNamespace(
        get_sandbox_endpoint=AsyncMock(return_value=SimpleNamespace(endpoint=endpoint, headers=headers))
    )
    handle = SandboxHandle("sandbox", "opensandbox", SimpleNamespace(_sandbox_service=sdk_service))
    provider = OpenSandboxProvider(networking={"enabled": True}, connection={"use_server_proxy": True})
    if expected is None:
        with pytest.raises(ValueError):
            await provider.network_address(handle)
    else:
        assert await provider.network_address(handle) == expected
    sdk_service.get_sandbox_endpoint.assert_awaited_once_with("sandbox", 44772, False)


@pytest.mark.asyncio
async def test_hosts_append_validated_mapping_as_root(monkeypatch):
    provider = OpenSandboxProvider(networking={"enabled": True})
    execute = AsyncMock(return_value=SandboxExecResult("", "", 0))
    monkeypatch.setattr(provider, "exec", execute)
    handle = SandboxHandle("sandbox", "opensandbox", None)
    await provider.set_hosts(handle, {"database": "10.2.3.4", "cache.internal": "fd00::1"})
    args, kwargs = execute.call_args
    assert args[0] is handle
    assert "10.2.3.4 database\n" in shlex.split(args[1])
    assert "fd00::1 cache.internal\n" in shlex.split(args[1])
    assert args[1].endswith(" >> /etc/hosts")
    assert kwargs["user"] == "root"
    execute.return_value = SandboxExecResult("uid change failed", "permission denied", 1)
    with pytest.raises(RuntimeError, match="hosts") as error:
        await provider.set_hosts(handle, {"database": "10.2.3.4"})
    assert "sandbox" in str(error.value)
    assert "uid change failed" in str(error.value)
    assert "permission denied" in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hosts", [{"bad\nname": "10.2.3.4"}, {"db": "127.0.0.1;touch /tmp/injected"}, {"-bad": "10.2.3.4"}]
)
async def test_hosts_reject_injection_before_execution(hosts, monkeypatch):
    provider = OpenSandboxProvider(networking={"enabled": True})
    execute = AsyncMock()
    monkeypatch.setattr(provider, "exec", execute)
    with pytest.raises(ValueError):
        await provider.set_hosts(SandboxHandle("sandbox", "opensandbox", None), hosts)
    execute.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("user,expected_uid", [("root", 0), (0, 0), (None, 1000)])
async def test_explicit_root_sets_sdk_uid_even_when_daemon_is_nonroot(monkeypatch, user, expected_uid):
    from nemo_gym.sandbox.providers.opensandbox import provider as provider_module

    async def run(command, *, opts):
        assert command == "id -u"
        return SimpleNamespace(
            logs=SimpleNamespace(stdout=[SimpleNamespace(text=str(getattr(opts, "uid", 1000)))], stderr=[]),
            error=None,
            exit_code=0,
        )

    monkeypatch.setattr(
        provider_module, "_require_opensandbox_sdk", lambda: (object, object, SimpleNamespace, object, object)
    )
    provider = OpenSandboxProvider()
    handle = SandboxHandle("nonroot-sandbox", "opensandbox", SimpleNamespace(commands=SimpleNamespace(run=run)))
    result = await provider.exec(handle, "id -u", user=user)
    assert result.stdout == str(expected_uid)


def test_runtime_requirements_fail_before_creation_without_operator_support():
    provider = OpenSandboxProvider()
    provider.validate_runtime_requirements(cap_add=(), shm_size=None)
    with pytest.raises(NotImplementedError, match="SYS_PTRACE"):
        provider.validate_runtime_requirements(cap_add=("SYS_PTRACE",), shm_size=None)
    with pytest.raises(NotImplementedError, match="resize_shared_memory"):
        provider.validate_runtime_requirements(cap_add=(), shm_size=1024)


@pytest.mark.asyncio
async def test_runtime_requirements_execute_configured_probe_and_resize(monkeypatch):
    provider = OpenSandboxProvider(
        runtime_requirements={
            "capability_probes": {"SYS_PTRACE": "gdb-probe"},
            "resize_shared_memory": True,
        }
    )
    execute = AsyncMock(
        side_effect=[
            SandboxExecResult("", "", 1),
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
        ]
    )
    monkeypatch.setattr(provider, "exec", execute)
    handle = SandboxHandle("sandbox", "opensandbox", None)
    await provider.configure_runtime(handle, cap_add=("SYS_PTRACE",), shm_size=1073741824)
    assert execute.call_args_list[0].kwargs["user"] is None
    assert execute.call_args_list[1].args[1] == "gdb-probe"
    assert execute.call_args_list[1].kwargs["user"] == "root"
    assert execute.call_args_list[2].args[1].startswith("mount -o remount,size=1073741824 /dev/shm && ")
    assert execute.call_args_list[2].args[1].endswith(execute.call_args_list[0].args[1])
    assert execute.call_args_list[2].kwargs["user"] == "root"


@pytest.mark.asyncio
async def test_shared_memory_matching_allocation_never_requests_root(monkeypatch):
    provider = OpenSandboxProvider(runtime_requirements={"resize_shared_memory": True})
    execute = AsyncMock(return_value=SandboxExecResult("", "", 0))
    monkeypatch.setattr(provider, "exec", execute)
    await provider.configure_runtime(SandboxHandle("nonroot", "opensandbox", None), cap_add=(), shm_size=67108864)
    assert execute.await_count == 1
    assert execute.call_args.kwargs["user"] is None
    assert "mount" not in execute.call_args.args[1]


@pytest.mark.asyncio
async def test_shared_memory_mismatch_attempts_privileged_resize_and_reports_failure(monkeypatch):
    provider = OpenSandboxProvider(runtime_requirements={"resize_shared_memory": True})
    execute = AsyncMock(
        side_effect=[
            SandboxExecResult("", "", 1),
            SandboxExecResult("", "fork/exec /bin/bash: operation not permitted", 1),
        ]
    )
    monkeypatch.setattr(provider, "exec", execute)
    with pytest.raises(RuntimeError, match="shm_size.*operation not permitted"):
        await provider.configure_runtime(
            SandboxHandle("nonroot", "opensandbox", None), cap_add=(), shm_size=1073741824
        )
    assert [call.kwargs["user"] for call in execute.call_args_list] == [None, "root"]


@pytest.mark.parametrize("size", [True, 0, -1, "1gb"])
def test_runtime_requirements_reject_invalid_shared_memory(size):
    provider = OpenSandboxProvider(runtime_requirements={"resize_shared_memory": True})
    with pytest.raises(ValueError, match="positive"):
        provider.validate_runtime_requirements(cap_add=(), shm_size=size)


@pytest.mark.asyncio
async def test_port_forwarding_requires_opt_in_and_reports_process_exit(monkeypatch):
    provider = OpenSandboxProvider()
    with pytest.raises(NotImplementedError, match="loopback_forwarding"):
        provider.validate_port_forwarding()
    with pytest.raises(NotImplementedError, match="background_exec"):
        OpenSandboxProvider(networking={"loopback_forwarding": True}).validate_port_forwarding()
    provider = OpenSandboxProvider(
        networking={"loopback_forwarding": True, "python_executable": "/opt/python"},
        operations={"background_exec": True},
    )
    execute = AsyncMock(return_value=SandboxExecResult("", "address in use", 1))
    monkeypatch.setattr(provider, "exec", execute)
    with pytest.raises(RuntimeError, match="address in use"):
        await provider.forward_ports(
            SandboxHandle("box", "opensandbox", None), "10.1.2.3", (5432,), ready_file="/tmp/ready"
        )
    args = shlex.split(execute.call_args.args[1])
    assert execute.call_args.kwargs["timeout_s"] is None
    assert args[:3] == ["/opt/python", "-u", "-c"]
    assert args[4:] == ["10.1.2.3", "/tmp/ready", "5432"]


@pytest.mark.asyncio
@pytest.mark.parametrize("user", [None, "root", 1000])
async def test_readiness_probe_uses_configured_user(monkeypatch, user):
    provider = OpenSandboxProvider(probe={"user": user})
    execute = AsyncMock(return_value=SandboxExecResult("nemo-gym-sandbox-ready", "", 0))
    monkeypatch.setattr(provider, "_exec", execute)
    await provider._verify_created_handle(SandboxHandle("box", "opensandbox", None))
    assert execute.call_args.kwargs["user"] == user


@pytest.mark.asyncio
async def test_port_forwarding_setup_failure_does_not_launch_relay(monkeypatch):
    provider = OpenSandboxProvider(
        networking={"loopback_forwarding": True, "setup_command": "install-python"},
        operations={"background_exec": True},
    )
    execute = AsyncMock(return_value=SandboxExecResult("", "package unavailable", 1))
    monkeypatch.setattr(provider, "exec", execute)
    handle = SandboxHandle("box", "opensandbox", None)
    with pytest.raises(RuntimeError, match="setup failed.*package unavailable"):
        await provider.forward_ports(handle, "10.1.2.3", (5432,), ready_file="/tmp/ready")
    execute.assert_awaited_once_with(handle, "install-python", user="root", timeout_s=180)
    execute.reset_mock()
    execute.side_effect = [SandboxExecResult("", "", 0), SandboxExecResult("", "listener stopped", 1)]
    with pytest.raises(RuntimeError, match="listener stopped"):
        await provider.forward_ports(handle, "10.1.2.3", (5432,), ready_file="/tmp/ready")
    assert execute.await_count == 2
    assert execute.call_args.kwargs["timeout_s"] is None
