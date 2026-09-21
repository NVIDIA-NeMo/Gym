# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec
from nemo_gym.sandbox.providers.docker import DockerProvider
from nemo_gym.sandbox.providers.docker import provider as module


pytestmark = pytest.mark.sandbox


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(module, "_require_docker", lambda: "/usr/bin/docker")
    return DockerProvider(exec={"exec_shell": "sh"}, probe={"command": None})


def handle():
    return SandboxHandle("nemo-gym-test", "docker", SimpleNamespace(name="nemo-gym-test"))


async def test_native_runtime_flags_and_verification(provider):
    metadata = provider.validate_runtime_requirements(cap_add=("SYS_PTRACE",), shm_size=67108864)
    provider._run = AsyncMock(return_value=(0, "id", ""))
    box = await provider.create(SandboxSpec(image="test", metadata=metadata))
    argv = provider._run.call_args.args[0]
    assert argv[argv.index("--shm-size") + 1] == "67108864"
    assert argv[argv.index("--cap-add") + 1] == "SYS_PTRACE"
    provider._run.return_value = (
        0,
        json.dumps({"HostConfig": {"ShmSize": 67108864, "CapAdd": ["CAP_SYS_PTRACE"]}}),
        "",
    )
    await provider.configure_runtime(box, cap_add=("SYS_PTRACE",), shm_size=67108864)
    assert provider._run.call_args.args[0][1] == "inspect"
    assert all("remount" not in str(call) for call in provider._run.call_args_list)


@pytest.mark.parametrize("config", [{"ShmSize": 1, "CapAdd": ["SYS_PTRACE"]}, {"ShmSize": 67108864, "CapAdd": []}])
async def test_runtime_mismatch_fails(provider, config):
    provider._run = AsyncMock(return_value=(0, json.dumps({"HostConfig": config}), ""))
    with pytest.raises(RuntimeError, match="did not apply"):
        await provider.configure_runtime(handle(), cap_add=("SYS_PTRACE",), shm_size=67108864)


@pytest.mark.parametrize("size", [0, -1, True, "64m"])
def test_invalid_shared_memory(provider, size):
    with pytest.raises(ValueError, match="shm_size"):
        provider.validate_runtime_requirements(cap_add=(), shm_size=size)


def test_invalid_capability(provider):
    with pytest.raises(ValueError, match="cap_add"):
        provider.validate_runtime_requirements(cap_add=("--privileged",), shm_size=None)


@pytest.mark.parametrize("network", ["none", "host", "container:other"])
def test_unsupported_network_fails_preflight(provider, network):
    provider._create_config = module.DockerCreateConfig(network=network)
    with pytest.raises(NotImplementedError, match="peer-reachable"):
        provider.validate_networking()


@pytest.mark.parametrize(
    "network,addresses,expected",
    [
        (None, {"bridge": {"IPAddress": "172.17.0.2"}}, "172.17.0.2"),
        ("custom", {"custom": {"IPAddress": "172.18.0.3"}}, "172.18.0.3"),
        ("network-id", {"custom": {"GlobalIPv6Address": "fd00::2"}}, "fd00::2"),
    ],
)
async def test_network_address(provider, network, addresses, expected):
    provider._create_config = module.DockerCreateConfig(network=network)
    provider._run = AsyncMock(return_value=(0, json.dumps({"NetworkSettings": {"Networks": addresses}}), ""))
    assert await provider.network_address(handle()) == expected


@pytest.mark.parametrize("address", ["", "0.0.0.0", "127.0.0.1", "224.0.0.1"])
async def test_unusable_network_address_fails(provider, address):
    provider._run = AsyncMock(
        return_value=(0, json.dumps({"NetworkSettings": {"Networks": {"bridge": {"IPAddress": address}}}}), "")
    )
    with pytest.raises(ValueError):
        await provider.network_address(handle())


async def test_inspect_and_hosts_errors(provider):
    provider._run = AsyncMock(return_value=(1, "", "missing"))
    with pytest.raises(RuntimeError, match="missing"):
        await provider.network_address(handle())
    provider.exec = AsyncMock(return_value=SandboxExecResult("", "", 0))
    await provider.set_hosts(handle(), {"api": "172.17.0.2"})
    assert "172.17.0.2 api" in provider.exec.call_args.args[1]
    assert provider.exec.call_args.kwargs["user"] == "root"
    with pytest.raises(ValueError, match="hostname"):
        await provider.set_hosts(handle(), {"bad;host": "172.17.0.2"})
    provider.exec.return_value = SandboxExecResult("", "read only", 1)
    with pytest.raises(RuntimeError, match="read only"):
        await provider.set_hosts(handle(), {"api": "172.17.0.2"})


def test_shared_volume_mapping_and_opt_in(provider):
    from nemo_gym.sandbox.providers.docker import DockerSharedStorageConfig

    with pytest.raises(NotImplementedError, match="host_path"):
        provider.shared_volume_metadata()
    provider._shared_storage = DockerSharedStorageConfig(host_path="/srv/compose")
    assert provider.shared_volume_metadata() == {}
    assert provider.shared_volume_options(None, "/root") == {"volumes": ["/srv/compose:/root"]}
    assert provider.shared_volume_options("project/data", "/data", read_only=True) == {
        "volumes": ["/srv/compose/project/data:/data:ro"]
    }
    for source in ["", "../outside", "/outside", "a:b", "a\\b"]:
        with pytest.raises(ValueError):
            provider.shared_volume_options(source, "/data")
    with pytest.raises(ValueError):
        provider.shared_volume_options("safe", "/data:rw")


async def test_forwarder_is_opt_in_unbounded_and_reports_exit(provider):
    from nemo_gym.sandbox.providers.docker import DockerNetworkingConfig

    with pytest.raises(NotImplementedError, match="loopback_forwarding"):
        provider.validate_port_forwarding()
    provider._networking = DockerNetworkingConfig(loopback_forwarding=True, setup_command="setup-python")
    provider.exec = AsyncMock(return_value=SandboxExecResult("", "", 0))
    provider._run = AsyncMock(return_value=(1, "", "bind failed"))
    with pytest.raises(RuntimeError, match="bind failed"):
        await provider.forward_ports(handle(), "172.17.0.2", (8000,), ready_file="/tmp/ready")
    assert provider._run.call_args.kwargs == {"timeout_s": None, "bounded": False}
    assert provider._run.call_args.args[0][-3:] == ["172.17.0.2", "/tmp/ready", "8000"]
    provider.exec.return_value = SandboxExecResult("", "setup denied", 1)
    with pytest.raises(RuntimeError, match="setup denied"):
        await provider.forward_ports(handle(), "172.17.0.2", (8000,), ready_file="/tmp/ready")


@pytest.mark.skipif(shutil.which("sleep") is None, reason="sleep not installed")
async def test_cancelled_cli_is_reaped_and_unbounded_relay_does_not_block_exec(provider, monkeypatch):
    real_create = asyncio.create_subprocess_exec
    processes = []
    started = asyncio.Event()

    async def create(*args, **kwargs):
        proc = await real_create(*args, **kwargs)
        processes.append(proc)
        started.set()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    provider._semaphore = asyncio.Semaphore(1)
    async with provider._semaphore:
        task = asyncio.create_task(provider._run([shutil.which("sleep"), "30"], timeout_s=None, bounded=False))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert processes[0].returncode is not None


@pytest.mark.skipif(
    os.environ.get("GYM_TEST_DOCKER_COMPOSE") != "1" or shutil.which("docker") is None,
    reason="set GYM_TEST_DOCKER_COMPOSE=1 with a reachable Docker daemon",
)
async def test_live_docker_compose(tmp_path, monkeypatch):
    import uuid

    import yaml

    from nemo_gym import server_utils
    from nemo_gym.sandbox import AsyncSandboxCompose
    from nemo_gym.server_utils import request

    monkeypatch.setattr(server_utils, "get_global_config_dict", lambda **kwargs: {})
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)

    image = "python:3.13-slim"
    root = "/tmp/gym-compose-test-" + uuid.uuid4().hex
    provider = DockerProvider(
        exec={"concurrency": 1},
        networking={"loopback_forwarding": True},
        shared_storage={"host_path": root},
    )
    document = {
        "services": {
            "api": {
                "image": image,
                "shm_size": 67108864,
                "cap_add": ["SYS_PTRACE"],
                "entrypoint": [
                    "sh",
                    "-c",
                    "printf shared-data > /data/payload; exec python3 -m http.server 8000 --directory /data",
                ],
                "expose": ["8000"],
                "volumes": [{"type": "volume", "source": "data", "target": "/data"}],
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python3",
                        "-c",
                        "import urllib.request; urllib.request.urlopen('http://localhost:8000/payload')",
                    ],
                    "interval": "1s",
                },
            },
            "other": {"image": image, "entrypoint": ["python3", "-m", "http.server", "8000"], "expose": ["8000"]},
            "main": {
                "image": image,
                "entrypoint": ["sleep", "infinity"],
                "working_dir": "/data",
                "environment": {"COMPOSE_GREETING": "hello"},
                "volumes": [{"type": "volume", "source": "data", "target": "/data", "read_only": True}],
                "depends_on": {"api": {"condition": "service_healthy"}},
            },
            "localhost": {"image": image, "entrypoint": ["sleep", "infinity"], "network_mode": "service:api"},
        }
    }
    document["volumes"] = {"data": {}}
    path = tmp_path / "compose.yaml"
    path.write_text(yaml.safe_dump(document))
    names = []
    try:
        async with AsyncSandboxCompose(provider, path, volume_init_image=image, timeout_s=120) as group:
            names = [box._require_handle().sandbox_id for box in group.services.values()]
            command = "python3 -c \"import urllib.request; print(urllib.request.urlopen('http://api:8000/payload').read().decode())\""
            assert (await group.services["main"].exec(command)).stdout.strip() == "shared-data"
            assert (await group.services["main"].exec("cat /data/payload")).stdout == "shared-data"
            assert (await group.services["main"].exec("touch /data/should-fail")).return_code != 0
            assert (
                await group.services["main"].exec(
                    "python3 -c \"import urllib.request; urllib.request.urlopen('http://other:8000')\""
                )
            ).return_code == 0
            assert (
                await group.services["localhost"].exec(command.replace("http://api:", "http://127.0.0.1:"))
            ).stdout.strip() == "shared-data"
            result = await group.services["api"].exec("stat -fc '%S %b' /dev/shm", user=65534)
            assert result.return_code == 0
            block, count = map(int, result.stdout.split())
            assert block * count == 67108864
            endpoint = await group.services["api"].endpoint(8000)
            async with await request("GET", endpoint.endpoint + "/payload") as response:
                assert response.status == 200
                assert await response.text() == "shared-data"
            descriptor = await group.serialize()
            attached = await AsyncSandboxCompose.connect(descriptor, provider={"docker": {}})
            try:
                assert (await attached.services["main"].exec(command)).stdout.strip() == "shared-data"
                assert (await attached.services["main"].exec("pwd")).stdout.strip() == "/data"
                assert (await attached.services["main"].exec("printenv COMPOSE_GREETING")).stdout.strip() == "hello"
                assert (await attached.services["api"].endpoint(8000)).endpoint == endpoint.endpoint
            finally:
                await attached.stop()
        for name in names:
            code, _, _ = await provider._run([provider._binary, "inspect", name], timeout_s=30)
            assert code != 0
        # The adapter removes its collection directory but preserves the configured shared root.
        code, out, err = await provider._run(
            [provider._binary, "run", "--rm", "-v", f"{root}:/data", image, "sh", "-c", 'test -z "$(ls -A /data)"'],
            timeout_s=30,
        )
        assert code == 0, (out, err)
    finally:
        # Clear only this test's unique daemon-host directory using a disposable helper.
        await provider._run(
            [provider._binary, "run", "--rm", "-v", f"{root}:/data", image, "sh", "-c", "rm -rf /data/*"],
            timeout_s=30,
        )
        await provider.aclose()
        if server_utils._GLOBAL_AIOHTTP_CLIENT is not None:
            await server_utils._GLOBAL_AIOHTTP_CLIENT.close()


@pytest.mark.parametrize("during_probe", [False, True])
async def test_cancelled_create_cleans_up_container(provider, during_probe):
    async def run(argv, **kwargs):
        if argv[1] == "run" and not during_probe:
            raise asyncio.CancelledError()
        return 0, "id", ""

    provider._run = AsyncMock(side_effect=run)
    if during_probe:
        provider._verify_created_handle = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await provider.create(SandboxSpec(image="test"))
    assert provider._run.call_args.args[0][1:3] == ["rm", "-f"]


async def test_explicit_unbounded_exec_preserves_default_for_ordinary_calls(provider):
    box = SandboxHandle("box", "docker", SimpleNamespace(name="box", shell="sh", env={}))
    provider._run = AsyncMock(return_value=(0, "", ""))
    await provider.exec(box, "short-command")
    assert provider._run.call_args.kwargs["timeout_s"] == 180
    assert "bounded" not in provider._run.call_args.kwargs
    await provider.exec(box, "service", timeout_s=None)
    assert provider._run.call_args.kwargs["timeout_s"] is None
    assert provider._run.call_args.kwargs["bounded"] is False


@pytest.mark.skipif(
    os.environ.get("GYM_TEST_DOCKER_COMPOSE") != "1" or shutil.which("docker") is None,
    reason="set GYM_TEST_DOCKER_COMPOSE=1 with a reachable Docker daemon",
)
async def test_live_failed_start_removes_containers(tmp_path, monkeypatch):
    from nemo_gym.sandbox import AsyncSandboxCompose

    provider = DockerProvider(exec={"exec_shell": "sh"})
    created = []
    original = provider.create

    async def create(spec):
        box = await original(spec)
        created.append(box.sandbox_id)
        return box

    monkeypatch.setattr(provider, "create", create)
    path = tmp_path / "bad.yaml"
    path.write_text("""services:
  failing:
    image: python:3.13-slim
    entrypoint: [sh, -c, 'exit 7']
    healthcheck:
      test: [CMD, 'false']
      interval: 1s
      retries: 1
""")
    group = AsyncSandboxCompose(provider, path, timeout_s=30)
    with pytest.raises(RuntimeError, match="(exited|unhealthy)"):
        await group.start()
    assert created
    for name in created:
        code, _, _ = await provider._run([provider._binary, "inspect", name], timeout_s=30)
        assert code != 0
    await group.stop()
