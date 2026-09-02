# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket
import time
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from nemo_gym.sandbox.providers.base import SandboxEndpoint, SandboxSpec
from resources_servers.enterpriseops_gym import runtime as runtime_module
from resources_servers.enterpriseops_gym.runtime import SERVICES, EnterpriseOpsAssets, EnterpriseOpsServiceRuntime


class FakeSandbox:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.started = False
        self.stopped = False
        self.commands = []

    async def start(self):
        self.started = True
        return self

    async def endpoint(self, port: int) -> SandboxEndpoint:
        return SandboxEndpoint(endpoint=f"http://127.0.0.1:{port}")

    async def exec(self, command: str, *, cwd: str, timeout_s: float):
        self.commands.append((command, cwd, timeout_s))
        return type("Result", (), {"return_code": 0, "stderr": ""})()

    async def stop(self) -> None:
        self.stopped = True


class TimeoutLaunchingSandbox(FakeSandbox):
    async def exec(self, command: str, *, cwd: str, timeout_s: float):
        self.commands.append((command, cwd, timeout_s))
        return type(
            "Result", (), {"return_code": 125, "stderr": "launch command timed out", "error_type": "timeout"}
        )()


class FailingLaunchingSandbox(FakeSandbox):
    async def exec(self, command: str, *, cwd: str, timeout_s: float):
        self.commands.append((command, cwd, timeout_s))
        return type("Result", (), {"return_code": 1, "stderr": "launch failed", "error_type": None})()


class FailingStoppingSandbox(FakeSandbox):
    async def stop(self) -> None:
        raise RuntimeError("stop failed")


class BlockingStoppingSandbox(FakeSandbox):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        self.started_stop = asyncio.Event()
        self.release_stop = asyncio.Event()

    async def stop(self) -> None:
        self.started_stop.set()
        await self.release_stop.wait()
        self.stopped = True


def test_drive_uses_its_package_main_module() -> None:
    assert SERVICES["drive"].app_target == "app.main:app"


def test_managed_services_use_a_clean_contained_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class CapturingSandbox:
        def __init__(self, provider, spec) -> None:
            captured["provider"] = provider
            captured["spec"] = spec

    monkeypatch.setattr(runtime_module, "AsyncSandbox", CapturingSandbox)
    spec = SandboxSpec(image="service.sif", ports=(8001,))

    EnterpriseOpsServiceRuntime._create_sandbox(spec)

    assert captured["spec"] is spec
    assert captured["provider"] == {
        "apptainer": {
            "create": {
                "extra_start_args": [
                    "--writable-tmpfs",
                    "--contain",
                    "--cleanenv",
                    "--no-home",
                    "--no-mount",
                    "hostfs,bind-paths",
                ]
            }
        }
    }


def test_services_keep_their_internal_api_urls_on_the_sandbox_port() -> None:
    assert SERVICES["csm"].environment == (("API_BASE_URL", "http://127.0.0.1:8001"),)
    assert SERVICES["teams"].environment == (("API_PORT", "8002"),)
    assert SERVICES["calendar"].environment == (("API_PORT", "8003"),)
    assert SERVICES["email"].environment == (("API_PORT", "8004"),)
    assert SERVICES["itsm"].environment == (("ITSM_API_BASE_URL", "http://127.0.0.1:8006"),)
    assert SERVICES["hr"].environment == (("HR_API_BASE_URL", "http://127.0.0.1:8008"),)
    assert SERVICES["drive"].environment == (
        ("FASTAPI_BASE_URL", "http://127.0.0.1:8009"),
        ("MCP_SERVER_HOST", "127.0.0.1"),
        ("MCP_SERVER_PORT", "8009"),
    )


@pytest.mark.asyncio
async def test_existing_digest_named_sif_is_reused_without_pulling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    service = SERVICES["csm"]
    sif_path = assets.sif_path(service)
    sif_path.parent.mkdir(parents=True)
    sif_path.write_bytes(b"cached-sif")

    async def unexpected_pull(*args: object, **kwargs: object) -> None:
        raise AssertionError("existing SIF must be reused")

    monkeypatch.setattr(assets, "_pull_sif", unexpected_pull)

    assert await assets.ensure_sif(service) == sif_path


@pytest.mark.asyncio
async def test_native_arm64_sif_is_used_without_pulling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    native_sif_dir = tmp_path / "native"
    native_sif_dir.mkdir()
    native_sif = native_sif_dir / "csm-arm64.sif"
    native_sif.write_bytes(b"native-arm64-sif")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path / "cache", native_sif_dir=native_sif_dir)

    async def unexpected_pull(*args: object, **kwargs: object) -> None:
        raise AssertionError("native ARM64 mode must not pull an upstream image")

    monkeypatch.setattr(assets, "_pull_sif", unexpected_pull)

    assert await assets.ensure_sif(SERVICES["csm"]) == native_sif


def test_native_sif_directory_does_not_override_amd64_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path / "cache", native_sif_dir=tmp_path / "native")

    assert assets.sif_path(SERVICES["csm"]) == tmp_path / "cache" / "images" / (
        "csm-eaa456ac9aa85728426e7d3813a0bbca0949d6a8695be30e26f03894e6e6b189.sif"
    )


@pytest.mark.asyncio
async def test_arm64_missing_cached_sifs_fail_before_downloading_or_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "aarch64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    created = []

    async def unexpected_seed_root() -> Path:
        raise AssertionError("ARM64 SIF validation must happen before downloading the seed archive")

    monkeypatch.setattr(assets, "ensure_seed_root", unexpected_seed_root)

    def sandbox_factory(spec):
        created.append(spec)
        return FakeSandbox(spec)

    runtime = EnterpriseOpsServiceRuntime(assets=assets, sandbox_factory=sandbox_factory)
    monkeypatch.setattr(runtime, "_reserve_service_ports", lambda: None)

    with pytest.raises(RuntimeError, match="missing native ARM64 EnterpriseOps SIFs") as error:
        await runtime.start()

    assert "csm-arm64.sif" in str(error.value)
    assert created == []


@pytest.mark.asyncio
async def test_seed_root_extracts_the_verified_database_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)

    def checkout(target: Path) -> None:
        target.mkdir()
        with ZipFile(target / "gym_dbs.zip", "w") as archive:
            archive.writestr("Domain Wise DBs and Task-DB Mappings/csm/dbs/example.sql", "select 1;")

    monkeypatch.setattr(assets, "_checkout_source", checkout)
    monkeypatch.setattr(assets, "database_archive_sha256", None)

    seed_root = await assets.ensure_seed_root()

    assert seed_root == assets.source_root
    assert (seed_root / "Domain Wise DBs and Task-DB Mappings/csm/dbs/example.sql").read_text() == "select 1;"


@pytest.mark.asyncio
async def test_seed_root_materialization_is_serialized_across_concurrent_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    checkouts = 0

    def checkout(target: Path) -> None:
        nonlocal checkouts
        checkouts += 1
        time.sleep(0.05)
        target.mkdir()
        with ZipFile(target / "gym_dbs.zip", "w") as archive:
            archive.writestr("Domain Wise DBs and Task-DB Mappings/csm/dbs/example.sql", "select 1;")

    monkeypatch.setattr(assets, "_checkout_source", checkout)
    monkeypatch.setattr(assets, "database_archive_sha256", None)

    await asyncio.gather(assets.ensure_seed_root(), assets.ensure_seed_root())

    assert checkouts == 1


@pytest.mark.asyncio
async def test_managed_services_reject_an_occupied_port_before_asset_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        service = replace(SERVICES["csm"], port=port)
        monkeypatch.setattr(runtime_module, "SERVICES", {"csm": service})
        assets = EnterpriseOpsAssets(cache_dir=tmp_path)
        created = []

        def unexpected_asset_setup() -> None:
            raise AssertionError("port availability must be checked before asset setup")

        monkeypatch.setattr(assets, "ensure_native_arm64_sifs", unexpected_asset_setup)

        def sandbox_factory(spec):
            created.append(spec)
            return FakeSandbox(spec)

        runtime = EnterpriseOpsServiceRuntime(assets=assets, sandbox_factory=sandbox_factory)

        with pytest.raises(RuntimeError, match=f"port {port} is already in use"):
            await runtime.start()

    assert created == []


@pytest.mark.asyncio
async def test_managed_services_stop_every_sandbox_when_one_stop_fails(tmp_path: Path) -> None:
    runtime = EnterpriseOpsServiceRuntime(assets=EnterpriseOpsAssets(cache_dir=tmp_path))
    clean_sandbox = FakeSandbox(SandboxSpec(image="clean"))
    failing_sandbox = FailingStoppingSandbox(SandboxSpec(image="failing"))
    runtime.sandboxes = [clean_sandbox, failing_sandbox]

    with pytest.raises(RuntimeError, match="failed to stop EnterpriseOps services"):
        await runtime.stop()

    assert clean_sandbox.stopped


@pytest.mark.asyncio
async def test_managed_services_start_all_sandbox_stops_concurrently(tmp_path: Path) -> None:
    runtime = EnterpriseOpsServiceRuntime(assets=EnterpriseOpsAssets(cache_dir=tmp_path))
    first_sandbox = BlockingStoppingSandbox(SandboxSpec(image="first"))
    second_sandbox = BlockingStoppingSandbox(SandboxSpec(image="second"))
    runtime.sandboxes = [first_sandbox, second_sandbox]

    stop_task = asyncio.create_task(runtime.stop())
    try:
        await asyncio.wait_for(
            asyncio.gather(first_sandbox.started_stop.wait(), second_sandbox.started_stop.wait()), timeout=0.1
        )
    finally:
        first_sandbox.release_stop.set()
        second_sandbox.release_stop.set()

    await stop_task
    assert first_sandbox.stopped
    assert second_sandbox.stopped


@pytest.mark.asyncio
async def test_managed_services_finish_stops_before_propagating_cancellation(tmp_path: Path) -> None:
    runtime = EnterpriseOpsServiceRuntime(assets=EnterpriseOpsAssets(cache_dir=tmp_path))
    sandbox = BlockingStoppingSandbox(SandboxSpec(image="service"))
    runtime.sandboxes = [sandbox]

    stop_task = asyncio.create_task(runtime.stop())
    await sandbox.started_stop.wait()
    stop_task.cancel()
    sandbox.release_stop.set()

    with pytest.raises(asyncio.CancelledError):
        await stop_task

    assert sandbox.stopped


@pytest.mark.asyncio
async def test_managed_services_start_from_cached_sifs_and_stop_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "is_arm64_host", lambda: False)
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    created = []

    async def seed_root() -> Path:
        return tmp_path / "source"

    async def sif_path(service) -> Path:
        return tmp_path / "images" / f"{service.domain}.sif"

    ready_urls = []

    async def readiness_probe(url: str) -> None:
        ready_urls.append(url)

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)
    monkeypatch.setattr(assets, "ensure_sif", sif_path)

    def sandbox_factory(spec):
        sandbox = FakeSandbox(spec)
        created.append(sandbox)
        return sandbox

    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_factory=sandbox_factory,
        readiness_probe=readiness_probe,
    )
    monkeypatch.setattr(runtime, "_reserve_service_ports", lambda: None)

    await runtime.start()

    assert runtime.seed_root == tmp_path / "source"
    assert runtime.urls == {service.gym_name: f"http://127.0.0.1:{service.port}" for service in SERVICES.values()}
    assert [sandbox.spec.ports for sandbox in created] == [(service.port,) for service in SERVICES.values()]
    expected_environment = {
        "csm": "API_BASE_URL=http://127.0.0.1:8001",
        "teams": "API_PORT=8002",
        "calendar": "API_PORT=8003",
        "email": "API_PORT=8004",
        "itsm": "ITSM_API_BASE_URL=http://127.0.0.1:8006",
        "hr": "HR_API_BASE_URL=http://127.0.0.1:8008",
        "drive": "FASTAPI_BASE_URL=http://127.0.0.1:8009 MCP_SERVER_HOST=127.0.0.1 MCP_SERVER_PORT=8009",
    }
    assert [sandbox.commands for sandbox in created] == [
        [
            (
                f"nohup env {expected_environment[service.domain]} python -m uvicorn "
                f"{service.app_target} --host 127.0.0.1 --port {service.port} "
                + f">/sandbox/{service.domain}.log 2>&1 &",
                "/app",
                30.0,
            )
        ]
        for service in SERVICES.values()
    ]
    assert ready_urls == [f"http://127.0.0.1:{service.port}" for service in SERVICES.values()]
    assert all(sandbox.started for sandbox in created)

    await runtime.stop()

    assert all(sandbox.stopped for sandbox in created)


@pytest.mark.asyncio
async def test_managed_services_accept_a_daemon_launch_timeout_when_the_endpoint_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "is_arm64_host", lambda: False)
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)

    async def seed_root() -> Path:
        return tmp_path / "source"

    async def sif_path(service) -> Path:
        return tmp_path / "images" / f"{service.domain}.sif"

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)
    monkeypatch.setattr(assets, "ensure_sif", sif_path)

    async def readiness_probe(_url: str) -> None:
        return None

    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_factory=TimeoutLaunchingSandbox,
        readiness_probe=readiness_probe,
    )
    monkeypatch.setattr(runtime, "_reserve_service_ports", lambda: None)

    await runtime.start()

    assert runtime.urls == {service.gym_name: f"http://127.0.0.1:{service.port}" for service in SERVICES.values()}


@pytest.mark.asyncio
async def test_managed_services_stop_the_current_sandbox_when_service_launch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "is_arm64_host", lambda: False)
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    created = []

    async def seed_root() -> Path:
        return tmp_path / "source"

    async def sif_path(service) -> Path:
        return tmp_path / "images" / f"{service.domain}.sif"

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)
    monkeypatch.setattr(assets, "ensure_sif", sif_path)

    def sandbox_factory(spec):
        sandbox = FailingLaunchingSandbox(spec)
        created.append(sandbox)
        return sandbox

    runtime = EnterpriseOpsServiceRuntime(assets=assets, sandbox_factory=sandbox_factory)
    monkeypatch.setattr(runtime, "_reserve_service_ports", lambda: None)

    with pytest.raises(RuntimeError, match="failed to start EnterpriseOps service csm"):
        await runtime.start()

    assert len(created) == 1
    assert created[0].stopped
