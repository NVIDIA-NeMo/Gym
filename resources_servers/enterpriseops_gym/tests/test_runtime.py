# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
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


class HeaderEndpointSandbox(FakeSandbox):
    async def endpoint(self, port: int) -> SandboxEndpoint:
        return SandboxEndpoint(endpoint=f"https://sandbox.example/services/{port}", headers={"X-Provider": "token"})


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


def test_managed_services_use_the_selected_sandbox_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    class CapturingSandbox:
        def __init__(self, provider, spec) -> None:
            captured["provider"] = provider
            captured["spec"] = spec

    monkeypatch.setattr(runtime_module, "AsyncSandbox", CapturingSandbox)
    spec = SandboxSpec(image="service.sif", ports=(8001,))
    provider = {"test-provider": {"connection": {"url": "https://sandbox.example"}}}
    runtime = EnterpriseOpsServiceRuntime(assets=EnterpriseOpsAssets(cache_dir=tmp_path), sandbox_provider=provider)

    runtime._create_sandbox(spec)

    assert captured["spec"] is spec
    assert captured["provider"] == provider


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
async def test_arm64_missing_native_images_fail_before_asset_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "aarch64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    created = []

    async def unexpected_seed_root() -> Path:
        raise AssertionError("ARM64 native-image validation must happen before downloading the seed archive")

    monkeypatch.setattr(assets, "ensure_seed_root", unexpected_seed_root)

    def sandbox_factory(spec):
        created.append(spec)
        return FakeSandbox(spec)

    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_provider={"test-provider": {}},
        sandbox_factory=sandbox_factory,
    )

    with pytest.raises(RuntimeError, match="missing native ARM64 EnterpriseOps service image") as error:
        await runtime.start()

    assert "csm" in str(error.value)
    assert created == []


@pytest.mark.asyncio
async def test_managed_services_declare_oci_images_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    created = []

    async def seed_root() -> Path:
        return tmp_path

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)

    def sandbox_factory(spec):
        sandbox = FakeSandbox(spec)
        created.append(sandbox)
        return sandbox

    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_provider={"test-provider": {}},
        sandbox_factory=sandbox_factory,
        readiness_probe=lambda _url: asyncio.sleep(0),
    )

    await runtime.start()

    csm = next(sandbox for sandbox in created if sandbox.spec.ports == (SERVICES["csm"].port,))
    assert csm.spec.image == SERVICES["csm"].image
    assert csm.spec.env["API_BASE_URL"] == "http://127.0.0.1:8001"
    assert csm.spec.env["NEMO_GYM_SERVICE_BIND_HOST"] == "127.0.0.1"

    await runtime.stop()


@pytest.mark.asyncio
async def test_managed_services_preserve_provider_endpoint_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)

    async def seed_root() -> Path:
        return tmp_path

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)
    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_provider={"test-provider": {}},
        sandbox_factory=HeaderEndpointSandbox,
        readiness_probe=lambda _url: asyncio.sleep(0),
    )

    await runtime.start()

    assert runtime.endpoint_headers["sn-csm-server"] == {"X-Provider": "token"}
    await runtime.stop()


@pytest.mark.asyncio
async def test_endpoint_readiness_uses_https_default_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Writer:
        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    async def open_connection(host: str, port: int):
        calls.append((host, port))
        return object(), Writer()

    monkeypatch.setattr(runtime_module.asyncio, "open_connection", open_connection)
    runtime = EnterpriseOpsServiceRuntime(
        assets=EnterpriseOpsAssets(cache_dir=tmp_path),
        sandbox_provider={"test-provider": {}},
    )

    await runtime._wait_for_endpoint("https://sandbox.example/services/8001")

    assert calls == [("sandbox.example", 443)]


def test_arm64_runtime_uses_configured_native_service_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "aarch64")
    native_image = tmp_path / "csm-arm64.sif"
    native_image.write_bytes(b"native-arm64-sif")
    runtime = EnterpriseOpsServiceRuntime(
        assets=EnterpriseOpsAssets(cache_dir=tmp_path),
        sandbox_provider={"test-provider": {}},
        native_service_images={"csm": str(native_image)},
    )

    assert runtime.service_image(SERVICES["csm"]) == str(native_image)


@pytest.mark.asyncio
async def test_managed_services_merge_generic_sandbox_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    created = []

    async def seed_root() -> Path:
        return tmp_path

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)

    def sandbox_factory(spec):
        sandbox = FakeSandbox(spec)
        created.append(sandbox)
        return sandbox

    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_provider={"test-provider": {}},
        sandbox_factory=sandbox_factory,
        readiness_probe=lambda _url: asyncio.sleep(0),
        sandbox_spec={
            "resources": {"cpu": 2, "memory_mib": 1024},
            "provider_options": {"resource_class": "small"},
            "workdir": "/app",
            "files": {"/sandbox/config.json": "{}"},
        },
    )

    await runtime.start()

    csm = next(sandbox for sandbox in created if sandbox.spec.ports == (SERVICES["csm"].port,))
    assert csm.spec.resources.cpu == 2
    assert csm.spec.resources.memory_mib == 1024
    assert csm.spec.provider_options == {"resource_class": "small"}
    assert csm.spec.workdir == "/app"
    assert csm.spec.files == {"/sandbox/config.json": "{}"}

    await runtime.stop()


def test_managed_services_reject_custom_sandbox_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    runtime = EnterpriseOpsServiceRuntime(
        assets=EnterpriseOpsAssets(cache_dir=tmp_path),
        sandbox_provider={"test-provider": {}},
        sandbox_spec={"entrypoint": ["/bin/sh"]},
    )

    with pytest.raises(ValueError, match="entrypoint"):
        runtime._build_sandbox_spec(SERVICES["csm"])


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
async def test_managed_services_start_from_provider_images_and_stop_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    created = []

    async def seed_root() -> Path:
        return tmp_path / "source"

    ready_urls = []

    async def readiness_probe(url: str) -> None:
        ready_urls.append(url)

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)

    def sandbox_factory(spec):
        sandbox = FakeSandbox(spec)
        created.append(sandbox)
        return sandbox

    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_provider={"test-provider": {}},
        sandbox_factory=sandbox_factory,
        readiness_probe=readiness_probe,
    )

    await runtime.start()

    assert runtime.seed_root == tmp_path / "source"
    assert runtime.urls == {service.gym_name: f"http://127.0.0.1:{service.port}" for service in SERVICES.values()}
    assert [sandbox.spec.ports for sandbox in created] == [(service.port,) for service in SERVICES.values()]
    assert [sandbox.commands for sandbox in created] == [
        [
            (
                f"nohup python -m uvicorn {service.app_target} "
                "--host $NEMO_GYM_SERVICE_BIND_HOST --port $NEMO_GYM_SERVICE_PORT "
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
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)

    async def seed_root() -> Path:
        return tmp_path / "source"

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)

    async def readiness_probe(_url: str) -> None:
        return None

    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_provider={"test-provider": {}},
        sandbox_factory=TimeoutLaunchingSandbox,
        readiness_probe=readiness_probe,
    )

    await runtime.start()

    assert runtime.urls == {service.gym_name: f"http://127.0.0.1:{service.port}" for service in SERVICES.values()}


@pytest.mark.asyncio
async def test_managed_services_stop_the_current_sandbox_when_service_launch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    assets = EnterpriseOpsAssets(cache_dir=tmp_path)
    created = []

    async def seed_root() -> Path:
        return tmp_path / "source"

    monkeypatch.setattr(assets, "ensure_seed_root", seed_root)

    def sandbox_factory(spec):
        sandbox = FailingLaunchingSandbox(spec)
        created.append(sandbox)
        return sandbox

    runtime = EnterpriseOpsServiceRuntime(
        assets=assets,
        sandbox_provider={"test-provider": {}},
        sandbox_factory=sandbox_factory,
    )

    with pytest.raises(RuntimeError, match="failed to start EnterpriseOps service csm"):
        await runtime.start()

    assert len(created) == 1
    assert created[0].stopped
