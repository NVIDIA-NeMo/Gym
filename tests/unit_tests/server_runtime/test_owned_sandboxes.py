# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a server holds the sandboxes it owns: ``SimpleServer.sandboxes``, created on first use and closed at shutdown."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Optional
from unittest.mock import MagicMock

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseRunServerInstanceConfig
from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.sandbox.providers.base import NETWORK_RULES, SandboxCapabilities, SandboxSpec
from nemo_gym.server_runtime import owned_sandboxes as owned_sandboxes_module
from nemo_gym.server_runtime.base_server import SimpleServer
from nemo_gym.server_runtime.owned_sandboxes import OwnedSandboxes, SandboxCheckpointConfig
from tests.unit_tests.server_runtime.test_checkpoint_sandbox import FakeSnapshotProvider


class _Config(BaseRunServerInstanceConfig):
    sandbox_provider: Optional[str] = None
    gpu_sandbox_provider: Optional[str] = None


class _Server(SimpleServer):
    config: _Config

    def model_post_init(self, context: Any, /) -> None:
        # Servers override this without calling the base; the sandboxes must still be there.
        self._started = True

    def setup_webserver(self) -> FastAPI:
        return FastAPI()


class _ClosableProvider(FakeSnapshotProvider):
    """Records whether its client was closed."""

    def __init__(self) -> None:
        super().__init__()
        self.client_closed = False

    async def aclose(self) -> None:
        self.client_closed = True


# The global config a server was started with; the fixture's resolver records which config it was handed.
SERVER_GLOBAL_CONFIG = {"opensandbox": {"opensandbox": {}}}


class _Built(dict[str, _ClosableProvider]):
    """The providers built, by block name, and the global configs their blocks were resolved from."""

    def __init__(self) -> None:
        super().__init__()
        self.resolved_from: list[Any] = []


@pytest.fixture
def providers(monkeypatch) -> _Built:
    """Every provider the servers build, by the name of the config block that named it."""
    built = _Built()
    resolved_from = built.resolved_from

    def resolve(reference: str, named_configs: Any = None) -> dict[str, Any]:
        resolved_from.append(named_configs)
        return {"fake": {"block": reference}}

    def create(config: dict[str, Any]) -> _ClosableProvider:
        provider = _ClosableProvider()
        built[config["fake"]["block"]] = provider
        return provider

    def global_config_of_the_process() -> Any:
        raise AssertionError("a server reads the config it was started with, not the process's")

    monkeypatch.setattr(owned_sandboxes_module, "resolve_provider_config", resolve)
    monkeypatch.setattr(owned_sandboxes_module, "create_provider", create)
    monkeypatch.setattr(
        owned_sandboxes_module,
        "resolve_provider_owner_concurrency",
        lambda reference, named: owned_sandboxes_module.OwnerConcurrency(),
    )
    monkeypatch.setattr(
        owned_sandboxes_module, "resolve_provider_metadata", lambda reference, named: {"provider_block": reference}
    )
    monkeypatch.setattr("nemo_gym.server_runtime.base_server.get_global_config_dict", global_config_of_the_process)
    return built


def _server(**config: Any) -> _Server:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = SERVER_GLOBAL_CONFIG
    return _Server(
        config=_Config(name="my_server", host="", port=0, entrypoint="", **config),
        server_client=server_client,
    )


def _spec() -> SandboxSpec:
    return SandboxSpec(image="image", ttl_s=60)


def test_the_sandboxes_are_created_once_from_the_servers_provider(providers) -> None:
    server = _server(sandbox_provider="opensandbox")

    owned = server.sandboxes

    assert isinstance(owned, OwnedSandboxes)
    assert server.sandboxes is owned
    assert owned.provider is providers["opensandbox"]


def test_a_server_that_never_uses_sandboxes_builds_no_provider(providers) -> None:
    server = _server(sandbox_provider="opensandbox")

    server.setup_owned_sandboxes(app := server.setup_webserver())
    with TestClient(app):
        pass

    assert providers == {}


def test_a_second_provider_field_gets_its_own_instance(providers) -> None:
    server = _server(sandbox_provider="opensandbox", gpu_sandbox_provider="gpu")

    gpu = server.sandboxes.on("gpu_sandbox_provider")

    assert gpu is not server.sandboxes
    assert gpu is server.sandboxes.on("gpu_sandbox_provider")
    assert gpu.provider is providers["gpu"]


def test_a_server_without_a_sandbox_provider_is_refused(providers) -> None:
    server = _server()

    with pytest.raises(ValueError, match="my_server sets no sandbox_provider"):
        server.sandboxes


async def test_a_non_exporting_owner_needs_no_checkpoint_state_dir(providers, monkeypatch) -> None:
    def no_state_dir(global_config: Any) -> None:
        raise ValueError("checkpoint.state_dir is required to checkpoint sandboxes")

    monkeypatch.setattr(owned_sandboxes_module, "checkpoint_state_dir", no_state_dir)
    owned = _server(sandbox_provider="opensandbox").sandboxes
    await owned.create("session", _spec())
    assert owned.indexed_snapshots == {}
    with pytest.raises(ValueError, match="checkpoint.state_dir is required"):
        await owned.export(["session"], checkpoint_id="checkpoint")
    assert providers["opensandbox"].snapshots == {}
    await owned.aclose()
    assert providers["opensandbox"].client_closed


@pytest.mark.parametrize(
    "resolver", ["resolve_provider_config", "resolve_provider_owner_concurrency", "resolve_provider_metadata"]
)
def test_a_refused_provider_config_builds_no_provider(providers, monkeypatch, resolver: str) -> None:
    def refuse(*args: Any) -> None:
        raise ValueError("unknown provider block")

    monkeypatch.setattr(owned_sandboxes_module, resolver, refuse)
    with pytest.raises(ValueError, match="unknown provider block"):
        _server(sandbox_provider="missing").sandboxes
    assert providers == {}


def test_an_instance_no_server_holds_has_no_other_providers() -> None:
    owned = OwnedSandboxes(FakeSnapshotProvider(), state_dir=None, server_name="server")

    with pytest.raises(RuntimeError, match="only an instance a server holds"):
        owned.on("gpu_sandbox_provider")


def test_the_shutdown_stops_every_sandbox_left_on_every_provider(providers) -> None:
    server = _server(sandbox_provider="opensandbox", gpu_sandbox_provider="gpu")
    app = server.setup_webserver()
    server.setup_owned_sandboxes(app)

    with TestClient(app) as client:
        client.portal.call(server.sandboxes.create, "session", _spec())
        client.portal.call(server.sandboxes.on("gpu_sandbox_provider").create, "session/gpu", _spec())

    assert [box["state"] for box in providers["opensandbox"].boxes.values()] == ["stopped"]
    assert [box["state"] for box in providers["gpu"].boxes.values()] == ["stopped"]
    assert server.sandboxes.keys == []


def test_access_lends_through_the_servers_provider_block(providers) -> None:
    server = _server(sandbox_provider="opensandbox")
    app = server.setup_webserver()

    with TestClient(app) as client:
        client.portal.call(server.sandboxes.create, "session", _spec())
        access = client.portal.call(lambda: server.sandboxes.access("session", workdir="/work"))

    assert access.connection.provider_config_ref == "opensandbox"
    assert access.connection.descriptor["scope"] == "operate"


def test_a_server_reads_the_config_it_was_started_with(providers) -> None:
    _server(sandbox_provider="opensandbox").sandboxes

    assert providers.resolved_from == [SERVER_GLOBAL_CONFIG]


def test_the_shutdown_closes_the_providers_the_server_made(providers) -> None:
    server = _server(sandbox_provider="opensandbox", gpu_sandbox_provider="gpu")
    app = server.setup_webserver()
    server.setup_owned_sandboxes(app)

    with TestClient(app) as client:
        client.portal.call(server.sandboxes.create, "session", _spec())
        server.sandboxes.on("gpu_sandbox_provider")

    assert providers["opensandbox"].client_closed
    assert providers["gpu"].client_closed


async def test_a_provider_stays_open_while_a_sandbox_is_left_to_stop(providers) -> None:
    owned = _server(sandbox_provider="opensandbox").sandboxes
    await owned.create("session", _spec())
    provider = providers["opensandbox"]
    provider.fail[("close", "sb-1")] = RuntimeError("provider unreachable")

    await owned.aclose()
    assert not provider.client_closed, "a later aclose can still retry the stop"

    await owned.aclose()
    assert provider.client_closed


async def test_each_provider_field_has_its_own_snapshot_index(providers, monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(owned_sandboxes_module, "checkpoint_state_dir", lambda config: state_dir)
    server = _server(sandbox_provider="opensandbox", gpu_sandbox_provider="gpu")

    main, gpu = server.sandboxes, server.sandboxes.on("gpu_sandbox_provider")

    assert not state_dir.exists()
    await main.export([], checkpoint_id="checkpoint")
    await gpu.delete_snapshots(frozenset())

    assert (state_dir / "my_server" / "snapshots").is_dir()
    assert (state_dir / "my_server.gpu_sandbox_provider" / "snapshots").is_dir()
    assert main.snapshot_name_prefix != gpu.snapshot_name_prefix


def test_a_field_that_is_not_a_provider_field_is_refused(providers) -> None:
    server = _server(sandbox_provider="opensandbox")

    with pytest.raises(ValueError, match="has no config field 'typo_provider'"):
        server.sandboxes.on("typo_provider")


def test_the_servers_own_shutdown_runs_before_its_sandboxes_are_closed(providers) -> None:
    server = _server(sandbox_provider="opensandbox")
    app = FastAPI()
    seen: list[str] = []

    @asynccontextmanager
    async def servers_own_lifespan(application: FastAPI) -> AsyncIterator[None]:
        yield
        # A server's own shutdown may still start or stop sandboxes; the sweep stops what it leaves.
        await server.sandboxes.create("from-own-shutdown", _spec())
        seen.append("own shutdown")

    app.router.lifespan_context = servers_own_lifespan
    server.setup_owned_sandboxes(app)
    with TestClient(app):
        pass

    assert seen == ["own shutdown"]
    assert [box["state"] for box in providers["opensandbox"].boxes.values()] == ["stopped"]


def test_no_instance_is_built_once_the_shutdown_began(providers) -> None:
    server = _server(sandbox_provider="opensandbox", gpu_sandbox_provider="gpu")
    app = server.setup_webserver()
    server.setup_owned_sandboxes(app)
    with TestClient(app):
        pass

    with pytest.raises(RuntimeError, match="shutting down"):
        server.sandboxes.on("gpu_sandbox_provider")


def test_the_shutdown_sweep_is_bounded(providers, monkeypatch, caplog) -> None:
    monkeypatch.setattr("nemo_gym.server_runtime.base_server.SANDBOX_SHUTDOWN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(owned_sandboxes_module, "STOP_TIMEOUT_S", 30)
    server = _server(sandbox_provider="opensandbox")
    app = server.setup_webserver()
    server.setup_owned_sandboxes(app)

    with caplog.at_level("ERROR", logger="nemo_gym.server_runtime.base_server"):
        with TestClient(app) as client:
            client.portal.call(server.sandboxes.create, "session", _spec())
            provider = providers["opensandbox"]

            async def hanging_close(handle: Any) -> None:
                await asyncio.sleep(10)

            provider.close = hanging_close  # type: ignore[method-assign]
            started = time.monotonic()

    assert time.monotonic() - started < 5
    assert "were not all stopped within" in caplog.text


def _run_webserver(monkeypatch, server_class: type, config: _Config) -> FastAPI:
    """Run ``server_class.run_webserver()`` as a launch does, and return the app it hands to uvicorn."""
    import nemo_gym.server_runtime.base_server as base_server_module
    import nemo_gym.server_runtime.ray_support as ray_support_module
    from nemo_gym.config.config_types import BaseServerConfig
    from nemo_gym.config.global_config import DRY_RUN_KEY_NAME

    monkeypatch.setattr(base_server_module, "get_global_config_dict", lambda: {DRY_RUN_KEY_NAME: False})
    monkeypatch.setattr(ray_support_module, "_get_ray", MagicMock(return_value=MagicMock()))
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = SERVER_GLOBAL_CONFIG
    server_client_class = MagicMock(return_value=server_client)
    server_client_class.load_head_server_config = MagicMock(return_value=BaseServerConfig(host="", port=0))
    monkeypatch.setattr(base_server_module, "ServerClient", server_client_class)
    monkeypatch.setattr(base_server_module, "is_nemo_gym_fastapi_worker", MagicMock(return_value=False))
    launched: dict[str, Any] = {}
    monkeypatch.setattr(base_server_module.uvicorn, "run", lambda **kwargs: launched.update(kwargs))
    monkeypatch.setattr(server_class, "load_config_from_global_config", classmethod(lambda cls: config))
    for quiet in ("setup_telemetry", "set_ulimit", "prefix_server_logs"):
        monkeypatch.setattr(server_class, quiet, lambda self: None)
    for quiet in ("setup_exception_middleware", "setup_cancellation_middleware", "instrument_app_for_telemetry"):
        monkeypatch.setattr(server_class, quiet, lambda self, app: None)

    server_class.run_webserver()
    return launched["app"]


class _LaunchedServer(_Server):
    """Remembers the instance run_webserver built, so a test can reach its sandboxes."""

    launched: Any = None

    def setup_webserver(self) -> FastAPI:
        type(self).launched = self
        return FastAPI()


def test_a_launched_servers_shutdown_stops_the_sandboxes_it_owns(providers, monkeypatch) -> None:
    config = _Config(
        name="my_server", host="127.0.0.1", port=8000, entrypoint="app.py", sandbox_provider="opensandbox"
    )

    app = _run_webserver(monkeypatch, _LaunchedServer, config)
    with TestClient(app) as client:
        client.portal.call(_LaunchedServer.launched.sandboxes.create, "session", _spec())

    assert [box["state"] for box in providers["opensandbox"].boxes.values()] == ["stopped"]
    assert providers["opensandbox"].client_closed


class _GatedProvider(FakeSnapshotProvider):
    """Holds every create and close until its gate opens, and counts how many run at once."""

    def __init__(self) -> None:
        super().__init__()
        self.create_gate = asyncio.Event()
        self.create_gate.set()
        self.close_gate = asyncio.Event()
        self.close_gate.set()
        self.in_flight = 0
        self.max_in_flight = 0

    async def _gated(self, gate: asyncio.Event) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await gate.wait()
        finally:
            self.in_flight -= 1

    async def create(self, spec: SandboxSpec) -> Any:
        await self._gated(self.create_gate)
        return await super().create(spec)

    async def close(self, handle: Any) -> None:
        await self._gated(self.close_gate)
        await super().close(handle)


def _owned(provider: FakeSnapshotProvider, state_dir: Optional[Path] = None, **kwargs: Any) -> OwnedSandboxes:
    kwargs.setdefault("snapshot_retry_delay_s", 0.0)
    return OwnedSandboxes(provider, state_dir=state_dir, server_name="server", **kwargs)


def _states(provider: FakeSnapshotProvider) -> list[str]:
    return [box["state"] for box in provider.boxes.values()]


async def _until(condition: Any) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.001)


class TestStop:
    """A client that disconnects cancels its request's task group; a stop the request started must still finish."""

    async def test_a_stop_in_a_cancelled_request_still_stops_and_forgets_the_sandbox(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider)
        await owned.create("session", _spec())
        provider.close_gate.clear()

        async with anyio.create_task_group() as request:
            request.start_soon(owned.stop, "session")
            await _until(lambda: provider.in_flight == 1)
            request.cancel_scope.cancel()
            provider.close_gate.set()

        assert _states(provider) == ["stopped"]
        assert owned.keys == []

    async def test_a_stop_cancelled_while_it_waits_for_a_slot_still_runs(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider, max_concurrent_stops=1)
        await owned.create("first", _spec())
        await owned.create("second", _spec())
        provider.close_gate.clear()
        holding = asyncio.create_task(owned.stop("first"))
        await _until(lambda: provider.in_flight == 1)

        async with anyio.create_task_group() as request:
            request.start_soon(owned.stop, "second")
            await asyncio.sleep(0.01)
            request.cancel_scope.cancel()
            provider.close_gate.set()
        await holding

        assert _states(provider) == ["stopped", "stopped"]
        assert owned.keys == []


class TestStartsAndStopsHaveTheirOwnLimits:
    async def test_starts_are_not_bounded_by_the_snapshot_parallelism(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider, parallelism=1)
        provider.create_gate.clear()

        creates = [asyncio.create_task(owned.create(f"session-{index}", _spec())) for index in range(4)]
        await _until(lambda: provider.in_flight == 4)
        provider.create_gate.set()
        await asyncio.gather(*creates)

        assert provider.max_in_flight == 4

    async def test_the_start_and_stop_limit_bounds_them(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider, max_concurrent_starts=2)
        provider.create_gate.clear()

        creates = [asyncio.create_task(owned.create(f"session-{index}", _spec())) for index in range(4)]
        await _until(lambda: provider.in_flight == 2)
        await asyncio.sleep(0.01)
        provider.create_gate.set()
        await asyncio.gather(*creates)

        assert provider.max_in_flight == 2

    def test_a_limit_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            _owned(FakeSnapshotProvider(), max_concurrent_starts=0)
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            _owned(FakeSnapshotProvider(), max_concurrent_stops=0)


class TestTrackedBeforeStart:
    async def test_a_start_that_fails_is_stopped_and_forgotten(self) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider)
        spec = SandboxSpec(image="image", ttl_s=60, files={"/task": "x"})

        async def failing_upload(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("upload refused")

        provider.upload_file = failing_upload  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="upload refused"):
            await owned.create("session", spec)

        assert _states(provider) == ["stopped"]
        assert owned.keys == []
        await owned.create("session", _spec())

    async def test_a_start_cancelled_by_its_request_is_stopped(self) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider)
        spec = SandboxSpec(image="image", ttl_s=60, files={"/task": "x"})
        uploading = asyncio.Event()
        real_close = provider.close

        async def hanging_upload(*args: Any, **kwargs: Any) -> None:
            uploading.set()
            await asyncio.Event().wait()

        async def slow_close(handle: Any) -> None:
            # A stop that has to await, so the cancelled request cuts the start's own stop short.
            await asyncio.sleep(0.01)
            await real_close(handle)

        provider.upload_file = hanging_upload  # type: ignore[method-assign]
        provider.close = slow_close  # type: ignore[method-assign]
        async with anyio.create_task_group() as request:
            request.start_soon(owned.create, "session", spec)
            await asyncio.wait_for(uploading.wait(), timeout=5)
            request.cancel_scope.cancel()

        assert _states(provider) == ["stopped"]
        assert owned.keys == []

    async def test_a_start_cancelled_inside_the_providers_create_is_let_finish_and_stopped(self) -> None:
        # The provider has made the sandbox before its create returns (Enroot's rootfs, a container still starting).
        # Cut short there, the create would leave a sandbox with no handle, which a provider without ttl_s never ends.
        provider = FakeSnapshotProvider()
        owned = _owned(provider)
        made = asyncio.Event()
        ready = asyncio.Event()
        real_create = provider.create

        async def create(spec: SandboxSpec) -> Any:
            handle = await real_create(spec)
            made.set()
            await ready.wait()
            return handle

        provider.create = create  # type: ignore[method-assign]
        async with anyio.create_task_group() as request:
            request.start_soon(owned.create, "session", _spec())
            await asyncio.wait_for(made.wait(), timeout=5)
            request.cancel_scope.cancel()
            ready.set()

        assert _states(provider) == ["stopped"]
        assert owned.keys == []

    async def test_a_second_cancellation_does_not_end_the_wait_for_the_start(self) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider)
        made = asyncio.Event()
        ready = asyncio.Event()
        real_create = provider.create

        async def create(spec: SandboxSpec) -> Any:
            handle = await real_create(spec)
            made.set()
            await ready.wait()
            return handle

        provider.create = create  # type: ignore[method-assign]
        creating = asyncio.create_task(owned.create("session", _spec()))
        await asyncio.wait_for(made.wait(), timeout=5)
        creating.cancel()
        await asyncio.sleep(0.01)
        creating.cancel()
        await asyncio.sleep(0.01)
        assert not creating.done(), "the create still waits for its start"
        ready.set()
        with pytest.raises(asyncio.CancelledError):
            await creating

        assert _states(provider) == ["stopped"]
        assert owned.keys == []

    async def test_a_cancelled_start_that_never_returns_is_cut_short_after_the_stop_limit(self, monkeypatch) -> None:
        monkeypatch.setattr(owned_sandboxes_module, "STOP_TIMEOUT_S", 0.05)
        provider = FakeSnapshotProvider()
        owned = _owned(provider)
        made = asyncio.Event()
        real_create = provider.create

        async def create(spec: SandboxSpec) -> Any:
            handle = await real_create(spec)
            made.set()
            await asyncio.Event().wait()
            return handle

        provider.create = create  # type: ignore[method-assign]
        async with asyncio.timeout(5):
            async with anyio.create_task_group() as request:
                request.start_soon(owned.create, "session", _spec())
                await asyncio.wait_for(made.wait(), timeout=5)
                request.cancel_scope.cancel()

        assert owned.keys == []

    async def test_a_failed_start_whose_stop_fails_stays_owned_until_a_stop_succeeds(self) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider)
        spec = SandboxSpec(image="image", ttl_s=60, files={"/task": "x"})
        real_close = provider.close
        unreachable = True

        async def close(handle: Any) -> None:
            if unreachable:
                raise RuntimeError("provider unreachable")
            await real_close(handle)

        async def failing_upload(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("upload refused")

        provider.close = close  # type: ignore[method-assign]
        provider.upload_file = failing_upload  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="upload refused") as raised:
            await owned.create("session", spec)

        assert str(raised.value.__context__) == "provider unreachable", "the failed stop rides along as context"
        assert owned.keys == ["session"], "the sandbox whose stop failed stays owned"
        assert owned.get("session") is None
        assert "session" not in owned
        assert _states(provider) == ["running"]
        unreachable = False
        await owned.stop("session")
        assert owned.keys == []
        assert _states(provider) == ["stopped"]


class TestExportAndCreateOfOneKey:
    async def test_an_export_waits_for_the_keys_create_and_snapshots_it(self, tmp_path: Path) -> None:
        provider = _GatedProvider()
        owned = _owned(provider, tmp_path / "state")
        provider.create_gate.clear()
        create = asyncio.create_task(owned.create("session", _spec()))
        await _until(lambda: provider.in_flight == 1)

        export = asyncio.create_task(owned.export(["session"], checkpoint_id="c1"))
        await asyncio.sleep(0.01)
        assert not export.done()
        provider.create_gate.set()
        await create

        assert (await export)["session"] is not None

    async def test_an_export_holds_back_only_the_keys_it_exports(self, tmp_path: Path) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider, tmp_path / "state")
        await owned.create("exported", _spec())
        provider.submit_gate = asyncio.Event()
        export = asyncio.create_task(owned.export(["exported"], checkpoint_id="c1"))
        await _until(lambda: provider.in_flight_submits == 1)

        await asyncio.wait_for(owned.create("other", _spec()), timeout=5)

        assert not export.done()
        provider.submit_gate.set()
        assert (await export)["exported"] is not None

    async def test_a_stop_waits_for_the_export_of_its_key(self, tmp_path: Path) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider, tmp_path / "state")
        await owned.create("session", _spec())
        provider.submit_gate = asyncio.Event()
        export = asyncio.create_task(owned.export(["session"], checkpoint_id="c1"))
        await _until(lambda: provider.in_flight_submits == 1)

        stop = asyncio.create_task(owned.stop("session"))
        await asyncio.sleep(0.01)
        assert _states(provider) == ["running"], "the sandbox is not stopped under its snapshot"
        provider.submit_gate.set()
        await stop

        assert (await export)["session"] is not None
        assert _states(provider) == ["stopped"]

    async def test_the_key_locks_are_forgotten_once_nobody_holds_them(self, tmp_path: Path) -> None:
        owned = _owned(FakeSnapshotProvider(), tmp_path / "state")
        await owned.create("session", _spec())
        await owned.export(["session", "never-started"], checkpoint_id="c1")
        await owned.stop("session")

        assert len(owned._key_locks) == 0


class TestEveryStopIsShieldedAndBoundedFromItsCall:
    async def test_a_stop_cancelled_while_it_waits_for_its_keys_export_still_runs(self, tmp_path: Path) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider, tmp_path / "state")
        await owned.create("session", _spec())
        provider.submit_gate = asyncio.Event()
        export = asyncio.create_task(owned.export(["session"], checkpoint_id="c1"))
        await _until(lambda: provider.in_flight_submits == 1)

        async with anyio.create_task_group() as request:
            request.start_soon(owned.stop, "session")
            await asyncio.sleep(0.01)
            request.cancel_scope.cancel()
            provider.submit_gate.set()

        assert (await export)["session"] is not None
        assert _states(provider) == ["stopped"]
        assert owned.keys == []

    async def test_a_stop_waiting_for_its_keys_lock_is_bounded(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(owned_sandboxes_module, "STOP_TIMEOUT_S", 0.05)
        provider = FakeSnapshotProvider()
        owned = _owned(provider, tmp_path / "state")
        await owned.create("session", _spec())
        provider.submit_gate = asyncio.Event()
        export = asyncio.create_task(owned.export(["session"], checkpoint_id="c1"))
        await _until(lambda: provider.in_flight_submits == 1)

        started = time.monotonic()
        # The shield holds off an anyio deadline, so a native one guards the test: without the stop's own bound it
        # fails after 5 s instead of hanging.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(owned.stop("session"), timeout=5)

        assert time.monotonic() - started < 2, "the stop's own bound ended it"
        assert owned.keys == ["session"], "a stop that timed out keeps its key"
        provider.submit_gate.set()
        await export
        await owned.stop("session")
        assert owned.keys == []

    async def test_a_stop_does_not_wait_behind_starts(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider, max_concurrent_starts=1, max_concurrent_stops=1)
        await owned.create("running", _spec())
        provider.create_gate.clear()
        hung = asyncio.create_task(owned.create("hung", _spec()))
        await _until(lambda: provider.in_flight == 1)

        await asyncio.wait_for(owned.stop("running"), timeout=5)

        assert owned.keys == ["hung"]
        provider.create_gate.set()
        await hung
        await owned.aclose()

    async def test_stops_take_the_stop_limit(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider, max_concurrent_stops=1)
        for index in range(3):
            await owned.create(f"session-{index}", _spec())
        provider.close_gate.clear()

        stops = [asyncio.create_task(owned.stop(f"session-{index}")) for index in range(3)]
        await _until(lambda: provider.in_flight == 1)
        await asyncio.sleep(0.01)
        provider.close_gate.set()
        await asyncio.gather(*stops)

        assert provider.max_in_flight == 1

    async def test_a_stop_queued_for_a_slot_longer_than_the_limit_still_stops(self, monkeypatch) -> None:
        # Each stop ahead holds its slot for at most the limit, so a queued stop is not cut short by the queue: under
        # load that would leave sandboxes running until their ttl_s.
        monkeypatch.setattr(owned_sandboxes_module, "STOP_TIMEOUT_S", 0.2)
        provider = _GatedProvider()
        owned = _owned(provider, max_concurrent_stops=1)
        await owned.create("session", _spec())
        assert owned._stop_limit is not None
        await owned._stop_limit.acquire()
        asyncio.get_running_loop().call_later(0.5, owned._stop_limit.release)

        await asyncio.wait_for(owned.stop("session"), timeout=5)

        assert owned.keys == []
        assert _states(provider) == ["stopped"]

    async def test_the_providers_stop_is_bounded_once_it_has_its_slot(self, monkeypatch) -> None:
        monkeypatch.setattr(owned_sandboxes_module, "STOP_TIMEOUT_S", 0.2)
        provider = _GatedProvider()
        owned = _owned(provider, max_concurrent_stops=1)
        await owned.create("session", _spec())
        provider.close_gate.clear()

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(owned.stop("session"), timeout=5)

        assert time.monotonic() - started < 2
        assert owned.keys == ["session"], "a stop that timed out keeps its key"
        provider.close_gate.set()
        await owned.stop("session")
        assert owned.keys == []


class TestTheOwnersOwnChecks:
    async def test_a_sandbox_without_ttl_s_is_refused_before_it_starts(self) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider)

        with pytest.raises(ValueError, match="needs a ttl_s"):
            await owned.create("session", SandboxSpec(image="image"))

        assert provider.boxes == {}
        assert owned.keys == []

    async def test_a_create_queued_for_a_slot_when_the_shutdown_began_starts_nothing(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider, max_concurrent_starts=1)
        provider.create_gate.clear()
        first = asyncio.create_task(owned.create("first", _spec()))
        await _until(lambda: provider.in_flight == 1)
        queued = asyncio.create_task(owned.create("queued", _spec()))
        await asyncio.sleep(0.01)

        closing = asyncio.create_task(owned.aclose())
        await asyncio.sleep(0.01)
        provider.create_gate.set()
        await first
        with pytest.raises(RuntimeError, match="closed"):
            await queued
        await closing

        assert len(provider.boxes) == 1, "only the create that had its slot started a sandbox"
        assert _states(provider) == ["stopped"]
        assert owned.keys == []

    async def test_the_shutdown_stops_what_runs_before_it_waits_for_a_pending_create(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider)
        await owned.create("running", _spec())
        provider.create_gate.clear()
        pending = asyncio.create_task(owned.create("pending", _spec()))
        await _until(lambda: provider.in_flight == 1)

        closing = asyncio.create_task(owned.aclose())
        await _until(lambda: "running" not in owned)
        assert not closing.done(), "the shutdown still waits for the pending create"
        provider.create_gate.set()
        await pending
        await closing

        assert owned.keys == []
        assert _states(provider) == ["stopped", "stopped"]

    async def test_a_sandbox_still_starting_is_not_lent(self) -> None:
        provider = _GatedProvider()
        owned = _owned(provider, provider_config_ref="sandbox")
        provider.create_gate.clear()
        creating = asyncio.create_task(owned.create("session", _spec()))
        await _until(lambda: provider.in_flight == 1)

        with pytest.raises(KeyError, match="has no running sandbox"):
            await owned.access("session", workdir="/work")

        provider.create_gate.set()
        await creating
        assert (await owned.access("session", workdir="/work")).workdir == "/work"
        await owned.aclose()

    async def test_an_instance_from_a_checkpoint_config_bounds_starts_and_stops_by_its_parallelism(self) -> None:
        provider = _GatedProvider()
        owned = OwnedSandboxes.from_config(
            provider, SandboxCheckpointConfig(parallelism=1), state_dir=None, server_name="server"
        )
        provider.create_gate.clear()
        creates = [asyncio.create_task(owned.create(f"session-{index}", _spec())) for index in range(3)]
        await _until(lambda: provider.in_flight == 1)
        await asyncio.sleep(0.01)
        provider.create_gate.set()
        await asyncio.gather(*creates)
        provider.close_gate.clear()
        stops = [asyncio.create_task(owned.stop(f"session-{index}")) for index in range(3)]
        await _until(lambda: provider.in_flight == 1)
        await asyncio.sleep(0.01)
        provider.close_gate.set()
        await asyncio.gather(*stops)

        assert provider.max_in_flight == 1
        assert owned.keys == []


class TestAFailedStartWhoseStopFails:
    async def test_it_is_never_exported(self, tmp_path: Path) -> None:
        provider = FakeSnapshotProvider()
        owned = _owned(provider, tmp_path / "state")
        real_close = provider.close

        async def unreachable(handle: Any) -> None:
            raise RuntimeError("provider unreachable")

        async def failing_upload(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("upload refused")

        provider.close = unreachable  # type: ignore[method-assign]
        provider.upload_file = failing_upload  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="upload refused"):
            await owned.create("session", SandboxSpec(image="image", ttl_s=60, files={"/task": "x"}))

        assert await owned.export(["session"], checkpoint_id="c1") == {"session": None}
        provider.close = real_close  # type: ignore[method-assign]
        await owned.stop("session")
        assert owned.keys == []


class TestStopDuringRestore:
    async def test_a_stop_waits_for_the_keys_restore_and_stops_its_fork(self, tmp_path: Path) -> None:
        provider = FakeSnapshotProvider()
        first = _owned(provider, tmp_path / "state")
        await first.create("session", _spec())
        states = await first.export(["session"], checkpoint_id="c1")
        gate = asyncio.Event()
        real_fork = provider.create_from_snapshot

        async def slow_fork(snapshot_id: str, spec: SandboxSpec) -> Any:
            await gate.wait()
            return await real_fork(snapshot_id, spec)

        provider.create_from_snapshot = slow_fork  # type: ignore[method-assign]
        fresh = _owned(provider, tmp_path / "state")
        restore = asyncio.create_task(fresh.restore(states))
        await asyncio.sleep(0.01)
        stop = asyncio.create_task(fresh.stop("session"))
        await asyncio.sleep(0.01)
        assert not stop.done(), "the stop waits for the restore of its key"

        gate.set()
        await restore
        await stop
        assert fresh.keys == []
        fork = [box for box in provider.boxes.values() if box["snapshot_id"] is not None]
        assert [box["state"] for box in fork] == ["stopped"]

    async def test_a_restore_cancelled_inside_a_forks_create_stops_the_fork(self, tmp_path: Path) -> None:
        # The provider has made the fork before its create returns; cut short there, the fork would run on untracked.
        provider = FakeSnapshotProvider()
        first = _owned(provider, tmp_path / "state")
        await first.create("session", _spec())
        states = await first.export(["session"], checkpoint_id="c1")
        made = asyncio.Event()
        ready = asyncio.Event()
        real_fork = provider.create_from_snapshot

        async def fork(snapshot_id: str, spec: SandboxSpec) -> Any:
            handle = await real_fork(snapshot_id, spec)
            made.set()
            await ready.wait()
            return handle

        provider.create_from_snapshot = fork  # type: ignore[method-assign]
        fresh = _owned(provider, tmp_path / "state")
        async with anyio.create_task_group() as request:
            request.start_soon(fresh.restore, states)
            await asyncio.wait_for(made.wait(), timeout=5)
            request.cancel_scope.cancel()
            ready.set()
        await fresh.settle()

        forks = [box for box in provider.boxes.values() if box["snapshot_id"] is not None]
        assert [box["state"] for box in forks] == ["stopped"]
        assert fresh.keys == []

    async def test_forks_take_the_start_limit(self, tmp_path: Path) -> None:
        provider = FakeSnapshotProvider()
        first = _owned(provider, tmp_path / "state")
        for index in range(3):
            await first.create(f"session-{index}", _spec())
        states = await first.export([f"session-{index}" for index in range(3)], checkpoint_id="c1")
        gate = asyncio.Event()
        in_flight = 0
        peak = 0
        real_fork = provider.create_from_snapshot

        async def counted_fork(snapshot_id: str, spec: SandboxSpec) -> Any:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await gate.wait()
                return await real_fork(snapshot_id, spec)
            finally:
                in_flight -= 1

        provider.create_from_snapshot = counted_fork  # type: ignore[method-assign]
        fresh = _owned(provider, tmp_path / "state", max_concurrent_starts=1)
        restore = asyncio.create_task(fresh.restore(states))
        await _until(lambda: in_flight == 1)
        await asyncio.sleep(0.01)
        gate.set()
        await restore

        assert peak == 1
        assert sorted(fresh.keys) == [f"session-{index}" for index in range(3)]


async def test_a_keys_lock_is_never_held_twice_when_a_third_caller_arrives() -> None:
    # A holds the lock and B waits; A releases, and C arrives while B holds it. The lock entry must outlive A.
    locks = owned_sandboxes_module._KeyLocks()
    held = 0
    peak = 0
    a_in, a_go, b_in = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def hold(entered: asyncio.Event, release: Optional[asyncio.Event]) -> None:
        nonlocal held, peak
        async with locks.locked("key"):
            held += 1
            peak = max(peak, held)
            entered.set()
            if release is not None:
                await release.wait()
            else:
                await asyncio.sleep(0.05)
            held -= 1

    a = asyncio.create_task(hold(a_in, a_go))
    await a_in.wait()
    b = asyncio.create_task(hold(b_in, None))
    await asyncio.sleep(0.01)
    a_go.set()
    await b_in.wait()
    c = asyncio.create_task(hold(asyncio.Event(), None))
    await asyncio.gather(a, b, c)

    assert peak == 1
    assert len(locks) == 0


def test_a_server_reads_its_providers_owner_concurrency(providers, monkeypatch) -> None:
    monkeypatch.setattr(
        owned_sandboxes_module,
        "resolve_provider_owner_concurrency",
        lambda reference, named: owned_sandboxes_module.OwnerConcurrency(starts=4, stops=2),
    )
    owned = _server(sandbox_provider="opensandbox").sandboxes

    assert owned._start_limit is not None and owned._start_limit._value == 4
    assert owned._stop_limit is not None and owned._stop_limit._value == 2


async def test_a_superseded_stop_waiting_for_a_slot_is_bounded(monkeypatch, caplog) -> None:
    monkeypatch.setattr(owned_sandboxes_module, "STOP_TIMEOUT_S", 0.05)
    provider = _GatedProvider()
    owned = _owned(provider, max_concurrent_stops=1)
    await owned.create("superseded", _spec())
    superseded_id = owned.get("superseded").sandbox_id
    # Every stop slot is taken, for longer than a stop may take.
    await owned._stop_limit.acquire()

    started = time.monotonic()
    with caplog.at_level("INFO", logger=owned_sandboxes_module.LOGGER.name):
        await asyncio.wait_for(
            owned._stop_superseded("key", SimpleNamespace(descriptor={"sandbox_id": superseded_id})), timeout=5
        )

    assert time.monotonic() - started < 2
    assert "was not stopped" in caplog.text
    owned._stop_limit.release()


class _RuleProvider(FakeSnapshotProvider):
    """A provider that applies every network rule, the way OpenSandbox does, and records what it was asked."""

    capabilities = SandboxCapabilities(network_rules=NETWORK_RULES)

    def network_options(self, rule: str, hosts: Any, provider_options: Any) -> dict[str, Any]:
        return {**provider_options, "network_policy": {"rule": rule, "hosts": list(hosts)}}


class _NeverReconnects(FakeSnapshotProvider):
    capabilities = SandboxCapabilities(reconnect="never")


def _servers_at(**hosts: str) -> dict[str, Any]:
    """A global config that defines each named server, the way a launch config nests it, on the given host."""
    return {name: {"servers": {name: {"host": host, "port": 8000}}} for name, host in hosts.items()}


class TestSpec:
    def _owned(self, provider: Any, global_config: Optional[dict[str, Any]] = None) -> OwnedSandboxes:
        return OwnedSandboxes(
            provider,
            state_dir=None,
            server_name="server",
            default_metadata={"team": "gym", "run": "provider"},
            global_config=global_config or _servers_at(policy_model="10.0.0.5", search="search.internal"),
        )

    def test_a_restricted_rule_allows_the_servers_hosts_and_the_listed_hosts(self) -> None:
        config = SandboxConfig(
            network_access="model_and_tools",
            model_server="policy_model",
            tool_servers=["search"],
            allowed_hosts=["proxy.internal"],
            ttl_s=60,
            metadata={"run": "config"},
            provider_options={"volumes": ["v"]},
        )

        spec = self._owned(_RuleProvider()).spec(config, image="image", workdir="/work", metadata={"task": "t1"})

        assert spec.provider_options == {
            "volumes": ["v"],
            "network_policy": {
                "rule": "model_and_tools",
                "hosts": ["10.0.0.5", "search.internal", "proxy.internal"],
            },
        }
        assert spec.metadata == {"team": "gym", "run": "config", "task": "t1"}
        assert (spec.image, spec.workdir, spec.ttl_s) == ("image", "/work", 60)

    def test_inherit_leaves_the_provider_options_as_they_are(self) -> None:
        config = SandboxConfig(network_access="inherit", provider_options={"volumes": ["v"]})

        spec = self._owned(_RuleProvider()).spec(config, image="image", workdir=None)

        assert spec.provider_options == {"volumes": ["v"]}

    def test_a_rule_the_provider_cannot_apply_is_refused(self) -> None:
        config = SandboxConfig(network_access="no_network")

        with pytest.raises(ValueError, match="cannot apply network 'no_network'"):
            self._owned(FakeSnapshotProvider()).spec(config, image="image", workdir=None)

    def test_a_server_on_a_loopback_address_is_refused(self) -> None:
        config = SandboxConfig(network_access="model_only", model_server="policy_model")
        owned = self._owned(_RuleProvider(), _servers_at(policy_model="127.0.0.1"))

        with pytest.raises(ValueError, match="sandbox-reachable host"):
            owned.spec(config, image="image", workdir=None)

    def test_a_server_the_config_does_not_define_is_named(self) -> None:
        config = SandboxConfig(network_access="model_only", model_server="typo_model")

        with pytest.raises(ValueError, match="names server 'typo_model', for which the config defines no host"):
            self._owned(_RuleProvider()).spec(config, image="image", workdir=None)

    def test_the_servers_are_read_from_the_config_the_instance_was_built_with(self, monkeypatch) -> None:
        def process_config() -> Any:
            raise AssertionError("the config the server was started with, not the process's")

        monkeypatch.setattr(owned_sandboxes_module, "get_global_config_dict", process_config)
        config = SandboxConfig(network_access="model_only", model_server="policy_model")

        spec = self._owned(_RuleProvider()).spec(config, image="image", workdir=None)

        assert spec.provider_options["network_policy"]["hosts"] == ["10.0.0.5"]

    def test_a_block_without_a_rule_keeps_its_options_and_warns_once_per_config_class(self, caplog) -> None:
        owned = self._owned(_RuleProvider())
        config = SandboxConfig(provider_options={"network_policy": {"defaultAction": "deny"}})

        with caplog.at_level("WARNING", logger=owned_sandboxes_module.LOGGER.name):
            first = owned.spec(config, image="image", workdir=None)
            owned.spec(config, image="image", workdir=None)

        assert first.provider_options == {"network_policy": {"defaultAction": "deny"}}
        assert caplog.text.count("sets no network rule") == 1


class TestCheckProvider:
    def test_each_rule_is_checked_on_its_own(self) -> None:
        owned = OwnedSandboxes(FakeSnapshotProvider(), state_dir=None, server_name="server")

        owned.check_provider(configs=[SandboxConfig(network_access="inherit"), SandboxConfig()])
        with pytest.raises(ValueError, match="cannot apply network 'no_network'"):
            owned.check_provider(configs=[SandboxConfig(network_access="inherit"), SandboxConfig(network_access="no_network")])

    def test_lending_is_refused_on_a_provider_that_never_reconnects(self) -> None:
        owned = OwnedSandboxes(_NeverReconnects(), state_dir=None, server_name="server")

        owned.check_provider()
        with pytest.raises(ValueError, match="cannot lend a sandbox"):
            owned.check_provider(lends=True)

    def test_lending_is_refused_on_a_provider_that_declares_nothing_and_cannot_reconnect(self) -> None:
        # A provider outside Gym that declares no capabilities reads reconnect="always"; only its methods tell.
        class _NoReconnect:
            name = "no_reconnect"

        owned = OwnedSandboxes(_NoReconnect(), state_dir=None, server_name="server")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="cannot lend a sandbox"):
            owned.check_provider(lends=True)

    async def test_the_first_access_refuses_too(self) -> None:
        owned = OwnedSandboxes(_NeverReconnects(), state_dir=None, server_name="server", provider_config_ref="p")
        await owned.create("session", _spec())

        with pytest.raises(ValueError, match="cannot lend a sandbox"):
            await owned.access("session", workdir="/work")
        await owned.aclose()


class _TypedConfig(_Config):
    sandbox_config: SandboxConfig = SandboxConfig()
    grader_sandbox_config: SandboxConfig = SandboxConfig(network_access="inherit")


class _TypedServer(_Server):
    config: _TypedConfig
    sandbox_config_providers = {"sandbox_config": "sandbox_provider", "grader_sandbox_config": "sandbox_provider"}


class _LendingServer(_Server):
    lends_sandboxes = True


class TestStartupCheck:
    @pytest.fixture(autouse=True)
    def provider_classes(self, monkeypatch) -> None:
        classes = {"rules": _RuleProvider, "plain": FakeSnapshotProvider, "never": _NeverReconnects}
        monkeypatch.setattr(
            "nemo_gym.sandbox.config.resolve_provider_config", lambda reference, named: {reference: {}}, raising=True
        )
        monkeypatch.setattr("nemo_gym.sandbox.providers.registry.get_provider_class", classes.__getitem__)
        monkeypatch.setattr("nemo_gym.server_runtime.base_server.get_global_config_dict", lambda: {})

    def _typed(self, provider: str, rule: str) -> _TypedServer:
        return _TypedServer(
            config=_TypedConfig(
                name="my_server",
                host="",
                port=0,
                entrypoint="",
                sandbox_provider=provider,
                sandbox_config=SandboxConfig(network_access=rule),
            ),
            server_client=MagicMock(spec=ServerClient),
        )

    def test_a_rule_the_provider_cannot_apply_fails_the_startup(self) -> None:
        with pytest.raises(ValueError, match="cannot apply network 'no_network'"):
            self._typed("plain", "no_network").check_sandbox_provider()

    def test_rules_the_provider_applies_pass(self) -> None:
        self._typed("rules", "no_network").check_sandbox_provider()

    def test_a_lending_server_on_a_provider_that_never_reconnects_fails_the_startup(self) -> None:
        server = _LendingServer(
            config=_Config(name="my_server", host="", port=0, entrypoint="", sandbox_provider="never"),
            server_client=MagicMock(spec=ServerClient),
        )

        with pytest.raises(ValueError, match="cannot lend a sandbox"):
            server.check_sandbox_provider()

    def test_a_server_without_typed_blocks_or_lending_checks_nothing(self, providers) -> None:
        _server(sandbox_provider="never").check_sandbox_provider()

        assert providers == {}


class _OwnNetworkConfig(SandboxConfig):
    """A server's block with its own network setting, as partner_if's tool sandbox has until it moves."""

    network_access: Literal["package_index"] = "package_index"  # type: ignore[assignment]


def test_a_blocks_own_network_setting_is_left_to_its_server() -> None:
    owned = OwnedSandboxes(FakeSnapshotProvider(), state_dir=None, server_name="server")

    owned.check_provider(configs=[_OwnNetworkConfig()])


class TestStartupCheckResolvesWhatItCan:
    @pytest.fixture(autouse=True)
    def provider_classes(self, monkeypatch) -> None:
        classes = {"rules": _RuleProvider, "plain": FakeSnapshotProvider}
        monkeypatch.setattr(
            "nemo_gym.sandbox.config.resolve_provider_config", lambda reference, named: {reference: {}}
        )
        monkeypatch.setattr("nemo_gym.sandbox.providers.registry.get_provider_class", classes.__getitem__)
        monkeypatch.setattr("nemo_gym.server_runtime.base_server.get_global_config_dict", lambda: {})

    def _typed(self, block: SandboxConfig, **fields: Any) -> _TypedServer:
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = _servers_at(policy="127.0.0.1")
        return _TypedServer(
            config=_TypedConfig(name="my_server", host="", port=0, entrypoint="", sandbox_config=block, **fields),
            server_client=server_client,
        )

    def test_a_model_server_a_sandbox_cannot_reach_fails_the_startup(self) -> None:
        server = self._typed(SandboxConfig(network_access="model_only", model_server="policy"), sandbox_provider="rules")

        with pytest.raises(ValueError, match="sandbox-reachable host"):
            server.check_sandbox_provider()

    def test_with_two_provider_fields_a_block_is_checked_where_it_is_used(self) -> None:
        # The block may be meant for the GPU provider, which applies the rule; startup cannot tell.
        server = self._typed(
            SandboxConfig(network_access="no_network"), sandbox_provider="plain", gpu_sandbox_provider="rules"
        )

        class GpuServer(_TypedServer):
            sandbox_config_providers = {"sandbox_config": "gpu_sandbox_provider"}

        GpuServer(config=server.config, server_client=server.server_client).check_sandbox_provider()

    def test_a_restricted_rule_on_a_server_that_has_not_adopted_the_owner_fails_startup(self) -> None:
        server = self._typed(SandboxConfig(network_access="no_network"), sandbox_provider="rules")
        unadopted = _Server(config=server.config, server_client=server.server_client)
        with pytest.raises(ValueError, match="has not declared"):
            unadopted.check_sandbox_provider()

    def test_a_provider_field_with_no_naming_convention_is_checked_explicitly(self) -> None:
        class Config(_TypedConfig):
            backend: str = "rules"

        class Server(_TypedServer):
            config: Config
            sandbox_config_providers = {"sandbox_config": "backend"}

        server = Server(
            config=Config(name="server", host="", port=0, entrypoint="", sandbox_provider="plain",
                          sandbox_config=SandboxConfig(network_access="no_network")),
            server_client=MagicMock(spec=ServerClient),
        )
        server.check_sandbox_provider()
        server.config.backend = "plain"
        with pytest.raises(ValueError, match="cannot apply"):
            server.check_sandbox_provider()


class TestSpecLeavesWhatIsNotATypedRule:
    def test_a_subclass_with_its_own_network_setting_gets_its_options_as_they_are(self) -> None:
        owned = OwnedSandboxes(_RuleProvider(), state_dir=None, server_name="server")
        config = _OwnNetworkConfig(provider_options={"volumes": ["v"]})

        spec = owned.spec(config, image="image", workdir=None)

        assert spec.provider_options == {"volumes": ["v"]}

    @pytest.mark.parametrize(
        ("policy", "suggestion"),
        [
            (None, "network_access: inherit"),
            ({"defaultAction": "deny", "egress": []}, "network_access: no_network"),
            (
                {"defaultAction": "deny", "egress": [{"action": "allow", "target": "proxy"}]},
                "network_access: hosts_only with allowed_hosts ['proxy']",
            ),
        ],
    )
    def test_the_warning_names_the_rule_that_keeps_todays_behavior(self, caplog, policy, suggestion) -> None:
        owned = OwnedSandboxes(_RuleProvider(), state_dir=None, server_name="server")
        options = {"network_policy": policy} if policy is not None else {}

        with caplog.at_level("WARNING", logger=owned_sandboxes_module.LOGGER.name):
            owned.spec(SandboxConfig(provider_options=options), image="image", workdir=None)

        assert suggestion in caplog.text


class _RulesWithoutOptions(FakeSnapshotProvider):
    """Declares a rule it has no way to apply."""

    capabilities = SandboxCapabilities(network_rules=NETWORK_RULES)


def test_a_provider_that_declares_a_rule_without_applying_it_is_refused() -> None:
    owned = OwnedSandboxes(_RulesWithoutOptions(), state_dir=None, server_name="server")

    with pytest.raises(ValueError, match="cannot apply network 'no_network'"):
        owned.spec(SandboxConfig(network_access="no_network"), image="image", workdir=None)


class _LendingLaunchedServer(_LaunchedServer):
    lends_sandboxes = True


def test_a_launch_whose_provider_cannot_lend_fails_at_startup(monkeypatch) -> None:
    monkeypatch.setattr("nemo_gym.sandbox.config.resolve_provider_config", lambda reference, named: {reference: {}})
    monkeypatch.setattr(
        "nemo_gym.sandbox.providers.registry.get_provider_class", {"never": _NeverReconnects}.__getitem__
    )
    config = _Config(name="my_server", host="127.0.0.1", port=8000, entrypoint="app.py", sandbox_provider="never")

    with pytest.raises(ValueError, match="cannot lend a sandbox"):
        _run_webserver(monkeypatch, _LendingLaunchedServer, config)


def test_the_old_checkpointer_name_resolves_with_a_deprecation_warning() -> None:
    import importlib

    with pytest.warns(DeprecationWarning, match="OwnedSandboxes"):
        old = getattr(importlib.import_module("nemo_gym.server_runtime"), "SandboxSessionCheckpointer")
    assert old is OwnedSandboxes
    with pytest.warns(DeprecationWarning):
        module = importlib.import_module("nemo_gym.server_runtime._checkpoint.sandbox")
        old = getattr(module, "SandboxSessionCheckpointer")
    assert old is OwnedSandboxes


async def test_discard_waits_for_a_pending_start_and_stops_the_sandbox() -> None:
    provider = _GatedProvider()
    owned = _owned(provider)
    provider.create_gate.clear()
    creating = asyncio.create_task(owned.create("session", _spec()))
    await _until(lambda: provider.in_flight == 1)
    assert owned.get("session") is None
    assert "session" not in owned
    assert owned.keys == ["session"]
    discarding = asyncio.create_task(owned.discard(["session"]))
    await asyncio.sleep(0)
    assert not discarding.done()
    provider.create_gate.set()
    sandbox = await creating
    await discarding
    assert not sandbox.started
    assert _states(provider) == ["stopped"]
    assert owned.keys == []
