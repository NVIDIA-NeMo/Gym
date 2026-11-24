# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a server holds the sandboxes it owns: ``SimpleServer.sandboxes``, created on first use and closed at shutdown."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseRunServerInstanceConfig
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.server_runtime import owned_sandboxes as owned_sandboxes_module
from nemo_gym.server_runtime.base_server import SimpleServer
from nemo_gym.server_runtime.owned_sandboxes import OwnedSandboxes
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


def test_a_refused_config_builds_no_provider(providers, monkeypatch) -> None:
    # Checkpointing on without checkpoint.state_dir is refused; a provider built first would never be closed, and a
    # later self.sandboxes would build another.
    def no_state_dir(global_config: Any) -> None:
        raise ValueError("checkpoint.state_dir is required to checkpoint sandboxes")

    monkeypatch.setattr(owned_sandboxes_module, "checkpoint_state_dir", no_state_dir)
    server = _server(sandbox_provider="opensandbox")

    for _ in range(2):
        with pytest.raises(ValueError, match="checkpoint.state_dir is required"):
            server.sandboxes

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


def test_each_provider_field_has_its_own_snapshot_index(providers, monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(owned_sandboxes_module, "checkpoint_state_dir", lambda config: state_dir)
    server = _server(sandbox_provider="opensandbox", gpu_sandbox_provider="gpu")

    main, gpu = server.sandboxes, server.sandboxes.on("gpu_sandbox_provider")

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
