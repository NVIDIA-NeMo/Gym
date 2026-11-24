# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a server holds the sandboxes it owns: ``SimpleServer.sandboxes``, created on first use and closed at shutdown."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

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


def test_a_refused_provider_config_builds_no_provider(providers, monkeypatch) -> None:
    def refuse(*args: Any) -> None:
        raise ValueError("unknown provider block")

    monkeypatch.setattr(owned_sandboxes_module, "resolve_provider_config", refuse)
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


def test_the_old_checkpointer_name_resolves_with_a_deprecation_warning() -> None:
    import importlib

    with pytest.warns(DeprecationWarning, match="OwnedSandboxes"):
        old = getattr(importlib.import_module("nemo_gym.server_runtime"), "SandboxSessionCheckpointer")
    assert old is OwnedSandboxes
    with pytest.warns(DeprecationWarning):
        module = importlib.import_module("nemo_gym.server_runtime._checkpoint.sandbox")
        old = getattr(module, "SandboxSessionCheckpointer")
    assert old is OwnedSandboxes
