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

"""Unit tests for the shared IdeGYM orchestrator session.

The session is where the provider's two structural claims live -- one registered
IdeGYM client per process, reference-counted, on a private event loop -- so these
tests are about that machinery rather than about sandbox behavior: refcounting,
cross-loop bridging, and the SDK call shapes. The SDK itself is faked at the
``IdeGYMClient`` boundary; the tests gated on ``idegym`` being installed check that
the provider's real calls bind against the published signatures.
"""

import asyncio
import importlib.util
import threading
from typing import Any

import pytest

from nemo_gym.sandbox.providers.idegym import session as idegym_session
from nemo_gym.sandbox.providers.idegym.config import (
    IdeGymAttributionConfig,
    IdeGymConnectionConfig,
    IdeGymPollingConfig,
)
from nemo_gym.sandbox.providers.idegym.errors import IdeGymError, IdeGymUnknownServerError
from nemo_gym.sandbox.providers.idegym.session import (
    IdeGymSession,
    _SessionLoop,
    acquire_session,
    active_session_count,
    build_transport,
    install_transport,
    release_session,
    resolve_client_name,
)


pytestmark = pytest.mark.sandbox

idegym = pytest.importorskip("idegym.client", reason="idegym-client optional dependency is not installed")

# Captured before the autouse fixture replaces it, so the tests that exercise the
# real SDK construction can still reach it.
REAL_BUILD_CLIENT = IdeGymSession._build_client


class FakeServer:
    """Stands in for the SDK's ``IdeGYMServer`` handle."""

    def __init__(self, server_id: int) -> None:
        self.server_id = server_id
        self.bash_calls: list[dict[str, Any]] = []
        self.capability_calls = 0

    async def execute_bash(self, **kwargs: Any) -> Any:
        self.bash_calls.append(kwargs)
        from idegym.api.tools.bash import BashCommandResponse

        return BashCommandResponse(stdout="out", stderr="err", exit_code=0)

    async def list_capabilities(self) -> Any:
        self.capability_calls += 1
        from idegym.api.capabilities import CapabilitiesResponse

        return CapabilitiesResponse(plugins=["tools"])


class FakeClient:
    """Stands in for the SDK's ``IdeGYMClient``, recording lifecycle transitions."""

    instances: list["FakeClient"] = []

    def __init__(self) -> None:
        from uuid import uuid4

        self.client_id = uuid4()
        self.entered = 0
        self.exited = 0
        self.start_calls: list[dict[str, Any]] = []
        self.stop_calls: list[Any] = []
        self.servers: list[FakeServer] = []
        self.next_server_id = 1
        self.start_error: Exception | None = None
        self.enter_error: Exception | None = None
        FakeClient.instances.append(self)

    async def __aenter__(self) -> "FakeClient":
        if self.enter_error is not None:
            raise self.enter_error
        self.entered += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.exited += 1

    async def health_check(self) -> Any:
        from idegym.api.health import HealthCheckResponse

        return HealthCheckResponse(status="healthy")

    async def start_server(self, **kwargs: Any) -> FakeServer:
        self.start_calls.append(kwargs)
        if self.start_error is not None:
            raise self.start_error
        server = FakeServer(self.next_server_id)
        self.next_server_id += 1
        self.servers.append(server)
        return server

    async def stop_server(self, server: Any, polling_config: Any = None) -> None:
        self.stop_calls.append(server)


@pytest.fixture(autouse=True)
def isolate_sessions(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Keep the module-level session registry from leaking between tests."""
    FakeClient.instances = []
    idegym_session._SESSIONS.clear()
    monkeypatch.setattr(IdeGymSession, "_build_client", lambda self: FakeClient())
    monkeypatch.setattr(idegym_session, "require_idegym_client", lambda: None)
    yield
    idegym_session._SESSIONS.clear()


def connection(**overrides: Any) -> IdeGymConnectionConfig:
    return IdeGymConnectionConfig(**{"orchestrator_url": "idegym.test", "client_name": "test-client", **overrides})


POLLING = IdeGymPollingConfig()


# --- private event loop ----------------------------------------------------


async def test_session_loop_runs_work_on_its_own_thread_and_propagates_results() -> None:
    loop = _SessionLoop("test-session-loop")
    caller_thread = threading.get_ident()
    try:

        async def where() -> int:
            return threading.get_ident()

        assert await asyncio.wrap_future(loop.submit(where)) != caller_thread

        async def boom() -> None:
            raise ValueError("from the session loop")

        with pytest.raises(ValueError, match="from the session loop"):
            await asyncio.wrap_future(loop.submit(boom))
    finally:
        loop.stop()
        loop.stop()  # idempotent


# --- registration and refcounting -----------------------------------------


async def test_sandboxes_sharing_a_connection_share_one_registration() -> None:
    config = connection()
    first = await acquire_session(config, IdeGymAttributionConfig())
    second = await acquire_session(config, IdeGymAttributionConfig())
    assert first is second
    assert active_session_count() == 1
    assert len(FakeClient.instances) == 1
    assert FakeClient.instances[0].entered == 1

    # Only the last release unregisters, otherwise stopping one sandbox would
    # terminate every other sandbox owned by the same IdeGYM client.
    await release_session(first)
    assert active_session_count() == 1
    assert FakeClient.instances[0].exited == 0

    await release_session(second)
    assert active_session_count() == 0
    assert FakeClient.instances[0].exited == 1


async def test_sessions_are_acquired_concurrently_without_duplicate_registration() -> None:
    config = connection()
    sessions = await asyncio.gather(*(acquire_session(config, IdeGymAttributionConfig()) for _ in range(5)))
    assert len({id(session) for session in sessions}) == 1
    assert len(FakeClient.instances) == 1
    for session in sessions:
        await release_session(session)
    assert active_session_count() == 0


async def test_different_connections_and_client_names_get_different_sessions() -> None:
    a = await acquire_session(connection(), IdeGymAttributionConfig())
    b = await acquire_session(connection(namespace="other"), IdeGymAttributionConfig())
    c = await acquire_session(connection(client_name="other-client"), IdeGymAttributionConfig())
    assert len({id(a), id(b), id(c)}) == 3
    assert active_session_count() == 3
    for session in (a, b, c):
        await release_session(session)


async def test_a_session_acquired_after_the_last_release_registers_again() -> None:
    config = connection()
    first = await acquire_session(config, IdeGymAttributionConfig())
    await release_session(first)
    second = await acquire_session(config, IdeGymAttributionConfig())
    assert second is not first
    assert len(FakeClient.instances) == 2
    await release_session(second)


async def test_failed_registration_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    def build_failing_client(self: IdeGymSession) -> FakeClient:
        client = FakeClient()
        client.enter_error = RuntimeError("orchestrator refused the registration")
        return client

    monkeypatch.setattr(IdeGymSession, "_build_client", build_failing_client)
    with pytest.raises(RuntimeError, match="refused the registration"):
        await acquire_session(connection(), IdeGymAttributionConfig())
    # A cached failure would make every later sandbox in the process fail too.
    assert active_session_count() == 0
    with pytest.raises(RuntimeError, match="refused the registration"):
        await acquire_session(connection(), IdeGymAttributionConfig())
    assert len(FakeClient.instances) == 2


async def test_releasing_an_unknown_session_is_a_no_op() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    await release_session(session)
    await release_session(session)
    assert active_session_count() == 0


async def test_stopping_a_session_that_never_started_is_safe() -> None:
    session = IdeGymSession(connection(), "unstarted")
    await session.stop()
    await session.stop()
    assert session.client_name == "unstarted"


async def test_a_factory_that_raises_before_awaiting_reaches_the_caller() -> None:
    """A synchronous raise inside the submitted factory must resolve the future.

    Left to the loop's exception handler it would never be reported, and the caller
    would wait forever instead of seeing the error.
    """
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        async with asyncio.timeout(5):
            with pytest.raises(IdeGymUnknownServerError, match="not held"):
                await session.list_capabilities(999)
    finally:
        await release_session(session)


async def test_an_operation_in_flight_when_the_session_stops_is_failed_not_hung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping the loop must fail anything still waiting on it."""
    release = threading.Event()

    async def blocking_capabilities(self: FakeServer) -> Any:
        while not release.is_set():
            await asyncio.sleep(0)

    session = await acquire_session(connection(), IdeGymAttributionConfig())
    ref = await session.start_server({"image_tag": "reg/env:1", "server_name": "s"}, polling=POLLING, timeout_s=5)
    monkeypatch.setattr(FakeServer, "list_capabilities", blocking_capabilities)
    try:
        pending = asyncio.create_task(session.list_capabilities(ref.server_id))
        await asyncio.sleep(0.05)
        await release_session(session)
        async with asyncio.timeout(5):
            with pytest.raises(IdeGymError, match="stopped while this operation was in flight"):
                await pending
    finally:
        release.set()


async def test_a_cancelled_acquirer_does_not_fail_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acquirers share one readiness future, and cancellation propagates into it.

    Without a per-waiter view, one cancelled acquirer would cancel the registration
    that its peers are waiting on.
    """
    gate = threading.Event()
    real_build = IdeGymSession._build_client

    def slow_client(self: IdeGymSession) -> Any:
        gate.wait(5)
        return real_build(self)

    monkeypatch.setattr(IdeGymSession, "_build_client", slow_client)
    config = connection()
    first = asyncio.create_task(acquire_session(config, IdeGymAttributionConfig()))
    second = asyncio.create_task(acquire_session(config, IdeGymAttributionConfig()))
    await asyncio.sleep(0.05)
    first.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    session = await second
    try:
        assert await session.health() == "healthy"
    finally:
        await release_session(session)


async def test_operations_on_a_stopped_session_fail_clearly() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    await release_session(session)
    with pytest.raises(IdeGymError, match="has been stopped"):
        await session.health()


# --- client naming ---------------------------------------------------------


def test_explicit_client_name_wins_over_attribution() -> None:
    name = resolve_client_name(connection(client_name="pinned-by-operator"), IdeGymAttributionConfig(team="t"))
    assert name == "pinned-by-operator"


def test_attribution_derived_client_name_excludes_the_run_id() -> None:
    attribution = IdeGymAttributionConfig(team="My Team", user="dev", workload="swebench", run="run-1234")
    name = resolve_client_name(connection(client_name=None), attribution)
    assert name == "nemo-gym-my-team-dev-swebench"
    # The name has to stay stable across launches for quota-rule matching, so the
    # per-launch run id must not appear in it.
    assert "run-1234" not in name


def test_attribution_can_be_disabled() -> None:
    name = resolve_client_name(connection(client_name=None), IdeGymAttributionConfig(enabled=False))
    assert name == "nemo-gym"


# --- SDK call shapes -------------------------------------------------------


async def test_session_calls_bind_against_the_installed_sdk() -> None:
    """Bind the session's exact SDK call shapes against the installed signatures.

    The provider talks to IdeGYM only through these calls, so this is what catches
    a pre-1.0 SDK changing its argument names underneath the pinned range.
    """
    import inspect

    from idegym.client import IdeGYMClient
    from idegym.client.operations.utils import PollingConfig
    from idegym.client.server import IdeGYMServer

    inspect.signature(IdeGYMClient.__init__).bind(
        None,
        orchestrator_url="idegym.test",
        name="n",
        namespace="ns",
        nodes_count=0,
        auth=None,
        heartbeat_interval_in_seconds=60,
        request_timeout_in_seconds=60,
        otel_config=None,
    )
    inspect.signature(IdeGYMClient.start_server).bind(
        None,
        image_tag="reg/env:1",
        server_name="nemo-gym-abcdef12",
        namespace="ns",
        runtime_class_name=None,
        run_as_root=True,
        service_port=80,
        container_port=8000,
        resources=None,
        node_selector=None,
        volumes=None,
        volume_mounts=None,
        env_from=None,
        service_account_name=None,
        pod_overrides=None,
        server_start_wait_timeout_in_seconds=60,
        retry_delay_in_seconds=15,
        polling_config=PollingConfig(),
        reuse_strategy="NONE",
        server_kind="idegym",
        snapshot=None,
        max_restarts=0,
    )
    inspect.signature(IdeGYMClient.health_check).bind(None)
    inspect.signature(IdeGYMServer.execute_bash).bind(
        None,
        script="true",
        command_timeout=1.0,
        graceful_termination_timeout=2.0,
        request_timeout=3,
        polling_config=PollingConfig(),
    )
    inspect.signature(IdeGYMServer.list_capabilities).bind(None)
    # `IdeGYMClient.stop_server` is wrapped by a retry decorator that does not
    # preserve the signature, so only its presence can be asserted here.
    assert inspect.iscoroutinefunction(IdeGYMClient.stop_server)
    PollingConfig(
        initial_delay_in_sec=0.25,
        wait_timeout_in_sec=60,
        poll_interval_in_sec=0.0,
        factor_for_exponential_wait=1.5,
        max_delay_for_exponential_wait_in_sec=30.0,
    )


@pytest.mark.parametrize("with_auth", [False, True])
def test_build_client_constructs_a_real_sdk_client(monkeypatch: pytest.MonkeyPatch, with_auth: bool) -> None:
    """Construct the real ``IdeGYMClient`` (no I/O) and check what it was handed.

    This is the one place the provider's constructor arguments meet the SDK's, and
    it is also where tracing is pinned off: the SDK ships a default OTLP endpoint
    that traces off-box.
    """
    from idegym.client import IdeGYMClient

    monkeypatch.delenv("IDEGYM_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("IDEGYM_AUTH_PASSWORD", raising=False)
    config = connection(
        client_name="build-client-test",
        username="user" if with_auth else None,
        password="secret" if with_auth else None,
        heartbeat_interval_s=17,
        request_timeout_s=23,
    )
    session = IdeGymSession(config, "build-client-test")
    client = None
    try:
        client = REAL_BUILD_CLIENT(session)
        assert isinstance(client, IdeGYMClient)
        assert client.name == "build-client-test"
        assert str(client._http_client.base_url).rstrip("/") == "http://idegym.test"
        assert client._http_client.timeout.read == 23
        assert session._connection.heartbeat_interval_s == 17
        assert client._otel_config.tracing.enabled is False
        expected = "Basic dXNlcjpzZWNyZXQ=" if with_auth else None
        assert client._http_client.headers.get("authorization") == expected
    finally:
        if client is not None:
            asyncio.run(client._http_client.aclose())
        session._loop.stop()


def test_build_client_requires_credentials_for_a_real_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK only allows anonymous access to its local-testing host."""
    monkeypatch.delenv("IDEGYM_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("IDEGYM_AUTH_PASSWORD", raising=False)
    session = IdeGymSession(connection(orchestrator_url="idegym.example.com"), "no-creds")
    try:
        with pytest.raises(ValueError, match="[Uu]sername and password"):
            REAL_BUILD_CLIENT(session)
    finally:
        session._loop.stop()


async def test_start_server_builds_real_sdk_models() -> None:
    """The plain-dict request is validated into the SDK's own pydantic models."""
    from idegym.api.orchestrator.servers import ServerKind, ServerReuseStrategy
    from idegym.api.pod_spec import KubernetesEnvFromSource, KubernetesPodOverrides, KubernetesVolume
    from idegym.api.resources import KubernetesResources

    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        ref = await session.start_server(
            {
                "image_tag": "reg/env:1",
                "server_name": "nemo-gym-abcdef12",
                "run_as_root": True,
                "resources": {"requests": {"cpu": "500m"}, "limits": {"cpu": "2", "memory": "8192Mi"}},
                "volumes": [{"name": "creds", "secret": {"secretName": "creds"}}],
                "volume_mounts": [{"name": "creds", "mountPath": "/etc/creds"}],
                "env_from": [{"secretRef": {"name": "creds"}}],
                "pod_overrides": {"tolerations": [{"key": "dedicated", "operator": "Exists"}]},
                "snapshot": {"id": "17"},
                "reuse_strategy": "NONE",
                "server_kind": "idegym",
            },
            polling=POLLING,
            timeout_s=42.4,
        )
        assert ref.server_id == 1
        assert ref.server_name == "nemo-gym-abcdef12"
        assert ref.namespace == "idegym"

        call = FakeClient.instances[0].start_calls[0]
        assert isinstance(call["resources"], KubernetesResources)
        assert str(call["resources"].limits.cpu) == "2"
        assert isinstance(call["volumes"][0], KubernetesVolume)
        assert isinstance(call["env_from"][0], KubernetesEnvFromSource)
        assert isinstance(call["pod_overrides"], KubernetesPodOverrides)
        assert call["reuse_strategy"] is ServerReuseStrategy.NONE
        assert call["server_kind"] is ServerKind.IDEGYM
        assert call["namespace"] == "idegym"
        # The readiness budget is handed to the SDK, which owns the wait, rounded
        # up to the whole seconds its API takes.
        assert call["server_start_wait_timeout_in_seconds"] == 43
        assert call["polling_config"].wait_timeout_in_sec == 43
    finally:
        await release_session(session)


async def test_execute_bash_stop_and_capabilities_use_the_tracked_server() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        ref = await session.start_server({"image_tag": "reg/env:1", "server_name": "s"}, polling=POLLING, timeout_s=5)
        result = await session.execute_bash(
            ref.server_id,
            "true",
            command_timeout_s=12.0,
            graceful_termination_timeout_s=2.0,
            request_timeout_s=30.5,
            polling=POLLING,
        )
        assert (result.stdout, result.stderr, result.exit_code) == ("out", "err", 0)
        server = FakeClient.instances[0].servers[0]
        assert server.bash_calls[0]["command_timeout"] == 12.0
        assert server.bash_calls[0]["request_timeout"] == 31

        assert await session.list_capabilities(ref.server_id) == ["tools"]
        assert await session.health() == "healthy"

        await session.stop_server(ref.server_id, polling=POLLING, timeout_s=5)
        assert FakeClient.instances[0].stop_calls == [server]
        # Stopping forgets the server, so a stale id fails loudly instead of
        # silently addressing a pod that is already gone.
        with pytest.raises(IdeGymUnknownServerError, match="not held by this session"):
            await session.list_capabilities(ref.server_id)
    finally:
        await release_session(session)


async def test_a_failed_start_does_not_track_a_server() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        FakeClient.instances[0].start_error = RuntimeError("no capacity")
        with pytest.raises(RuntimeError, match="no capacity"):
            await session.start_server({"image_tag": "reg/env:1", "server_name": "s"}, polling=POLLING, timeout_s=5)
        with pytest.raises(IdeGymUnknownServerError, match="not held"):
            await session.stop_server(1, polling=POLLING, timeout_s=5)
    finally:
        await release_session(session)


async def test_a_slow_operation_does_not_block_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrency is bounded by the HTTP pool, not by serializing whole operations.

    A create can poll for the whole readiness timeout, so holding a slot for an entire
    operation would let provisioning block the exec calls of running sandboxes.
    """
    release = threading.Event()
    started = 0

    async def slow_capabilities(self: FakeServer) -> Any:
        nonlocal started
        started += 1
        while not release.is_set():
            await asyncio.sleep(0)
        from idegym.api.capabilities import CapabilitiesResponse

        return CapabilitiesResponse(plugins=[])

    monkeypatch.setattr(FakeServer, "list_capabilities", slow_capabilities)
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        refs = [
            await session.start_server(
                {"image_tag": "reg/env:1", "server_name": f"s{index}"}, polling=POLLING, timeout_s=5
            )
            for index in range(5)
        ]
        pending = [asyncio.create_task(session.list_capabilities(ref.server_id)) for ref in refs]
        await asyncio.sleep(0.05)
        assert started == len(refs)
        # An unrelated operation still gets through while all five are in flight.
        assert await session.health() == "healthy"
    finally:
        release.set()
        await asyncio.gather(*pending)
        await release_session(session)


# --- HTTP transport --------------------------------------------------------


requires_aiohttp_bridge = pytest.mark.skipif(
    importlib.util.find_spec("httpx_aiohttp") is None,
    reason="httpx-aiohttp (nemo-gym[sandbox]) is not installed",
)


@requires_aiohttp_bridge
def test_build_transport_honors_the_configured_backend() -> None:
    aiohttp_transport = build_transport(connection(transport_backend="aiohttp"))
    httpx_transport = build_transport(connection(transport_backend="httpx"))
    assert type(aiohttp_transport).__name__ == "AiohttpTransport"
    assert type(httpx_transport).__name__ == "AsyncHTTPTransport"


def test_build_transport_applies_the_configured_pool_limits() -> None:
    transport = build_transport(
        connection(
            transport_backend="httpx",
            max_connections=7,
            max_keepalive_connections=3,
            keepalive_expiry_s=4.0,
        )
    )
    # httpx exposes the resolved limits only on the pool, hence the private reach.
    pool = transport._pool
    assert (pool._max_connections, pool._max_keepalive_connections, pool._keepalive_expiry) == (7, 3, 4.0)


def test_install_transport_replaces_the_sdk_transport() -> None:
    import httpx

    client = type("Client", (), {})()
    client._http_client = httpx.AsyncClient()
    original = client._http_client._transport
    install_transport(client, connection(transport_backend="httpx"))
    assert client._http_client._transport is not original
    assert type(client._http_client._transport).__name__ == "AsyncHTTPTransport"


def test_install_transport_degrades_when_the_sdk_shape_changes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Reaching into the SDK's private httpx client must never be fatal."""
    monkeypatch.setattr(idegym_session, "_TRANSPORT_WARNED", False)
    with caplog.at_level("WARNING"):
        install_transport(type("Client", (), {})(), connection())
    assert "falling back to the SDK's own transport" in caplog.text


@requires_aiohttp_bridge
def test_aiohttp_backend_falls_back_when_the_bridge_is_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "httpx_aiohttp":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level("WARNING"):
        transport = build_transport(connection(transport_backend="aiohttp"))
    assert type(transport).__name__ == "AsyncHTTPTransport"
    assert "httpx-aiohttp is not installed" in caplog.text


def test_tracing_stays_off_unless_an_endpoint_is_configured() -> None:
    """The SDK traces to its default off-box collector; the provider must not."""
    from idegym.api.config import OTELConfig, TracingConfig

    assert OTELConfig(tracing=TracingConfig(endpoint=None)).tracing.enabled is False
    assert connection().tracing_enabled is False
    assert connection(tracing_endpoint="http://collector.example/v1/traces").tracing_enabled is True
