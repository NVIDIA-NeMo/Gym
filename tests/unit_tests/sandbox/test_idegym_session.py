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

"""Unit tests for the shared IdeGYM orchestrator session and its HTTP transport.

The session is where the provider's two structural claims live -- one registered
IdeGYM client per process, reference-counted, driven through the SDK's
``SharedIdeGYMClient`` loop thread -- so these tests are about that machinery rather
than about sandbox behavior: refcounting, cancellation, shutdown, and the SDK call
shapes. The real ``SharedIdeGYMClient`` runs; only the ``IdeGYMClient`` it builds is
faked. The tests that bind against the installed SDK check that the provider's real
calls match the published signatures.
"""

import asyncio
import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any

import aiohttp
import httpx
import pytest

from nemo_gym.sandbox.providers.idegym import http as idegym_http
from nemo_gym.sandbox.providers.idegym import session as idegym_session
from nemo_gym.sandbox.providers.idegym.config import (
    IdeGymAttributionConfig,
    IdeGymConnectionConfig,
    IdeGymPollingConfig,
)
from nemo_gym.sandbox.providers.idegym.errors import (
    IdeGymError,
    IdeGymInvalidRequestError,
    IdeGymUnknownServerError,
    is_invalid_request,
)
from nemo_gym.sandbox.providers.idegym.http import RetryingTransport, build_transport
from nemo_gym.sandbox.providers.idegym.session import (
    IdeGymServerStatus,
    IdeGymSession,
    acquire_session,
    active_session_count,
    release_session,
    resolve_client_name,
)


pytestmark = pytest.mark.sandbox

idegym = pytest.importorskip("idegym.client", reason="idegym-client optional dependency is not installed")

from idegym.client import shared as idegym_shared  # noqa: E402 - after the importorskip guard


REAL_IDEGYM_CLIENT = idegym_shared.IdeGYMClient
SDK_LOOP_THREAD = "idegym-client-loop"


class FakeServer:
    """Stands in for the SDK's ``IdeGYMServer`` handle."""

    def __init__(self, server_id: int, client_id: Any) -> None:
        self.server_id = server_id
        self.client_id = client_id
        self.bash_calls: list[dict[str, Any]] = []
        self.keepalive_calls: list[float] = []
        self.upload_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.download_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.usable = True

    async def execute_bash(self, **kwargs: Any) -> Any:
        self.bash_calls.append(kwargs)
        from idegym.api.tools.bash import BashCommandResponse

        return BashCommandResponse(stdout="  out\n", stderr="err", exit_code=3)

    async def status(self) -> Any:
        from idegym.api.orchestrator.servers import ServerStatusResponse

        return ServerStatusResponse(
            server_id=self.server_id,
            server_name="s",
            generated_name=f"s-{self.server_id}",
            namespace="idegym",
            availability="ALIVE" if self.usable else "CRASHED",
            usable=self.usable,
            created_at=1,
            last_activity_at=2,
            details=None if self.usable else "OOMKilled",
            idle_seconds=0.5,
            pod_phase="Running" if self.usable else None,
            pod_ready=True if self.usable else None,
        )

    async def keepalive(self, minutes: float = 15.0) -> Any:
        self.keepalive_calls.append(minutes)
        from idegym.api.orchestrator.servers import KeepaliveServerResponse

        return KeepaliveServerResponse(server_id=self.server_id, keepalive_until=1_700_000_000_000, minutes=minutes)

    async def upload_file(self, *args: Any, **kwargs: Any) -> int:
        self.upload_calls.append((args, kwargs))
        return 11

    async def download_file(self, *args: Any, **kwargs: Any) -> int:
        self.download_calls.append((args, kwargs))
        return 10


class FakeClient:
    """Stands in for the ``IdeGYMClient`` that ``SharedIdeGYMClient`` builds on its loop."""

    instances: list["FakeClient"] = []
    enter_error: BaseException | None = None
    enter_gate: threading.Event | None = None
    exit_error: BaseException | None = None

    def __init__(self, **arguments: Any) -> None:
        from uuid import uuid4

        self.arguments = arguments
        self.client_id = uuid4()
        self.entered = 0
        self.exited = 0
        self.start_calls: list[dict[str, Any]] = []
        self.stop_calls: list[Any] = []
        self.servers: list[FakeServer] = []
        self.next_server_id = 1
        self.start_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        FakeClient.instances.append(self)

    async def __aenter__(self) -> "FakeClient":
        gate = FakeClient.enter_gate
        while gate is not None and not gate.is_set():
            await asyncio.sleep(0.01)
        if FakeClient.enter_error is not None:
            raise FakeClient.enter_error
        self.entered += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.exited += 1
        if FakeClient.exit_error is not None:
            raise FakeClient.exit_error

    async def health_check(self) -> Any:
        from idegym.api.health import HealthCheckResponse

        return HealthCheckResponse(status="healthy")

    async def start_server(self, **kwargs: Any) -> FakeServer:
        self.start_calls.append(kwargs)
        if self.start_error is not None:
            raise self.start_error
        server = FakeServer(self.next_server_id, self.client_id)
        self.next_server_id += 1
        self.servers.append(server)
        return server

    async def stop_server(self, server: Any, polling_config: Any = None) -> Any:
        self.stop_calls.append(server)
        if self.stop_error is not None:
            raise self.stop_error


@pytest.fixture(autouse=True)
def isolate_sessions(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Keep the module-level session registry from leaking between tests."""
    FakeClient.instances = []
    FakeClient.enter_error = None
    FakeClient.enter_gate = None
    FakeClient.exit_error = None
    idegym_session._SESSIONS.clear()
    monkeypatch.setattr(idegym_shared, "IdeGYMClient", FakeClient)
    yield
    idegym_session._SESSIONS.clear()


def connection(**overrides: Any) -> IdeGymConnectionConfig:
    return IdeGymConnectionConfig(**{"orchestrator_url": "idegym.test", "client_name": "test-client", **overrides})


def sdk_loop_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == SDK_LOOP_THREAD]


async def wait_until(predicate: Any, timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.01)


POLLING = IdeGymPollingConfig()


async def started(session: IdeGymSession, **request: Any) -> Any:
    return await session.start_server(
        {"image_tag": "reg/env:1", "server_name": "s", **request}, polling=POLLING, timeout_s=300
    )


# --- registration and refcounting -----------------------------------------


async def test_sandboxes_sharing_a_connection_share_one_registration() -> None:
    loops_before = len(sdk_loop_threads())
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
    # The SDK's loop thread is joined, not leaked.
    assert len(sdk_loop_threads()) == loops_before

    third = await acquire_session(config, IdeGymAttributionConfig())
    assert third is not first
    assert len(FakeClient.instances) == 2
    await release_session(third)
    assert FakeClient.instances[0].exited == 1
    assert FakeClient.instances[1].exited == 1


async def test_sessions_are_acquired_concurrently_without_duplicate_registration() -> None:
    config = connection()
    sessions = await asyncio.gather(*(acquire_session(config, IdeGymAttributionConfig()) for _ in range(5)))
    assert len({id(session) for session in sessions}) == 1
    assert len(FakeClient.instances) == 1
    for session in sessions:
        await release_session(session)
    assert active_session_count() == 0
    assert FakeClient.instances[0].exited == 1
    # Releasing a session that is no longer registered is a no-op.
    await release_session(sessions[0])
    assert FakeClient.instances[0].exited == 1


async def test_different_connections_and_client_names_get_different_sessions() -> None:
    a = await acquire_session(connection(), IdeGymAttributionConfig())
    b = await acquire_session(connection(namespace="other"), IdeGymAttributionConfig())
    c = await acquire_session(connection(client_name="other-client"), IdeGymAttributionConfig())
    assert len({id(a), id(b), id(c)}) == 3
    assert active_session_count() == 3
    assert FakeClient.instances[2].arguments["name"] == "other-client"
    for session in (a, b, c):
        await release_session(session)


async def test_failed_registration_is_not_cached() -> None:
    loops_before = len(sdk_loop_threads())
    FakeClient.enter_error = RuntimeError("orchestrator refused the registration")
    with pytest.raises(RuntimeError, match="refused the registration"):
        await acquire_session(connection(), IdeGymAttributionConfig())
    # A cached failure would make every later sandbox in the process fail too.
    assert active_session_count() == 0
    assert len(sdk_loop_threads()) == loops_before

    FakeClient.enter_error = None
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    assert len(FakeClient.instances) == 2
    assert await session.health() == "healthy"
    await release_session(session)


async def test_a_real_client_without_credentials_fails_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK only allows anonymous access to its local-testing host; the error reaches the acquirer."""
    monkeypatch.setattr(idegym_shared, "IdeGYMClient", REAL_IDEGYM_CLIENT)
    monkeypatch.delenv("IDEGYM_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("IDEGYM_AUTH_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="[Uu]sername and password"):
        await acquire_session(connection(orchestrator_url="idegym.example.com"), IdeGymAttributionConfig())
    assert active_session_count() == 0


async def test_a_failed_deregistration_is_logged_and_still_stops_the_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loops_before = len(sdk_loop_threads())
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    FakeClient.exit_error = idegym.IdeGYMServerError("orchestrator down", status_code=500)
    with caplog.at_level(logging.WARNING, logger=idegym_session.LOGGER.name):
        await release_session(session)
    assert "Failed to unregister the IdeGYM client 'test-client' cleanly" in caplog.text
    assert active_session_count() == 0
    assert len(sdk_loop_threads()) == loops_before
    with pytest.raises(IdeGymError, match="has been stopped"):
        await session.health()


async def test_stopping_a_session_that_never_started_is_safe() -> None:
    session = IdeGymSession(connection(), "unstarted")
    await session.stop()
    await session.stop()
    assert session.client_name == "unstarted"


async def test_a_cancelled_acquirer_does_not_fail_the_others() -> None:
    """Acquirers share one readiness future, and cancellation propagates into it.

    Without a per-waiter view, one cancelled acquirer would cancel the registration
    that its peers are waiting on.
    """
    FakeClient.enter_gate = threading.Event()
    config = connection()
    first = asyncio.create_task(acquire_session(config, IdeGymAttributionConfig()))
    second = asyncio.create_task(acquire_session(config, IdeGymAttributionConfig()))
    await asyncio.sleep(0.05)
    first.cancel()
    FakeClient.enter_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    session = await second
    try:
        assert await session.health() == "healthy"
        assert FakeClient.instances[0].exited == 0
    finally:
        await release_session(session)
    assert FakeClient.instances[0].exited == 1


async def test_a_registration_whose_only_acquirer_was_cancelled_is_undone() -> None:
    """The registration finishes after its acquirer gave up; it must still be unregistered."""
    FakeClient.enter_gate = threading.Event()
    acquirer = asyncio.create_task(acquire_session(connection(), IdeGymAttributionConfig()))
    await wait_until(lambda: FakeClient.instances)
    acquirer.cancel()
    await asyncio.sleep(0.05)
    FakeClient.enter_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await acquirer
    assert active_session_count() == 0
    assert (FakeClient.instances[0].entered, FakeClient.instances[0].exited) == (1, 1)


async def test_work_the_sdk_refuses_after_close_is_reported_as_a_stopped_session() -> None:
    """``SharedIdeGYMClient.submit`` raises a bare RuntimeError once closed; callers see IdeGymError."""
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        await asyncio.to_thread(session._shared.__exit__, None, None, None)
        with pytest.raises(IdeGymError, match="has been stopped") as raised:
            await session.health()
        assert isinstance(raised.value.__cause__, RuntimeError)
    finally:
        await release_session(session)


async def test_an_unknown_server_id_reaches_the_caller() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        async with asyncio.timeout(5):
            with pytest.raises(IdeGymUnknownServerError, match="not held"):
                await session.server_status(999)
    finally:
        await release_session(session)


async def test_an_operation_in_flight_when_the_session_stops_is_failed_not_hung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK cancels what is still running when it closes; the caller sees IdeGymError, not CancelledError."""
    monkeypatch.setattr(idegym_shared._LoopThread, "_DRAIN_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()

    async def blocking_status(self: FakeServer) -> Any:
        while not release.is_set():
            await asyncio.sleep(0.01)

    session = await acquire_session(connection(), IdeGymAttributionConfig())
    ref = await started(session)
    monkeypatch.setattr(FakeServer, "status", blocking_status)
    try:
        pending = asyncio.create_task(session.server_status(ref.server_id))
        await asyncio.sleep(0.05)
        await release_session(session)
        async with asyncio.timeout(5):
            with pytest.raises(IdeGymError, match="stopped while this operation was in flight"):
                await pending
    finally:
        release.set()


async def test_a_cancelled_caller_does_not_cancel_the_sdk_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The start keeps going on the SDK loop, so the server it creates is still tracked and can be stopped."""
    gate = threading.Event()
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    client = FakeClient.instances[0]
    real_start = client.start_server

    async def gated_start(**kwargs: Any) -> FakeServer:
        while not gate.is_set():
            await asyncio.sleep(0.01)
        return await real_start(**kwargs)

    monkeypatch.setattr(client, "start_server", gated_start)
    try:
        caller = asyncio.create_task(started(session))
        await asyncio.sleep(0.05)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        gate.set()
        await wait_until(lambda: 1 in session._servers)
        await session.stop_server(1, polling=POLLING, timeout_s=5)
        assert client.stop_calls == [client.servers[0]]
    finally:
        gate.set()
        await release_session(session)


async def test_a_slow_operation_does_not_block_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session adds no lock of its own around an SDK call.

    A create can poll for the whole readiness timeout, so serializing whole operations
    would let provisioning block the exec calls of running sandboxes.
    """
    release = threading.Event()
    in_flight = 0
    real_status = FakeServer.status

    async def slow_status(self: FakeServer) -> Any:
        nonlocal in_flight
        in_flight += 1
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await real_status(self)

    monkeypatch.setattr(FakeServer, "status", slow_status)
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    pending: list[asyncio.Task[Any]] = []
    try:
        refs = [await started(session, server_name=f"s{index}") for index in range(5)]
        pending = [asyncio.create_task(session.server_status(ref.server_id)) for ref in refs]
        await wait_until(lambda: in_flight == len(refs))
        # An unrelated operation still gets through while all five are in flight.
        assert await session.health() == "healthy"
    finally:
        release.set()
        await asyncio.gather(*pending)
        await release_session(session)


def test_a_missing_sdk_is_reported_with_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "idegym.client", None)
    with pytest.raises(ModuleNotFoundError, match="nemo-gym\\[idegym\\]"):
        idegym_session.require_idegym_client()


def test_an_sdk_without_the_shared_client_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(idegym, "SharedIdeGYMClient")
    with pytest.raises(ImportError, match=r"idegym-client >= 0\.13\.1"):
        idegym_session.require_idegym_client()


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


def test_session_calls_bind_against_the_installed_sdk() -> None:
    """Bind the session's exact SDK call shapes against the installed signatures.

    The provider talks to IdeGYM only through these calls, and the dependency carries no
    upper bound, so this is the guard against a pre-1.0 SDK renaming an argument.
    """
    import inspect

    from idegym.client import IdeGYMClient, SharedIdeGYMClient
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
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        otel_config=None,
    )
    shared = SharedIdeGYMClient(orchestrator_url="idegym.test", name="n", namespace="ns")
    inspect.signature(shared.submit).bind(lambda client: client.health_check())
    assert callable(shared.__enter__) and callable(shared.__exit__)
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
        labels={},
        annotations={},
    )
    inspect.signature(IdeGYMClient.health_check).bind(None)
    inspect.signature(IdeGYMServer.execute_bash).bind(
        None,
        script="true",
        command_timeout=1.0,
        graceful_termination_timeout=2.0,
        request_timeout=3,
        polling_config=PollingConfig(),
        max_output_bytes=None,
        strip_output=False,
        cwd="/work",
        env={"A": "1"},
        user="root",
    )
    inspect.signature(IdeGYMServer.status).bind(None)
    inspect.signature(IdeGYMServer.keepalive).bind(None, minutes=5.0)
    inspect.signature(IdeGYMServer.upload_file).bind(
        None, Path("/tmp/a"), "/work/a", chunk_bytes=1024, request_timeout=3, polling_config=PollingConfig()
    )
    inspect.signature(IdeGYMServer.download_file).bind(
        None, "/work/a", Path("/tmp/a"), chunk_bytes=1024, request_timeout=3, polling_config=PollingConfig()
    )
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


def test_the_sdk_exception_hierarchy_is_what_errors_relies_on() -> None:
    """``errors`` classifies by these types; a regrouped hierarchy would silently reclassify failures."""
    from idegym.client import exceptions

    # Timeouts are read as builtin TimeoutErrors, and 408/504 have no type but that one.
    assert issubclass(idegym.IdeGYMTimeoutError, TimeoutError)
    for status in (408, 504):
        assert type(exceptions.http_error("m", status_code=status)) is idegym.IdeGYMTimeoutError
    # 502 is only recognisable by its status.
    assert type(exceptions.http_error("m", status_code=502)) is idegym.IdeGYMServerError
    for status in (429, 503):
        assert type(exceptions.http_error("m", status_code=status)) is idegym.IdeGYMBusyError
    for status in (404, 410):
        assert type(exceptions.http_error("m", status_code=status)) is idegym.IdeGYMNotFoundError
    # A sandbox-reported failure is checked first, so it must not be mistaken for a status type.
    assert issubclass(idegym.IdeGYMSandboxError, idegym.IdeGYMHTTPError)
    assert not issubclass(idegym.IdeGYMSandboxError, (idegym.IdeGYMNotFoundError, idegym.IdeGYMServerError))
    # IdeGYMConnectionError is not a builtin ConnectionError, so it is matched by name.
    assert not issubclass(idegym.IdeGYMConnectionError, ConnectionError)


@pytest.mark.parametrize("with_auth", [False, True])
def test_client_arguments_build_a_real_sdk_client(monkeypatch: pytest.MonkeyPatch, with_auth: bool) -> None:
    """Construct the real ``IdeGYMClient`` (no I/O) from the arguments ``SharedIdeGYMClient`` gets."""
    monkeypatch.delenv("IDEGYM_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("IDEGYM_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("IDEGYM_OTEL_TRACING_ENDPOINT", raising=False)
    config = connection(
        client_name="build-client-test",
        username="user" if with_auth else None,
        password="secret" if with_auth else None,  # pragma: allowlist secret
        heartbeat_interval_s=17,
        request_timeout_s=23,
        transport_backend="httpx",
        max_connections=7,
    )
    arguments = IdeGymSession(config, "build-client-test")._client_arguments()
    # The SDK rejects `limits` next to `transport`, and tracing is left to its default (off).
    assert "limits" not in arguments
    assert "otel_config" not in arguments
    client = REAL_IDEGYM_CLIENT(**arguments)
    try:
        assert client.name == "build-client-test"
        assert str(client._http_client.base_url).rstrip("/") == "http://idegym.test"
        assert client._http_client.timeout.read == 23
        assert client._heartbeat_interval_in_seconds == 17
        assert client._otel_config.tracing.enabled is False
        transport = client._http_client._transport
        assert isinstance(transport, RetryingTransport)
        assert transport._transport._pool._max_connections == 7
        expected = "Basic dXNlcjpzZWNyZXQ=" if with_auth else None
        assert client._http_client.headers.get("authorization") == expected
    finally:
        asyncio.run(client._http_client.aclose())


def test_a_configured_tracing_endpoint_is_passed_to_the_sdk() -> None:
    config = connection(tracing_endpoint="http://collector.example/v1/traces", tracing_timeout_s=4)
    otel = IdeGymSession(config, "traced")._client_arguments()["otel_config"]
    assert otel.tracing.endpoint == "http://collector.example/v1/traces"
    assert otel.tracing.enabled is True
    assert otel.service_name == "nemo-gym-traced"


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
                "volumes": [{"name": "creds", "secret": {"secretName": "creds"}}],  # pragma: allowlist secret
                "volume_mounts": [{"name": "creds", "mountPath": "/etc/creds"}],
                "env_from": [{"secretRef": {"name": "creds"}}],
                "pod_overrides": {"tolerations": [{"key": "dedicated", "operator": "Exists"}]},
                "snapshot": {"id": "17"},
                "reuse_strategy": "NONE",
                "server_kind": "idegym",
                "labels": {"nemo-gym.nvidia.com/team": "t", "instance_id": "django__django-11099"},
                "annotations": {"nemo-gym.nvidia.com/note": "free text"},
            },
            polling=POLLING,
            timeout_s=300,
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
        assert call["labels"] == {"nemo-gym.nvidia.com/team": "t", "instance_id": "django__django-11099"}
        assert call["annotations"] == {"nemo-gym.nvidia.com/note": "free text"}
        # 300 s minus the SDK's 60 s minimum grace.
        assert call["server_start_wait_timeout_in_seconds"] == 240
        # Only the backoff reaches the SDK here: it polls start-server for its own remaining time.
        polling = call["polling_config"]
        assert (polling.initial_delay_in_sec, polling.factor_for_exponential_wait) == (
            POLLING.initial_delay_s,
            POLLING.backoff_factor,
        )
    finally:
        await release_session(session)


@pytest.mark.parametrize("timeout_s", [120, 300, 600, 1800, 3600, 36000])
def test_the_sdk_start_deadline_fits_inside_the_readiness_budget(timeout_s: float) -> None:
    from idegym.client.operations.servers import _client_start_deadline

    wait = idegym_session._start_wait_timeout_s(timeout_s)
    assert wait >= timeout_s / 2
    assert _client_start_deadline(wait) <= timeout_s


@pytest.mark.parametrize(("timeout_s", "expected_wait"), [(1, 1), (40, 20), (60, 30), (61, 30), (119, 59)])
def test_a_short_budget_gives_the_orchestrator_half_of_it(timeout_s: float, expected_wait: int) -> None:
    from idegym.client.operations.servers import _client_start_deadline

    wait = idegym_session._start_wait_timeout_s(timeout_s)
    assert wait == expected_wait
    # The SDK's fixed grace is what the call can overrun the budget by, and no more.
    assert _client_start_deadline(wait) <= timeout_s + idegym_session.START_DEADLINE_GRACE_MIN_S


async def test_execute_bash_sends_context_as_request_fields() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        ref = await started(session)
        env = {"A": "1", "B": "two words"}
        result = await session.execute_bash(
            ref.server_id,
            "echo hi",
            cwd="/work",
            env=env,
            user="agent",
            max_output_bytes=None,
            command_timeout_s=12.0,
            graceful_termination_timeout_s=2.0,
            request_timeout_s=30.5,
            polling=POLLING,
        )
        # Verbatim: the leading spaces and trailing newline survive.
        assert (result.stdout, result.stderr, result.exit_code) == ("  out\n", "err", 3)
        call = FakeClient.instances[0].servers[0].bash_calls[0]
        assert call["script"] == "echo hi"
        assert (call["cwd"], call["env"], call["user"]) == ("/work", env, "agent")
        assert call["env"] is not env
        assert call["max_output_bytes"] is None
        assert call["strip_output"] is False
        assert call["command_timeout"] == 12.0
        assert call["graceful_termination_timeout"] == 2.0
        assert call["request_timeout"] == 31
        assert call["polling_config"].wait_timeout_in_sec == 31

        await session.execute_bash(
            ref.server_id,
            "true",
            cwd=None,
            env=None,
            user=None,
            max_output_bytes=1024,
            command_timeout_s=1.0,
            graceful_termination_timeout_s=2.0,
            request_timeout_s=5,
            polling=POLLING,
        )
        call = FakeClient.instances[0].servers[0].bash_calls[1]
        assert (call["cwd"], call["env"], call["user"], call["max_output_bytes"]) == (None, None, None, 1024)
    finally:
        await release_session(session)


@pytest.mark.parametrize(
    ("env", "problem"),
    [({"NOT AN IDENT": "hidden-1"}, "must match"), ({"NAME": "hidden\0-2"}, "must not contain NUL")],
    ids=["bad-name", "nul-value"],
)
async def test_an_invalid_bash_request_is_rejected_before_it_is_sent(env: dict[str, str], problem: str) -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        ref = await started(session)
        with pytest.raises(IdeGymInvalidRequestError, match=problem) as raised:
            await session.execute_bash(
                ref.server_id,
                "true",
                cwd=None,
                env=env,
                user=None,
                max_output_bytes=None,
                command_timeout_s=1.0,
                graceful_termination_timeout_s=2.0,
                request_timeout_s=5,
                polling=POLLING,
            )
        assert is_invalid_request(raised.value)
        # The message names the variable, never its value.
        assert "hidden" not in str(raised.value)
        assert FakeClient.instances[0].servers[0].bash_calls == []
    finally:
        await release_session(session)


async def test_a_malformed_bash_response_is_not_an_invalid_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from idegym.api.tools.bash import BashCommandResponse

    async def garbled(self: FakeServer, **kwargs: Any) -> Any:
        return BashCommandResponse.model_validate({"stdout": "x"})

    monkeypatch.setattr(FakeServer, "execute_bash", garbled)
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        ref = await started(session)
        with pytest.raises(ValueError) as raised:
            await session.execute_bash(
                ref.server_id,
                "true",
                cwd=None,
                env={"OK": "1"},
                user=None,
                max_output_bytes=None,
                command_timeout_s=1.0,
                graceful_termination_timeout_s=2.0,
                request_timeout_s=5,
                polling=POLLING,
            )
        assert not is_invalid_request(raised.value)
    finally:
        await release_session(session)


async def test_server_status_and_keepalive_map_the_sdk_responses() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        ref = await started(session)
        assert await session.server_status(ref.server_id) == IdeGymServerStatus(
            usable=True, availability="ALIVE", pod_ready=True, details=None
        )
        server = FakeClient.instances[0].servers[0]
        server.usable = False
        assert await session.server_status(ref.server_id) == IdeGymServerStatus(
            usable=False, availability="CRASHED", pod_ready=None, details="OOMKilled"
        )

        assert await session.keepalive(ref.server_id, 12.5) == 1_700_000_000_000
        assert server.keepalive_calls == [12.5]
    finally:
        await release_session(session)


async def test_file_transfers_use_the_sdk_chunked_calls(tmp_path: Path) -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        ref = await started(session)
        server = FakeClient.instances[0].servers[0]
        local = tmp_path / "payload.bin"

        uploaded = await session.upload_file(
            ref.server_id, local, "/work/payload.bin", chunk_bytes=4096, request_timeout_s=9.2, polling=POLLING
        )
        assert uploaded == 11
        args, kwargs = server.upload_calls[0]
        assert args == (local, "/work/payload.bin")
        assert (kwargs["chunk_bytes"], kwargs["request_timeout"]) == (4096, 10)

        downloaded = await session.download_file(
            ref.server_id, "/work/out.bin", local, chunk_bytes=4, request_timeout_s=2.2, polling=POLLING
        )
        assert downloaded == 10
        args, kwargs = server.download_calls[0]
        assert args == ("/work/out.bin", local)
        assert (kwargs["chunk_bytes"], kwargs["request_timeout"]) == (4, 3)
    finally:
        await release_session(session)


async def test_stop_server_forgets_the_server_only_on_success() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        ref = await started(session)
        client = FakeClient.instances[0]
        server = client.servers[0]

        client.stop_error = idegym.IdeGYMServerError("delete failed", status_code=500)
        with pytest.raises(idegym.IdeGYMServerError, match="delete failed"):
            await session.stop_server(ref.server_id, polling=POLLING, timeout_s=5)
        # Still held, so the provider's retry addresses the pod instead of reading "already stopped".
        assert ref.server_id in session._servers

        client.stop_error = None
        await session.stop_server(ref.server_id, polling=POLLING, timeout_s=5)
        assert client.stop_calls == [server, server]
        with pytest.raises(IdeGymUnknownServerError, match="not held by this session"):
            await session.stop_server(ref.server_id, polling=POLLING, timeout_s=5)
    finally:
        await release_session(session)


async def test_a_failed_start_does_not_track_a_server() -> None:
    session = await acquire_session(connection(), IdeGymAttributionConfig())
    try:
        FakeClient.instances[0].start_error = idegym.IdeGYMBusyError("no capacity", status_code=429)
        with pytest.raises(idegym.IdeGYMBusyError, match="no capacity"):
            await started(session)
        with pytest.raises(IdeGymUnknownServerError, match="not held"):
            await session.stop_server(1, polling=POLLING, timeout_s=5)
    finally:
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
    assert isinstance(aiohttp_transport, RetryingTransport)
    assert type(aiohttp_transport._transport).__name__ == "AiohttpTransport"
    assert type(httpx_transport._transport).__name__ == "AsyncHTTPTransport"


@requires_aiohttp_bridge
@pytest.mark.parametrize(("max_connections", "limit"), [(7, 7), (None, 0)], ids=["capped", "uncapped"])
async def test_the_aiohttp_backend_applies_the_connection_cap(max_connections: int | None, limit: int) -> None:
    """aiohttp spells "no cap" as 0; passing None on would leave its default cap of 100."""
    transport = build_transport(
        connection(transport_backend="aiohttp", max_connections=max_connections, keepalive_expiry_s=4.0)
    )
    client = transport._transport.get_client()
    try:
        assert (client.connector.limit, client.connector._keepalive_timeout) == (limit, 4.0)
    finally:
        await client.close()


def test_build_transport_applies_the_configured_pool_limits_and_retries() -> None:
    transport = build_transport(
        connection(
            transport_backend="httpx",
            max_connections=7,
            max_keepalive_connections=3,
            keepalive_expiry_s=4.0,
            connect_retries=5,
            poll_retries=6,
            retry_delay_s=0.25,
        )
    )
    # httpx exposes the resolved limits only on the pool, hence the private reach.
    pool = transport._transport._pool
    assert (pool._max_connections, pool._max_keepalive_connections, pool._keepalive_expiry) == (7, 3, 4.0)
    assert (transport._connect_retries, transport._poll_retries, transport._retry_delay_s) == (5, 6, 0.25)


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
    assert type(transport._transport).__name__ == "AsyncHTTPTransport"
    assert "httpx-aiohttp is not installed" in caplog.text


class Script:
    """An ``httpx.MockTransport`` handler that plays back one outcome per attempt."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.attempts = 0
        self.responses: list[httpx.Response] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        # A streamed body, so the response stays open until someone closes it.
        response = httpx.Response(
            outcome[0], headers=outcome[1] if len(outcome) > 1 else None, stream=httpx.ByteStream(b"{}")
        )
        self.responses.append(response)
        return response


def retrying(script: Script, *, connect_retries: int = 2, poll_retries: int = 3, delay: float = 0.0) -> Any:
    return RetryingTransport(
        httpx.MockTransport(script), connect_retries=connect_retries, poll_retries=poll_retries, retry_delay_s=delay
    )


URL = "http://idegym.test/api/operations/status/1"


async def test_a_get_is_retried_after_a_read_error() -> None:
    script = Script(httpx.ReadError("reset"), (200,))
    response = await retrying(script).handle_async_request(httpx.Request("GET", URL))
    assert response.status_code == 200
    assert script.attempts == 2


async def test_a_get_gives_up_after_poll_retries() -> None:
    script = Script(*(httpx.ReadError("reset") for _ in range(3)))
    with pytest.raises(httpx.ReadError):
        await retrying(script, poll_retries=2).handle_async_request(httpx.Request("GET", URL))
    assert script.attempts == 3


async def test_a_post_that_reached_the_server_is_never_sent_again() -> None:
    script = Script(httpx.ReadError("reset"), (200,))
    with pytest.raises(httpx.ReadError):
        await retrying(script).handle_async_request(httpx.Request("POST", URL, json={"script": "rm -rf x"}))
    assert script.attempts == 1


async def test_a_post_whose_connection_never_opened_is_retried_up_to_connect_retries() -> None:
    script = Script(*(httpx.ConnectError("refused") for _ in range(4)))
    with pytest.raises(httpx.ConnectError):
        await retrying(script, connect_retries=3).handle_async_request(httpx.Request("POST", URL, json={}))
    assert script.attempts == 4

    script = Script(httpx.ConnectError("refused"), (200,))
    response = await retrying(script).handle_async_request(httpx.Request("POST", URL, json={}))
    assert (response.status_code, script.attempts) == (200, 2)


async def test_an_aiohttp_disconnect_after_send_is_not_a_connect_failure() -> None:
    """httpx-aiohttp reports a server dropping a sent request as ConnectTimeout; the cause decides."""
    after_send = httpx.ConnectTimeout("disconnected")
    after_send.__cause__ = aiohttp.ServerDisconnectedError()
    script = Script(after_send, (200,))
    with pytest.raises(httpx.ConnectTimeout):
        await retrying(script).handle_async_request(httpx.Request("POST", URL, json={}))
    assert script.attempts == 1

    never_sent = httpx.ConnectTimeout("connect timed out")
    never_sent.__cause__ = aiohttp.ConnectionTimeoutError()
    script = Script(never_sent, (200,))
    response = await retrying(script).handle_async_request(httpx.Request("POST", URL, json={}))
    assert (response.status_code, script.attempts) == (200, 2)


async def test_a_streamed_body_is_never_retried() -> None:
    async def body() -> Any:
        yield b"chunk"

    script = Script(httpx.ConnectError("refused"), (200,))
    with pytest.raises(httpx.ConnectError):
        await retrying(script).handle_async_request(httpx.Request("POST", URL, content=body()))
    assert script.attempts == 1


async def test_a_get_answered_503_is_retried_and_the_503_is_closed() -> None:
    script = Script((503,), (200,))
    response = await retrying(script).handle_async_request(httpx.Request("GET", URL))
    assert response.status_code == 200
    assert script.responses[0].is_closed

    script = Script((503,), (200,))
    response = await retrying(script, poll_retries=0).handle_async_request(httpx.Request("GET", URL))
    assert response.status_code == 503
    assert response is script.responses[0]
    assert not response.is_closed


async def test_a_post_answered_503_is_returned_to_the_sdk() -> None:
    script = Script((503,), (200,))
    response = await retrying(script).handle_async_request(httpx.Request("POST", URL, json={}))
    assert (response.status_code, script.attempts) == (503, 1)


async def test_the_last_retryable_response_is_returned_not_raised() -> None:
    script = Script((502,), (504,), (429,))
    response = await retrying(script, poll_retries=2).handle_async_request(httpx.Request("GET", URL))
    assert (response.status_code, script.attempts) == (429, 3)


async def test_retry_after_overrides_the_backoff(caplog: pytest.LogCaptureFixture) -> None:
    # A 60 s backoff would time this test out; Retry-After: 0 must win.
    script = Script((429, {"Retry-After": "0"}), (200,))
    with caplog.at_level(logging.WARNING, logger=idegym_http.LOGGER.name):
        async with asyncio.timeout(5):
            response = await retrying(script, delay=60.0).handle_async_request(httpx.Request("GET", URL))
    assert response.status_code == 200
    assert "method=GET path=/api/operations/status/1 reason=HTTP 429 retry=1 sleep_s=0.0" in caplog.text


async def test_aclose_closes_the_inner_transport() -> None:
    class Closing(httpx.MockTransport):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    inner = Closing(Script())
    await RetryingTransport(inner, connect_retries=0, poll_retries=0, retry_delay_s=0).aclose()
    assert inner.closed
