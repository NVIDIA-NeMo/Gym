# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The registered IdeGYM client shared by every sandbox in a process.

The only module that imports the ``idegym`` SDK, so everything above works in the
provider-owned types declared here and the tests have one seam to fake.

*One client per process, not per sandbox.* An IdeGYM client is a heartbeating owner of
N server pods; registering one takes a table-level lock on the orchestrator's database,
and the resource quota is matched against its name. Sharing then forces the reference
count: stopping a client also terminates every server it owns, so it may only be
unregistered once the last provider has let go.

*The client lives on a private event loop* in a daemon thread. It is loop-bound -- an
httpx session plus a heartbeat task -- while the sync ``Sandbox`` facade gives every
sandbox a loop of its own.
"""

import asyncio
import concurrent.futures
import logging
import math
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from nemo_gym.sandbox.attribution import RUN_KEY, log_attribution_once, resolve_attribution, resolve_run_id
from nemo_gym.sandbox.providers.idegym.config import (
    IdeGymAttributionConfig,
    IdeGymConnectionConfig,
    IdeGymPollingConfig,
    TransportBackend,
)
from nemo_gym.sandbox.providers.idegym.errors import IdeGymError, IdeGymUnknownServerError
from nemo_gym.sandbox.providers.idegym.naming import clamp_client_name, sanitize_name


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

SESSION_THREAD_NAME = "nemo-gym-idegym-session"
# Grace period for the private loop thread to unwind once the session shuts down.
LOOP_JOIN_TIMEOUT_S = 10.0


def resolve_client_name(connection: IdeGymConnectionConfig, attribution: IdeGymAttributionConfig) -> str:
    """Return the IdeGYM client name this process registers under.

    An explicit ``connection.client_name`` always wins, because IdeGYM matches
    resource-quota rules by regex against this name. Otherwise it is derived from job
    attribution, which is the closest IdeGYM gets to the per-sandbox Kubernetes labels
    other providers offer.
    """
    if connection.client_name:
        return clamp_client_name(connection.client_name)
    if not attribution.enabled:
        return clamp_client_name(attribution.client_name_prefix)
    resolved = resolve_attribution(
        team=attribution.team,
        user=attribution.user,
        workload=attribution.workload,
    )
    resolved[RUN_KEY] = resolve_run_id(attribution.run)
    log_attribution_once(resolved)
    # `run` is deliberately left out of the name: it changes per launch, and a name
    # that changes per launch defeats both quota-rule matching and the dashboard
    # grouping the name exists for.
    parts = [attribution.client_name_prefix, *(resolved.get(key) or "" for key in ("team", "user", "workload"))]
    name = sanitize_name("-".join(part for part in parts if part))
    return clamp_client_name(name or attribution.client_name_prefix)


@dataclass(frozen=True)
class IdeGymServerRef:
    """Identity of one started IdeGYM server pod."""

    server_id: int
    server_name: str
    namespace: str


@dataclass(frozen=True)
class IdeGymBashResult:
    """Outcome of one ``/api/tools/bash`` call."""

    stdout: str
    stderr: str
    exit_code: int


def _resolve(
    future: "concurrent.futures.Future[Any]", exception: BaseException | None = None, result: Any = None
) -> None:
    """Complete ``future`` unless a racing cancellation already finished it."""
    try:
        if exception is not None:
            future.set_exception(exception)
        else:
            future.set_result(result)
    except concurrent.futures.InvalidStateError:
        pass


async def _settle(coro: Awaitable[T], future: "concurrent.futures.Future[T]") -> None:
    """Run ``coro`` and report its outcome through ``future``.

    A cancelled caller cancels ``future`` but not this coroutine, so the orchestrator
    call finishes and its result is dropped. That is deliberate: the sandbox-side work
    has already started, and abandoning it mid-flight would leave the orchestrator's
    view and ours disagreeing.
    """
    try:
        result = await coro
    except BaseException as e:  # noqa: BLE001 - the future is the only channel back to the caller
        _resolve(future, e)
    else:
        _resolve(future, result=result)


class _SessionLoop:
    """A private asyncio event loop running in a daemon thread."""

    def __init__(self, name: str) -> None:
        self._loop = asyncio.new_event_loop()
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._pending_lock = threading.Lock()
        started = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(started,), name=name, daemon=True)
        self._thread.start()
        started.wait()

    def _run(self, started: threading.Event) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(started.set)
        self._loop.run_forever()

    def call_soon(self, callback: Callable[[], None]) -> None:
        self._loop.call_soon_threadsafe(callback)

    def submit(self, factory: Callable[[], Awaitable[T]]) -> "concurrent.futures.Future[T]":
        """Schedule ``factory()`` on the private loop and return its future.

        The coroutine is created *on* the loop, so anything the factory allocates
        (locks, tasks) is bound to the right loop.
        """
        if self._loop.is_closed():
            # Reported as a session error rather than asyncio's bare "Event loop is
            # closed", which says nothing about which session was stopped or why.
            raise IdeGymError("The IdeGYM session has been stopped and can no longer run operations")
        future: concurrent.futures.Future[T] = concurrent.futures.Future()
        with self._pending_lock:
            self._pending.add(future)

        def forget(_: Any) -> None:
            with self._pending_lock:
                self._pending.discard(future)

        future.add_done_callback(forget)

        def start() -> None:
            try:
                coro = factory()
            except BaseException as e:  # noqa: BLE001 - the future is the only channel back
                # A factory can raise before it ever awaits (an unknown server id, say).
                # Left to the loop's exception handler, that would never resolve the
                # future and the caller would wait forever.
                _resolve(future, e)
                return
            task = self._loop.create_task(_settle(coro, future))
            # Hold a reference until the future completes so the task cannot be
            # garbage-collected mid-flight.
            future.add_done_callback(lambda _: task)

        self._loop.call_soon_threadsafe(start)
        return future

    def stop(self) -> None:
        """Stop the loop, join its thread, and fail anything still in flight.

        Without the last part a caller awaiting an operation when the session is
        released would wait forever: the loop stops before its task can resolve the
        future, and nothing else ever will. Safe to call more than once.
        """
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=LOOP_JOIN_TIMEOUT_S)
        if self._thread.is_alive():  # pragma: no cover - only reachable with a wedged loop
            LOGGER.warning(
                f"IdeGYM session loop thread did not exit within {LOOP_JOIN_TIMEOUT_S:g}s; leaving it running"
            )
            return
        self._loop.close()
        with self._pending_lock:
            abandoned, self._pending = self._pending, set()
        for future in abandoned:
            _resolve(future, IdeGymError("The IdeGYM session was stopped while this operation was in flight"))


class IdeGymSession:
    """A registered IdeGYM client plus the private loop it runs on.

    Instances are created and released through :func:`acquire_session` and
    :func:`release_session` rather than directly: the registry is what makes them
    shared and reference-counted.
    """

    def __init__(self, connection: IdeGymConnectionConfig, client_name: str) -> None:
        self._connection = connection
        self._client_name = client_name
        self._loop = _SessionLoop(SESSION_THREAD_NAME)
        self._client: Any = None
        self._shutdown: asyncio.Event | None = None
        self._serving: concurrent.futures.Future[None] | None = None
        # SDK server handles, keyed by server id. Only ever touched on the private
        # loop, so the dict needs no lock of its own.
        self._servers: dict[int, Any] = {}

    @property
    def client_name(self) -> str:
        """The IdeGYM client name this session registered under. Used in log lines."""
        return self._client_name

    def _require_client(self) -> Any:
        client = self._client
        if client is None:
            raise IdeGymError("The IdeGYM session is not running")
        return client

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> "concurrent.futures.Future[None]":
        """Register the client on the private loop.

        The returned future resolves once the client is registered and
        heartbeating; the session then keeps serving until :meth:`stop`.
        """
        ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        serving = self._loop.submit(lambda: self._serve(ready))
        self._serving = serving
        # An unexpected raise before `ready` is resolved would otherwise leave
        # every acquirer waiting forever.
        serving.add_done_callback(lambda finished: _propagate_failure(finished, ready))
        return ready

    async def _serve(self, ready: "concurrent.futures.Future[None]") -> None:
        """Own the SDK client's context for the whole session lifetime."""
        self._shutdown = asyncio.Event()
        client = self._build_client()
        # `async with` registers the client and starts its heartbeat; on exit it stops
        # the client, which also terminates any server the process failed to close.
        # Holding the context for the session's lifetime makes that a guarantee.
        try:
            async with client:
                self._client = client
                _resolve(ready)
                await self._shutdown.wait()
        finally:
            self._client = None
            self._servers.clear()

    async def stop(self) -> None:
        """Unregister the client, then stop the private loop.

        Stopping the loop joins its thread, so it runs in a worker thread rather than
        blocking the caller's, and it happens even if unregistering failed — otherwise
        the thread leaks for the life of the process.
        """
        serving, self._serving = self._serving, None
        try:
            if serving is not None and self._shutdown is not None:
                shutdown = self._shutdown
                self._loop.call_soon(shutdown.set)
                await asyncio.wrap_future(serving)
        finally:
            await asyncio.to_thread(self._loop.stop)

    # --- SDK construction --------------------------------------------------

    def _build_client(self) -> Any:
        from idegym.api.auth import BasicAuth
        from idegym.api.config import OTELConfig, TracingConfig
        from idegym.client import IdeGYMClient

        connection = self._connection
        auth = None
        if connection.username is not None or connection.password is not None:
            auth = BasicAuth(username=connection.username, password=connection.password)
        # Always pass an explicit OTEL config: left to its own devices the SDK traces
        # to its default off-box collector, and NeMo Gym does not ship telemetry to a
        # third party unless it was configured to.
        otel_config = OTELConfig(
            service_name=f"nemo-gym-{self._client_name}",
            tracing=TracingConfig(
                endpoint=connection.tracing_endpoint,
                timeout=connection.tracing_timeout_s,
                auth=BasicAuth(username=connection.tracing_username, password=connection.tracing_password),
            ),
        )
        client = IdeGYMClient(
            orchestrator_url=connection.orchestrator_url,
            name=self._client_name,
            namespace=connection.namespace,
            nodes_count=connection.nodes_count,
            auth=auth,
            heartbeat_interval_in_seconds=connection.heartbeat_interval_s,
            request_timeout_in_seconds=connection.request_timeout_s,
            otel_config=otel_config,
        )
        install_transport(client, connection)
        return client

    # --- request plumbing --------------------------------------------------

    async def _call(self, factory: Callable[[], Awaitable[T]]) -> T:
        """Run ``factory()`` on the session loop and await it on the caller's."""
        return await asyncio.wrap_future(self._loop.submit(factory))

    def _polling_config(self, polling: IdeGymPollingConfig, wait_timeout_s: float) -> Any:
        from idegym.client.operations.utils import PollingConfig

        return PollingConfig(
            initial_delay_in_sec=polling.initial_delay_s,
            wait_timeout_in_sec=max(1, math.ceil(wait_timeout_s)),
            poll_interval_in_sec=polling.interval_s,
            factor_for_exponential_wait=polling.backoff_factor,
            max_delay_for_exponential_wait_in_sec=polling.max_delay_s,
        )

    # --- operations --------------------------------------------------------

    async def health(self) -> str:
        """Return the orchestrator's reported health status."""
        response = await self._call(lambda: self._require_client().health_check())
        return str(getattr(response, "status", "") or "")

    async def start_server(
        self,
        request: Mapping[str, Any],
        *,
        polling: IdeGymPollingConfig,
        timeout_s: float,
    ) -> IdeGymServerRef:
        """Start one server pod and return its identity once it is up.

        ``request`` is the plain-dict form of the IdeGYM start-server arguments
        produced by :mod:`nemo_gym.sandbox.providers.idegym.spec`; the SDK models
        it needs are built here so nothing above this module imports them.
        """
        kwargs = self._build_start_kwargs(request, polling=polling, timeout_s=timeout_s)
        server = await self._call(lambda: self._start_and_track(kwargs))
        return IdeGymServerRef(
            server_id=int(server.server_id),
            server_name=str(kwargs["server_name"]),
            namespace=self._connection.namespace,
        )

    async def _start_and_track(self, kwargs: dict[str, Any]) -> Any:
        server = await self._require_client().start_server(**kwargs)
        self._servers[int(server.server_id)] = server
        return server

    def _build_start_kwargs(
        self,
        request: Mapping[str, Any],
        *,
        polling: IdeGymPollingConfig,
        timeout_s: float,
    ) -> dict[str, Any]:
        from idegym.api.orchestrator.servers import ServerKind, ServerReuseStrategy, SnapshotRef
        from idegym.api.pod_spec import (
            KubernetesEnvFromSource,
            KubernetesPodOverrides,
            KubernetesVolume,
            KubernetesVolumeMount,
        )
        from idegym.api.resources import KubernetesResources

        models = {"resources": KubernetesResources, "pod_overrides": KubernetesPodOverrides, "snapshot": SnapshotRef}
        model_lists = {
            "volumes": KubernetesVolume,
            "volume_mounts": KubernetesVolumeMount,
            "env_from": KubernetesEnvFromSource,
        }
        enums = {"reuse_strategy": ServerReuseStrategy, "server_kind": ServerKind}

        kwargs = dict(request)
        for key, model in models.items():
            if (value := kwargs.get(key)) is not None:
                kwargs[key] = model.model_validate(value)
        for key, model in model_lists.items():
            if value := kwargs.get(key):
                kwargs[key] = [model.model_validate(entry) for entry in value]
        for key, enum in enums.items():
            if (value := kwargs.get(key)) is not None:
                kwargs[key] = enum(value)
        kwargs["namespace"] = self._connection.namespace
        # The SDK enforces the readiness budget itself, including its retries on
        # the orchestrator's 429 back-pressure, so it gets the whole budget.
        kwargs["server_start_wait_timeout_in_seconds"] = max(1, math.ceil(timeout_s))
        kwargs["polling_config"] = self._polling_config(polling, timeout_s)
        return kwargs

    async def execute_bash(
        self,
        server_id: int,
        script: str,
        *,
        command_timeout_s: float,
        graceful_termination_timeout_s: float,
        request_timeout_s: float,
        polling: IdeGymPollingConfig,
    ) -> IdeGymBashResult:
        """Run ``script`` through the server's bash tool."""
        request_timeout = max(1, math.ceil(request_timeout_s))
        polling_config = self._polling_config(polling, request_timeout_s)
        response = await self._call(
            lambda: self._server(server_id).execute_bash(
                script=script,
                command_timeout=command_timeout_s,
                graceful_termination_timeout=graceful_termination_timeout_s,
                request_timeout=request_timeout,
                polling_config=polling_config,
            )
        )
        return IdeGymBashResult(
            stdout=response.stdout or "",
            stderr=response.stderr or "",
            exit_code=int(response.exit_code),
        )

    async def list_capabilities(self, server_id: int) -> list[str]:
        """Return the plugins loaded in the running server container.

        This is the cheapest orchestrator call that touches both the server's
        database record and the pod itself, which makes it the provider's liveness
        probe.
        """
        response = await self._call(lambda: self._server(server_id).list_capabilities())
        return [str(plugin) for plugin in (response.plugins or [])]

    async def stop_server(self, server_id: int, *, polling: IdeGymPollingConfig, timeout_s: float) -> None:
        """Delete the server pod and its Kubernetes resources."""
        polling_config = self._polling_config(polling, timeout_s)
        await self._call(lambda: self._stop_and_forget(server_id, polling_config))

    async def _stop_and_forget(self, server_id: int, polling_config: Any) -> None:
        server = self._server(server_id)
        response = await self._require_client().stop_server(server, polling_config=polling_config)
        # The SDK reports a failed delete operation by *returning* an ErrorResponse
        # rather than raising, so an unchecked return records a live pod as stopped.
        self._raise_for_error_response(response, f"Deleting IdeGYM server {server_id}")
        # Forgotten only on success: a server dropped after a failed delete would make
        # the caller's next attempt look like "already stopped".
        self._servers.pop(server_id, None)

    @staticmethod
    def _raise_for_error_response(response: Any, action: str) -> None:
        from idegym.api.orchestrator.servers import ErrorResponse

        if isinstance(response, ErrorResponse):
            # The dump carries `status_code`, which the error classifier reads to tell
            # an already-gone server from a real failure.
            raise IdeGymError(f"{action} failed: {response.model_dump()}")

    def _server(self, server_id: int) -> Any:
        server = self._servers.get(server_id)
        if server is None:
            raise IdeGymUnknownServerError(
                f"IdeGYM server {server_id} is not held by this session; it was already stopped or "
                f"belongs to another process"
            )
        return server


def _propagate_failure(source: "concurrent.futures.Future[Any]", target: "concurrent.futures.Future[None]") -> None:
    if target.done():
        return
    exception = None if source.cancelled() else source.exception()
    _resolve(target, exception or IdeGymError("The IdeGYM session ended before it became ready"))


_TRANSPORT_WARNED = False


def install_transport(client: Any, connection: IdeGymConnectionConfig) -> None:
    """Replace the SDK's HTTP transport with a pool-bounded one.

    The IdeGYM client builds its own ``httpx.AsyncClient`` and exposes no transport
    hook, so the pool limits and the aiohttp backend Gym expects can only be installed
    by swapping it out afterwards. That reaches into a private attribute, so a shape
    change in the SDK degrades to a warning rather than failing the run.
    """
    global _TRANSPORT_WARNED

    http_client = getattr(client, "_http_client", None)
    if http_client is None or not hasattr(http_client, "_transport"):
        if not _TRANSPORT_WARNED:
            _TRANSPORT_WARNED = True
            LOGGER.warning(
                "The installed idegym-client does not expose the httpx client this provider configures "
                "connection pooling on; falling back to the SDK's own transport. Pool limits and "
                "connection.transport_backend have no effect."
            )
        return
    # The transport being replaced was built moments ago and has never issued a
    # request, so its connection pool holds nothing that needs closing.
    http_client._transport = build_transport(connection)


def build_transport(connection: IdeGymConnectionConfig) -> Any:
    """Build the httpx transport for the configured backend and pool limits."""
    import httpx

    limits = httpx.Limits(
        max_connections=connection.max_connections,
        max_keepalive_connections=connection.max_keepalive_connections,
        keepalive_expiry=connection.keepalive_expiry_s,
    )
    if connection.transport_backend == TransportBackend.AIOHTTP:
        try:
            from httpx_aiohttp import AiohttpTransport
        except ImportError:
            LOGGER.warning(
                "connection.transport_backend=aiohttp requested but httpx-aiohttp is not installed; "
                "falling back to the httpx transport"
            )
        else:
            return AiohttpTransport(limits=limits, retries=connection.connect_retries)
    return httpx.AsyncHTTPTransport(limits=limits, retries=connection.connect_retries)


@dataclass
class _SessionEntry:
    session: IdeGymSession
    ready: "concurrent.futures.Future[None]"
    refcount: int = field(default=0)


_SESSIONS: dict[tuple[IdeGymConnectionConfig, str], _SessionEntry] = {}
_SESSIONS_LOCK = threading.Lock()


def _observe(source: "concurrent.futures.Future[None]") -> "concurrent.futures.Future[None]":
    """A private view of ``source`` for one waiter.

    Concurrent acquirers of a session all wait on the same readiness future, and
    ``asyncio.wrap_future`` propagates cancellation into what it wraps — so without a
    per-waiter view, one cancelled acquirer would cancel registration for the rest.
    """
    view: concurrent.futures.Future[None] = concurrent.futures.Future()

    def relay(finished: "concurrent.futures.Future[None]") -> None:
        if finished.cancelled():
            view.cancel()
            return
        _resolve(view, finished.exception())

    source.add_done_callback(relay)
    return view


def require_idegym_client() -> None:
    """Fail with an actionable message when the IdeGYM SDK is not installed."""
    try:
        import idegym.client  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The idegym-client SDK is required for the idegym sandbox provider. Install it with "
            "`uv pip install 'nemo-gym[idegym]'` (or `pip install idegym-client`) in the runtime "
            "environment before selecting the idegym sandbox provider."
        ) from e


async def acquire_session(connection: IdeGymConnectionConfig, attribution: IdeGymAttributionConfig) -> IdeGymSession:
    """Return the shared session for ``connection``, registering it if needed.

    Every acquired session must be handed back to :func:`release_session` exactly
    once; the last release unregisters the IdeGYM client.
    """
    require_idegym_client()
    key = (connection, resolve_client_name(connection, attribution))
    with _SESSIONS_LOCK:
        entry = _SESSIONS.get(key)
        if entry is None:
            session = IdeGymSession(connection, key[1])
            entry = _SessionEntry(session=session, ready=session.start())
            _SESSIONS[key] = entry
        entry.refcount += 1
    try:
        await asyncio.wrap_future(_observe(entry.ready))
    except BaseException:
        # Registration failed for everyone waiting on this entry. Drop the
        # reference so a later acquire starts a fresh session rather than
        # inheriting the failure.
        await _release_entry(key, entry)
        raise
    return entry.session


async def release_session(session: IdeGymSession) -> None:
    """Give back one reference to a shared session."""
    with _SESSIONS_LOCK:
        key = next((candidate for candidate, entry in _SESSIONS.items() if entry.session is session), None)
        entry = _SESSIONS.get(key) if key is not None else None
    if key is None or entry is None:
        return
    await _release_entry(key, entry)


async def _release_entry(key: tuple[IdeGymConnectionConfig, str], entry: _SessionEntry) -> None:
    with _SESSIONS_LOCK:
        if _SESSIONS.get(key) is not entry:
            return
        entry.refcount -= 1
        if entry.refcount > 0:
            return
        # Remove before tearing down: an acquire racing this release must start a
        # new session rather than take a reference on a dying one.
        _SESSIONS.pop(key, None)
    try:
        await entry.session.stop()
    except Exception:
        LOGGER.warning(f"Failed to unregister the IdeGYM client {entry.session.client_name!r} cleanly", exc_info=True)


def active_session_count() -> int:
    """Number of live shared sessions. Exposed for tests and diagnostics."""
    with _SESSIONS_LOCK:
        return len(_SESSIONS)
