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

The only module that calls the ``idegym`` SDK, so everything above works in the
provider-owned types declared here and the tests have one seam to fake.
(:mod:`~nemo_gym.sandbox.providers.idegym.errors` also imports the SDK's exception
types, lazily, to classify what these calls raise.)

*One client per process, not per sandbox.* An IdeGYM client is a heartbeating owner of
N server pods; registering one takes a table-level lock on the orchestrator's database,
and the resource quota is matched against its name. Sharing then forces the reference
count: stopping a client also terminates every server it owns, so it may only be
unregistered once the last provider has let go.

*The client lives on the SDK's own loop thread.* It is loop-bound -- an httpx session
plus a heartbeat task -- while the sync ``Sandbox`` facade gives every sandbox a loop of
its own, so it is driven through ``SharedIdeGYMClient``, which owns a loop in a daemon
thread and marshals every call onto it.
"""

import asyncio
import concurrent.futures
import logging
import math
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from nemo_gym.sandbox.attribution import RUN_KEY, log_attribution_once, resolve_attribution, resolve_run_id
from nemo_gym.sandbox.providers.idegym.config import (
    IdeGymAttributionConfig,
    IdeGymConnectionConfig,
    IdeGymPollingConfig,
)
from nemo_gym.sandbox.providers.idegym.errors import (
    IdeGymError,
    IdeGymInvalidRequestError,
    IdeGymUnknownServerError,
)
from nemo_gym.sandbox.providers.idegym.http import build_transport
from nemo_gym.sandbox.providers.idegym.naming import clamp_client_name, sanitize_name


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

SESSION_THREAD_NAME = "nemo-gym-idegym-session"

# The SDK waits max(60 s, 10%) past the wait it hands the orchestrator, so the readiness
# budget is split to keep the SDK's own deadline inside it.
START_DEADLINE_GRACE_MIN_S = 60.0
START_DEADLINE_GRACE_FACTOR = 1.1

_STOPPED_MESSAGE = "The IdeGYM session has been stopped and can no longer run operations"


def resolve_client_name(connection: IdeGymConnectionConfig, attribution: IdeGymAttributionConfig) -> str:
    """Return the IdeGYM client name this process registers under.

    An explicit ``connection.client_name`` always wins, because IdeGYM matches
    resource-quota rules by regex against this name. Otherwise it is derived from job
    attribution, so quota rules and the orchestrator's client list group by job.
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


@dataclass(frozen=True)
class IdeGymServerStatus:
    """Whether one server accepts requests right now, from the orchestrator's status endpoint."""

    usable: bool
    availability: str
    # None when the pod was not checked: the server is terminal or Kubernetes was unreachable.
    pod_ready: bool | None
    details: str | None


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


def _start_wait_timeout_s(timeout_s: float) -> int:
    """The orchestrator's start wait for a start-server call budgeted at ``timeout_s``.

    The SDK waits ``max(60 s, 10%)`` longer than the orchestrator does. From 120 s up,
    the wait leaves room for that, so the SDK's client deadline stays within
    ``timeout_s``. Below 120 s, fitting the grace would leave the orchestrator almost no
    time and every start would fail. The orchestrator gets half the budget instead, and
    the call can run up to 60 s past ``timeout_s``.
    """
    fitted = min(timeout_s - START_DEADLINE_GRACE_MIN_S, timeout_s / START_DEADLINE_GRACE_FACTOR)
    return max(1, math.floor(max(fitted, timeout_s / 2)))


class IdeGymSession:
    """A registered IdeGYM client plus the SDK loop thread it runs on.

    Instances are created and released through :func:`acquire_session` and
    :func:`release_session` rather than directly: the registry is what makes them
    shared and reference-counted.
    """

    def __init__(self, connection: IdeGymConnectionConfig, client_name: str) -> None:
        self._connection = connection
        self._client_name = client_name
        self._shared: Any = None
        self._starter: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        # SDK server handles, keyed by server id. Only ever touched on the SDK loop, or
        # after it has stopped, so the dict needs no lock of its own.
        self._servers: dict[int, Any] = {}

    @property
    def client_name(self) -> str:
        """The IdeGYM client name this session registered under. Used in log lines."""
        return self._client_name

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> "concurrent.futures.Future[None]":
        """Register the client on the SDK's loop thread.

        Registration blocks until the orchestrator answers, so it runs in a one-shot
        thread. The returned future resolves once the client is registered and
        heartbeating; the session then keeps serving until :meth:`stop`.
        """
        ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        starter = threading.Thread(target=self._enter, args=(ready,), name=SESSION_THREAD_NAME, daemon=True)
        self._starter = starter
        starter.start()
        return ready

    def _enter(self, ready: "concurrent.futures.Future[None]") -> None:
        try:
            from idegym.client import SharedIdeGYMClient

            shared = SharedIdeGYMClient(**self._client_arguments())
            # Registers the client and starts its heartbeat; the SDK closes its own loop
            # when this raises, so a failed registration leaves nothing behind.
            shared.__enter__()
        except BaseException as e:  # noqa: BLE001 - the future is the only channel back to the acquirers
            _resolve(ready, e)
            return
        self._shared = shared
        _resolve(ready)

    async def stop(self) -> None:
        """Unregister the client, which also terminates any server still open, and stop its loop.

        The SDK drains in-flight work for several seconds and joins its thread, so this
        runs in a worker thread rather than blocking the caller's loop. Safe to call
        more than once, and before :meth:`start`.
        """
        await asyncio.to_thread(self._exit)

    def _exit(self) -> None:
        with self._lifecycle_lock:
            # A registration still in flight (its only acquirer was cancelled, say) has to
            # finish before it can be undone; skipping it would leave a client heartbeating.
            if self._starter is not None:
                self._starter.join()
            shared, self._shared = self._shared, None
            if shared is None:
                return
            try:
                shared.__exit__(None, None, None)
            finally:
                self._servers.clear()

    # --- SDK construction --------------------------------------------------

    def _client_arguments(self) -> dict[str, Any]:
        """Keyword arguments for ``IdeGYMClient``, which ``SharedIdeGYMClient`` builds on its loop."""
        from idegym.api.auth import BasicAuth

        connection = self._connection
        auth = None
        if connection.username is not None or connection.password is not None:
            auth = BasicAuth(username=connection.username, password=connection.password)
        arguments: dict[str, Any] = {
            "orchestrator_url": connection.orchestrator_url,
            "name": self._client_name,
            "namespace": connection.namespace,
            "nodes_count": connection.nodes_count,
            "auth": auth,
            "heartbeat_interval_in_seconds": connection.heartbeat_interval_s,
            "request_timeout_in_seconds": connection.request_timeout_s,
            # Pool limits are set on the transport: the SDK rejects `limits` next to it.
            "transport": build_transport(connection),
        }
        # Left out, the SDK keeps tracing off unless IDEGYM_OTEL_TRACING_ENDPOINT is set.
        if connection.tracing_enabled:
            from idegym.api.config import OTELConfig, TracingConfig

            arguments["otel_config"] = OTELConfig(
                service_name=f"nemo-gym-{self._client_name}",
                tracing=TracingConfig(
                    endpoint=connection.tracing_endpoint,
                    timeout=connection.tracing_timeout_s,
                    auth=BasicAuth(username=connection.tracing_username, password=connection.tracing_password),
                ),
            )
        return arguments

    # --- request plumbing --------------------------------------------------

    async def _call(self, call: Callable[[Any], Awaitable[T]]) -> T:
        """Run ``call(client)`` on the SDK loop and await it on the caller's.

        A cancelled caller does not cancel the SDK call: the sandbox-side work has
        already started, and abandoning it mid-flight would leave the orchestrator's
        view and ours disagreeing. The result is dropped instead.
        """
        shared = self._shared
        if shared is None:
            raise IdeGymError(_STOPPED_MESSAGE)
        try:
            future = shared.submit(call)
        except RuntimeError as e:
            # The SDK refuses work once it has started closing, with a message that
            # says nothing about which session was stopped or why.
            raise IdeGymError(_STOPPED_MESSAGE) from e
        try:
            return await asyncio.shield(asyncio.wrap_future(future))
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if future.cancelled() and (task is None or not task.cancelling()):
                # The SDK cancelled it while draining at stop; the caller was not cancelled.
                raise IdeGymError("The IdeGYM session was stopped while this operation was in flight") from None
            raise

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
        response = await self._call(lambda client: client.health_check())
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
        server = await self._call(lambda client: self._start_and_track(client, kwargs))
        ref = IdeGymServerRef(
            server_id=int(server.server_id),
            server_name=str(kwargs["server_name"]),
            namespace=self._connection.namespace,
        )
        return ref

    async def _start_and_track(self, client: Any, kwargs: dict[str, Any]) -> Any:
        server = await client.start_server(**kwargs)
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
        # The SDK enforces the readiness budget itself, including its retries on the
        # orchestrator's 429 back-pressure, but adds a grace period on top of the wait.
        kwargs["server_start_wait_timeout_in_seconds"] = _start_wait_timeout_s(timeout_s)
        kwargs["polling_config"] = self._polling_config(polling, timeout_s)
        return kwargs

    async def execute_bash(
        self,
        server_id: int,
        script: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        user: str | None,
        max_output_bytes: int | None,
        command_timeout_s: float,
        graceful_termination_timeout_s: float,
        request_timeout_s: float,
        polling: IdeGymPollingConfig,
    ) -> IdeGymBashResult:
        """Run ``script`` through the server's bash tool, in ``cwd`` with ``env``, as ``user``.

        ``max_output_bytes`` bounds each stream (head and tail around a truncation
        marker); ``None`` returns them whole. Output is returned verbatim. A request the
        SDK would refuse to build raises ``IdeGymInvalidRequestError`` before anything is
        sent.
        """
        request_timeout = max(1, math.ceil(request_timeout_s))
        polling_config = self._polling_config(polling, request_timeout_s)
        bash_env = dict(env) if env is not None else None
        self._validate_bash_request(
            script=script,
            cwd=cwd,
            env=bash_env,
            user=user,
            max_output_bytes=max_output_bytes,
            command_timeout_s=command_timeout_s,
            graceful_termination_timeout_s=graceful_termination_timeout_s,
        )
        response = await self._call(
            lambda _: self._server(server_id).execute_bash(
                script=script,
                command_timeout=command_timeout_s,
                graceful_termination_timeout=graceful_termination_timeout_s,
                request_timeout=request_timeout,
                polling_config=polling_config,
                max_output_bytes=max_output_bytes,
                strip_output=False,
                cwd=cwd,
                env=bash_env,
                user=user,
            )
        )
        return IdeGymBashResult(
            stdout=response.stdout or "",
            stderr=response.stderr or "",
            exit_code=int(response.exit_code),
        )

    @staticmethod
    def _validate_bash_request(
        *,
        script: str,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | None,
        max_output_bytes: int | None,
        command_timeout_s: float,
        graceful_termination_timeout_s: float,
    ) -> None:
        """Build the SDK's bash request here, so only its validation reads as an invalid request.

        The SDK builds the same model inside ``execute_bash``, where its ``ValidationError``
        would be indistinguishable from one raised while parsing the response.
        """
        from idegym.api.tools.bash import BashCommandRequest
        from pydantic import ValidationError

        try:
            BashCommandRequest(
                command=script,
                timeout=command_timeout_s,
                graceful_termination_timeout=graceful_termination_timeout_s,
                max_output_bytes=max_output_bytes,
                strip_output=False,
                cwd=cwd,
                env=env or {},
                user=user,
            )
        except ValidationError as e:
            # Messages only: pydantic's own rendering repeats the input, env values included.
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in e.errors()
            )
            raise IdeGymInvalidRequestError(f"Invalid bash request: {problems}") from e

    async def server_status(self, server_id: int) -> IdeGymServerStatus:
        """Report whether the server is usable. Reading it does not count as activity."""
        response = await self._call(lambda _: self._server(server_id).status())
        return IdeGymServerStatus(
            usable=bool(response.usable),
            availability=str(response.availability),
            pod_ready=response.pod_ready,
            details=response.details,
        )

    async def keepalive(self, server_id: int, minutes: float) -> int:
        """Hold the server against the inactivity reaper and return the hold's end, in epoch ms.

        The orchestrator only ever extends a hold, so the returned end can be later than
        ``minutes`` from now.
        """
        response = await self._call(lambda _: self._server(server_id).keepalive(minutes=minutes))
        return int(response.keepalive_until)

    async def upload_file(
        self,
        server_id: int,
        local_path: Path,
        remote_path: str,
        *,
        chunk_bytes: int,
        request_timeout_s: float,
        polling: IdeGymPollingConfig,
    ) -> int:
        """Upload ``local_path`` byte-for-byte and return the remote file's size."""
        request_timeout = max(1, math.ceil(request_timeout_s))
        polling_config = self._polling_config(polling, request_timeout_s)
        return await self._call(
            lambda _: self._server(server_id).upload_file(
                local_path,
                remote_path,
                chunk_bytes=chunk_bytes,
                request_timeout=request_timeout,
                polling_config=polling_config,
            )
        )

    async def download_file(
        self,
        server_id: int,
        remote_path: str,
        local_path: Path,
        *,
        chunk_bytes: int,
        request_timeout_s: float,
        polling: IdeGymPollingConfig,
    ) -> int:
        """Download ``remote_path`` into ``local_path`` and return the byte count."""
        request_timeout = max(1, math.ceil(request_timeout_s))
        polling_config = self._polling_config(polling, request_timeout_s)
        return await self._call(
            lambda _: self._server(server_id).download_file(
                remote_path,
                local_path,
                chunk_bytes=chunk_bytes,
                request_timeout=request_timeout,
                polling_config=polling_config,
            )
        )

    async def stop_server(self, server_id: int, *, polling: IdeGymPollingConfig, timeout_s: float) -> None:
        """Delete the server pod and its Kubernetes resources. The SDK raises if that fails."""
        polling_config = self._polling_config(polling, timeout_s)
        await self._call(lambda client: self._stop_and_forget(client, server_id, polling_config))

    async def _stop_and_forget(self, client: Any, server_id: int, polling_config: Any) -> None:
        server = self._server(server_id)
        await client.stop_server(server, polling_config=polling_config)
        # Forgotten only on success: a server dropped after a failed delete would make
        # the caller's next attempt look like "already stopped".
        self._servers.pop(server_id, None)

    def _server(self, server_id: int) -> Any:
        server = self._servers.get(server_id)
        if server is None:
            raise IdeGymUnknownServerError(
                f"IdeGYM server {server_id} is not held by this session; it was already stopped or "
                f"belongs to another process"
            )
        return server


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
    """Fail with an actionable message when the IdeGYM SDK is missing or too old."""
    install_hint = (
        "Install it with `uv pip install 'nemo-gym[idegym]'` (or `pip install 'idegym-client>=0.13.1'`) in the "
        "runtime environment before selecting the idegym sandbox provider."
    )
    try:
        import idegym.client
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"The idegym-client SDK is required for the idegym sandbox provider. {install_hint}"
        ) from e
    if not hasattr(idegym.client, "SharedIdeGYMClient"):
        version = getattr(idegym.client, "__version__", "unknown")
        raise ImportError(
            f"The idegym sandbox provider requires idegym-client >= 0.13.1, found {version}. {install_hint}"
        )


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
