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

"""Sandbox provider backed by the E2B Python SDK.

Works against e2b.dev itself and against any e2b-compatible gateway (point
``connection.api_url``/``connection.sandbox_url`` at it).

Two E2B concepts differ from the provider-neutral :class:`SandboxSpec` and are
handled explicitly rather than silently:

**Templates, not images.** E2B starts sandboxes from a pre-built *template*
name or ID, not from an arbitrary registry reference. Because tagged template
names and OCI image references can both contain ``:``, the direct
``SandboxSpec.image`` shortcut is deliberately limited to unambiguous names in
``[A-Za-z0-9_-]``. Use ``provider_options.template`` or
``create.template_map`` for tagged names, IDs, and image references.

**Resources are fixed at template build time.** ``cpu_count``/``memory_mb`` are
arguments to the template *build*, so a per-sandbox
``SandboxSpec.resources`` cannot be honoured at create time. Requests are
reported once per provider instance (or raise, with ``create.strict_resources``)
instead of being dropped quietly.

**Idle sandboxes.** An agent's sandbox sits idle while the agent waits on its
model. With ``create.idle_policy`` the provider parks a sandbox that no request
has touched for ``create.idle_timeout_s`` and brings it back before the next
request, so an idle sandbox holds no CPU or memory on the backend.
"""

import asyncio
import contextlib
import logging
import math
import os
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from time import monotonic
from typing import Any, Awaitable, Callable, TypeVar

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.e2b._sdk import E2B_SDK_CONSTRAINT, require_e2b_sdk


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# Conservative direct-template shortcut. E2B also supports tagged names such
# as ``name:v1`` and template IDs, but those must be explicit because ``:`` and
# other punctuation overlap with OCI image syntax in ``SandboxSpec.image``.
_DIRECT_TEMPLATE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# How a deployment publishes sandbox ports. See ``E2BProvider.endpoint``.
_PORT_ROUTING_MODES = ("auto", "hostname", "gateway")

# Return code for "no process exit status", shared with the other providers.
SANDBOX_RUNTIME_RETURN_CODE = 125

# What happens to an idle sandbox. See ``E2BCreateConfig.idle_policy``.
_IDLE_POLICIES = ("never", "pause")
# How long an explicit pause() waits for in-flight requests before giving up.
_PAUSE_DRAIN_TIMEOUT_S = 120.0

# Passed straight through to the SDK (``ApiParams``) on every call.
_API_PARAM_KEYS = (
    "api_key",
    "api_url",
    "sandbox_url",
    "domain",
    "debug",
    "validate_api_key",
    "headers",
    "api_headers",
    "proxy",
)


class E2BCreateError(SandboxCreateError):
    """Raised when a sandbox cannot be created."""


class E2BResumeError(RuntimeError):
    """Raised when a parked sandbox cannot be brought back for a request."""


def _require_e2b_sdk() -> Any:
    """Load the optional SDK through the shared helper.

    Keep this small wrapper local so provider unit tests can replace the SDK
    without importing or contacting E2B.
    """
    return require_e2b_sdk("The e2b sandbox provider")


def _config_from_mapping(cls: type[T], value: Any) -> T:
    """Build a config dataclass from a mapping, rejecting unknown keys."""
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(value).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {', '.join(sorted(unknown))}. Expected: {', '.join(sorted(allowed))}"
        )
    return cls(**dict(value))


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and (
        isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    )


def _validate_optional_number(name: str, value: Any, *, positive: bool) -> None:
    operator = "> 0" if positive else ">= 0"
    if value is not None and (not _is_finite_number(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be {operator}")


def _validate_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_nonnegative_number(name: str, value: Any) -> None:
    if not _is_finite_number(value) or value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class E2BConnectionConfig:
    """Connection settings forwarded to the SDK.

    Any field left ``None`` falls back to the SDK's own environment variables
    (``E2B_API_KEY``, ``E2B_API_URL``, ``E2B_SANDBOX_URL``, ``E2B_DOMAIN``, ...).
    """

    api_key: str | None = None
    api_url: str | None = None
    sandbox_url: str | None = None
    domain: str | None = None
    debug: bool | None = None
    validate_api_key: bool | None = None
    headers: dict[str, str] | None = None
    api_headers: dict[str, str] | None = None
    proxy: str | None = None
    request_timeout_s: float | None = None
    # How ``endpoint()`` publishes a sandbox port. "auto" resolves to "gateway"
    # when a sandbox URL is configured and "hostname" otherwise, mirroring the
    # SDK's own rule for whether one origin fronts every sandbox.
    port_routing: str = "auto"
    # Scheme for "hostname" routing. None mirrors the SDK (http in debug mode,
    # https otherwise); set it for a deployment that serves wildcard DNS over
    # plain HTTP. Ignored by "gateway" routing, which takes the scheme from the
    # gateway URL itself.
    port_scheme: str | None = None

    def __post_init__(self) -> None:
        _validate_optional_number("connection.request_timeout_s", self.request_timeout_s, positive=False)
        if self.port_routing not in _PORT_ROUTING_MODES:
            raise ValueError(f"connection.port_routing must be one of {', '.join(_PORT_ROUTING_MODES)}")
        if self.port_scheme is not None and self.port_scheme not in {"http", "https"}:
            raise ValueError("connection.port_scheme must be 'http' or 'https'")


@dataclass(frozen=True)
class E2BCreateConfig:
    """Sandbox creation settings."""

    # Template name or ID used only when SandboxSpec.image is omitted.
    template: str | None = None
    # Explicit ``SandboxSpec.image`` -> template name or ID mapping. Needed whenever
    # image references are not themselves valid aliases (registry refs contain
    # '/' and ':', which E2B rejects).
    template_map: dict[str, str] = field(default_factory=dict)
    # Sandbox lifetime in seconds; E2B kills the sandbox when it elapses.
    # ``SandboxSpec.ttl_s`` overrides it per sandbox.
    timeout_s: float | None = 3600.0
    allow_internet_access: bool = True
    secure: bool = True
    # Raise instead of warning when a spec requests resources E2B cannot apply
    # per sandbox (they are fixed when the template is built).
    strict_resources: bool = False
    # What happens to a sandbox that no request has touched for
    # ``idle_timeout_s`` since its create or its last request. ``pause`` pauses
    # it and resumes it before the next request: it holds no CPU or memory while
    # its agent waits on a model, each idle period costs one pause and one
    # resume, and it resumes on the node that paused it. ``never`` leaves
    # sandboxes running until they are closed.
    idle_policy: str = "never"
    idle_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if self.template is not None and (not isinstance(self.template, str) or not self.template.strip()):
            raise ValueError("create.template must be a non-empty string when provided")
        if not isinstance(self.template_map, Mapping):
            raise TypeError("create.template_map must be a mapping")
        for image, template in self.template_map.items():
            if not isinstance(image, str) or not image.strip():
                raise ValueError("create.template_map keys must be non-empty strings")
            if not isinstance(template, str) or not template.strip():
                raise ValueError("create.template_map values must be non-empty template strings")
        _validate_optional_number("create.timeout_s", self.timeout_s, positive=True)
        if self.idle_policy not in _IDLE_POLICIES:
            raise ValueError(f"create.idle_policy must be one of {', '.join(_IDLE_POLICIES)}")
        _validate_nonnegative_number("create.idle_timeout_s", self.idle_timeout_s)


@dataclass(frozen=True)
class E2BExecConfig:
    """Command execution settings."""

    # Applied when the provider receives ``timeout_s=None``. The public NeMo Gym
    # sandbox facade passes its own 180-second default unless callers override it.
    default_timeout_s: float | None = 180.0
    user: str | None = None
    request_timeout_s: float | None = None
    # Start commands detached and reattach by pid if the output stream drops.
    # The command keeps running inside the sandbox when the stream dies, so its
    # exit code survives a control-plane restart that would otherwise fail the
    # command outright.
    #
    # On by default, matching Harbor's own e2b environment, which dispatches
    # every command with ``background=True``. Set False to have the SDK block
    # on the stream instead; note that reattaching is lossy (see
    # :meth:`_run_background`).
    background: bool = True
    # How many times to reattach before giving up.
    reconnect_attempts: int = 2

    def __post_init__(self) -> None:
        _validate_optional_number("exec.default_timeout_s", self.default_timeout_s, positive=False)
        _validate_optional_number("exec.request_timeout_s", self.request_timeout_s, positive=False)
        _validate_nonnegative_int("exec.reconnect_attempts", self.reconnect_attempts)


@dataclass(frozen=True)
class E2BOperationConfig:
    """Retry policy for transient SDK/transport failures."""

    retries: int = 2
    retry_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0

    def __post_init__(self) -> None:
        _validate_nonnegative_int("operations.retries", self.retries)
        _validate_nonnegative_number("operations.retry_delay_s", self.retry_delay_s)
        _validate_nonnegative_number("operations.retry_max_delay_s", self.retry_max_delay_s)


@dataclass(eq=False)
class _E2BSandboxState:
    """Idle-policy bookkeeping for a sandbox this provider created.

    Kept by sandbox ID because :meth:`E2BProvider.resume` replaces
    ``handle.raw``. Every field is touched only on the sandbox's own event loop.
    """

    # Lifetime sent at create; reapplied on every resume, because E2B's
    # connect otherwise resets it to the SDK default.
    ttl_s: int | None
    # Held across a pause, a resume or a kill, never across a command, so a
    # request waits out a transition already sent instead of hitting a sandbox
    # that the backend reports as mid-transition.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Requests currently running; the sandbox is never parked while nonzero.
    inflight: int = 0
    # True after a park was attempted and nothing has brought the sandbox back:
    # the next request does that first. A pause whose outcome is unknown sets
    # it too; connecting to a sandbox that is still running only extends its
    # lifetime.
    parked: bool = False
    # Bumped by every request and explicit pause/resume. An idle park applies
    # only to the idle period it was scheduled for.
    epoch: int = 0
    park_task: asyncio.Task | None = None
    # Idle parks past their idle wait; those are no longer cancelled.
    parking: int = 0
    closed: bool = False
    # Set once traffic that bypasses this provider may reach the sandbox (an
    # endpoint, or a handle serialized to another process). The provider can
    # then no longer tell when the sandbox is idle, so it stops parking it.
    pinned: bool = False
    parked_at: float | None = None
    pauses: int = 0
    pause_s: float = 0.0
    resumes: int = 0
    resume_s: float = 0.0


class E2BProvider:
    """Provider backed by the E2B Python SDK."""

    name = "e2b"

    def __init__(
        self,
        *,
        connection: E2BConnectionConfig | Mapping[str, Any] | None = None,
        create: E2BCreateConfig | Mapping[str, Any] | None = None,
        exec: E2BExecConfig | Mapping[str, Any] | None = None,
        operations: E2BOperationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _config_from_mapping(E2BConnectionConfig, connection)
        self._create = _config_from_mapping(E2BCreateConfig, create)
        self._exec = _config_from_mapping(E2BExecConfig, exec)
        self._operations = _config_from_mapping(E2BOperationConfig, operations)
        self._warned_resource_specs: set[str] = set()
        # Sandboxes this provider created under an idle policy other than never.
        self._states: dict[str, _E2BSandboxState] = {}
        self._park_tasks: set[asyncio.Task] = set()

    # ---------------------------------------------------------------- helpers

    def _api_params(self) -> dict[str, Any]:
        """SDK ``ApiParams`` for connection-scoped calls; omitted keys fall back to env.

        Only ``create``/``connect``/``kill`` open a connection and accept these.
        Everything else runs against an already-connected sandbox -- see
        :meth:`_request_params`.
        """
        params = {key: getattr(self._connection, key) for key in _API_PARAM_KEYS}
        params = {key: value for key, value in params.items() if value is not None}
        if self._connection.request_timeout_s is not None:
            params["request_timeout"] = self._connection.request_timeout_s
        return params

    def _request_params(self) -> dict[str, Any]:
        """Return the request timeout; data-plane methods reject connection parameters."""
        if self._connection.request_timeout_s is None:
            return {}
        return {"request_timeout": self._connection.request_timeout_s}

    def _exec_request_timeout(self) -> float | None:
        """Return the E2B 2.36 stream-open timeout for command requests."""
        if self._exec.request_timeout_s is not None:
            return self._exec.request_timeout_s
        return self._connection.request_timeout_s

    def _resolve_template(self, spec: SandboxSpec) -> str:
        """Map a spec onto an E2B template.

        Precedence: ``provider_options.template`` -> ``create.template_map`` ->
        an unambiguous direct ``spec.image``. ``create.template`` is used only
        when ``spec.image`` is omitted; an unmapped image must not silently
        select an unrelated fallback template.

        Building a template from an image is provisioning, not part of starting
        a sandbox, so it lives in :mod:`nemo_gym.sandbox.providers.e2b.build`
        and never runs on this path.
        """
        options = spec.provider_options or {}
        if not isinstance(options, Mapping):
            raise TypeError("E2B provider_options must be a mapping")
        unknown = set(options) - {"template"}
        if unknown:
            raise ValueError(f"Unknown E2B provider option(s): {', '.join(sorted(unknown))}. Supported: template")
        option = options.get("template")
        if option is not None:
            if not isinstance(option, str) or not option:
                raise ValueError("E2B provider option 'template' must be a non-empty string")
            return option

        if spec.image:
            mapped = self._create.template_map.get(spec.image)
            if mapped:
                return str(mapped)
            if _DIRECT_TEMPLATE_RE.match(spec.image):
                return spec.image
            raise E2BCreateError(
                f"E2B starts sandboxes from a template name or ID, but SandboxSpec.image={spec.image!r} "
                "cannot be treated as an unambiguous direct template name. Map it with create.template_map, "
                "set provider_options.template (including for tagged names or template IDs), or build a "
                "template first with "
                "nemo_gym.sandbox.providers.e2b.build (its output is a ready-made template_map)."
            )

        if self._create.template:
            return self._create.template

        raise E2BCreateError(
            "No E2B template to start from: set SandboxSpec.image, provider_options.template, or create.template."
        )

    def _check_resources(self, spec: SandboxSpec, template: str) -> None:
        """Surface resource requests E2B cannot honour per sandbox."""
        resources = spec.resources
        requested = {
            name: getattr(resources, name)
            for name in ("cpu", "memory_mib", "disk_gib", "gpu", "gpu_type")
            if getattr(resources, name, None) is not None
        }
        if not requested:
            return
        detail = ", ".join(f"{key}={value}" for key, value in sorted(requested.items()))
        message = (
            f"E2B fixes sandbox resources when the template is built, so {detail} requested for template "
            f"{template!r} cannot be applied at create time. The bundled builder can set cpu_count and "
            "memory_mb; disk_gib, gpu, and gpu_type require a suitable pre-built E2B template."
        )
        if self._create.strict_resources:
            raise E2BCreateError(message)
        if template not in self._warned_resource_specs:
            self._warned_resource_specs.add(template)
            LOGGER.warning("%s", message)

    async def _with_retries(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        operation: str,
        retry_timeouts: bool = False,
    ) -> T:
        """Retry transient failures with exponential backoff."""
        e2b = _require_e2b_sdk()
        # Never retry these: they are deterministic and retrying only adds latency.
        non_retryable_candidates = [
            getattr(e2b, "NotFoundException", None),
            getattr(e2b, "SandboxNotFoundException", None),
            getattr(e2b, "AuthenticationException", None),
            getattr(e2b, "InvalidArgumentException", None),
            # Let the caller choose a backoff window for 429s instead of
            # amplifying a deployment-wide limit with fast local retries.
            getattr(e2b, "RateLimitException", None),
        ]
        if not retry_timeouts:
            non_retryable_candidates.append(getattr(e2b, "TimeoutException", None))
        non_retryable = tuple(exc for exc in non_retryable_candidates if isinstance(exc, type))
        attempts = self._operations.retries + 1
        delay = self._operations.retry_delay_s
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return await factory()
            except non_retryable:
                raise
            except Exception as exc:  # noqa: BLE001 - transport/5xx errors are provider-specific
                last_exc = exc
                if attempt == attempts - 1:
                    break
                LOGGER.debug("e2b %s failed (attempt %d/%d): %s", operation, attempt + 1, attempts, exc)
                await asyncio.sleep(min(delay, self._operations.retry_max_delay_s))
                delay *= 2
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _sandbox(handle: SandboxHandle) -> Any:
        sandbox = handle.raw
        if sandbox is None:
            raise RuntimeError(f"Sandbox handle {handle.sandbox_id} carries no e2b sandbox object")
        return sandbox

    # ------------------------------------------------------------- lifecycle

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.entrypoint:
            raise E2BCreateError(
                "SandboxSpec.entrypoint is not supported by the e2b provider; "
                "the E2B template defines the sandbox entrypoint"
            )
        template = self._resolve_template(spec)

        timeout_s = spec.ttl_s if spec.ttl_s is not None else self._create.timeout_s
        if timeout_s is not None and (not _is_finite_number(timeout_s) or timeout_s <= 0):
            raise E2BCreateError("E2B sandbox ttl_s must be > 0")
        if spec.ready_timeout_s is not None and (
            not _is_finite_number(spec.ready_timeout_s) or spec.ready_timeout_s <= 0
        ):
            raise E2BCreateError("E2B sandbox ready_timeout_s must be > 0")
        self._check_resources(spec, template)
        kwargs: dict[str, Any] = {
            "template": template,
            "timeout": max(1, math.ceil(timeout_s)) if timeout_s is not None else None,
            "allow_internet_access": self._create.allow_internet_access,
            "secure": self._create.secure,
            **self._api_params(),
        }
        if spec.ready_timeout_s is not None:
            # E2B retains this on the returned sandbox connection. Subsequent
            # provider calls explicitly reapply connection.request_timeout_s
            # when configured; otherwise the SDK-retained value remains.
            kwargs["request_timeout"] = float(spec.ready_timeout_s)
        if spec.env:
            kwargs["envs"] = {str(k): str(v) for k, v in spec.env.items()}
        if spec.metadata:
            kwargs["metadata"] = {str(k): str(v) for k, v in spec.metadata.items()}

        e2b = _require_e2b_sdk()
        try:
            # Creating a sandbox is not idempotent. Retrying an ambiguous
            # transport failure can leak the first, billable sandbox.
            sandbox = await e2b.AsyncSandbox.create(**kwargs)
        except Exception as exc:
            raise E2BCreateError(f"Failed to create e2b sandbox from template {template!r}: {exc}") from exc

        handle = SandboxHandle(sandbox_id=sandbox.sandbox_id, provider_name=self.name, raw=sandbox)
        if self._create.idle_policy != "never":
            self._states[handle.sandbox_id] = _E2BSandboxState(ttl_s=kwargs["timeout"])
            # The first command comes after the agent's first model call; until
            # then the sandbox would hold its resources for nothing.
            await self._after_request(handle, wait=True)
        return handle

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Return a descriptor for attaching to this sandbox from another process.

        Under an idle policy the sandbox stays running from then on: this
        provider cannot see the other process's requests.
        """
        await self._keep_running(handle)
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Attach to the sandbox described by ``descriptor``.

        E2B's public connect API applies its default sandbox timeout when none
        is supplied, so attaching may renew a sandbox that is close to expiry.
        """
        e2b = _require_e2b_sdk()
        sandbox_id = str(descriptor["sandbox_id"])
        sandbox = await self._with_retries(
            lambda: e2b.AsyncSandbox.connect(sandbox_id, **self._api_params()),
            operation="connect",
        )
        return SandboxHandle(sandbox_id=str(sandbox.sandbox_id), provider_name=self.name, raw=sandbox)

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        """Resolve a declared service port to a caller-reachable endpoint.

        E2B deployments publish sandbox ports in one of two shapes, and the
        endpoint differs accordingly:

        ``hostname``
            One DNS name per port, ``{port}-{sandbox_id}.{domain}``. The caller
            dials it directly, so no headers are needed.
        ``gateway``
            A single origin fronts every sandbox and every port, and the proxy
            picks the target from request headers. A self-hosted deployment
            without wildcard DNS for sandbox subdomains uses this, and points
            ``sandbox_url`` at that origin. The SDK mints ``X-Access-Token``
            and ``E2b-Sandbox-Id`` per sandbox and sets ``E2b-Sandbox-Port`` to
            envd's own port, so the target port must be re-pointed here.

        ``auto`` selects ``gateway`` exactly when a sandbox URL is configured,
        which is the same signal the SDK uses to decide that one origin serves
        every sandbox.

        The returned headers are the caller's to send verbatim. ``X-Access-Token``
        is minted per sandbox and is not the deployment's API key.

        Under an idle policy the sandbox stays running from then on: traffic to
        the endpoint bypasses this provider.
        """
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"Invalid sandbox TCP port: {port!r}")
        if port < 1 or port > 65535:
            raise ValueError(f"Sandbox TCP port must be between 1 and 65535, got {port}")

        await self._keep_running(handle)
        sandbox = self._sandbox(handle)
        connection_config = getattr(sandbox, "connection_config", None)
        if connection_config is None:
            raise RuntimeError(f"Sandbox {handle.sandbox_id} exposes no e2b connection config")

        mode = self._connection.port_routing
        if mode == "auto":
            # Mirror ConnectionConfig's own resolution order rather than reading
            # its private attribute: an explicit config value, else the SDK env var.
            configured_sandbox_url = self._connection.sandbox_url or os.environ.get("E2B_SANDBOX_URL")
            mode = "gateway" if configured_sandbox_url else "hostname"

        if mode == "hostname":
            scheme = self._connection.port_scheme or ("http" if connection_config.debug else "https")
            return SandboxEndpoint(endpoint=f"{scheme}://{sandbox.get_host(port)}")

        base_url = connection_config.get_sandbox_url(sandbox.sandbox_id, sandbox.sandbox_domain)
        try:
            headers = dict(connection_config.sandbox_headers)
        except AttributeError as exc:
            raise RuntimeError(
                "Gateway port routing needs ConnectionConfig.sandbox_headers, which this "
                f"e2b SDK does not expose. Install '{E2B_SDK_CONSTRAINT}', "
                "or set connection.port_routing to 'hostname'."
            ) from exc
        headers["E2b-Sandbox-Port"] = str(port)
        return SandboxEndpoint(endpoint=base_url, headers=headers)

    async def pause(self, handle: SandboxHandle) -> None:
        """Pause the sandbox through the E2B SDK without replaying failed requests.

        Under an idle policy, requests in flight finish first (pausing would
        cut them off), and the next request resumes the sandbox.
        """
        sandbox = self._sandbox(handle)
        state = self._states.get(handle.sandbox_id)
        if state is None:
            await sandbox.pause(**self._request_params())
            return
        self._cancel_pending_park(state)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _PAUSE_DRAIN_TIMEOUT_S
        while True:
            async with state.lock:
                self._sandbox(handle)
                if not state.inflight:
                    state.epoch += 1
                    if not await self._park_locked(handle, state, reason="pause()"):
                        raise RuntimeError(f"e2b sandbox {handle.sandbox_id!r} was not paused in its current state")
                    return
            if loop.time() >= deadline:
                raise RuntimeError(f"e2b sandbox {handle.sandbox_id!r} still has requests in flight; not pausing")
            await asyncio.sleep(0.1)

    async def resume(self, handle: SandboxHandle) -> None:
        """Resume the sandbox and replace its SDK handle only after success."""
        self._sandbox(handle)
        state = self._states.get(handle.sandbox_id)
        if state is None:
            e2b = _require_e2b_sdk()
            handle.raw = await e2b.AsyncSandbox.connect(handle.sandbox_id, **self._api_params())
            return
        self._cancel_pending_park(state)
        async with state.lock:
            self._sandbox(handle)
            state.epoch += 1
            await self._unpark_locked(handle, state, reason="resume()")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        e2b = _require_e2b_sdk()
        sandbox = self._sandbox(handle)
        not_found = tuple(
            exc
            for exc in (getattr(e2b, "SandboxNotFoundException", None), getattr(e2b, "NotFoundException", None))
            if isinstance(exc, type)
        )
        try:
            # Control-plane status avoids waking paused sandboxes with a health check.
            info = await sandbox.get_info(**self._request_params())
        except not_found:
            return SandboxStatus.STOPPED
        except Exception:  # noqa: BLE001 - status must not raise for transient issues
            return SandboxStatus.UNKNOWN
        return {"running": SandboxStatus.RUNNING, "paused": SandboxStatus.PAUSED}.get(
            info.state, SandboxStatus.UNKNOWN
        )

    async def close(self, handle: SandboxHandle) -> None:
        if handle.raw is None:
            return
        state = self._states.get(handle.sandbox_id)
        if state is None:
            await self._kill(handle)
            return
        state.closed = True
        self._cancel_pending_park(state)
        try:
            # A pause or resume already sent finishes first; killing under it
            # would only collide with the transition.
            async with state.lock:
                await self._kill(handle)
        except BaseException:
            state.closed = False
            raise
        self._states.pop(handle.sandbox_id, None)
        LOGGER.debug(
            "e2b: closed %s after %d pauses (%.1fs) and %d resumes (%.1fs)",
            handle.sandbox_id,
            state.pauses,
            state.pause_s,
            state.resumes,
            state.resume_s,
        )

    async def _kill(self, handle: SandboxHandle) -> None:
        """Kill the sandbox in any state; clear the handle only after confirmed cleanup."""
        e2b = _require_e2b_sdk()
        sandbox = handle.raw
        if sandbox is None:
            return
        not_found = tuple(
            exc
            for exc in (getattr(e2b, "SandboxNotFoundException", None), getattr(e2b, "NotFoundException", None))
            if isinstance(exc, type)
        )
        try:
            killed = await self._with_retries(
                lambda: sandbox.kill(**self._api_params()),
                operation="kill",
                retry_timeouts=True,
            )
        except not_found:
            # Already gone (expired TTL or killed elsewhere) - closing is idempotent.
            LOGGER.debug("e2b sandbox %s already gone on close", handle.sandbox_id)
        else:
            if killed is False:
                LOGGER.debug("e2b sandbox %s already gone on close", handle.sandbox_id)
        handle.raw = None

    async def aclose(self) -> None:
        """Wait out idle parks already sent; sandboxes own their connections."""
        for state in self._states.values():
            self._cancel_pending_park(state)
        loop = asyncio.get_running_loop()
        pending = [task for task in self._park_tasks if task.get_loop() is loop]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # ------------------------------------------------------------ idle policy

    @contextlib.asynccontextmanager
    async def _in_flight(self, handle: SandboxHandle) -> AsyncIterator[Any]:
        """Hold a request on the sandbox, bringing it back first if it was parked.

        Yields the SDK sandbox to use. An idle park still waiting out its delay
        is dropped; one already sent is waited out on the lock. While a request
        is held the sandbox is never parked.
        """
        state = self._states.get(handle.sandbox_id)
        if state is None:
            yield self._sandbox(handle)
            return
        self._cancel_pending_park(state)
        async with state.lock:
            self._sandbox(handle)
            state.epoch += 1
            if state.parked:
                try:
                    await self._unpark_locked(handle, state, reason="request")
                except Exception as exc:
                    raise E2BResumeError(f"e2b sandbox {handle.sandbox_id} could not be resumed: {exc}") from exc
            state.inflight += 1
        try:
            yield self._sandbox(handle)
        finally:
            state.inflight -= 1

    async def _keep_running(self, handle: SandboxHandle) -> None:
        """Bring a parked sandbox back and stop parking it for good."""
        state = self._states.get(handle.sandbox_id)
        if state is None:
            return
        state.pinned = True
        async with self._in_flight(handle):
            pass

    def _cancel_pending_park(self, state: _E2BSandboxState) -> None:
        """Cancel an idle park that is still waiting out its delay."""
        task = state.park_task
        if task is not None and not task.done() and state.parking == 0:
            task.cancel()
            state.park_task = None

    async def _after_request(self, handle: SandboxHandle, *, wait: bool) -> None:
        """Schedule the idle park that follows create or a request.

        The park always runs in its own task, shielded from the caller: a pause
        cancelled client-side can leave the sandbox mid-transition with nothing
        to finish it. With no idle delay and ``wait`` set, the caller waits for
        it.
        """
        state = self._states.get(handle.sandbox_id)
        if state is None or state.closed or state.pinned:
            return
        self._cancel_pending_park(state)
        task = asyncio.get_running_loop().create_task(self._park_after(handle, state, state.epoch))
        state.park_task = task
        self._park_tasks.add(task)
        task.add_done_callback(self._park_tasks.discard)
        if wait and self._create.idle_timeout_s <= 0:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
                # A newer request superseded the park.

    async def _park_after(self, handle: SandboxHandle, state: _E2BSandboxState, epoch: int) -> None:
        delay = self._create.idle_timeout_s
        try:
            if delay > 0:
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        state.parking += 1
        try:
            async with state.lock:
                if (
                    state.closed
                    or state.pinned
                    or state.inflight
                    or state.parked
                    or state.epoch != epoch
                    or handle.raw is None
                ):
                    return
                await self._park_locked(handle, state, reason="idle")
        except Exception as exc:  # noqa: BLE001 - a background task must not die silently
            # A failed park costs capacity, not correctness.
            LOGGER.warning("e2b: idle park of %s failed: %s", handle.sandbox_id, exc)
        finally:
            state.parking -= 1

    async def _park_locked(self, handle: SandboxHandle, state: _E2BSandboxState, *, reason: str) -> bool:
        """Pause the sandbox; the caller holds ``state.lock`` with nothing in flight.

        Returns whether the sandbox is now paused. An exception leaves the
        outcome unknown, so the next request reconnects first.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        sandbox = self._sandbox(handle)
        try:
            paused = await sandbox.pause(**self._request_params())
            if paused is False:
                # The SDK reports a 409 as "already paused", but gateways such
                # as AgentENV also answer 409 mid-transition.
                info = await sandbox.get_info(**self._request_params())
                if getattr(info, "state", None) != "paused":
                    LOGGER.warning("e2b: pause of %s (%s) was refused; it keeps running", handle.sandbox_id, reason)
                    return False
        except BaseException:
            state.parked = True
            raise
        elapsed = loop.time() - started
        state.parked = True
        state.parked_at = loop.time()
        state.pauses += 1
        state.pause_s += elapsed
        LOGGER.debug("e2b: paused %s (%s) in %.2fs", handle.sandbox_id, reason, elapsed)
        return True

    async def _unpark_locked(self, handle: SandboxHandle, state: _E2BSandboxState, *, reason: str) -> None:
        """Reconnect to a paused sandbox with its own lifetime; the caller holds ``state.lock``."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        e2b = _require_e2b_sdk()
        kwargs = self._api_params()
        if state.ttl_s is not None:
            kwargs["timeout"] = state.ttl_s
        handle.raw = await e2b.AsyncSandbox.connect(handle.sandbox_id, **kwargs)
        elapsed = loop.time() - started
        parked_for = loop.time() - state.parked_at if state.parked_at is not None else 0.0
        state.parked = False
        state.parked_at = None
        state.resumes += 1
        state.resume_s += elapsed
        LOGGER.debug(
            "e2b: resumed %s (%s) in %.2fs after %.1fs paused", handle.sandbox_id, reason, elapsed, parked_for
        )

    # -------------------------------------------------------------- commands

    async def _run_background(self, sandbox: Any, kwargs: dict[str, Any]) -> Any:
        """Run a command detached, reattaching by pid if the stream drops.

        ``commands.run(background=True)`` returns as soon as the process has
        started, handing back its pid. The command then keeps running inside
        the sandbox independently of the stream carrying its output, so losing
        that stream -- a gateway rollout, a proxy restart, a network blip --
        no longer destroys the command: reattach with ``commands.connect(pid)``
        and the real exit code still arrives.

        Reattaching has two inherent limits:

        * **Output emitted while disconnected is lost.** Output already
          received by the previous handle is retained and combined with the
          reattached stream, but the stream is live rather than replayed.
        * **The process must still be running.** ``connect`` raises
          not-found once it has exited, so a command that finishes during the
          gap cannot be recovered.
        """
        e2b = _require_e2b_sdk()
        command_timeout = kwargs.get("timeout")
        deadline = (
            monotonic() + float(command_timeout)
            if command_timeout is not None and float(command_timeout) > 0
            else None
        )
        handle = await sandbox.commands.run(**kwargs, background=True)
        pid = getattr(handle, "pid", None)
        stdout = ""
        stderr = ""

        # Never swallow these while reattaching: a non-zero exit and a command
        # timeout are real outcomes, not transport failures.
        exit_exc = getattr(e2b, "CommandExitException", None)
        terminal = tuple(
            exc
            for exc in (
                TimeoutError,
                exit_exc,
                getattr(e2b, "TimeoutException", None),
            )
            if isinstance(exc, type)
        )

        for attempt in range(self._exec.reconnect_attempts + 1):
            try:
                result = await handle.wait()
                result.stdout = stdout + (getattr(result, "stdout", "") or "")
                result.stderr = stderr + (getattr(result, "stderr", "") or "")
                return result
            except terminal as exc:
                if isinstance(exit_exc, type) and isinstance(exc, exit_exc):
                    exc.stdout = stdout + (getattr(exc, "stdout", "") or "")
                    exc.stderr = stderr + (getattr(exc, "stderr", "") or "")
                raise
            except Exception as exc:  # noqa: BLE001 - transport failure; try to reattach
                stdout += getattr(handle, "stdout", "") or ""
                stderr += getattr(handle, "stderr", "") or ""
                if pid is None or attempt >= self._exec.reconnect_attempts:
                    raise
                LOGGER.warning(
                    "e2b command stream lost (pid=%s, attempt %d/%d): %s; reattaching",
                    pid,
                    attempt + 1,
                    self._exec.reconnect_attempts,
                    exc,
                )
                reconnect_timeout = command_timeout
                reconnect_request_timeout = kwargs.get("request_timeout")
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"e2b command timed out after {command_timeout}s") from exc
                    reconnect_timeout = remaining
                    if reconnect_request_timeout in (None, 0):
                        reconnect_request_timeout = remaining
                    else:
                        reconnect_request_timeout = min(float(reconnect_request_timeout), remaining)
                try:
                    handle = await sandbox.commands.connect(
                        pid,
                        timeout=reconnect_timeout,
                        request_timeout=reconnect_request_timeout,
                    )
                except Exception as reconnect_exc:
                    # Typically not-found: the command finished while we were
                    # disconnected, so its result is gone. Report the original
                    # transport failure, which explains what actually happened.
                    raise exc from reconnect_exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        self._sandbox(handle)

        effective_timeout = timeout_s if timeout_s is not None else self._exec.default_timeout_s
        if effective_timeout is not None and (not _is_finite_number(effective_timeout) or effective_timeout < 0):
            raise ValueError("e2b command timeout_s must be >= 0")
        effective_user = user if user is not None else self._exec.user
        kwargs: dict[str, Any] = {"cmd": command}
        if cwd is not None:
            kwargs["cwd"] = cwd
        if env:
            kwargs["envs"] = {str(k): str(v) for k, v in env.items()}
        if effective_user is not None:
            kwargs["user"] = str(effective_user)
        # E2B treats ``timeout`` as "no timeout" when falsy; keep None explicit.
        kwargs["timeout"] = float(effective_timeout) if effective_timeout is not None else None
        kwargs["request_timeout"] = self._exec_request_timeout()

        completed = False
        try:
            result = await self._run_command(handle, kwargs, effective_timeout)
            completed = True
        except E2BResumeError as exc:
            return SandboxExecResult(
                stdout=None, stderr=str(exc), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="sandbox"
            )
        finally:
            await self._after_request(handle, wait=completed)
        return result

    async def _run_command(
        self, handle: SandboxHandle, kwargs: dict[str, Any], effective_timeout: float | None
    ) -> SandboxExecResult:
        e2b = _require_e2b_sdk()
        timeout_exc = getattr(e2b, "TimeoutException", None)
        exit_exc = getattr(e2b, "CommandExitException", None)
        try:
            async with self._in_flight(handle) as sandbox:
                if self._exec.background:
                    result = await self._run_background(sandbox, kwargs)
                else:
                    result = await sandbox.commands.run(**kwargs)
        except E2BResumeError:
            raise
        except Exception as exc:
            # A non-zero exit is a normal outcome, not a provider failure.
            if isinstance(exit_exc, type) and isinstance(exc, exit_exc):
                return SandboxExecResult(
                    stdout=getattr(exc, "stdout", None),
                    stderr=getattr(exc, "stderr", None),
                    return_code=int(getattr(exc, "exit_code", 1) or 1),
                )
            # E2B uses one TimeoutException for stream-open timeouts, the
            # running stream deadline, sandbox expiry and server cancellation.
            # Preserve that SDK detail instead of claiming every case was the
            # configured command deadline.
            if isinstance(timeout_exc, type) and isinstance(exc, timeout_exc):
                raise TimeoutError(
                    f"e2b command did not complete: {exc} (configured wait/stream budget={effective_timeout}s)"
                ) from exc
            raise

        return SandboxExecResult(
            stdout=getattr(result, "stdout", None),
            stderr=getattr(result, "stderr", None),
            return_code=int(getattr(result, "exit_code", 0) or 0),
        )

    # ----------------------------------------------------------------- files

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        self._sandbox(handle)
        completed = False
        try:
            async with self._in_flight(handle) as sandbox:
                await self._with_retries(
                    lambda: sandbox.files.write(target_path, data, **self._request_params()),
                    operation="write_file",
                )
            completed = True
        finally:
            await self._after_request(handle, wait=completed)

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        self._sandbox(handle)
        completed = False
        try:
            async with self._in_flight(handle) as sandbox:
                data = await self._with_retries(
                    lambda: sandbox.files.read(source_path, format="bytes", **self._request_params()),
                    operation="read_file",
                )
            completed = True
        finally:
            await self._after_request(handle, wait=completed)
        return data if isinstance(data, bytes) else bytes(data)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        await self.write_file(handle, target_path, source.read_bytes())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self.read_file(handle, source_path))
