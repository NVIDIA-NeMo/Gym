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

"""OpenShell sandbox provider: sandboxes managed by an OpenShell gateway (github.com/NVIDIA/OpenShell).

The provider talks to the gateway's gRPC control plane through the synchronous ``openshell``
SDK; blocking SDK calls run on a thread pool bounded by ``exec.concurrency``. The client and
thread pool are cached at module scope keyed on the connection config, so concurrent sandboxes
created from identical provider configs share one gRPC channel and one pool instead of
allocating one per sandbox. The SDK has no file-transfer API, so uploads stream bytes through
``exec`` stdin (chunked to stay under the gateway's gRPC message size limit) and downloads
round-trip through ``base64`` on the sandbox's stdout.
"""

import asyncio
import base64
import binascii
import dataclasses
import functools
import logging
import math
import posixpath
import shlex
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.attribution import PROCESS_KEY, AttributionConfig, resolve_process_id, sanitize_label_value
from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

# OpenShell includes sandbox names in service-routing hostnames and accepts at most 19 characters.
MAX_SANDBOX_NAME_LENGTH = 19
# Allowed shortfall between the elapsed time and the timeout (clock and integer-second rounding).
EXEC_TIMEOUT_SLACK_S = 0.5
EXEC_TIMEOUT_SLACK_FRACTION = 0.1
MAX_DELETE_ATTEMPTS = 3
SANDBOX_NAME_PREFIX = "ng-"
SANDBOX_NAME_RANDOM_HEX_LENGTH = MAX_SANDBOX_NAME_LENGTH - len(SANDBOX_NAME_PREFIX)
SANDBOX_LABEL = "nemo-gym.sandbox"
READY_PROBE_COMMAND = "printf openshell-sandbox-ready"
READY_PROBE_EXPECTED = "openshell-sandbox-ready"
SANDBOX_RUNTIME_RETURN_CODE = 125
# Each upload chunk travels as one ExecSandboxRequest.stdin proto field, so chunks (plus the
# rest of the request message) must stay under the gateway's gRPC max message decode size --
# observed at 1 MiB on current gateway builds. 512 KiB leaves comfortable headroom.
DEFAULT_UPLOAD_CHUNK_BYTES = 512 * 1024


class OpenShellCreateError(SandboxCreateError):
    """Raised when the OpenShell gateway cannot create a sandbox."""


class OpenShellCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a new sandbox fails its readiness probe."""


def _generate_sandbox_name() -> str:
    """Return a DNS-safe name within OpenShell's routable-name limit."""
    return SANDBOX_NAME_PREFIX + uuid.uuid4().hex[:SANDBOX_NAME_RANDOM_HEX_LENGTH]


def _require_openshell() -> None:
    try:
        import openshell  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The openshell SDK is required for the openshell sandbox provider. Install "
            "nemo-gym[openshell] in the runtime image before using env.sandbox.provider.name=openshell."
        ) from e


def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


def _normalize_image(image: str) -> str:
    prefix = "docker://"
    return image[len(prefix) :] if image.startswith(prefix) else image


@functools.cache
def _phase_to_status_map() -> dict[int, SandboxStatus]:
    """SandboxPhase -> SandboxStatus, built from the SDK's generated proto constants."""
    from openshell._proto import openshell_pb2

    return {
        openshell_pb2.SANDBOX_PHASE_UNSPECIFIED: SandboxStatus.UNKNOWN,
        openshell_pb2.SANDBOX_PHASE_PROVISIONING: SandboxStatus.STARTING,
        openshell_pb2.SANDBOX_PHASE_STARTING: SandboxStatus.STARTING,
        openshell_pb2.SANDBOX_PHASE_READY: SandboxStatus.RUNNING,
        openshell_pb2.SANDBOX_PHASE_ERROR: SandboxStatus.ERROR,
        openshell_pb2.SANDBOX_PHASE_DELETING: SandboxStatus.STOPPED,
        openshell_pb2.SANDBOX_PHASE_STOPPING: SandboxStatus.STOPPED,
        openshell_pb2.SANDBOX_PHASE_STOPPED: SandboxStatus.STOPPED,
        openshell_pb2.SANDBOX_PHASE_SUSPENDING: SandboxStatus.STOPPED,
        openshell_pb2.SANDBOX_PHASE_SUSPENDED: SandboxStatus.STOPPED,
        openshell_pb2.SANDBOX_PHASE_COMPLETED: SandboxStatus.STOPPED,
        openshell_pb2.SANDBOX_PHASE_UNKNOWN: SandboxStatus.UNKNOWN,
    }


def _format_conditions(ref: Any) -> str:
    """The sandbox's reported conditions as a message suffix (the gateway's own reason, e.g. a pull failure)."""
    parts = [
        f"{c.type}={c.status}" + (f" ({c.reason}: {c.message})" if c.reason or c.message else "")
        for c in getattr(ref, "conditions", ())
    ]
    return f"; conditions: {'; '.join(parts)}" if parts else ""


def _phase(name: str) -> int:
    from openshell._proto import openshell_pb2

    return getattr(openshell_pb2, name)


def _grpc_status_code(exc: BaseException) -> Any | None:
    """The ``grpc.StatusCode`` of an RPC error, else None."""
    import grpc

    if not isinstance(exc, grpc.RpcError):
        return None
    code = getattr(exc, "code", None)
    if not callable(code):
        return None
    try:
        return code()
    except Exception:
        return None


def _is_grpc_error(exc: BaseException) -> bool:
    import grpc

    return isinstance(exc, grpc.RpcError)


def _is_not_found(exc: BaseException) -> bool:
    import grpc

    return _grpc_status_code(exc) == grpc.StatusCode.NOT_FOUND


def _is_already_exists(exc: BaseException) -> bool:
    import grpc

    return _grpc_status_code(exc) == grpc.StatusCode.ALREADY_EXISTS


def _is_grpc_timeout(exc: BaseException) -> bool:
    import grpc

    return _grpc_status_code(exc) == grpc.StatusCode.DEADLINE_EXCEEDED


def _is_sdk_error(exc: BaseException) -> bool:
    from openshell import SandboxError

    return isinstance(exc, SandboxError)


def _is_runtime_failure(exc: BaseException) -> bool:
    return _is_grpc_error(exc) or _is_sdk_error(exc)


def _is_retryable_create_error(exc: BaseException) -> bool:
    """Whether a CreateSandbox RPC failure is likely transient (safe to retry with the same name)."""
    import grpc

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return _grpc_status_code(exc) in {
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.UNAVAILABLE,
    }


@dataclass(frozen=True)
class OpenShellConnectionConfig:
    """Gateway connection settings. Defaults target a local plaintext gateway (deploy/docker compose)."""

    endpoint: str = "localhost:8080"
    workspace: str = "default"
    bearer_token: str | None = None
    tls_ca_path: str | None = None
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    request_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("connection.endpoint must be a non-empty host:port")
        if not self.workspace:
            raise ValueError("connection.workspace must be a non-empty workspace name")
        if self.request_timeout_s <= 0:
            raise ValueError("connection.request_timeout_s must be > 0")
        if bool(self.tls_cert_path) != bool(self.tls_key_path):
            raise ValueError("connection.tls_cert_path and connection.tls_key_path must be set together")


def _exec_timed_out(return_code: int, elapsed_s: float | None, timeout_s: int | None) -> bool:
    """Whether the exec stream ended at the ``timeout_seconds`` the provider passed it.

    The 0.1.x exec stream reports the timeout as exit code 124 with no timeout flag. The gateway
    only ends the stream; it does not kill the process, which may keep running in the sandbox. Only elapsed time
    tells it from a command that exits 124 on its own, so a quick ``exit 124`` is not a timeout.
    """
    if not timeout_s or return_code != 124 or elapsed_s is None:
        return False
    return elapsed_s >= timeout_s - min(EXEC_TIMEOUT_SLACK_S, EXEC_TIMEOUT_SLACK_FRACTION * timeout_s)


@dataclass(frozen=True)
class OpenShellCreateConfig:
    ready_timeout_s: float = 300
    poll_interval_s: float = 1.0
    retries: int = 2
    retry_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0

    def __post_init__(self) -> None:
        if self.ready_timeout_s <= 0:
            raise ValueError("create.ready_timeout_s must be > 0")
        if self.poll_interval_s <= 0:
            raise ValueError("create.poll_interval_s must be > 0")
        if self.retries < 0:
            raise ValueError("create.retries must be >= 0")
        if self.retry_delay_s < 0:
            raise ValueError("create.retry_delay_s must be >= 0")
        if self.retry_max_delay_s < 0:
            raise ValueError("create.retry_max_delay_s must be >= 0")


@dataclass(frozen=True)
class OpenShellExecConfig:
    default_timeout_s: float | None = 180
    # Bounds in-flight gateway RPCs across ALL operations sharing this connection config --
    # exec as well as control-plane calls (create/get/delete and probe polling). The pool
    # queue is unbounded: time spent waiting for a worker is not counted against timeouts
    # (the gateway's timeout_seconds only starts once the RPC is issued).
    concurrency: int = 32
    exec_shell: str = "/bin/sh"
    upload_chunk_bytes: int = DEFAULT_UPLOAD_CHUNK_BYTES

    def __post_init__(self) -> None:
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")
        if not self.exec_shell:
            raise ValueError("exec.exec_shell must be a non-empty shell name/path")
        if self.upload_chunk_bytes < 1:
            raise ValueError("exec.upload_chunk_bytes must be >= 1")


@dataclass(frozen=True)
class OpenShellProbeConfig:
    command: str | None = READY_PROBE_COMMAND
    expected_stdout: str | None = READY_PROBE_EXPECTED
    timeout_s: int = 30
    deadline_s: float | None = 60
    stable_count: int = 1
    # Non-zero by default: the probe polls a remote gateway, so back-to-back retries would
    # hammer it for the full deadline when a sandbox is slow to become exec-ready.
    stable_delay_s: float = 1.0

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("probe.deadline_s must be > 0")
        if self.stable_count < 1:
            raise ValueError("probe.stable_count must be >= 1")
        if self.stable_delay_s < 0:
            raise ValueError("probe.stable_delay_s must be >= 0")


@dataclass(frozen=True)
class OpenShellOperationsConfig:
    # DeleteSandbox takes the sandbox pod's termination grace period (~31s on a Kubernetes-backed gateway). The gateway
    # finishes a delete it has started even after the client deadline, but a delete still waiting on a lifecycle lock
    # can hit the deadline first, so deletes run on their own client with a long deadline.
    delete_rpc_timeout_s: float = 120
    # Workers for deletes, separate from exec.concurrency. Each holds a worker for the delete plus the wait until
    # gone, so this bounds how fast queued deletes drain.
    delete_concurrency: int = 32
    close_wait_deleted: bool = True
    close_timeout_s: float = 60
    poll_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if self.delete_rpc_timeout_s <= 0:
            raise ValueError("operations.delete_rpc_timeout_s must be > 0")
        if self.delete_concurrency < 1:
            raise ValueError("operations.delete_concurrency must be >= 1")
        if self.close_timeout_s <= 0:
            raise ValueError("operations.close_timeout_s must be > 0")
        if self.poll_interval_s <= 0:
            raise ValueError("operations.poll_interval_s must be > 0")


@dataclass(frozen=True)
class OpenShellProviderOptions:
    """Validated per-sandbox options carried in ``SandboxSpec.provider_options``."""

    providers: list[str] = field(default_factory=list)
    policy: Any | None = None
    template_resources: dict[str, Any] = field(default_factory=dict)
    driver_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> "OpenShellProviderOptions":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                f"Unknown openshell provider_options keys: {sorted(unknown)}. Allowed keys: {sorted(allowed)}"
            )
        providers = options.get("providers") or []
        if isinstance(providers, str):
            providers = [providers]
        if not isinstance(providers, (list, tuple)):
            raise TypeError(f"provider_options['providers'] must be a string or list, got {type(providers).__name__}")
        policy = options.get("policy")
        if policy is not None and not isinstance(policy, (str, Mapping)):
            raise TypeError(
                f"provider_options['policy'] must be a policy YAML path or a mapping, got {type(policy).__name__}"
            )
        template_resources = options.get("template_resources") or {}
        if not isinstance(template_resources, Mapping):
            raise TypeError(
                f"provider_options['template_resources'] must be a mapping, got {type(template_resources).__name__}"
            )
        driver_config = options.get("driver_config") or {}
        if not isinstance(driver_config, Mapping):
            raise TypeError(f"provider_options['driver_config'] must be a mapping, got {type(driver_config).__name__}")
        return cls(
            providers=[str(p) for p in providers],
            policy=policy,
            template_resources=dict(template_resources),
            driver_config=dict(driver_config),
        )


@dataclass
class _OpenShellSandbox:
    name: str
    sandbox_id: str
    workspace: str
    image: str | None
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None


@dataclass
class _SharedClientState:
    """One gRPC client + worker pool shared by every provider with the same connection config."""

    key: tuple[Any, ...]
    client: Any
    executor: ThreadPoolExecutor
    refcount: int = 0


_SHARED_CLIENTS: dict[tuple[Any, ...], _SharedClientState] = {}
_SHARED_CLIENTS_LOCK = threading.Lock()

# Deletes run on process-lifetime workers that are never released: CPython runs queued work items and joins
# the (non-daemon) worker threads at interpreter exit, so a delete handed to a worker finishes even after the
# event loop is gone. The SDK fixes one deadline per client, hence a separate long-deadline client for deletes.
_DELETE_STATES: dict[tuple[Any, ...], _SharedClientState] = {}
# Sandboxes this process is creating or has created and not closed, and deletes handed to a worker, by
# (workspace, name). A name is added before CreateSandbox is issued (the client generates it), so the orphan
# reap never takes a sandbox that is still being created.
_LIVE_SANDBOXES: set[tuple[str, str]] = set()
_PENDING_DELETES: dict[tuple[str, str], Future[None]] = {}


def _build_client(connection: OpenShellConnectionConfig) -> Any:
    from openshell import SandboxClient, TlsConfig

    tls = None
    if connection.tls_ca_path or connection.tls_cert_path:
        tls = TlsConfig(
            ca_path=Path(connection.tls_ca_path) if connection.tls_ca_path else None,
            cert_path=Path(connection.tls_cert_path) if connection.tls_cert_path else None,
            key_path=Path(connection.tls_key_path) if connection.tls_key_path else None,
        )
    return SandboxClient(
        connection.endpoint,
        tls=tls,
        bearer_token=connection.bearer_token,
        timeout=connection.request_timeout_s,
    )


def _acquire_shared_client(connection: OpenShellConnectionConfig, concurrency: int) -> _SharedClientState:
    key = (connection, concurrency)
    with _SHARED_CLIENTS_LOCK:
        state = _SHARED_CLIENTS.get(key)
        if state is None:
            state = _SharedClientState(
                key=key,
                client=_build_client(connection),
                executor=ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="openshell-sandbox"),
            )
            _SHARED_CLIENTS[key] = state
        state.refcount += 1
        return state


def _delete_state(connection: OpenShellConnectionConfig, concurrency: int) -> _SharedClientState:
    key = (connection, concurrency)
    with _SHARED_CLIENTS_LOCK:
        state = _DELETE_STATES.get(key)
        if state is None:
            state = _SharedClientState(
                key=key,
                client=_build_client(connection),
                executor=ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="openshell-delete"),
            )
            _DELETE_STATES[key] = state
        return state


def _release_shared_client(state: _SharedClientState) -> None:
    with _SHARED_CLIENTS_LOCK:
        state.refcount -= 1
        if state.refcount > 0:
            return
        _SHARED_CLIENTS.pop(state.key, None)
    # Stop the workers before closing the channel so queued futures don't run against a dead channel.
    state.executor.shutdown(wait=False, cancel_futures=True)
    state.client.close()


def _forget_delete(key: tuple[str, str], future: Future[None]) -> None:
    with _SHARED_CLIENTS_LOCK:
        if _PENDING_DELETES.get(key) is future:
            del _PENDING_DELETES[key]


def _delete_sandbox(client: Any, inst: _OpenShellSandbox, operations: OpenShellOperationsConfig) -> None:
    """Delete one sandbox and (optionally) wait until it is gone, logging the outcome; runs in a worker thread.

    A client deadline does not cancel a delete the gateway has already started, so after one the worker waits for
    the sandbox to disappear and only re-issues the delete if it is still there; ``allow_missing`` makes the
    re-issue idempotent. Transient transport failures and FAILED_PRECONDITION (a create's driver operation is
    still pending) are retried until ``operations.close_timeout_s``. Each job can use up to
    ``MAX_DELETE_ATTEMPTS`` delete deadlines, plus retry delays and the deletion wait.
    """
    try:
        _delete_and_wait(client, inst, operations)
    except Exception as e:
        LOGGER.error(f"openshell delete of {inst.name!r} failed; the sandbox may be leaked on the gateway: {e}")
    else:
        LOGGER.info(f"openshell sandbox {inst.name!r} deleted")


def _is_delete_retryable(e: Exception) -> bool:
    """Transient transport failures, or FAILED_PRECONDITION while a create's driver operation is pending."""
    import grpc

    return _is_retryable_create_error(e) or _grpc_status_code(e) == grpc.StatusCode.FAILED_PRECONDITION


def _delete_and_wait(client: Any, inst: _OpenShellSandbox, operations: OpenShellOperationsConfig) -> None:
    from openshell.mutations import DeletionOutcome

    deadlines = 0
    retry_until = time.monotonic() + operations.close_timeout_s
    while True:
        try:
            result = client.delete(inst.name, workspace=inst.workspace, allow_missing=True)
            break
        except Exception as e:
            if _is_not_found(e):
                return
            if _is_grpc_timeout(e):
                deadlines += 1
                LOGGER.warning(
                    f"DeleteSandbox for {inst.name!r} hit its deadline (attempt {deadlines}/{MAX_DELETE_ATTEMPTS}); "
                    "waiting to see whether the gateway deleted it"
                )
                try:
                    _wait_deleted(client, inst, operations)
                    return
                except RuntimeError:
                    pass
                if deadlines >= MAX_DELETE_ATTEMPTS:
                    raise RuntimeError(
                        f"DeleteSandbox for {inst.name!r} hit its deadline {deadlines} times and the sandbox is still "
                        f"there (operations.delete_rpc_timeout_s={operations.delete_rpc_timeout_s:g})"
                    ) from e
                continue
            if _is_delete_retryable(e) and time.monotonic() < retry_until:
                LOGGER.warning(f"Retrying DeleteSandbox for {inst.name!r} after {e}")
                time.sleep(operations.poll_interval_s)
                continue
            if _is_runtime_failure(e):
                raise RuntimeError(f"DeleteSandbox failed for {inst.name!r}: {e}") from e
            raise
    if result.outcome == DeletionOutcome.ALREADY_ABSENT:
        return
    if result.outcome not in (DeletionOutcome.COMPLETED, DeletionOutcome.ACCEPTED):
        LOGGER.warning(
            f"DeleteSandbox returned outcome={result.outcome!r} for {inst.name!r}; skipping the deletion wait."
        )
        return
    if operations.close_wait_deleted:
        _wait_deleted(client, inst, operations)


def _wait_deleted(client: Any, inst: _OpenShellSandbox, operations: OpenShellOperationsConfig) -> None:
    """Poll GetSandbox until NOT_FOUND (transient RPC failures keep polling until the deadline)."""
    deadline = time.monotonic() + operations.close_timeout_s
    while True:
        try:
            client.get(inst.name, workspace=inst.workspace)
        except Exception as e:
            if _is_not_found(e):
                return
            if not _is_runtime_failure(e):
                raise
            LOGGER.debug(f"GetSandbox failed while waiting for {inst.name!r} to be deleted: {e}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"sandbox {inst.name!r} was not deleted within {operations.close_timeout_s:g}s")
        time.sleep(operations.poll_interval_s)


class OpenShellProvider:
    """Sandbox provider backed by an OpenShell gateway's gRPC control plane."""

    name = "openshell"

    def __init__(
        self,
        *,
        connection: OpenShellConnectionConfig | Mapping[str, Any] | None = None,
        create: OpenShellCreateConfig | Mapping[str, Any] | None = None,
        exec: OpenShellExecConfig | Mapping[str, Any] | None = None,
        probe: OpenShellProbeConfig | Mapping[str, Any] | None = None,
        operations: OpenShellOperationsConfig | Mapping[str, Any] | None = None,
        attribution: AttributionConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _coerce_config(connection, OpenShellConnectionConfig)
        self._create_config = _coerce_config(create, OpenShellCreateConfig)
        self._exec_config = _coerce_config(exec, OpenShellExecConfig)
        self._probe = _coerce_config(probe, OpenShellProbeConfig)
        self._operations = _coerce_config(operations, OpenShellOperationsConfig)
        self._attribution = _coerce_config(attribution, AttributionConfig)
        _require_openshell()
        self._shared = _acquire_shared_client(self._connection, self._exec_config.concurrency)
        self._closed = False
        # DeleteSandbox blocks for the pod's termination grace period (~31s on a Kubernetes-backed gateway), so
        # close() hands it to a delete worker (see _DELETE_STATES) and returns; aclose() never waits for it.
        self._delete_shared = _delete_state(
            dataclasses.replace(self._connection, request_timeout_s=self._operations.delete_rpc_timeout_s),
            self._operations.delete_concurrency,
        )

    @property
    def _client(self) -> Any:
        return self._shared.client

    async def _call(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._shared.executor, functools.partial(func, *args, **kwargs))

    async def _call_create(self, name: str, workspace: str, **kwargs: Any) -> Any:
        """Run CreateSandbox on a worker. Cancelling the await does not stop a call already running, so a create
        that commits after the cancel is deleted when it returns."""
        future = self._shared.executor.submit(
            functools.partial(self._client.create, workspace=workspace, name=name, **kwargs)
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            inst = _OpenShellSandbox(name=name, sandbox_id="", workspace=workspace, image=None)
            future.add_done_callback(
                lambda done: None if done.cancelled() or done.exception() else self._submit_delete(inst)
            )
            raise

    def _build_policy(self, policy: str | Mapping[str, Any]) -> Any:
        from google.protobuf import json_format
        from openshell._proto import sandbox_pb2

        if isinstance(policy, str):
            import yaml

            policy = yaml.safe_load(Path(policy).read_text())
        if not isinstance(policy, Mapping):
            raise OpenShellCreateError(
                f"provider_options['policy'] must load to a mapping, got {type(policy).__name__}"
            )
        try:
            return json_format.ParseDict(dict(policy), sandbox_pb2.SandboxPolicy())
        except json_format.ParseError as e:
            raise OpenShellCreateError(f"provider_options['policy'] is not a valid SandboxPolicy: {e}") from e

    def _build_sandbox_spec(self, spec: SandboxSpec, image: str | None, options: OpenShellProviderOptions) -> Any:
        from openshell._proto import openshell_pb2

        kwargs: dict[str, Any] = {"environment": {str(k): str(v) for k, v in spec.env.items()}}
        if image or options.template_resources or options.driver_config:
            template = openshell_pb2.SandboxTemplate()
            if image:
                template.image = image
            if options.template_resources:
                template.resources.update(options.template_resources)
            if options.driver_config:
                template.driver_config.update(options.driver_config)
            kwargs["template"] = template
        if options.policy is not None:
            kwargs["policy"] = self._build_policy(options.policy)
        if options.providers:
            kwargs["providers"] = options.providers
        resources = spec.resources
        if resources.gpu:
            kwargs["resource_requirements"] = openshell_pb2.ResourceRequirements(
                gpu=openshell_pb2.GpuResourceRequirements(count=resources.gpu)
            )
        ignored = [
            key
            for key, value in (
                ("cpu", resources.cpu),
                ("memory_mib", resources.memory_mib),
                ("disk_gib", resources.disk_gib),
                ("gpu_type", resources.gpu_type),
            )
            if value is not None
        ]
        if ignored:
            LOGGER.warning(
                "%s resource requests are not mapped by this provider; OpenShell exposes driver-specific "
                "limits through SandboxTemplate.resources — pass provider_options.template_resources instead.",
                ", ".join(ignored),
            )
        return openshell_pb2.SandboxSpec(**kwargs)

    async def _create_sandbox_with_retries(self, pb_spec: Any, name: str, labels: dict[str, str]) -> Any:
        """Issue CreateSandbox, retrying transient gRPC failures with the same name.

        Retrying with the same name is safe: if an earlier attempt actually committed, the
        retry fails ALREADY_EXISTS and the sandbox is recovered via GetSandbox.
        """
        cfg = self._create_config
        workspace = self._connection.workspace
        delay = cfg.retry_delay_s
        attempt = 0
        while True:
            try:
                return await self._call_create(name, workspace, spec=pb_spec, labels=labels)
            except Exception as e:
                if _is_already_exists(e):
                    return await self._call(self._client.get, name, workspace=workspace)
                if attempt >= cfg.retries or not _is_retryable_create_error(e):
                    raise
                attempt += 1
                LOGGER.warning(
                    f"CreateSandbox attempt {attempt}/{cfg.retries + 1} for {name!r} failed with a "
                    f"transient error; retrying in {delay:g}s: {e}"
                )
                await asyncio.sleep(min(delay, cfg.retry_max_delay_s))
                delay = min(delay * 2, cfg.retry_max_delay_s) if delay > 0 else cfg.retry_delay_s

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox through the gateway, wait for the READY phase, then probe exec readiness.

        ``spec.image`` is optional (the gateway's configured default image is used when unset).
        ``spec.ttl_s`` is not enforced (OpenShell sandboxes live until deleted) and only logs a
        warning. ``spec.entrypoint`` is unsupported: the OpenShell supervisor owns the sandbox
        entrypoint. ``spec.provider_options`` accepts ``providers`` (OpenShell credential-provider
        names), ``policy`` (a SandboxPolicy mapping or YAML path), and ``template_resources`` /
        ``driver_config`` (free-form driver passthrough Structs). A half-created sandbox is
        deleted on any failure.
        """
        if spec.entrypoint:
            raise OpenShellCreateError(
                "spec.entrypoint is not supported by the openshell provider; the OpenShell "
                "supervisor owns the sandbox entrypoint"
            )
        if spec.ttl_s is not None:
            LOGGER.warning("ttl_s is not enforced by the openshell provider; sandboxes live until close().")

        image = _normalize_image(spec.image) if spec.image else None
        options = OpenShellProviderOptions.from_mapping(spec.provider_options)
        pb_spec = self._build_sandbox_spec(spec, image, options)
        name = _generate_sandbox_name()
        # Explicit spec keys win over attribution keys. The marker label and the process label (which the orphan
        # reap selects on) go last so no caller metadata replaces them. Label values must be valid Kubernetes label
        # values (at most 63 characters), so sanitize them all.
        attribution = self._attribution.metadata()
        owner = {f"{self._attribution.key_prefix}{PROCESS_KEY}": resolve_process_id()} if attribution else {}
        labels = {
            str(key): sanitize_label_value(value)
            for key, value in {**attribution, **spec.metadata, **owner, SANDBOX_LABEL: "1"}.items()
        }
        workspace = self._connection.workspace
        key = (workspace, name)
        _LIVE_SANDBOXES.add(key)

        stub = _OpenShellSandbox(name=name, sandbox_id="", workspace=workspace, image=image)
        try:
            ref = await self._create_sandbox_with_retries(pb_spec, name, labels)
        except BaseException as e:
            # The gateway may have committed the sandbox even though no ref came back; delete is idempotent.
            self._submit_delete(stub)
            if isinstance(e, Exception) and (_is_runtime_failure(e) or isinstance(e, (ConnectionError, TimeoutError))):
                raise OpenShellCreateError(f"CreateSandbox failed for image={image!r}: {e}") from e
            raise

        workspace = getattr(ref, "workspace", "") or self._connection.workspace
        _LIVE_SANDBOXES.add((workspace, ref.name))
        handle = SandboxHandle(
            sandbox_id=ref.id,
            provider_name=self.name,
            raw=_OpenShellSandbox(
                name=ref.name,
                sandbox_id=ref.id,
                workspace=workspace,
                image=image,
                env=dict(spec.env),
                workdir=spec.workdir,
            ),
        )
        try:
            ready_timeout_s = spec.ready_timeout_s or self._create_config.ready_timeout_s
            await self._wait_ready(handle, timeout_s=ready_timeout_s)
            await self._verify_created_handle(handle)
        except BaseException:
            await self._cleanup_failed_create_handle(handle)
            raise
        return handle

    async def _wait_ready(self, handle: SandboxHandle, *, timeout_s: int | float) -> None:
        """Poll GetSandbox until READY, raising on the ERROR/DELETING phases or the deadline."""
        inst = handle.raw
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_phase: int | None = None
        conditions = ""
        while True:
            try:
                ref = await self._call(self._client.get, inst.name, workspace=inst.workspace)
                last_phase = ref.phase
                conditions = _format_conditions(ref)
            except Exception as e:
                if not _is_runtime_failure(e):
                    raise
                LOGGER.debug(f"GetSandbox failed while waiting for {inst.name!r} to become ready: {e}")
            else:
                if ref.phase == _phase("SANDBOX_PHASE_READY"):
                    return
                if ref.phase == _phase("SANDBOX_PHASE_ERROR"):
                    raise OpenShellCreateError(
                        f"sandbox {inst.name!r} entered the ERROR phase while provisioning{conditions}"
                    )
                if _phase_to_status_map().get(ref.phase) == SandboxStatus.STOPPED:
                    raise OpenShellCreateError(
                        f"sandbox {inst.name!r} is being deleted or stopped while provisioning (phase={ref.phase})"
                    )
            if loop.time() >= deadline:
                raise OpenShellCreateError(
                    f"sandbox {inst.name!r} was not READY within {timeout_s:g}s (last phase={last_phase}){conditions}"
                )
            await asyncio.sleep(self._create_config.poll_interval_s)

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        """Poll the readiness probe until it passes ``stable_count`` times or the deadline elapses."""
        probe = self._probe
        if probe.command is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + probe.deadline_s if probe.deadline_s is not None else None
        consecutive = 0
        last_detail = "no probe attempt completed"
        while True:
            result = await self.exec(handle, probe.command, timeout_s=probe.timeout_s)
            passed = result.return_code == 0 and (
                probe.expected_stdout is None or probe.expected_stdout in (result.stdout or "")
            )
            if passed:
                consecutive += 1
                if consecutive >= probe.stable_count:
                    return
            else:
                consecutive = 0
                last_detail = f"return_code={result.return_code}, stderr={(result.stderr or '').strip()!r}"
                if deadline is None:
                    raise OpenShellCreateVerificationError(
                        f"sandbox {handle.sandbox_id!r} failed readiness probe: {last_detail}"
                    )
            if deadline is not None and loop.time() >= deadline:
                raise OpenShellCreateVerificationError(
                    f"sandbox {handle.sandbox_id!r} did not pass readiness probe within "
                    f"{probe.deadline_s:g}s: {last_detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        self._submit_delete(handle.raw)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        """Run ``<shell> -c <command>`` through the gateway's streaming exec; never raises for command failure.

        The timeout is enforced by the gateway (``timeout_seconds``); the SDK extends its gRPC
        deadline past it. ``user`` is ignored with a warning: the OpenShell exec API has no user
        field, so commands run as the sandbox's default user.
        """
        inst = handle.raw
        if user is not None:
            LOGGER.warning(
                f"The openshell provider cannot run commands as user={user!r}; the OpenShell exec API "
                "has no user field. Running as the sandbox default user."
            )
        merged_env = {str(k): str(v) for k, v in inst.env.items()}
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        effective_timeout = timeout_s if timeout_s is not None else self._exec_config.default_timeout_s
        timeout_seconds = max(1, math.ceil(effective_timeout)) if effective_timeout is not None else None
        workdir = cwd if cwd is not None else inst.workdir
        try:
            result, elapsed_s = await self._call(
                self._timed_exec,
                inst.name,
                [self._exec_config.exec_shell, "-c", command],
                workspace=inst.workspace,
                workdir=workdir,
                env=merged_env or None,
                stdin=stdin,
                timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            if _is_grpc_timeout(e):
                return SandboxExecResult(
                    stdout=None, stderr=str(e), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="timeout"
                )
            if _is_runtime_failure(e):
                return SandboxExecResult(
                    stdout=None, stderr=str(e), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="sandbox"
                )
            raise
        if _exec_timed_out(result.exit_code, elapsed_s, timeout_seconds):
            note = (
                f"Command timed out after {timeout_seconds}s; the OpenShell gateway ended the exec "
                "but did not kill the process, which may still be running in the sandbox"
            )
            stderr = f"{result.stderr}\n{note}" if result.stderr else note
            return SandboxExecResult(stdout=result.stdout, stderr=stderr, return_code=124, error_type="timeout")
        return SandboxExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)

    def _timed_exec(self, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Run the SDK exec and time it inside the worker thread, so executor queueing is not counted."""
        started = time.monotonic()
        result = self._client.exec(*args, **kwargs)
        return result, time.monotonic() - started

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file by streaming its bytes through exec stdin (creates the parent dir).

        Bytes are sent in ``exec.upload_chunk_bytes`` chunks because each chunk travels as a
        single gRPC message that must stay under the gateway's max decode size.
        """
        data = await asyncio.to_thread(Path(source_path).read_bytes)
        quoted = shlex.quote(target_path)
        parent = posixpath.dirname(target_path)
        chunk_size = self._exec_config.upload_chunk_bytes
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)] or [b""]
        for index, chunk in enumerate(chunks):
            if index == 0:
                command = f"cat > {quoted}"
                if parent:
                    command = f"mkdir -p {shlex.quote(parent)} && {command}"
            else:
                command = f"cat >> {quoted}"
            result = await self.exec(handle, command, stdin=chunk)
            if result.return_code != 0:
                raise RuntimeError(
                    f"openshell upload to {target_path!r} failed (chunk {index + 1}/{len(chunks)}, "
                    f"code={result.return_code}): {(result.stderr or '').strip()}"
                )

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file via a base64 round-trip (binary-safe over the text exec stream).

        The whole file is buffered in memory (inflated 4/3 by base64), so this is intended for
        small-to-medium artifacts rather than large archives.
        """
        result = await self.exec(handle, f"base64 {shlex.quote(source_path)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"openshell download from {source_path!r} failed (code={result.return_code}): "
                f"{(result.stderr or '').strip()}"
            )
        try:
            data = base64.b64decode("".join((result.stdout or "").split()), validate=True)
        except (binascii.Error, ValueError) as e:
            raise RuntimeError(f"openshell download from {source_path!r} returned invalid base64: {e}") from e
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target_path.write_bytes, data)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Share gateway identity without serializing gateway credentials."""
        inst = handle.raw
        return {
            "provider": self.name,
            "sandbox_id": inst.sandbox_id,
            "name": inst.name,
            "endpoint": self._connection.endpoint,
            "workspace": inst.workspace,
            "image": inst.image,
            "workdir": inst.workdir,
        }

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Reconnect through the configured gateway, retaining owner-managed lifetime."""
        if (
            descriptor.get("provider") != self.name
            or descriptor.get("endpoint") != self._connection.endpoint
            or descriptor.get("workspace") != self._connection.workspace
        ):
            raise ValueError("OpenShell reconnect requires the same gateway and workspace")
        name, sandbox_id = descriptor["name"], descriptor["sandbox_id"]
        ref = await self._call(self._client.get, name, workspace=self._connection.workspace)
        if ref.id != sandbox_id or ref.name != name:
            raise ValueError("OpenShell sandbox identity changed")
        return SandboxHandle(
            sandbox_id=sandbox_id,
            provider_name=self.name,
            raw=_OpenShellSandbox(
                name=name,
                sandbox_id=sandbox_id,
                workspace=self._connection.workspace,
                image=descriptor.get("image"),
                env={},
                workdir=descriptor.get("workdir"),
            ),
        )

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Sandbox phase via GetSandbox (missing -> STOPPED; RPC failure -> UNKNOWN)."""
        inst = handle.raw
        try:
            ref = await self._call(self._client.get, inst.name, workspace=inst.workspace)
        except Exception as e:
            if _is_not_found(e):
                return SandboxStatus.STOPPED
            if _is_runtime_failure(e):
                return SandboxStatus.UNKNOWN
            raise
        return _phase_to_status_map().get(ref.phase, SandboxStatus.UNKNOWN)

    async def close(self, handle: SandboxHandle) -> None:
        """Hand the sandbox's delete to a worker thread and return without waiting for the gateway.

        The worker deletes, waits until the sandbox is gone when ``operations.close_wait_deleted`` is set, and
        retries transient failures. It is already queued when this returns and runs to completion at interpreter
        exit even if the event loop is gone. Every outcome is logged with the sandbox name; a delete that fails
        for good is logged as possibly leaked and then forgotten.
        """
        self._submit_delete(handle.raw)

    def _submit_delete(self, inst: _OpenShellSandbox) -> None:
        key = (inst.workspace, inst.name)
        with _SHARED_CLIENTS_LOCK:
            pending = _PENDING_DELETES.get(key)
            if pending is not None and not pending.done():
                _LIVE_SANDBOXES.discard(key)
                return
            try:
                future = self._delete_shared.executor.submit(
                    _delete_sandbox, self._delete_shared.client, inst, self._operations
                )
            except RuntimeError:
                # The interpreter is shutting down and takes no new work: delete here, blocking.
                LOGGER.warning(f"Deleting openshell sandbox {inst.name!r} synchronously at interpreter shutdown.")
                future = None
            else:
                _PENDING_DELETES[key] = future
            _LIVE_SANDBOXES.discard(key)
        if future is None:
            _delete_sandbox(self._delete_shared.client, inst, self._operations)
            return
        queued = len(_PENDING_DELETES)
        if queued > self._operations.delete_concurrency:
            LOGGER.warning(
                f"{queued} openshell deletes are pending on {self._operations.delete_concurrency} workers; "
                "sandboxes stay allocated until they drain."
            )
        future.add_done_callback(lambda done, key=key: _forget_delete(key, done))

    async def _reap_orphans(self) -> None:
        """Delete this process's orphaned sandboxes: ones it no longer tracks, for example a create cancelled
        after the gateway committed it. Selected by the process label, which needs attribution to be enabled."""
        if not self._attribution.enabled:
            return
        prefix = self._attribution.key_prefix
        selector = f"{prefix}{PROCESS_KEY}={resolve_process_id()},{SANDBOX_LABEL}=1"
        workspace = self._connection.workspace
        try:
            refs = await self._call(self._client.list_all, workspace=workspace, label_selector=selector)
        except Exception as e:
            LOGGER.warning(f"Could not list orphaned OpenShell sandboxes ({selector}): {e}")
            return
        for ref in refs:
            key = (ref.workspace or workspace, ref.name)
            if key in _LIVE_SANDBOXES or key in _PENDING_DELETES:
                continue
            LOGGER.warning(f"Deleting orphaned OpenShell sandbox {ref.name!r} of this process ({selector}).")
            self._submit_delete(_OpenShellSandbox(name=ref.name, sandbox_id=ref.id, workspace=key[0], image=None))

    async def aclose(self) -> None:
        """Release the shared client/pool (closed for real when the last provider releases it).

        Does not wait for deletes: they run on their own workers. When no sandbox of this process is still
        live or being created, it first deletes this process's orphaned sandboxes (see ``_reap_orphans``).
        """
        if self._closed:
            return
        self._closed = True
        if not _LIVE_SANDBOXES:
            await self._reap_orphans()
        await asyncio.to_thread(_release_shared_client, self._shared)
