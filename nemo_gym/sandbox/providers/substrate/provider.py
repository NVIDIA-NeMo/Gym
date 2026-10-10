# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Sandbox provider backed by Substrate actors."""

import asyncio
import contextlib
import copy
import hashlib
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
import httpx

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.substrate._ateapi import ateapi_pb2, ateapi_pb2_grpc
from nemo_gym.sandbox.providers.utils import coerce_config


LOGGER = logging.getLogger(__name__)

_TOKEN_TTL_S = 3600

TARGET_ACTOR_HEADER = "ate-target-actor"

SANDBOX_RUNTIME_RETURN_CODE = 125

TEMPLATE_NAME_PREFIX = "swb-"
TEMPLATE_DIGEST_LENGTH = 16

# ateapi answers ABORTED on an optimistic-concurrency conflict ("please retry")
# and UNAVAILABLE while it restarts; both are safe to retry for every RPC the
# provider sends.
_RETRYABLE_CONTROL_CODES = (grpc.StatusCode.ABORTED, grpc.StatusCode.UNAVAILABLE)
_CONTROL_ATTEMPTS = 5
_CONTROL_BACKOFF_S = 0.5

# The router answers 503 "actor <id> unavailable[: <reason>]" when it could not
# place a request: the pool had no room within the parking budget, the actor
# was mid-transition, a concurrent update conflicted, or ateapi was away. None
# of those requests reached the actor, so sending them again is safe. A reason
# that names a terminal state is the one exception: nothing will make that
# actor routable again.
_CAPACITY_MARKERS = ("router at capacity", "no worker has room")
# Reasons that should clear within seconds: a transition in progress, a
# concurrent-update conflict, ateapi being away. Retried for a bounded time
# only, so an actor stranded mid-transition fails the step instead of hanging
# it; anything else, a CRASHED actor included, fails at once.
_TRANSIENT_MARKERS = (
    "got: ACTOR_STATE_PAUSING",
    "got: ACTOR_STATE_SUSPENDING",
    "got: ACTOR_STATE_RESUMING",
    "got: ACTOR_STATE_REVERTING",
    "please retry",
    "another operation is in progress",
)
_TRANSITION_RETRY_DELAY_S = 1.0
_TRANSIENT_RETRY_BUDGET_S = 60.0
# How long a park may stay mid-transition before the provider stops waiting.
_PARK_SETTLE_BUDGET_S = 30.0

_STATUS_MAP = {
    ateapi_pb2.ActorState.ACTOR_STATE_RUNNING: SandboxStatus.RUNNING,
    ateapi_pb2.ActorState.ACTOR_STATE_RESUMING: SandboxStatus.STARTING,
    ateapi_pb2.ActorState.ACTOR_STATE_REVERTING: SandboxStatus.STARTING,
    # A paused or suspended actor is alive and one request away: the router
    # resumes it on the next exec. PAUSED says so without implying it is gone.
    ateapi_pb2.ActorState.ACTOR_STATE_SUSPENDED: SandboxStatus.PAUSED,
    ateapi_pb2.ActorState.ACTOR_STATE_SUSPENDING: SandboxStatus.PAUSED,
    ateapi_pb2.ActorState.ACTOR_STATE_PAUSED: SandboxStatus.PAUSED,
    ateapi_pb2.ActorState.ACTOR_STATE_PAUSING: SandboxStatus.PAUSED,
    ateapi_pb2.ActorState.ACTOR_STATE_CRASHED: SandboxStatus.ERROR,
    ateapi_pb2.ActorState.ACTOR_STATE_DELETING: SandboxStatus.STOPPED,
}

_PARKED_STATES = {
    ateapi_pb2.ActorState.ACTOR_STATE_PAUSED,
    ateapi_pb2.ActorState.ACTOR_STATE_SUSPENDED,
}

_TRANSITIONAL_STATES = {
    ateapi_pb2.ActorState.ACTOR_STATE_PAUSING,
    ateapi_pb2.ActorState.ACTOR_STATE_SUSPENDING,
}


class SubstrateCreateError(SandboxCreateError):
    """Raised when a Substrate actor cannot be created or resumed."""


class SubstrateCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a created actor never becomes ready."""


def template_name_for_image(image: str) -> str:
    """Derive the ActorTemplate name that serves `image`.

    An ActorTemplate spec is immutable and an actor names a template rather than
    an image, so templates are built ahead of a run. Deriving the name from the
    image lets both sides agree without a shared lookup table. Substrate's
    demos/sandbox/gym-templates.sh computes the same name; a unit test pins it.
    """
    digest = hashlib.sha256(image.encode()).hexdigest()
    return TEMPLATE_NAME_PREFIX + digest[:TEMPLATE_DIGEST_LENGTH]


def normalize_image(image: str) -> str:
    """Strip a runtime scheme and the default registry.

    `docker.io/foo:tag` and `foo:tag` name the same image; the SWE-bench
    environment spells it the first way and operators the second.
    """
    for prefix in ("docker://", "oci://"):
        if image.startswith(prefix):
            image = image[len(prefix) :]
            break
    if image.startswith("docker.io/"):
        image = image[len("docker.io/") :]
    return image


def _classify_503(response: httpx.Response) -> str | None:
    """Return "capacity", "transient", or None for a 503 that is not worth retrying.

    Every 503 the router makes before forwarding reads "actor <id>
    unavailable[: <reason>]"; a bare one with no reason is ateapi being away.
    """
    if response.status_code != 503:
        return None
    text = response.text
    if any(marker in text for marker in _CAPACITY_MARKERS):
        return "capacity"
    if any(marker in text for marker in _TRANSIENT_MARKERS) or text.rstrip().endswith("unavailable"):
        return "transient"
    return None


def _read_kube_config() -> dict[str, Any]:
    """Parse the active kubeconfig, honoring KUBECONFIG's first entry."""
    import yaml

    kubeconfig = os.environ.get("KUBECONFIG", "").split(os.pathsep)[0] or "~/.kube/config"
    with open(Path(kubeconfig).expanduser()) as f:
        return yaml.safe_load(f)


def _without_null_credentials(kubeconfig: dict[str, Any]) -> dict[str, Any]:
    """Drop credential fields explicitly set to null from every user entry."""
    cleaned = copy.deepcopy(kubeconfig)
    for entry in cleaned.get("users") or []:
        user = entry.get("user")
        if isinstance(user, dict):
            entry["user"] = {k: v for k, v in user.items() if v is not None}
    return cleaned


@dataclass(frozen=True)
class SubstrateConnectionConfig:
    """Where the control plane and the router are, and who we are to them."""

    ateapi_addr: str = "localhost:8080"
    atenet_addr: str = "localhost:8000"
    atespace: str = "nemo-gym"
    template_namespace: str = "nemo-gym"

    # ateapi always serves TLS. Against a port-forward the certificate names the
    # in-cluster service, so the expected name is overridden rather than
    # verification being skipped outright.
    tls_server_name: str = "api.ate-system.svc"

    # The serving certificate is issued by a cluster-internal CA that a client
    # outside the cluster does not trust. Left unset, the CA is read from the
    # same ClusterTrustBundle the in-cluster clients mount.
    ca_cert_path: str | None = None
    ca_trust_bundle_signer: str = "servicedns.podcert.ate.dev/identity"

    # ateapi rejects unauthenticated calls. A ServiceAccount token is minted
    # per provider instance through the Kubernetes TokenRequest API, the same
    # way kubectl-ate authenticates.
    service_account: str = "ate-client"
    service_account_namespace: str = "ate-system"
    token: str | None = None

    grpc_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not self.atespace:
            raise ValueError("connection.atespace must be set")
        if not self.template_namespace:
            raise ValueError("connection.template_namespace must be set")


@dataclass(frozen=True)
class SubstrateCreateConfig:
    # How long create() and resume() wait for the actor to answer its readiness
    # probe through the router. Under load this is mostly waiting for a worker
    # slot. SandboxSpec.ready_timeout_s overrides it per sandbox.
    ready_timeout_s: float = 1200.0

    # never: leave the actor running between calls.
    # after_exec: SuspendActor once the actor goes idle. The snapshot goes to
    #   object storage and the worker is freed.
    # pause_after_exec: PauseActor instead. The snapshot stays on the node, so
    #   the next resume skips the download but has to land on the same node.
    # Either way the next exec goes through the router, which resumes the
    # actor before routing, so nothing has to resume it explicitly.
    suspend_policy: str = "never"

    # An exec must be followed by this much idle time before the actor is
    # suspended or paused, so back-to-back steps (an eval's test runs, say) do
    # not each pay a checkpoint and a restore. 0 suspends inline as the exec
    # returns.
    idle_before_suspend_s: float = 1.0

    delete_on_close: bool = True

    def __post_init__(self) -> None:
        allowed = {"never", "after_exec", "pause_after_exec"}
        if self.suspend_policy not in allowed:
            raise ValueError(f"create.suspend_policy must be one of {sorted(allowed)}, got {self.suspend_policy!r}")
        if self.idle_before_suspend_s < 0:
            raise ValueError("create.idle_before_suspend_s must be >= 0")


@dataclass(frozen=True)
class SubstrateExecConfig:
    default_timeout_s: float | None = 1800.0
    concurrency: int = 32

    # Wait this long before resending a request the router could not place
    # within its parking budget (a 503 that never reached the actor). The
    # router has already held the request for its whole budget, so an
    # immediate retry would only park again.
    capacity_retry_delay_s: float = 5.0

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")


@dataclass(frozen=True)
class SubstrateProbeConfig:
    """Readiness probe run against the actor through the router."""

    path: str = "/readyz"
    # Longer than the router's parking budget, so a probe waiting for a worker
    # slot is held by the router rather than cut by the client and re-parked.
    timeout_s: float = 150.0
    stable_count: int = 1
    stable_delay_s: float = 1.0


@dataclass
class _SubstrateActor:
    """Provider-owned state behind a SandboxHandle."""

    atespace: str
    name: str
    template: str
    image: str
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    # Requests to the in-actor server currently in flight; the actor is never
    # suspended while this is nonzero.
    inflight: int = 0
    suspend_task: asyncio.Task | None = None
    # Held while a pause or suspend is in progress. Requests wait on it, since
    # the router refuses to resume an actor that is still PAUSING.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Number of suspend tasks past their idle wait; those are left to finish.
    parking: int = 0
    # True once a park settled and no request has gone out since, so two idle
    # parks that both found the actor idle do not both call the control plane.
    parked: bool = False
    # Set when the in-actor server reports that its worker is being evicted;
    # the next idle park then suspends to storage, whatever the policy says.
    evicting: bool = False

    @property
    def target(self) -> str:
        """Routing key the atenet router expects: ``<atespace>/<actor>``."""
        return f"{self.atespace}/{self.name}"

    @property
    def object_ref(self) -> Any:
        return ateapi_pb2.ObjectRef(atespace=self.atespace, name=self.name)


class SubstrateProvider:
    """Runs sandboxes as Substrate actors.

    Control operations go to ateapi over gRPC; commands and file transfers go
    to a small HTTP server inside the actor, reached through the atenet router.
    """

    name = "substrate"

    def __init__(
        self,
        *,
        connection: SubstrateConnectionConfig | dict[str, Any] | None = None,
        create: SubstrateCreateConfig | dict[str, Any] | None = None,
        exec: SubstrateExecConfig | dict[str, Any] | None = None,
        probe: SubstrateProbeConfig | dict[str, Any] | None = None,
    ) -> None:
        self._connection = coerce_config(connection, SubstrateConnectionConfig)
        self._create_config = coerce_config(create, SubstrateCreateConfig)
        self._exec_config = coerce_config(exec, SubstrateExecConfig)
        self._probe = coerce_config(probe, SubstrateProbeConfig)

        self._semaphore = asyncio.Semaphore(self._exec_config.concurrency)
        self._suspend_tasks: set[asyncio.Task] = set()
        self._channel: grpc.aio.Channel | None = None
        self._stub: Any = None
        self._http_client: httpx.AsyncClient | None = None
        self._token: str | None = self._connection.token
        self._token_deadline: float | None = None

    # ---- I/O chokepoints ---- #

    def _kube_client(self):
        """Load Kubernetes credentials, from a kubeconfig or from the pod.

        Both failures are reported together: falling back silently turns a
        readable kubeconfig error into "Service host/port is not set", which
        describes the fallback rather than what actually went wrong.
        """
        from kubernetes import client as k8s_client
        from kubernetes import config as k8s_config

        try:
            try:
                k8s_config.load_kube_config()
            except k8s_config.ConfigException:
                # Some generators emit explicit nulls for unused credential
                # fields (`token: null` beside a client cert). kubectl ignores
                # them; this client reports them as a missing key. Retry
                # without them before treating the kubeconfig as unusable.
                k8s_config.load_kube_config_from_dict(_without_null_credentials(_read_kube_config()))
        except Exception as kubeconfig_error:
            try:
                k8s_config.load_incluster_config()
            except Exception as incluster_error:
                raise RuntimeError(
                    f"could not load Kubernetes credentials: from kubeconfig: {kubeconfig_error}; "
                    f"in-cluster: {incluster_error}"
                ) from kubeconfig_error
        return k8s_client

    def _bearer_token(self) -> str:
        """Mint a ServiceAccount token for ateapi.

        Resolved here rather than through config interpolation: the per-instance
        config is written to disk, and a token in it would be written in
        plaintext and would not reach a remote worker anyway.

        Minted tokens are re-minted shortly before they expire; a provider can
        outlive the token TTL by hours (long eval runs), and ateapi rejects an
        expired JWT on every control call. A token supplied via config is the
        caller's to manage and is never replaced.
        """
        if self._token and self._token_deadline is None:
            return self._token
        if self._token and time.monotonic() < self._token_deadline:
            return self._token

        k8s_client = self._kube_client()
        api = k8s_client.CoreV1Api()
        # ateapi expects its own service name as the audience, which is also
        # the name its serving certificate carries.
        request = k8s_client.AuthenticationV1TokenRequest(
            spec=k8s_client.V1TokenRequestSpec(
                audiences=[self._connection.tls_server_name], expiration_seconds=_TOKEN_TTL_S
            )
        )
        response = api.create_namespaced_service_account_token(
            self._connection.service_account,
            self._connection.service_account_namespace,
            request,
        )
        self._token = response.status.token
        # 5 min slack so a token is never presented in its final seconds.
        self._token_deadline = time.monotonic() + _TOKEN_TTL_S - 300
        return self._token

    def _ca_bundle(self) -> bytes | None:
        """Return the CA that signed ateapi's serving certificate.

        The certificate is issued by a cluster-internal CA, so a client outside
        the cluster has to be told about it. In-cluster components mount it from
        a ClusterTrustBundle; read the same object rather than making the caller
        extract it by hand.
        """
        if self._connection.ca_cert_path:
            return Path(self._connection.ca_cert_path).read_bytes()

        k8s_client = self._kube_client()
        selector = f"spec.signerName={self._connection.ca_trust_bundle_signer}"
        # ClusterTrustBundle is served under different versions depending on the
        # cluster, and the client only has the ones its own version knows about.
        errors = []
        for api_name in ("CertificatesV1Api", "CertificatesV1beta1Api", "CertificatesV1alpha1Api"):
            api = getattr(k8s_client, api_name, None)
            if api is None:
                continue
            try:
                bundles = api().list_cluster_trust_bundle(field_selector=selector)
            except Exception as e:
                errors.append(f"{api_name}: {e}")
                continue
            pem = "".join(item.spec.trust_bundle for item in bundles.items)
            if pem:
                return pem.encode()

        raise RuntimeError(
            f"no ClusterTrustBundle for signer {self._connection.ca_trust_bundle_signer!r} "
            f"({'; '.join(errors) or 'none found'}); set connection.ca_cert_path instead"
        )

    def _channel_credentials(self) -> grpc.ChannelCredentials:
        return grpc.ssl_channel_credentials(root_certificates=self._ca_bundle())

    async def _control(self, rpc: str, request: Any) -> Any:
        """Call one ateapi Control RPC, retrying transient failures."""
        if self._stub is None:
            options = [("grpc.ssl_target_name_override", self._connection.tls_server_name)]
            self._channel = grpc.aio.secure_channel(
                self._connection.ateapi_addr, self._channel_credentials(), options=options
            )
            self._stub = ateapi_pb2_grpc.ControlStub(self._channel)

        method = getattr(self._stub, rpc)
        for attempt in range(_CONTROL_ATTEMPTS):
            metadata = (("authorization", f"Bearer {self._bearer_token()}"),)
            try:
                return await method(request, metadata=metadata, timeout=self._connection.grpc_timeout_s)
            except grpc.aio.AioRpcError as e:
                if e.code() not in _RETRYABLE_CONTROL_CODES or attempt == _CONTROL_ATTEMPTS - 1:
                    raise
                delay = _CONTROL_BACKOFF_S * 2**attempt + random.uniform(0, _CONTROL_BACKOFF_S)
                LOGGER.warning("substrate: %s failed with %s; retrying in %.1fs", rpc, e.code().name, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _http(
        self,
        method: str,
        actor: _SubstrateActor,
        path: str,
        *,
        timeout_s: float | None,
        json: Any = None,
        content: bytes | None = None,
        params: dict[str, str] | None = None,
        limited: bool = True,
        retry: bool = True,
    ) -> httpx.Response:
        """Call the in-actor server through the router.

        The ate-target-actor header names the actor, and is also what makes
        the router resume a suspended or paused actor before forwarding. A 503
        the router produced before forwarding is retried (see _classify_503);
        the request never reached the actor, so that is safe. Readiness probes
        pass retry=False: their own loop retries, and each attempt must get
        the router's whole parking budget.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(base_url=f"http://{self._connection.atenet_addr}")

        headers = {TARGET_ACTOR_HEADER: actor.target}
        # Join the caller's trace (W3C traceparent) so the router's resume and
        # atelet's restore/checkpoint spans nest under the client exec span.
        try:
            from opentelemetry import propagate

            propagate.inject(headers)
        except ImportError:
            pass

        loop = asyncio.get_running_loop()
        budget = timeout_s if timeout_s is not None else (self._exec_config.default_timeout_s or 1800.0)
        deadline = loop.time() + budget
        transient_deadline = loop.time() + min(budget, _TRANSIENT_RETRY_BUDGET_S)
        while True:
            remaining = max(deadline - loop.time(), 1.0)
            if limited:
                async with self._semaphore:
                    response = await self._http_client.request(
                        method, path, headers=headers, json=json, content=content, params=params, timeout=remaining
                    )
            else:
                response = await self._http_client.request(
                    method, path, headers=headers, json=json, content=content, params=params, timeout=remaining
                )
            kind = _classify_503(response) if retry else None
            if kind is None:
                return response
            if kind == "capacity":
                delay, limit = self._exec_config.capacity_retry_delay_s, deadline
            else:
                delay, limit = _TRANSITION_RETRY_DELAY_S, transient_deadline
            if loop.time() + delay >= limit:
                return response
            LOGGER.info("substrate: %s not routable yet (%s); retrying in %.0fs", actor.name, path, delay)
            await asyncio.sleep(delay)

    # ---- lifecycle ---- #

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if not spec.image:
            raise SubstrateCreateError("spec.image is required for the substrate provider")
        self._warn_unsupported(spec)

        image = normalize_image(spec.image)
        template = template_name_for_image(image)
        actor_name = f"gym-{uuid.uuid4().hex[:12]}"

        actor = _SubstrateActor(
            atespace=self._connection.atespace,
            name=actor_name,
            template=template,
            image=image,
            env=dict(spec.env),
            workdir=spec.workdir,
        )
        handle = SandboxHandle(sandbox_id=actor_name, provider_name=self.name, raw=actor)

        try:
            await self._control(
                "CreateActor",
                ateapi_pb2.CreateActorRequest(
                    actor=ateapi_pb2.Actor(
                        metadata=ateapi_pb2.ResourceMetadata(atespace=actor.atespace, name=actor.name),
                        actor_template=ateapi_pb2.ObjectRef(
                            atespace=self._connection.template_namespace, name=template
                        ),
                    )
                ),
            )
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise SubstrateCreateError(
                    f"no ActorTemplate {self._connection.template_namespace}/{template} for image {image!r}; "
                    "templates are built ahead of a run"
                ) from e
            raise SubstrateCreateError(f"CreateActor failed for image {image!r}: {e.details()}") from e

        try:
            # First activation happens through the readiness probe: the router
            # resumes a not-yet-running actor on request and parks the request
            # while the pool is saturated, retrying far more aggressively than
            # any client-side loop could. Calling ateapi's fail-fast ResumeActor
            # here instead would make new actors race parked resumes for freed
            # slots at a hopeless cadence, capping the live-actor population at
            # the worker count under load.
            await self._wait_until_ready(handle, spec.ready_timeout_s or self._create_config.ready_timeout_s)
            await self._upload_spec_files(handle, spec)
        except Exception:
            await self._cleanup_failed_create(handle)
            raise

        return handle

    def _warn_unsupported(self, spec: SandboxSpec) -> None:
        """Say plainly which parts of the spec Substrate does not honor."""
        if spec.ttl_s is not None:
            LOGGER.debug("substrate: ttl_s is ignored; a suspended actor costs almost nothing")
        if spec.entrypoint:
            LOGGER.warning("substrate: spec.entrypoint is ignored; the container command runs the sandbox server")
        resources = spec.resources
        if any(getattr(resources, f, None) is not None for f in ("cpu", "memory_mib", "disk_gib", "gpu")):
            LOGGER.debug("substrate: spec.resources is ignored; actors are sized by their ActorTemplate")

    async def _wait_until_ready(self, handle: SandboxHandle, deadline_s: float) -> None:
        """Poll the readiness probe through the router until it passes.

        Probes bypass the exec concurrency limit: they are what triggers the
        resume, and a probe parked at the router must not hold a slot that a
        running sandbox's exec needs.
        """
        probe = self._probe
        loop = asyncio.get_running_loop()
        deadline = loop.time() + deadline_s
        consecutive = 0
        last_detail = "no probe attempt completed"

        while True:
            try:
                response = await self._http(
                    "GET", handle.raw, probe.path, timeout_s=probe.timeout_s, limited=False, retry=False
                )
                if response.status_code == 200:
                    consecutive += 1
                    if consecutive >= probe.stable_count:
                        return
                else:
                    consecutive = 0
                    last_detail = f"status={response.status_code}, body={response.text.strip()!r}"
            except httpx.HTTPError as e:
                consecutive = 0
                last_detail = repr(e)

            if loop.time() >= deadline:
                raise SubstrateCreateVerificationError(
                    f"actor {handle.sandbox_id!r} did not become ready within {deadline_s:g}s: {last_detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

    async def _upload_spec_files(self, handle: SandboxHandle, spec: SandboxSpec) -> None:
        for target_path, content in (spec.files or {}).items():
            await self._write_file(handle, target_path, content.encode() if isinstance(content, str) else content)

    async def _cleanup_failed_create(self, handle: SandboxHandle) -> None:
        try:
            await self.close(handle)
        except Exception as e:
            LOGGER.warning("substrate: cleaning up failed create of %s: %s", handle.sandbox_id, e)

    # ---- operations ---- #

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
        actor: _SubstrateActor = handle.raw
        effective_timeout = timeout_s if timeout_s is not None else self._exec_config.default_timeout_s

        merged_env = dict(actor.env)
        if env:
            merged_env.update(env)

        payload = {
            "command": command,
            "cwd": cwd if cwd is not None else actor.workdir,
            "env": merged_env,
            "user": "" if user is None else str(user),
        }
        if effective_timeout is not None:
            payload["timeout_s"] = float(effective_timeout)

        # The request budget exceeds the command's own timeout so the agent gets
        # to report a timeout itself, rather than the client giving up first and
        # losing the output.
        request_timeout = None if effective_timeout is None else float(effective_timeout) + 30.0

        async with self._in_flight(actor):
            try:
                response = await self._http("POST", actor, "/exec", timeout_s=request_timeout, json=payload)
            except httpx.TimeoutException as e:
                return SandboxExecResult(
                    stdout=None, stderr=str(e), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="timeout"
                )
            except httpx.HTTPError as e:
                return SandboxExecResult(
                    stdout=None, stderr=str(e), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="sandbox"
                )

            if response.status_code != 200:
                # A router 5xx means the actor could not be reached or resumed;
                # that is a sandbox failure, not the command's exit status.
                return SandboxExecResult(
                    stdout=None,
                    stderr=f"router returned {response.status_code}: {response.text.strip()}",
                    return_code=SANDBOX_RUNTIME_RETURN_CODE,
                    error_type="sandbox",
                )

            body = response.json()
            result = SandboxExecResult(
                stdout=body.get("stdout"),
                stderr=body.get("stderr"),
                return_code=body.get("exit_code", SANDBOX_RUNTIME_RETURN_CODE),
                error_type=body.get("error_type") or None,
            )
            if body.get("evicting") and not actor.evicting:
                actor.evicting = True
                LOGGER.warning("substrate: worker of %s is being evicted; it will be suspended when idle", actor.name)

        try:
            await self._suspend_when_idle(actor)
        except Exception as e:  # noqa: BLE001 - the command ran; parking is best effort
            LOGGER.warning("substrate: parking %s after exec failed: %s", actor.name, e)
        return result

    # ---- idle handling ---- #

    @contextlib.asynccontextmanager
    async def _in_flight(self, actor: _SubstrateActor):
        """Hold a request slot on the actor.

        A suspend still waiting out its idle delay is dropped; one already under
        way finishes first, since the router cannot resume an actor that is
        mid-transition. While the slot is held the actor is never suspended.
        """
        self._cancel_pending_suspend(actor)
        async with actor.lock:
            actor.inflight += 1
            actor.parked = False  # the router resumes it for this request
        try:
            yield
        finally:
            actor.inflight -= 1

    def _cancel_pending_suspend(self, actor: _SubstrateActor) -> None:
        """Cancel a suspend that is still waiting out the idle delay.

        One that has already reached the control plane is left to finish; the
        caller then waits on the actor's lock instead, because the router
        cannot resume an actor that is mid-transition.
        """
        if actor.suspend_task is not None and not actor.suspend_task.done() and actor.parking == 0:
            actor.suspend_task.cancel()
            actor.suspend_task = None

    async def _suspend_when_idle(self, actor: _SubstrateActor) -> None:
        """Free the actor's worker once it has been idle long enough.

        Nothing resumes it explicitly: the next exec routes through the router,
        which resumes the actor before forwarding. The cost is a checkpoint
        here and a restore on the next call.
        """
        policy = self._create_config.suspend_policy
        if actor.evicting:
            # The node is going away: a node-local pause would go with it.
            rpc = "SuspendActor"
        elif policy == "never":
            return
        else:
            rpc = "SuspendActor" if policy == "after_exec" else "PauseActor"
        delay = self._create_config.idle_before_suspend_s
        self._cancel_pending_suspend(actor)
        # Always a task, shielded from the caller: a PauseActor RPC that is
        # cancelled client-side leaves the actor stranded mid-transition on
        # the server, with nothing to finish it.
        task = asyncio.create_task(self._suspend_after(actor, rpc, delay))
        actor.suspend_task = task
        self._suspend_tasks.add(task)
        task.add_done_callback(self._suspend_tasks.discard)
        if delay <= 0:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    return  # a newer request superseded the park; the result stands
                raise

    async def _suspend_after(self, actor: _SubstrateActor, rpc: str, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        actor.parking += 1
        try:
            await self._suspend_if_idle(actor, rpc)
        except Exception as e:  # noqa: BLE001 - a background task must not die silently
            LOGGER.warning("substrate: idle %s of %s failed: %s", rpc, actor.name, e)
        finally:
            actor.parking -= 1

    async def _suspend_if_idle(self, actor: _SubstrateActor, rpc: str) -> None:
        async with actor.lock:
            if actor.inflight or actor.parked:
                return
            try:
                state = await self._park(actor, rpc)
            except grpc.aio.AioRpcError as e:
                # A failed suspend costs density, not correctness: the actor
                # stays running and the next exec still works.
                LOGGER.warning("substrate: %s on %s failed: %s", rpc, actor.name, e.details())
                return
            actor.parked = state in _PARKED_STATES
            if actor.parked and rpc == "SuspendActor":
                actor.evicting = False  # the snapshot is in storage; the node can go
        if state not in _PARKED_STATES:
            LOGGER.warning(
                "substrate: %s on %s did not park it: %s", rpc, actor.name, ateapi_pb2.ActorState.Name(state)
            )

    async def _park(self, actor: _SubstrateActor, rpc: str) -> int:
        """Pause or suspend the actor and wait out the transition.

        Returns the state the actor settled in. The caller holds actor.lock.
        """
        if rpc == "PauseActor":
            request: Any = ateapi_pb2.PauseActorRequest(actor=actor.object_ref)
        else:
            request = ateapi_pb2.SuspendActorRequest(actor=actor.object_ref)
        try:
            await self._control(rpc, request)
        except grpc.aio.AioRpcError as e:
            # Already parked, by a previous call or by the platform.
            if e.code() != grpc.StatusCode.FAILED_PRECONDITION:
                raise

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _PARK_SETTLE_BUDGET_S
        while True:
            reply = await self._control("GetActor", ateapi_pb2.GetActorRequest(actor=actor.object_ref))
            state = reply.status.state
            if state not in _TRANSITIONAL_STATES or loop.time() >= deadline:
                return state
            await asyncio.sleep(0.25)

    async def pause(self, handle: SandboxHandle) -> None:
        """Pause the actor once it is idle and return when the control plane reports it parked.

        A pause while a command runs would checkpoint the command mid-flight
        and cut its response, so in-flight requests finish first.
        """
        actor: _SubstrateActor = handle.raw
        self._cancel_pending_suspend(actor)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._connection.grpc_timeout_s
        # A worker that is going away cannot keep a node-local snapshot.
        rpc = "SuspendActor" if actor.evicting else "PauseActor"
        while True:
            async with actor.lock:
                if not actor.inflight:
                    state = await self._park(actor, rpc)
                    actor.parked = state in _PARKED_STATES
                    if actor.parked and rpc == "SuspendActor":
                        actor.evicting = False
                    break
            if loop.time() >= deadline:
                raise RuntimeError(f"actor {actor.name!r} still has requests in flight; not pausing")
            await asyncio.sleep(0.1)
        if state not in _PARKED_STATES:
            raise RuntimeError(f"actor {actor.name!r} is {ateapi_pb2.ActorState.Name(state)} after pause")

    async def resume(self, handle: SandboxHandle) -> None:
        """Resume through the router, which parks the request until a worker has room."""
        self._cancel_pending_suspend(handle.raw)
        await self._wait_until_ready(handle, self._create_config.ready_timeout_s)

    async def _write_file(self, handle: SandboxHandle, target_path: str, content: bytes) -> None:
        async with self._in_flight(handle.raw):
            response = await self._http(
                "POST",
                handle.raw,
                "/upload",
                params={"path": target_path},
                timeout_s=self._exec_config.default_timeout_s,
                content=content,
            )
        if response.status_code not in (200, 204):
            raise RuntimeError(f"upload to {target_path!r} failed: {response.status_code} {response.text.strip()}")

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        await self._write_file(handle, target_path, Path(source_path).read_bytes())
        await self._suspend_when_idle(handle.raw)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        async with self._in_flight(handle.raw):
            response = await self._http(
                "GET",
                handle.raw,
                "/download",
                params={"path": source_path},
                timeout_s=self._exec_config.default_timeout_s,
            )
        if response.status_code != 200:
            raise RuntimeError(f"download of {source_path!r} failed: {response.status_code} {response.text.strip()}")
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(response.content)
        await self._suspend_when_idle(handle.raw)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        actor: _SubstrateActor = handle.raw
        try:
            reply = await self._control("GetActor", ateapi_pb2.GetActorRequest(actor=actor.object_ref))
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return SandboxStatus.STOPPED
            return SandboxStatus.UNKNOWN
        return _STATUS_MAP.get(reply.status.state, SandboxStatus.UNKNOWN)

    async def close(self, handle: SandboxHandle) -> None:
        actor: _SubstrateActor = handle.raw
        self._cancel_pending_suspend(actor)
        # A pause that already reached the control plane holds the lock until
        # the actor settles; deleting under it would only collide with it.
        async with actor.lock:
            if not self._create_config.delete_on_close:
                # Keep the sandbox for inspection, but not its worker.
                try:
                    await self._park(actor, "SuspendActor")
                except grpc.aio.AioRpcError as e:
                    if e.code() != grpc.StatusCode.NOT_FOUND:
                        LOGGER.warning("substrate: suspending %s on close: %s", actor.name, e.details())
                return

            # any_state removes a running, paused or mid-transition actor
            # outright; nothing about a finished sandbox is worth a final
            # snapshot.
            try:
                await self._control(
                    "DeleteActor", ateapi_pb2.DeleteActorRequest(actor=actor.object_ref, any_state=True)
                )
            except grpc.aio.AioRpcError as e:
                if e.code() != grpc.StatusCode.NOT_FOUND:
                    raise RuntimeError(f"deleting actor {actor.name!r} failed: {e.details()}") from e

    async def aclose(self) -> None:
        # A park still in flight must finish before its channel goes away, or
        # the actor is left mid-transition.
        if self._suspend_tasks:
            await asyncio.gather(*self._suspend_tasks, return_exceptions=True)
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
