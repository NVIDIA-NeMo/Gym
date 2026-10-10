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

"""Sandbox provider backed by Substrate actors."""

import asyncio
import copy
import hashlib
import logging
import os
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

# Actors are reachable through the atenet router by a name derived from the
# actor and its atespace. The router resolves it, resuming the actor first if
# it is suspended.
TARGET_ACTOR_HEADER = "ate-target-actor"

# Reported when no process exit code exists, because the command never ran or
# the sandbox itself failed.
SANDBOX_RUNTIME_RETURN_CODE = 125

# Template names are DNS labels, so the digest is truncated rather than used
# whole. Substrate's template generator derives names the same way; the two must
# stay in step.
TEMPLATE_NAME_PREFIX = "swb-"
TEMPLATE_DIGEST_LENGTH = 16

_STATUS_MAP = {
    ateapi_pb2.ActorState.ACTOR_STATE_RUNNING: SandboxStatus.RUNNING,
    ateapi_pb2.ActorState.ACTOR_STATE_RESUMING: SandboxStatus.STARTING,
    # A suspended or paused actor is alive and one request away: the router
    # resumes it on the next exec. Reporting it as STOPPED would make callers
    # discard a perfectly good sandbox.
    ateapi_pb2.ActorState.ACTOR_STATE_SUSPENDED: SandboxStatus.RUNNING,
    ateapi_pb2.ActorState.ACTOR_STATE_SUSPENDING: SandboxStatus.RUNNING,
    ateapi_pb2.ActorState.ACTOR_STATE_PAUSED: SandboxStatus.RUNNING,
    ateapi_pb2.ActorState.ACTOR_STATE_PAUSING: SandboxStatus.RUNNING,
    ateapi_pb2.ActorState.ACTOR_STATE_CRASHED: SandboxStatus.ERROR,
    ateapi_pb2.ActorState.ACTOR_STATE_DELETING: SandboxStatus.STOPPED,
}


class SubstrateCreateError(SandboxCreateError):
    """Raised when a Substrate actor cannot be created or resumed."""


class SubstrateCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a created actor never becomes ready."""


def template_name_for_image(image: str) -> str:
    """Derive the ActorTemplate name that serves `image`.

    An ActorTemplate spec is immutable and an actor names a template rather than
    an image, so templates are built ahead of a run. Deriving the name from the
    image lets both sides agree without a shared lookup table.
    """
    digest = hashlib.sha256(image.encode()).hexdigest()
    return TEMPLATE_NAME_PREFIX + digest[:TEMPLATE_DIGEST_LENGTH]


def normalize_image(image: str) -> str:
    """Strip a runtime scheme prefix, matching what the other providers accept."""
    for prefix in ("docker://", "oci://"):
        if image.startswith(prefix):
            return image[len(prefix) :]
    return image


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
    template_namespace: str = "ate-swebench"

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
    ready_timeout_s: float = 1200.0

    # never: leave the actor running between calls.
    # after_exec: SuspendActor once each exec returns. The snapshot goes to
    #   object storage and the worker is freed.
    # pause_after_exec: PauseActor instead. The snapshot stays on the node, so
    #   the next resume skips the download but has to land on the same node.
    # Either way the next exec goes through the router, which resumes the
    # actor before routing, so nothing has to resume it explicitly.
    suspend_policy: str = "never"

    delete_on_close: bool = True

    def __post_init__(self) -> None:
        allowed = {"never", "after_exec", "pause_after_exec"}
        if self.suspend_policy not in allowed:
            raise ValueError(f"create.suspend_policy must be one of {sorted(allowed)}, got {self.suspend_policy!r}")


@dataclass(frozen=True)
class SubstrateExecConfig:
    default_timeout_s: float | None = 1800.0
    concurrency: int = 32
    # The workload image supplies the interpreter; SWE-bench harnesses need bash
    # because conda activation is not portable to dash.
    shell: str = "bash"

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")


@dataclass(frozen=True)
class SubstrateProbeConfig:
    """Readiness probe run against the actor through the router."""

    path: str = "/readyz"
    timeout_s: float = 30.0
    deadline_s: float | None = 300.0
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
    suspended: bool = False

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
    to an agent inside the actor, reached through the atenet router.
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
        """Call one ateapi Control RPC."""
        if self._stub is None:
            options = [("grpc.ssl_target_name_override", self._connection.tls_server_name)]
            self._channel = grpc.aio.secure_channel(
                self._connection.ateapi_addr, self._channel_credentials(), options=options
            )
            self._stub = ateapi_pb2_grpc.ControlStub(self._channel)

        metadata = (("authorization", f"Bearer {self._bearer_token()}"),)
        method = getattr(self._stub, rpc)
        return await method(request, metadata=metadata, timeout=self._connection.grpc_timeout_s)

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
    ) -> httpx.Response:
        """Call the in-actor agent through the router.

        The ate-target-actor header names the actor, and is also what makes
        the router resume a suspended or paused actor before forwarding.
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

        async with self._semaphore:
            return await self._http_client.request(
                method,
                path,
                headers=headers,
                json=json,
                content=content,
                params=params,
                timeout=timeout_s,
            )

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
            await self._verify_created_handle(handle)
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
            LOGGER.warning("substrate: spec.entrypoint is ignored; the container command runs the sandbox agent")
        resources = spec.resources
        if any(getattr(resources, f, None) is not None for f in ("cpu", "memory_mib", "disk_gib", "gpu")):
            LOGGER.debug("substrate: spec.resources is ignored; actors are sized by their WorkerPool")

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        """Poll the readiness probe until it passes or the deadline elapses."""
        probe = self._probe
        loop = asyncio.get_running_loop()
        deadline = loop.time() + probe.deadline_s if probe.deadline_s is not None else None
        consecutive = 0
        last_detail = "no probe attempt completed"

        while True:
            try:
                response = await self._http("GET", handle.raw, probe.path, timeout_s=probe.timeout_s)
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

            if deadline is not None and loop.time() >= deadline:
                raise SubstrateCreateVerificationError(
                    f"actor {handle.sandbox_id!r} did not become ready within {probe.deadline_s:g}s: {last_detail}"
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
            # A router 5xx means the actor could not be reached or resumed; that
            # is a sandbox failure, not the command's exit status.
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

        await self._maybe_suspend(actor)
        return result

    async def _maybe_suspend(self, actor: _SubstrateActor) -> None:
        """Free the actor's worker between steps when asked to.

        Nothing resumes it explicitly: the next exec routes through the router,
        which resumes the actor before forwarding. The cost is a checkpoint here
        and a restore on the next call, both on the critical path.
        """
        policy = self._create_config.suspend_policy
        if policy == "after_exec":
            rpc, request = "SuspendActor", ateapi_pb2.SuspendActorRequest(actor=actor.object_ref)
        elif policy == "pause_after_exec":
            rpc, request = "PauseActor", ateapi_pb2.PauseActorRequest(actor=actor.object_ref)
        else:
            return
        try:
            await self._control(rpc, request)
            actor.suspended = True
        except grpc.aio.AioRpcError as e:
            # A failed suspend costs density, not correctness: the actor stays
            # running and the next exec still works.
            LOGGER.warning("substrate: %s on %s failed: %s", rpc, actor.name, e.details())

    async def _write_file(self, handle: SandboxHandle, target_path: str, content: bytes) -> None:
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

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
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
        if not self._create_config.delete_on_close:
            # Keep the sandbox for inspection, but not its worker.
            try:
                await self._control("SuspendActor", ateapi_pb2.SuspendActorRequest(actor=actor.object_ref))
            except grpc.aio.AioRpcError as e:
                if e.code() not in (grpc.StatusCode.NOT_FOUND, grpc.StatusCode.FAILED_PRECONDITION):
                    LOGGER.warning("substrate: suspending %s on close: %s", actor.name, e.details())
            return

        # any_state removes a running, paused or mid-transition actor outright;
        # nothing about a finished sandbox is worth a final snapshot.
        try:
            await self._control("DeleteActor", ateapi_pb2.DeleteActorRequest(actor=actor.object_ref, any_state=True))
        except grpc.aio.AioRpcError as e:
            if e.code() != grpc.StatusCode.NOT_FOUND:
                raise RuntimeError(f"deleting actor {actor.name!r} failed: {e.details()}") from e

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
