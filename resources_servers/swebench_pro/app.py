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

"""SWE-bench Pro resources server."""

import asyncio
import logging
import socket
import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from time import time
from traceback import format_exc
from typing import Annotated, Any, Literal, TypeVar
from uuid import uuid4

import aiohttp
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.agents import lifecycle_observation, restricted_network_policy, sandbox_server_url
from nemo_gym.agents.utils.verify_call import is_verify_transport_error
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    VERIFIER_ERROR,
    EpisodeId,
    FailureBoundary,
    NgFailureEvent,
    TaskId,
    current_verifier_invocation_id,
    observe_failure,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    ConnectableProvider,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import explicit_causes
from resources_servers.swebench_pro.image_cache import digest_hex, verify_local_image
from resources_servers.swebench_pro.verification import (
    DEFAULT_ENVIRONMENT_REPAIRS,
    VerificationInputs,
    VerificationResult,
    assemble_workspace_files,
    build_seed_normalization,
    drop_patch_sections,
    inconclusive_reason,
    run_verification,
)


# K8s maps localhost to ::1 and Node 17+ honours that, but servers under test bind IPv4.
SANDBOX_ENV_OVERRIDES = {"NODE_OPTIONS": "--dns-result-order=ipv4first"}
LOG = logging.getLogger(__name__)


# Blanked on the spec so the agent sees a clean env; the entryscript unsets them for real.
HARNESS_ENV_TO_SCRUB = (
    "OTEL_SERVICE_NAME",
    "OTEL_SERVICE_VERSION",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "OTEL_METRIC_EXPORT_INTERVAL",
    "OTEL_TRACES_EXPORTER",
    "OTEL_METRICS_EXPORTER",
    "OTEL_LOGS_EXPORTER",
)


def _verification_deadline(total_timeout: float | None) -> float | None:
    """Wall-clock instant by which all attempts for one rollout must be done."""
    return None if total_timeout is None else time() + total_timeout


def _budget_spent(deadline: float | None) -> bool:
    return deadline is not None and time() >= deadline


def _attempt_budget(attempt_timeout: float | None, deadline: float | None) -> float | None:
    """The smaller of this attempt's ceiling and what is left of the rollout's budget.

    ``asyncio.timeout(None)`` is a no-op, so both being unset preserves the old
    unbounded behaviour for anyone who wants it back.
    """
    remaining = None if deadline is None else max(deadline - time(), 0.0)
    if attempt_timeout is None:
        return remaining
    if remaining is None:
        return attempt_timeout
    return min(attempt_timeout, remaining)


class PatchCaptureLostError(RuntimeError):
    """The agent sandbox did not report how ``git diff`` ended or did not return its output, so the patch is unknown."""


PATCH_CAPTURE_TRANSPORT_RETRIES = 2
"""Extra attempts of the patch capture's ``git diff`` exec, and of its download, after a transport failure.

The agent sandbox is stopped once the capture ends, so a capture lost to a transient network failure loses the
rollout's patch for good."""
PATCH_CAPTURE_RETRY_DELAY_S = 0.5
"""Wait before the first retry of a patch capture step; it doubles before each later one."""

# Failures to reach the sandbox at all: no request was sent, so nothing ran. These are the connect failures Gym's HTTP
# client resends unasked (``ClientConnectorError``, which covers refused connections, DNS and "Connect call failed")
# and the unsent timeouts the OpenSandbox provider recognises (connect and pool timeouts).
_CONNECT_ERRORS: tuple[type[BaseException], ...] = (
    aiohttp.ClientConnectorError,
    aiohttp.ConnectionTimeoutError,
    ConnectionRefusedError,
    socket.gaierror,
)
_SENT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = ()
try:  # Sandbox SDKs speak httpx; Gym's aiohttp transport maps its errors onto httpx ones.
    import httpx

    _CONNECT_ERRORS += (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
    _SENT_TRANSPORT_ERRORS = (httpx.TransportError,)
except ModuleNotFoundError:  # pragma: no cover - httpx comes with the sandbox SDKs this server requires
    pass

_T = TypeVar("_T")


def _transport_failure_phase(error: BaseException) -> Literal["connect", "sent"] | None:
    """How ``error`` failed in transport: ``"connect"`` before any request was sent, ``"sent"`` after one may have
    been (a dropped connection, a read timeout), or ``None`` when it is no transport failure.

    The innermost transport error in the explicit cause chain decides. A sandbox SDK wraps its client's error, and
    Gym's aiohttp transport re-raises an aiohttp error as an httpx one, so an outer ``httpx.ConnectError`` may carry a
    connection reset after the request was written; the aiohttp error underneath says what happened on the wire.
    """
    phase: Literal["connect", "sent"] | None = None
    for link in explicit_causes(error):
        if isinstance(link, _CONNECT_ERRORS):
            phase = "connect"
        elif is_verify_transport_error(link) or isinstance(link, _SENT_TRANSPORT_ERRORS):
            phase = "sent"
    return phase


async def _retry_patch_capture_step(
    step: str, call: Callable[[], Awaitable[_T]], *, idempotent: bool
) -> tuple[_T, int]:
    """``call()`` and its attempt count, retried after a transport failure another attempt can repair.

    A connect failure sent nothing, so it is always retried. A failure after sending is retried only for an
    idempotent ``step``. The last failure, and any other one, propagates.
    """
    attempts = PATCH_CAPTURE_TRANSPORT_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            return await call(), attempt
        except Exception as exc:
            phase = _transport_failure_phase(exc)
            if attempt == attempts or not (phase == "connect" or (phase == "sent" and idempotent)):
                raise
            delay_s = PATCH_CAPTURE_RETRY_DELAY_S * 2 ** (attempt - 1)
            LOG.warning(
                "Patch capture %s failed in transport (%s, attempt %d/%d); retrying in %.1fs: %r",
                step,
                phase,
                attempt,
                attempts,
                delay_s,
                exc,
            )
            await asyncio.sleep(delay_s)
    raise AssertionError("unreachable")  # pragma: no cover


class SWEBenchProTaskNetworkPolicy(BaseModel):
    """Allow only the model server host for task egress, on every port and route.

    Requires an OpenSandbox deployment enforcing network policies with dns+nft.
    Grading sandboxes retain the original sandbox_config.provider_options.
    """

    model_config = ConfigDict(extra="forbid")

    model_server: ModelServerRef


class SWEBenchProResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    apply_anti_cheating: bool = True
    prefetch_go_modules: bool = False
    evaluation_timeout: int | None = None
    # A verdict-less run is retried on a new sandbox; see `inconclusive_reason`.
    inconclusive_verification_retries: int = 2
    # Ceiling on ONE verification attempt, covering sandbox creation as well as
    # the verification run. `evaluation_timeout` bounds only the test command
    # inside the sandbox, so creation is otherwise unbounded here. 1200s is
    # ~2.6x the p99 of observed per-rollout verification (461s) and above the
    # healthy maximum (735s), while still cutting the multi-attempt pile-ups
    # that leave dozens of verifications in flight at a job's wall clock.
    verification_attempt_timeout: float | None = 1200.0
    # Ceiling on ALL attempts for one rollout. Without it the worst case is
    # `1 + inconclusive_verification_retries` times the per-attempt ceiling,
    # which can exceed what remains of the job's wall clock -- and a rollout
    # that never returns holds the whole run open, because collection ends only
    # when the last rollout does.
    verification_total_timeout: float | None = 2700.0
    # Cleanup gets its own, smaller ceiling: a stop() that hangs in `finally`
    # would defeat the attempt timeout it runs after.
    verification_stop_timeout: float | None = 120.0
    # Which container repairs to apply; see `ENVIRONMENT_REPAIRS`.
    environment_repairs: tuple[str, ...] = DEFAULT_ENVIRONMENT_REPAIRS
    image_repository: str = "docker.io/jefzda/sweap-images"
    image_template: str | None = None  # Local SIF path with a checked provenance manifest.
    sandbox_provider: str
    sandbox_config: dict[str, Any]
    task_network_policy: SWEBenchProTaskNetworkPolicy | None = None


class SWEBenchProInstanceRequest(BaseModel):
    """One row from ScaleAI/SWE-bench_Pro plus pinned evaluator assets."""

    model_config = ConfigDict(extra="allow")

    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str = ""
    problem_statement: str
    requirements: str = ""
    interface: str = ""
    repo_language: str = ""
    fail_to_pass: str | list[str]
    pass_to_pass: str | list[str]
    issue_specificity: str = ""
    issue_categories: str = ""
    before_repo_set_cmd: str = ""
    selected_test_files_to_run: str | list[str]
    dockerhub_tag: str
    image_digest: str = ""
    run_script: str
    parser_script: str
    base_dockerfile: str = ""
    instance_dockerfile: str = ""
    subset: str = "pro"
    split: str = "test"


class SWEBenchProSeedSessionRequest(SWEBenchProInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class SWEBenchProSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    image_provenance: dict[str, Any] = Field(default_factory=dict)


class SWEBenchProVerifyRequest(SWEBenchProInstanceRequest, BaseVerifyRequest):
    pass


class SWEBenchProVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(serialize_by_alias=True)

    verification_attempts: int | None = Field(default=None, exclude_if=lambda value: value is None)
    image_provenance: dict[str, Any] = Field(default_factory=dict)
    evaluation_completed: bool
    eval_timed_out: bool = False
    resolved: bool
    patch_applied: bool
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float
    instance_id: str
    model_patch: str | None
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None
    log_dir: str
    # Lifecycle facts of the sandbox that produced this grade; the agent attaches it as a verifier SandboxObservation.
    verifier_sandbox_observation: dict[str, Any] | None = None
    verifier_sandbox_observations: list[dict[str, Any] | None] | None = None


class SWEBenchProResourcesServer(SimpleResourcesServer):
    config: SWEBenchProResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("SWE-bench Pro process-local sessions require num_workers=1")
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        # Untracked files the image ships, per session. Leading underscore: pydantic needs it.
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}
        # Typed sessions, which an Environment Server seeds and closes by resources_session_id.
        self._session_id_to_identity: dict[str, tuple[EpisodeId, TaskId]] = {}
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def close_resources_session(
        self,
        request: Request,
        body: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> ResourcesCloseSessionResponse | dict[str, bool]:
        # Agents that seeded through /run close their cookie session with an empty body.
        if not body:
            session_id = request.session.get(SESSION_ID_KEY)
            if session_id is None:
                return {"closed": True}
            await self._stop_session_sandbox(session_id)
            return {"closed": True}

        try:
            typed = ResourcesCloseSessionRequest.model_validate(body)
        except ValidationError as error:
            raise RequestValidationError(error.errors()) from error
        session_id = typed.resources_session_id
        # The body names the session, so a close without a cookie (after a lost seed reply) still releases it.
        async with self._fence.locked(session_id):
            closed_episode_id = self._fence.closed_episode(session_id)
            if closed_episode_id is not None:
                if typed.episode_id != closed_episode_id:
                    raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
                request.session.pop(SESSION_ID_KEY, None)
                return ResourcesCloseSessionResponse(resources_session_id=session_id)
            identity = self._session_id_to_identity.get(session_id)
            if identity is not None and typed.episode_id != identity[0]:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            await self._stop_session_sandbox(session_id)
            self._session_id_to_identity.pop(session_id, None)
            self._fence.mark_closed(session_id, typed.episode_id)
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _stop_session_sandbox(self, session_id: str) -> None:
        sandbox = self._session_id_to_sandbox.get(session_id)
        if sandbox is not None:
            async with asyncio.timeout(self.config.verification_stop_timeout):
                await sandbox.stop()
            self._session_id_to_sandbox.pop(session_id, None)
        self._session_id_to_pristine_untracked.pop(session_id, None)

    async def shutdown(self) -> None:
        self._session_id_to_identity.clear()
        self._fence.clear()
        for session_id in list(self._session_id_to_sandbox):
            try:
                await self._stop_session_sandbox(session_id)
            except Exception:
                LOG.exception("Failed to stop abandoned SWE-bench Pro sandbox %s", session_id)

    def _image(self, body: SWEBenchProInstanceRequest) -> str:
        if self.config.image_template:
            return self.config.image_template.format(
                instance_id=body.instance_id,
                dockerhub_tag=body.dockerhub_tag,
                image_digest_hex=digest_hex(body.image_digest),
            )
        if body.image_digest:
            return f"{self.config.image_repository}@{body.image_digest}"
        return f"{self.config.image_repository}:{body.dockerhub_tag}"

    def _image_info(self, body: SWEBenchProInstanceRequest) -> dict[str, Any]:
        image = self._image(body)
        if self.config.image_template:
            source = f"docker://{self.config.image_repository}@{body.image_digest}"
            return verify_local_image(Path(image), source)
        return {"image": image, "source_uri": image}

    async def _create_sandbox(
        self,
        body: SWEBenchProInstanceRequest,
        files: dict[str, str] | None = None,
        *,
        task: bool = False,
    ) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        provider_options = self.config.sandbox_config.get("provider_options", {})
        if task and self.config.task_network_policy is not None:
            model_url = sandbox_server_url(self.config.task_network_policy.model_server.name, require_reachable=True)
            # Replace rather than extend any baseline allowlist. Grading keeps the original options.
            provider_options = dict(provider_options or {})
            provider_options["network_policy"] = restricted_network_policy(next(iter(provider_config)), [model_url])
        image_info = await asyncio.to_thread(self._image_info, body)
        spec = SandboxSpec(
            image=image_info["image"],
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir="/app",
            env=dict.fromkeys(HARNESS_ENV_TO_SCRUB, "") | SANDBOX_ENV_OVERRIDES,
            files=files or {},
            metadata=provider_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": body.instance_id[:63],
            },
            resources=SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {})),
            entrypoint=None,
            provider_options=provider_options,
        )
        sandbox = AsyncSandbox(provider_config)
        await sandbox.start(spec)
        return sandbox

    def _verification_inputs(self, body: SWEBenchProInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            base_commit=body.base_commit,
            patch=patch,
            run_script=body.run_script,
            parser_script=body.parser_script,
            selected_test_files_to_run=body.selected_test_files_to_run,
            fail_to_pass=body.fail_to_pass,
            pass_to_pass=body.pass_to_pass,
            before_repo_set_cmd=body.before_repo_set_cmd,
            base_dockerfile=body.base_dockerfile,
            instance_dockerfile=body.instance_dockerfile,
            repo_language=body.repo_language,
            prefetch_go_modules=self.config.prefetch_go_modules,
            environment_repairs=tuple(self.config.environment_repairs),
        )

    async def seed_session(
        self,
        request: Request,
        body: SWEBenchProSeedSessionRequest | ResourcesSeedSessionRequest,
    ) -> SWEBenchProSeedSessionResponse | ResourcesSeedSessionResponse:
        if not isinstance(body, ResourcesSeedSessionRequest):
            return await self._seed_session(request.session[SESSION_ID_KEY], body)

        session_id = body.resources_session_id
        # The cookie is bound only once the seed succeeds: a refused seed's reply would otherwise point this client at
        # the session it was refused, which may belong to another episode.
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            identity = self._session_id_to_identity.get(session_id)
            if identity is not None:
                if identity != (body.episode_id, body.task_id):
                    raise ResourcesSessionConflictError(
                        "resources_session_id is already bound to another episode or task"
                    )
                response = await self._typed_seed_response(session_id)
                request.session[SESSION_ID_KEY] = session_id
                return response
            try:
                task = SWEBenchProInstanceRequest.model_validate(body.task_data)
            except ValidationError as error:
                raise ResourcesSeedRejectedError(f"task_data is not a SWE-bench Pro instance: {error}") from error
            if body.task_id.task_id != task.instance_id:
                raise ResourcesSeedRejectedError("TaskId does not match the SWE-bench Pro instance_id")
            await self._seed_session(session_id, task)
            try:
                response = await self._typed_seed_response(session_id)
            except BaseException:
                # The agent cannot reach a sandbox it was never handed, so stop it here.
                await self._stop_session_sandbox(session_id)
                raise
            self._session_id_to_identity[session_id] = (body.episode_id, body.task_id)
            request.session[SESSION_ID_KEY] = session_id
            return response

    async def _typed_seed_response(self, session_id: str) -> ResourcesSeedSessionResponse:
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider,
                    descriptor=await self._session_id_to_sandbox[session_id].serialize(),
                ),
                workdir="/app",
            ),
        )

    async def _seed_session(self, session_id: str, body: SWEBenchProInstanceRequest) -> SWEBenchProSeedSessionResponse:
        await self._stop_session_sandbox(session_id)
        sandbox = await self._create_sandbox(body, task=True)
        # Keep ownership before initialization, including when cleanup itself fails.
        self._session_id_to_sandbox[session_id] = sandbox
        try:
            if self.config.apply_anti_cheating:
                anti_cheat_setup_fpath = Path(__file__).parent.parent / "swebench" / "anti_cheat_setup.sh"
                await sandbox.upload(anti_cheat_setup_fpath, "/app/anti_cheat_setup.sh")
                result = await sandbox.exec(
                    "git reset --hard && WORKING_DIRECTORY=/app bash anti_cheat_setup.sh && rm anti_cheat_setup.sh",
                    timeout_s=600,
                )
                if result.return_code != 0:
                    print(
                        f"Failed to setup anti-cheating for {body.instance_id}. Return code: {result.return_code}\n"
                        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
                    )
            await self.normalize_sandbox_environment(sandbox, body.instance_id)
            pristine = await self.pristine_untracked_files(sandbox)
            descriptor = (
                await sandbox.serialize()
                if isinstance(getattr(sandbox, "_provider", None), ConnectableProvider)
                else None
            )
            response = SWEBenchProSeedSessionResponse(
                sandbox_handle=sandbox._handle.sandbox_id,
                sandbox_descriptor=descriptor,
                image_provenance=await asyncio.to_thread(self._image_info, body),
            )
        except BaseException:
            try:
                await self._stop_session_sandbox(session_id)
            except Exception:
                LOG.exception("Failed to stop partially initialized SWE-bench Pro sandbox %s", session_id)
            raise
        self._session_id_to_pristine_untracked[session_id] = pristine
        self._session_id_to_sandbox[session_id] = sandbox
        return response

    async def normalize_sandbox_environment(self, sandbox: AsyncSandbox, instance_id: str) -> None:
        """Give the agent container the same repairs the verifier gets; best effort."""
        try:
            script = build_seed_normalization(self.config.environment_repairs)
            result = await sandbox.exec(f"bash -c {quote(script)}", timeout_s=300)
            if result.return_code != 0:
                print(
                    f"Failed to normalize sandbox environment for {instance_id}. "
                    f"Return code: {result.return_code}\nStderr:\n{result.stderr}",
                    file=sys.stderr,
                )
        except Exception:
            print(f"Failed to normalize sandbox environment for {instance_id}", format_exc(), file=sys.stderr)

    async def pristine_untracked_files(self, sandbox: AsyncSandbox) -> frozenset[str]:
        """List the untracked files ``/app`` holds before the agent touches it."""
        try:
            result = await sandbox.exec("git -C /app ls-files --others --exclude-standard")
            if result.return_code != 0:
                print(f"Failed to list pristine untracked files: {result.stderr}", file=sys.stderr)
                return frozenset()
            return frozenset(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
        except Exception:
            print("Failed to list pristine untracked files", format_exc(), file=sys.stderr)
            return frozenset()

    async def _extract_model_patch(self, session_id: str, base_commit: str) -> str:
        original_sandbox = self._session_id_to_sandbox[session_id]
        pristine_untracked = self._session_id_to_pristine_untracked.get(session_id, frozenset())
        patch_path = f"/tmp/nemo-gym-swebench-pro-{uuid4().hex}.diff"
        try:
            # Only a connect failure is retried. After a request may have been sent, the first git process could
            # still hold the index lock while a second one starts.
            try:
                result, exec_attempts = await _retry_patch_capture_step(
                    "git diff exec",
                    lambda: original_sandbox.exec(
                        f"umask 077; git -C /app add -N . && git -C /app --no-pager diff {quote(base_commit)}"
                        f" > {quote(patch_path)}"
                    ),
                    idempotent=False,
                )
            except Exception as exc:
                raise PatchCaptureLostError(f"git diff exec failed: {exc}") from exc
            error_type = getattr(result, "error_type", None)
            if error_type is not None:
                raise PatchCaptureLostError(f"git diff exec reported no exit code ({error_type})")
            if result.return_code != 0:
                if exec_attempts > 1:
                    # With background exec a connect failure can come from a status poll of a command that did run,
                    # so the retry may have collided with it; that failure says nothing about the agent.
                    raise PatchCaptureLostError(
                        f"git diff failed on a retry after a transport failure: {result.stderr or result.return_code}"
                    )
                raise RuntimeError(result.stderr or "git diff failed")
            # Command logs can drop line endings; download the patch bytes instead. A failed cleanup of the local copy
            # must not discard bytes already read.
            try:
                with TemporaryDirectory(prefix="nemo-gym-swebench-pro-", ignore_cleanup_errors=True) as directory:
                    local_patch = Path(directory) / "model.diff"
                    # Reading the patch file again is idempotent, so any transport failure is retried.
                    await _retry_patch_capture_step(
                        "patch download", lambda: original_sandbox.download(patch_path, local_patch), idempotent=True
                    )
                    patch = local_patch.read_bytes().decode("utf-8", errors="replace")
            except Exception as exc:
                raise PatchCaptureLostError(f"patch download failed: {exc}") from exc
            return drop_patch_sections(patch, pristine_untracked)
        finally:
            try:
                cleanup = await original_sandbox.exec(f"rm -f -- {quote(patch_path)}")
                if cleanup.return_code != 0:
                    LOG.warning("Failed to remove agent patch file: %s", cleanup.stderr)
            except Exception:
                LOG.exception("Failed to remove agent patch file")
            finally:
                try:
                    await self._stop_session_sandbox(session_id)
                except Exception:
                    LOG.exception("Failed to stop agent sandbox %s", session_id)

    async def verify(
        self,
        request: Request,
        body: SWEBenchProVerifyRequest,
    ) -> SWEBenchProVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        # A session with no task sandbox (an earlier verify consumed it, a close released it, or the server never held
        # it) has nothing left to grade: grading would score an empty patch as a policy that changed nothing, so the
        # sample is masked. So is an extraction whose git diff exec reported no exit code. Any other failed extraction
        # from a live sandbox stays an unmasked failure, since the agent can cause it.
        if not self.config.is_verifying_golden_patch and session_id not in self._session_id_to_sandbox:
            return _lost_session_response(body, "SWE-bench Pro task sandbox is no longer available")
        extraction_error = None
        if self.config.is_verifying_golden_patch:
            model_patch = body.patch
        else:
            try:
                model_patch = await self._extract_model_patch(session_id, body.base_commit)
            except PatchCaptureLostError as exc:
                # Grading "" here would score a patch nobody captured as a policy that changed nothing.
                return _lost_session_response(body, f"Failed to extract model patch: {exc}")
            except Exception as exc:
                model_patch = ""
                extraction_error = f"Failed to extract model patch: {exc}"

        inputs = self._verification_inputs(body, model_patch)
        workspace_files, _ = assemble_workspace_files(body.instance_id, None, model_patch, asdict(inputs))
        sandbox_files = {
            f"/workspace/{relative_path}": contents for relative_path, contents in workspace_files.items()
        }

        run_log_dir = Path(__file__).parent / "logs" / "run_evaluation" / session_id / body.instance_id
        eval_sandbox_start_time_taken = 0.0
        patch_verification_time_taken = 0.0
        attempts = 1 + max(self.config.inconclusive_verification_retries, 0)
        deadline = _verification_deadline(self.config.verification_total_timeout)
        verifier_sandbox_observation = None
        verifier_sandbox_observations = []
        for attempt in range(1, attempts + 1):
            attempt_error: Exception | None = None
            eval_sandbox: AsyncSandbox | None = None
            start_time = time()
            try:
                async with asyncio.timeout(_attempt_budget(self.config.verification_attempt_timeout, deadline)):
                    eval_sandbox = await self._create_sandbox(body, files=sandbox_files)
                    eval_sandbox_start_time_taken = time() - start_time
                    verification_start = time()
                    result = await run_verification(
                        sandbox=eval_sandbox,
                        inputs=inputs,
                        log_dir=run_log_dir,
                        timeout_s=self.config.evaluation_timeout,
                    )
                    patch_verification_time_taken = time() - verification_start
            except Exception as exc:
                attempt_error = exc
                eval_sandbox_start_time_taken = time() - start_time
                patch_verification_time_taken = 0.0
                result = VerificationResult(
                    completed=False,
                    resolved=False,
                    patch_applied=False,
                    test_results=None,
                    error=f"Verification failed: {exc}",
                )
            finally:
                if eval_sandbox is not None:
                    try:
                        async with asyncio.timeout(self.config.verification_stop_timeout):
                            await eval_sandbox.stop()
                    except Exception:
                        print("Failed to stop verification sandbox", format_exc(), file=sys.stderr)
            # Observation is best effort and cannot replace the calculated grade or original error.
            try:
                verifier_sandbox_observation = lifecycle_observation(eval_sandbox, role="verifier")
            except Exception:
                print("Failed to observe verification sandbox", format_exc(), file=sys.stderr)
                verifier_sandbox_observation = None
            verifier_sandbox_observations.append(verifier_sandbox_observation)

            reason = inconclusive_reason(result)
            if reason is not None:
                # Record the actual failed attempt without selecting a retry or claiming a model failure.
                # Only an exception before sandbox assignment identifies allocation as the failing operation.
                allocation_failed = attempt_error is not None and eval_sandbox is None
                operation_id = current_verifier_invocation_id()
                observed = NgFailureEvent(
                    boundary=FailureBoundary.RESOURCES_SERVER,
                    failure_kind=VERIFIER_ERROR,
                    failure_reason=(
                        "Verifier sandbox allocation failed"
                        if allocation_failed
                        else "Verification produced no conclusive verdict"
                    ),
                    operation="verify.sandbox.allocate" if allocation_failed else "verify",
                    operation_id=operation_id,
                    attempt_id=f"{operation_id}:{attempt}" if operation_id is not None else None,
                    stage="verification",
                    exception_type=type(attempt_error).__name__ if attempt_error is not None else None,
                )
                observed.__cause__ = attempt_error
                observe_failure(observed)
            if reason is not None and _budget_spent(deadline):
                print(
                    f"Verification for {body.instance_id} gave up after {attempt} attempt(s): "
                    f"the {self.config.verification_total_timeout}s budget for this rollout is spent "
                    f"({reason})",
                    file=sys.stderr,
                )
                break
            if reason is None or attempt == attempts:
                if reason is not None:
                    print(
                        f"Verification for {body.instance_id} still inconclusive after {attempt} attempt(s): {reason}",
                        file=sys.stderr,
                    )
                break
            print(
                f"Retrying verification for {body.instance_id} on a new sandbox "
                f"(attempt {attempt}/{attempts} was inconclusive: {reason})",
                file=sys.stderr,
            )

        # The request allows extra fields, so response-owned fields a caller sent are overwritten, not passed through.
        response_data = body.model_dump() | {
            "image_provenance": await asyncio.to_thread(self._image_info, body),
            "reward": float(result.resolved),
            "mask_sample": False,
            "failure_kind": None,
            "failure_reason": None,
            "verification_attempts": attempt,
            "evaluation_completed": result.completed and reason is None,
            "eval_timed_out": result.timed_out,
            "resolved": result.resolved,
            "patch_applied": result.patch_applied,
            "eval_sandbox_start_time_taken": eval_sandbox_start_time_taken,
            "patch_verification_time_taken": patch_verification_time_taken,
            "instance_id": body.instance_id,
            "model_patch": model_patch or None,
            "test_results": result.test_results,
            "test_output": result.test_output,
            "error": extraction_error or result.error or reason,
            "log_dir": str(run_log_dir),
            # Older observation schemas forbid unknown fields; the plural field retains the full facts.
            "verifier_sandbox_observation": {
                key: value
                for key, value in verifier_sandbox_observation.items()
                if key not in {"observed_client_interval_s", "teardown_confirmed", "cleanup_errors"}
            }
            if verifier_sandbox_observation is not None
            else None,
            "verifier_sandbox_observations": verifier_sandbox_observations,
        }
        if reason is not None:
            # The saved patch has no conclusive verdict. A collection retry would run the agent again;
            # this stateful verifier cannot yet reverify that patch through the public CLI.
            response_data |= {
                "mask_sample": True,
                "failure_kind": VERIFIER_ERROR,
                "failure_reason": result.error or reason,
                "model_patch": model_patch,
                "verification_attempts": attempt,
            }
        # The adapter supplies typed verifier facts; collector-owned routing markers cannot cross a native episode.
        response_data.pop("_ng_failure_class", None)
        response_data.pop("_ng_failure_judge_error", None)
        return SWEBenchProVerifyResponse.model_validate(response_data)


def _lost_session_response(body: SWEBenchProVerifyRequest, reason: str) -> SWEBenchProVerifyResponse:
    """A masked, unscored verify response for a rollout whose task sandbox state could not be read."""
    return SWEBenchProVerifyResponse.model_validate(
        body.model_dump()
        | lost_session_verification(reason)
        | {
            "evaluation_completed": False,
            "eval_timed_out": False,
            "image_provenance": {},
            "resolved": False,
            "patch_applied": False,
            "eval_sandbox_start_time_taken": 0.0,
            "patch_verification_time_taken": 0.0,
            "instance_id": body.instance_id,
            "model_patch": None,
            "test_results": None,
            "test_output": "",
            "error": reason,
            "log_dir": "",
        }
    )


if __name__ == "__main__":
    SWEBenchProResourcesServer.run_webserver()
