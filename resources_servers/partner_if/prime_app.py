# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository lifecycle and verified Prime parent-delivery evidence for the shared IF grader."""

import json
import re
import shlex
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import ClassVar

from fastapi import FastAPI, Request
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import ConfigDict, Field, PrivateAttr, ValidationError, field_validator, model_validator

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxAccess,
    SandboxConfig,
    read_text,
    upload_text,
)
from nemo_gym.server_runtime import (
    BaseSeedSessionRequest,
    LentSandboxes,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ReverifyMode,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.partner_if.app import (
    PartnerIFResourcesServer,
    PartnerIFResourcesServerConfig,
    VerifyRequest,
    VerifyResponse,
)
from resources_servers.partner_if.evidence import Artifact, EvidenceBundle, EvidenceUnavailable
from resources_servers.partner_if.schema import VerifierMetadata


TASK_PYTHON_DIRECTORY = "/tmp/partner-if-prime-python"
TASK_REQUIREMENTS_PATH = "/tmp/partner-if-prime-requirements.txt"
TASK_INSTALLER_DIRECTORY = "/tmp/partner-if-prime-package-installer"


class PrimeIFConfig(PartnerIFResourcesServerConfig):
    """Prepare a pinned public repository; never put task prompts in adapter configuration."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    num_workers: int = 1
    image: str = "node:22-bookworm"
    sandbox_provider: str
    sandbox_config: SandboxConfig
    repository_url: str
    repository_commit: str
    workdir: str = "/workspace/repo"
    capture_directory: str = "/tmp/partner-if-prime-capture"
    setup_timeout_seconds: float = 1800
    # Task-specific pure-Python wheels, separate from PrimeSession's stock runtime and kernel.
    task_python_packages: list[str] = Field(default_factory=list)
    task_python_package_hashes: dict[str, str] = Field(default_factory=dict)

    @field_validator("task_python_packages")
    @classmethod
    def exact_task_packages(cls, value: list[str]) -> list[str]:
        names = set()
        for text in value:
            try:
                requirement = Requirement(text)
                specs = list(requirement.specifier)
                if (
                    "\n" in text
                    or "\r" in text
                    or requirement.url
                    or requirement.marker
                    or requirement.extras
                    or len(specs) != 1
                    or specs[0].operator != "=="
                    or "*" in specs[0].version
                ):
                    raise ValueError("Task Python packages require an exact name==version pin")
                Version(specs[0].version)
            except (InvalidRequirement, InvalidVersion) as error:
                raise ValueError("Task Python packages require an exact name==version pin") from error
            name = canonicalize_name(requirement.name)
            if name in names:
                raise ValueError("Task Python packages cannot repeat a package name")
            names.add(name)
        return value

    @model_validator(mode="after")
    def task_package_hashes_match(self):
        if set(self.task_python_packages) != set(self.task_python_package_hashes):
            raise ValueError("Every task Python package requires exactly one SHA-256 wheel hash")
        if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in self.task_python_package_hashes.values()):
            raise ValueError("Task Python wheel hashes must be lowercase SHA-256 digests")
        return self

    @field_validator("repository_commit")
    @classmethod
    def pinned_commit(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("repository_commit must be a full Git commit SHA")
        return value

    @field_validator("workdir", "capture_directory")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("Sandbox paths must be absolute")
        return value


def _delivered_report_evidence(capture: dict, context: list[dict]) -> EvidenceBundle:
    """Corroborate successful IPython sends with the receiving controller's receipts.

    Drafts, text that merely resembles a send, failed tools and uncorroborated receipts never count.
    Capture and provider failures remain unscorable. A completed turn budget without a send is a valid failure.
    """
    if capture.get("version") != 1:
        raise EvidenceUnavailable("Unsupported or missing Prime capture version")
    events, deliveries = capture.get("events"), capture.get("deliveries")
    if not isinstance(events, list) or not isinstance(deliveries, list):
        raise EvidenceUnavailable("Prime capture requires events and receiving-controller receipts")
    delivered = {
        item["id"]: item
        for item in deliveries
        if item.get("source") == "agent_message"
        and item.get("deliveryStatus") == "delivered"
        and isinstance(item.get("id"), str)
    }
    started = set()
    reports = {}
    proof = []
    model_errors = []
    for event in events:
        if event.get("type") == "message_end" and event.get("message", {}).get("role") == "assistant":
            message = event["message"]
            if message.get("stopReason") == "error" or message.get("errorMessage"):
                model_errors.append(message.get("errorMessage", "Model request failed"))
        call = event.get("toolCallId")
        if event.get("toolName") != "ipython":
            continue
        if event.get("type") == "tool_execution_start" and isinstance(call, str):
            started.add(call)
        result = event.get("result") or {}
        if (
            event.get("type") != "tool_execution_end"
            or call not in started
            or event.get("isError")
            or result.get("isError")
        ):
            continue
        details = result.get("details") or {}
        if details.get("status") != "ok":
            continue
        for sent in details.get("sentAgentMessages") or []:
            receipt = delivered.get(sent.get("id"))
            if receipt and sent.get("receiverRole") == "parent" and sent.get("deliveryStatus") == "delivered":
                if not isinstance(receipt.get("message"), str):
                    raise EvidenceUnavailable("Delivered parent report has no text")
                reports[receipt["id"]] = receipt["message"]
                proof.append({"tool_call_id": call, "receipt_id": receipt["id"]})
    if not proof and (model_errors or capture.get("termination") not in {"completed", "turn_limit"}):
        raise EvidenceUnavailable("Prime run/capture did not complete; no negative model verdict")
    return EvidenceBundle(
        context=context,
        artifacts={
            "parent_send": Artifact(status="present", text="verified_parent_send\n" + json.dumps(proof))
            if proof
            else Artifact(status="absent"),
            "parent_reports": Artifact(status="present", text=json.dumps(reports))
            if proof
            else Artifact(status="absent"),
        },
    )


def delivered_report_evidence(capture: dict, context: list[dict]) -> EvidenceBundle:
    """Convert malformed captures to an explicit infrastructure gap, never a scored absence."""
    try:
        return _delivered_report_evidence(capture, context)
    except (AttributeError, KeyError, TypeError) as error:
        raise EvidenceUnavailable("Malformed Prime controller capture") from error


class PrimeIFSeedRequest(BaseSeedSessionRequest):
    """The task data bound to a prepared repository and its verification contract."""

    # Preserve the other row fields so a repeated seed cannot replace the task's inputs.
    model_config = ConfigDict(extra="allow")
    verifier_metadata: VerifierMetadata | None = None
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None


@dataclass
class _PrimeTaskState:
    task: PrimeIFSeedRequest
    access: SandboxAccess | None = None


@dataclass(frozen=True)
class _PrimeVerificationEvidence:
    verifier_metadata: VerifierMetadata | None
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    bundle: EvidenceBundle | None
    error: str | None


class PrimeIFResourcesServer(PartnerIFResourcesServer):
    """Own the task sandbox, then grade only delivered report contents through the shared registry and judge."""

    config: PrimeIFConfig
    checkpoint_mode = "restart_only"
    checkpoint_verify = "wait"
    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = PrimeIFSeedRequest
    _sandboxes: LentSandboxes = PrivateAttr()
    # Task data and prepared access only; the base owns identity, locks, fences and cleanup retries.
    _tasks: dict[str, _PrimeTaskState] = PrivateAttr(default_factory=dict)
    _shutting_down: bool = PrivateAttr(default=False)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers != 1:
            raise ValueError("Prime resource sessions require num_workers=1")
        self._sandboxes = LentSandboxes(
            self.config.sandbox_provider,
            self.config.sandbox_config,
            image=self.config.image,
            workdir=self.config.workdir,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifecycle(application: FastAPI):
            try:
                async with original(application) as state:
                    yield state
            finally:
                self._shutting_down = True
                # Wait for each base-managed initializer/verification before stopping its sandbox.
                for identifier in list(self._tasks):
                    async with self.session_lock(identifier):
                        pass
                await self._sandboxes.release_all()
                self._tasks.clear()

        app.router.lifespan_context = lifecycle
        return app

    async def _prepare_repository(self, sandbox: AsyncSandbox) -> None:
        commands = [
            "apt-get update && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates",
            f"git clone --no-checkout {shlex.quote(self.config.repository_url)} {shlex.quote(self.config.workdir)} && git -C {shlex.quote(self.config.workdir)} checkout --detach {shlex.quote(self.config.repository_commit)}",
        ]
        for command in commands:
            result = await sandbox.exec(command, cwd="/", timeout_s=self.config.setup_timeout_seconds)
            if result.return_code != 0 or result.error_type:
                raise RuntimeError(f"Repository setup failed: {(result.stderr or result.stdout or '')[-4000:]}")
        if self.config.task_python_packages:
            requirements = "".join(
                f"{package} --hash=sha256:{self.config.task_python_package_hashes[package]}\n"
                for package in self.config.task_python_packages
            )
            await upload_text(sandbox, path=TASK_REQUIREMENTS_PATH, text=requirements)
            command = (
                f"python3 -m venv {shlex.quote(TASK_INSTALLER_DIRECTORY)} && "
                f"{shlex.quote(TASK_INSTALLER_DIRECTORY + '/bin/python')} -m pip install "
                "--disable-pip-version-check --no-input --require-hashes --no-deps --only-binary=:all: "
                "--platform any --implementation py --abi none --python-version 3.13 "
                f"--target {shlex.quote(TASK_PYTHON_DIRECTORY)} -r {shlex.quote(TASK_REQUIREMENTS_PATH)}"
            )
            result = await sandbox.exec(command, cwd="/", timeout_s=self.config.setup_timeout_seconds)
            if result.return_code != 0 or result.error_type:
                raise RuntimeError(f"Task Python setup failed: {(result.stderr or result.stdout or '')[-4000:]}")

    def _task_access(self, access: SandboxAccess) -> SandboxAccess:
        if not self.config.task_python_packages:
            return access
        return access.model_copy(update={"env": {**access.env, "PYTHONPATH": TASK_PYTHON_DIRECTORY}})

    def has_session_state(self, session_id: str) -> bool:
        """Whether repository setup completed and the session can be verified."""
        state = self._tasks.get(session_id)
        return state is not None and state.access is not None

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Prepare task state under the base lifecycle's lock and failure cleanup."""
        if self._shutting_down:
            raise ResourcesSessionConflictError("Prime resources server is shutting down")
        identity = self.session_identity(session_id)
        if identity is None:
            raise ResourcesSeedRejectedError("Prime requires an environment server's typed episode seed")
        if not isinstance(body, PrimeIFSeedRequest):
            raise TypeError("Expected PrimeIFSeedRequest")
        state = _PrimeTaskState(body.model_copy(deep=True))
        self._tasks[session_id] = state
        access = await self._sandboxes.lend(session_id, identity[1])
        sandbox = self._sandboxes.get(session_id)
        if sandbox is None:
            raise RuntimeError("Lent Prime sandbox is unavailable during repository setup")
        await self._prepare_repository(sandbox)
        state.access = self._task_access(access)

    async def release_session_state(self, session_id: str) -> None:
        """Keep task and sandbox references until release succeeds, so the base can retry."""
        await self._sandboxes.release(session_id)
        self._tasks.pop(session_id, None)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        """Use the inherited seed lifecycle and add the prepared sandbox's access."""
        if self._shutting_down:
            raise ResourcesSessionConflictError("Prime resources server is shutting down")
        task_fields = body.task_data
        if body.responses_create_params is not None:
            task_fields = {**task_fields, "responses_create_params": body.responses_create_params}
        try:
            task = PrimeIFSeedRequest.model_validate(task_fields)
        except ValidationError as error:
            raise ResourcesSeedRejectedError(f"Invalid Prime task data: {error}") from error
        previous = self._tasks.get(body.resources_session_id)
        if previous is not None and previous.task != task:
            raise ResourcesSessionConflictError("Resources session belongs to a different task")
        response = await super().seed_session(body, request=request)
        state = self._tasks.get(body.resources_session_id)
        if state is None or state.access is None:
            raise ResourcesSessionConflictError("Prime task sandbox is not prepared")
        # Recheck after the base lock: concurrently queued seeds must agree with the winning initializer.
        if state.task != task:
            raise ResourcesSessionConflictError("Resources session belongs to a different task")
        response.sandbox_access = state.access
        return response

    async def verify(self, request: Request, body: VerifyRequest) -> VerifyResponse:
        identifier = request.session[SESSION_ID_KEY]
        async with self.verifying(identifier) as verification:
            retained = verification.retained
            if verification.is_live:
                state = self._tasks[identifier]
                params = state.task.responses_create_params or body.responses_create_params
                bundle, error = None, None
                try:
                    sandbox = self._sandboxes.get(identifier)
                    if sandbox is None:
                        raise EvidenceUnavailable("Prime task sandbox is not prepared")
                    capture = json.loads(
                        await read_text(sandbox, path=f"{self.config.capture_directory}/capture.json")
                    )
                    bundle = delivered_report_evidence(
                        capture, params.model_dump(mode="json", exclude_none=True)["input"]
                    )
                    for name in ("parent_send", "parent_reports"):
                        bundle.artifact_text(name)
                except (KeyError, OSError, ValueError, RuntimeError) as failure:
                    bundle, error = None, str(failure)
                retained = _PrimeVerificationEvidence(
                    state.task.verifier_metadata,
                    params.model_copy(deep=True),
                    bundle,
                    error,
                )
                # Retain bounded report evidence, not the SDK event stream or a live sandbox handle.
                verification.retain(retained)

            def evidence() -> EvidenceBundle:
                if verification.is_lost or not isinstance(retained, _PrimeVerificationEvidence):
                    raise EvidenceUnavailable(
                        "Prime task sandbox is not prepared or its verification evidence expired"
                    )
                if retained.verifier_metadata != body.verifier_metadata:
                    raise EvidenceUnavailable("Verification contract differs from the seeded task")
                if retained.responses_create_params != body.responses_create_params:
                    raise EvidenceUnavailable("Verification input differs from the retained task evidence")
                if retained.error:
                    raise EvidenceUnavailable(retained.error)
                if retained.bundle is None:
                    raise EvidenceUnavailable("Prime parent-delivery evidence is unavailable")
                return retained.bundle

            fields = await self.grade_response(body.verifier_metadata.partner_if, evidence)
            if verification.is_lost:
                fields.update(lost_session_verification("Prime session is closed or its evidence expired"))
            return VerifyResponse.model_validate(body.model_dump() | fields)


if __name__ == "__main__":
    PrimeIFResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PrimeIFResourcesServer.run_webserver()
