# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository lifecycle and verified Prime parent-delivery evidence for the shared IF grader."""

import json
import re
import shlex
from contextlib import asynccontextmanager
from typing import ClassVar

from fastapi import FastAPI, Request
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import Field, PrivateAttr, field_validator, model_validator

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import EpisodeId
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxAccess,
    SandboxConfig,
    read_text,
    upload_text,
)
from nemo_gym.server_runtime import (
    LentSandboxes,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    ReverifyMode,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.partner_if.app import (
    PartnerIFResourcesServer,
    PartnerIFResourcesServerConfig,
    VerifyRequest,
    VerifyResponse,
)
from resources_servers.partner_if.evidence import Artifact, EvidenceBundle, EvidenceUnavailable


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


class PrimeIFResourcesServer(PartnerIFResourcesServer):
    """Own the task sandbox, then grade only delivered report contents through the shared registry and judge."""

    config: PrimeIFConfig
    checkpoint_mode = "restart_only"
    checkpoint_verify = "wait"
    _sandboxes: LentSandboxes = PrivateAttr()
    _seeds: dict[str, ResourcesSeedSessionRequest] = PrivateAttr(default_factory=dict)
    # Repository readiness is task state; LentSandboxes owns every sandbox handle.
    _prepared: set[str] = PrivateAttr(default_factory=set)
    _locks: ResourcesSessionLocks = PrivateAttr()
    _fences: ResourcesSessionFences[EpisodeId] = PrivateAttr()
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
        self._locks = ResourcesSessionLocks()
        self._fences = ResourcesSessionFences(self.config.session_close_retry_window_seconds)

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
                # Each seed records its owner before awaiting, including a setup whose cleanup is pending.
                for identifier in list(self._seeds):
                    async with self._locks.locked(identifier):
                        pass
                await self._sandboxes.release_all()

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

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        identifier = body.resources_session_id
        async with self._locks.locked(identifier):
            if self._shutting_down:
                raise ResourcesSessionConflictError("Prime Resources is shutting down")
            self._fences.raise_if_fenced(identifier)
            previous = self._seeds.get(identifier)
            if previous is not None and previous != body:
                raise ResourcesSessionConflictError("Resource session belongs to a different task")
            # Retain the owner during setup and failed cleanup, so close and shutdown can find it.
            self._seeds.setdefault(identifier, body.model_copy(deep=True))
            if identifier in self._prepared:
                access = await self._sandboxes.lend(identifier, body.task_id)
            else:
                # A prior checkout whose stop failed must not be lent as a prepared task.
                await self._sandboxes.release(identifier)
                try:
                    access = await self._sandboxes.lend(identifier, body.task_id)
                    sandbox = self._sandboxes.get(identifier)
                    if sandbox is None:
                        raise RuntimeError("Lent Prime sandbox is unavailable during repository setup")
                    await self._prepare_repository(sandbox)
                except BaseException:
                    await self._sandboxes.release(identifier)
                    self._seeds.pop(identifier, None)
                    raise
                self._prepared.add(identifier)
            request.session[SESSION_ID_KEY] = identifier
            return ResourcesSeedSessionResponse(
                resources_session_id=identifier, sandbox_access=self._task_access(access)
            )

    async def _release(self, identifier: str) -> None:
        await self._sandboxes.release(identifier)
        self._prepared.discard(identifier)
        self._seeds.pop(identifier, None)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        identifier = body.resources_session_id
        async with self._locks.locked(identifier):
            seed = self._seeds.get(identifier)
            episode = seed.episode_id if seed is not None else self._fences.get(identifier)
            if episode is not None and episode != body.episode_id:
                raise ResourcesSessionConflictError("Resource session belongs to another episode")
            await self._release(identifier)
            self._fences.fence(identifier, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=identifier)

    async def retire_session_state(self, session_id: str) -> None:
        """A retired/checkpointed attempt must not leave a task sandbox running."""
        async with self._locks.locked(session_id):
            seed = self._seeds.get(session_id)
            await self._release(session_id)
            if seed is not None:
                self._fences.fence(session_id, seed.episode_id)

    async def verify(self, request: Request, body: VerifyRequest) -> VerifyResponse:
        identifier = request.session[SESSION_ID_KEY]
        async with self._locks.locked(identifier):
            try:
                sandbox = self._sandboxes.get(identifier)
                seed = self._seeds[identifier]
                if sandbox is None or identifier not in self._prepared:
                    raise EvidenceUnavailable("Prime task sandbox is not prepared")
                from resources_servers.partner_if.schema import VerifierMetadata

                if VerifierMetadata.model_validate(seed.task_data.get("verifier_metadata")) != body.verifier_metadata:
                    raise EvidenceUnavailable("Verification contract differs from the seeded task")
                capture = json.loads(await read_text(sandbox, path=f"{self.config.capture_directory}/capture.json"))
                error = None
            except (KeyError, OSError, ValueError, RuntimeError) as failure:
                capture, error = None, str(failure)

            def evidence() -> EvidenceBundle:
                if error:
                    raise EvidenceUnavailable(error)
                return delivered_report_evidence(
                    capture, body.responses_create_params.model_dump(mode="json", exclude_none=True)["input"]
                )

            return VerifyResponse.model_validate(
                body.model_dump() | await self.grade_response(body.verifier_metadata.partner_if, evidence)
            )


if __name__ == "__main__":
    PrimeIFResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PrimeIFResourcesServer.run_webserver()
