# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository lifecycle and verified Prime parent-delivery evidence for the shared IF grader."""

import asyncio
import json
import re
import shlex
from contextlib import asynccontextmanager
from typing import ClassVar

from fastapi import FastAPI, HTTPException, Request
from pydantic import PrivateAttr, field_validator

from nemo_gym.base_resources_server import (
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import SandboxConfig, resolve_provider_config
from nemo_gym.sandbox.utils import read_text
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.partner_if.app import (
    PartnerIFResourcesServer,
    PartnerIFResourcesServerConfig,
    VerifyRequest,
    VerifyResponse,
)
from resources_servers.partner_if.evidence import Artifact, EvidenceBundle, EvidenceUnavailable


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
    _sandboxes: dict[str, AsyncSandbox] = PrivateAttr(default_factory=dict)
    _seeds: dict[str, ResourcesSeedSessionRequest] = PrivateAttr(default_factory=dict)
    _access: dict[str, SandboxAccess] = PrivateAttr(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _closed: ClosedResourcesSessionRecords[EpisodeId | None] = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers != 1:
            raise ValueError("Prime resource sessions require num_workers=1")
        self._closed = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifecycle(application: FastAPI):
            async with original(application) as state:
                try:
                    yield state
                finally:
                    await asyncio.gather(*(sandbox.stop() for sandbox in list(self._sandboxes.values())))

        app.router.lifespan_context = lifecycle
        return app

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        identifier = body.resources_session_id
        async with self._locks.setdefault(identifier, asyncio.Lock()):
            if identifier in self._closed:
                raise HTTPException(409, "Resource session is closed")
            if identifier in self._seeds and self._seeds[identifier] != body:
                raise HTTPException(409, "Resource session belongs to a different task")
            if identifier not in self._seeds:
                # A failed setup whose stop failed still belongs to this session.
                # Dispose of it before replacing the handle on a seed retry.
                if previous := self._sandboxes.get(identifier):
                    await previous.stop()
                    self._sandboxes.pop(identifier)
                sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
                self._sandboxes[identifier] = sandbox
                spec = self.config.sandbox_config.spec(
                    self.config.sandbox_provider,
                    image=self.config.image,
                    workdir=None,
                    named_configs=get_global_config_dict(),
                )

                async def setup(connection: AsyncSandbox) -> None:
                    commands = [
                        "apt-get update && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates",
                        f"git clone --no-checkout {shlex.quote(self.config.repository_url)} {shlex.quote(self.config.workdir)} && git -C {shlex.quote(self.config.workdir)} checkout --detach {shlex.quote(self.config.repository_commit)}",
                    ]
                    for command in commands:
                        result = await connection.exec(command, timeout_s=self.config.setup_timeout_seconds)
                        if result.return_code != 0 or result.error_type:
                            raise RuntimeError(
                                f"Repository setup failed: {(result.stderr or result.stdout or '')[-4000:]}"
                            )

                try:
                    await sandbox.start_with_setup(spec, setup)
                    access = SandboxAccess(
                        connection=DirectSandboxConnection(
                            provider_config_ref=self.config.sandbox_provider, descriptor=await sandbox.serialize()
                        ),
                        workdir=self.config.workdir,
                    )
                except BaseException:
                    await sandbox.stop()
                    self._sandboxes.pop(identifier, None)
                    raise
                self._seeds[identifier] = body
                self._access[identifier] = access
            request.session[SESSION_ID_KEY] = identifier
            return ResourcesSeedSessionResponse(
                resources_session_id=identifier, sandbox_access=self._access[identifier]
            )

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        identifier = body.resources_session_id
        async with self._locks.setdefault(identifier, asyncio.Lock()):
            seed = self._seeds.get(identifier)
            episode = seed.episode_id if seed else self._closed.get(identifier)
            if episode is not None and episode != body.episode_id:
                raise HTTPException(409, "Resource session belongs to another episode")
            if sandbox := self._sandboxes.get(identifier):
                await sandbox.stop()
            self._sandboxes.pop(identifier, None)
            self._seeds.pop(identifier, None)
            self._access.pop(identifier, None)
            self._closed.remember(identifier, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=identifier)

    async def retire_session_state(self, session_id: str) -> None:
        """A retired/checkpointed attempt must not leave a task sandbox running."""
        seed = self._seeds.get(session_id)
        if seed is not None:
            await self.close_resources_session(
                ResourcesCloseSessionRequest(resources_session_id=session_id, episode_id=seed.episode_id)
            )

    async def verify(self, request: Request, body: VerifyRequest) -> VerifyResponse:
        identifier = request.session[SESSION_ID_KEY]
        async with self._locks.setdefault(identifier, asyncio.Lock()):
            try:
                sandbox = self._sandboxes[identifier]
                seed = self._seeds[identifier]
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
