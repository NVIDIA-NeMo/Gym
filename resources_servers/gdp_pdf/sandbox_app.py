# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lend a PDF workspace to a session-based harness; reuse GDP.pdf's rubric grader."""

import asyncio
import logging
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import Field

from nemo_gym import PARENT_DIR
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import EpisodeId
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxConfig,
    resolve_provider_config,
)
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.gdp_pdf.app import (
    CriterionEvaluation,
    GdpPdfConfig,
    GdpPdfResourcesServer,
    GdpPdfVerifyRequest,
    GdpPdfVerifyResponse,
)


LOG = logging.getLogger(__name__)
WORKDIR = "/workspace"
DOCUMENT_PATH = f"{WORKDIR}/document.pdf"


class GdpPdfSandboxConfig(GdpPdfConfig):
    sandbox_provider: str
    sandbox_image: str
    sandbox_config: SandboxConfig = Field(default_factory=lambda: SandboxConfig(ttl_s=14400))
    documents_base_dir: str = "resources_servers/gdp_pdf/data"
    sandbox_stop_timeout_seconds: float = Field(default=60, gt=0)
    allow_unisolated_sandbox: bool = False


@dataclass
class _Session:
    seed: ResourcesSeedSessionRequest
    sandbox: AsyncSandbox
    access: SandboxAccess | None = None


class GdpPdfSandboxResourcesServer(GdpPdfResourcesServer):
    config: GdpPdfSandboxConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("GDP.pdf sandbox sessions require num_workers=1")
        if self.config.sandbox_config.ttl_s is None:
            raise ValueError("GDP.pdf sandboxes require ttl_s for abandoned episodes")
        self._sessions: dict[str, _Session] = {}
        self._locks = ResourcesSessionLocks()
        self._fences: ResourcesSessionFences[EpisodeId] = ResourcesSessionFences(
            self.config.session_close_retry_window_seconds
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent(app) as state:
                    yield state
            finally:
                for session_id in list(self._sessions):
                    try:
                        async with self._locks.locked(session_id):
                            await self._stop(session_id)
                    except Exception:
                        LOG.exception("Could not stop GDP.pdf sandbox %s", session_id)

        app.router.lifespan_context = lifespan
        return app

    def _source_pdf(self, body: ResourcesSeedSessionRequest) -> Path:
        metadata = body.task_data.get("verifier_metadata")
        source = metadata.get("source_pdf") if isinstance(metadata, dict) else None
        if not isinstance(source, str) or not source:
            raise ResourcesSeedRejectedError("verifier_metadata.source_pdf must name a PDF")
        root = (PARENT_DIR / self.config.documents_base_dir).resolve()
        path = (root / source).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() != ".pdf":
            raise ResourcesSeedRejectedError("source_pdf must be an existing PDF within documents_base_dir")
        return path

    async def verify(self, body: GdpPdfVerifyRequest) -> GdpPdfVerifyResponse:
        criteria = (body.verifier_metadata or {}).get("criteria") or []
        complete = body.response.status in (None, "completed") and not body.response.incomplete_details
        if not complete and criteria:
            evaluations = [
                CriterionEvaluation(index=c.get("index"), criterion=str(c.get("criterion", "")), passed=False)
                for c in criteria
            ]
            result = self._build_response(body, evaluations)
        else:
            result = await super().verify(body)
        return result.model_copy(update={"allow_unisolated_sandbox": self.config.allow_unisolated_sandbox})

    async def _check_isolation(self, provider: dict) -> None:
        if self.config.allow_unisolated_sandbox:
            LOG.warning(
                "GDP.pdf sandbox isolation explicitly disabled; this opt-out is recorded in verification results"
            )
            return
        create = provider.get("docker", {}).get("create", {})
        network = create.get("network")
        options = self.config.sandbox_config.provider_options
        if not network or create.get("extra_run_args") or options.get("volumes") or options.get("run_args"):
            raise ResourcesSeedRejectedError(
                "GDP.pdf requires a Docker internal network and no extra mounts/run flags"
            )
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "network", "inspect", network, "--format", "{{.Internal}}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ResourcesSeedRejectedError("Could not verify the Docker internal network") from exc
        if result.stdout.strip() != "true":
            raise ResourcesSeedRejectedError("Create the GDP.pdf network with docker network create --internal")

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            self._fences.raise_if_fenced(session_id)
            session = self._sessions.get(session_id)
            if session is not None and session.seed != body:
                raise ResourcesSessionConflictError("Session is already bound to another episode or task")
            if session is None or session.access is None:
                # A failed start whose cleanup failed remains owned until a retry can release it.
                await self._stop(session_id)
                pdf = self._source_pdf(body)
                global_config = get_global_config_dict()
                provider = resolve_provider_config(self.config.sandbox_provider, global_config)
                await self._check_isolation(provider)
                sandbox = AsyncSandbox(provider)
                session = _Session(body.model_copy(deep=True), sandbox)
                self._sessions[session_id] = session
                spec = self.config.sandbox_config.spec(
                    self.config.sandbox_provider,
                    image=self.config.sandbox_image,
                    workdir=WORKDIR,
                    named_configs=global_config,
                    metadata={"nemo_gym_task": body.task_id.task_id[:63]},
                )

                async def setup(sandbox: AsyncSandbox) -> None:
                    # Only the source document enters the sandbox: no rubric, reference answer or host mounts.
                    await sandbox.upload(pdf, DOCUMENT_PATH)

                try:
                    await sandbox.start_with_setup(spec, setup)
                    session.access = SandboxAccess(
                        connection=DirectSandboxConnection(
                            provider_config_ref=self.config.sandbox_provider, descriptor=await sandbox.serialize()
                        ),
                        workdir=WORKDIR,
                    )
                except BaseException:
                    try:
                        await self._stop(session_id)
                    except Exception:
                        LOG.exception("Could not stop partially initialized GDP.pdf sandbox %s", session_id)
                    raise
            request.session[SESSION_ID_KEY] = session_id
            return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=session.access)

    async def _stop(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            async with asyncio.timeout(self.config.sandbox_stop_timeout_seconds):
                await session.sandbox.stop()
            self._sessions.pop(session_id)

    async def retire_session_state(self, session_id: str) -> None:
        async with self._locks.locked(session_id):
            session = self._sessions.get(session_id)
            if session is not None:
                self._fences.fence(session_id, session.seed.episode_id)
            await self._stop(session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            session = self._sessions.get(session_id)
            owner = session.seed.episode_id if session else self._fences.get(session_id)
            if owner is not None and owner != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the resources session")
            await self._stop(session_id)
            self._fences.fence(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)


if __name__ == "__main__":
    GdpPdfSandboxResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GdpPdfSandboxResourcesServer.run_webserver()
