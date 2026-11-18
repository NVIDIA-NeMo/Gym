# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lend a PDF workspace to a session-based harness; reuse GDP.pdf's rubric grader."""

import asyncio
import logging
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import Field

from nemo_gym import PARENT_DIR
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import EpisodeId
from nemo_gym.sandbox import SandboxConfig, resolve_provider_config
from nemo_gym.server_runtime import (
    LentSandboxes,
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
    allow_unisolated_sandbox: bool = False


class GdpPdfSandboxResourcesServer(GdpPdfResourcesServer):
    config: GdpPdfSandboxConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("GDP.pdf sandbox sessions require num_workers=1")
        if self.config.sandbox_config.ttl_s is None:
            raise ValueError("GDP.pdf sandboxes require ttl_s for abandoned episodes")
        # The seed of each session that holds a lent sandbox or is starting one.
        self._seeds: dict[str, ResourcesSeedSessionRequest] = {}
        self._sandboxes = LentSandboxes(
            self.config.sandbox_provider, self.config.sandbox_config, image=self.config.sandbox_image, workdir=WORKDIR
        )
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
                # A seed, close or retirement still running finishes first, so no sandbox is stopped twice at once.
                for session_id in list(self._seeds):
                    async with self._locks.locked(session_id):
                        pass
                await self._sandboxes.release_all()

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
            seed = self._seeds.get(session_id)
            if seed is not None and seed != body:
                raise ResourcesSessionConflictError("Session is already bound to another episode or task")
            if seed is not None:
                access = await self._sandboxes.lend(session_id, body.task_id)
            else:
                # Recorded before the first await, so that shutdown waits for a seed that is still starting its sandbox.
                self._seeds[session_id] = body.model_copy(deep=True)
                try:
                    pdf = self._source_pdf(body)
                    await self._check_isolation(
                        resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
                    )
                    # Only the source document enters the sandbox: no rubric, reference answer or host mounts.
                    files = {DOCUMENT_PATH: await asyncio.to_thread(pdf.read_bytes)}
                    access = await self._sandboxes.lend(session_id, body.task_id, files=files)
                except BaseException:
                    del self._seeds[session_id]
                    raise
            request.session[SESSION_ID_KEY] = session_id
            return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=access)

    async def _release(self, session_id: str) -> None:
        await self._sandboxes.release(session_id)
        self._seeds.pop(session_id, None)

    async def retire_session_state(self, session_id: str) -> None:
        async with self._locks.locked(session_id):
            seed = self._seeds.get(session_id)
            if seed is not None:
                self._fences.fence(session_id, seed.episode_id)
            await self._release(session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            seed = self._seeds.get(session_id)
            owner = seed.episode_id if seed else self._fences.get(session_id)
            if owner is not None and owner != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the resources session")
            await self._release(session_id)
            self._fences.fence(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)


if __name__ == "__main__":
    GdpPdfSandboxResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GdpPdfSandboxResourcesServer.run_webserver()
