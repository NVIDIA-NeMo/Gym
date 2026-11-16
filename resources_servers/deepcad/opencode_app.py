# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lend a CAD workspace to stock harnesses and grade only its final model.py."""

import base64
import io
import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from fastapi import FastAPI, Request
from PIL import Image
from pydantic import BeforeValidator, Field, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.sandbox import SandboxAccess, SandboxConfig
from nemo_gym.server_runtime import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    LentSandboxes,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.deepcad.app import (
    DeepCADResourcesServer,
    DeepCADResourcesServerConfig,
    DeepCADVerifyRequest,
    DeepCADVerifyResponse,
)
from resources_servers.deepcad.task_data import TaskData


LOG = logging.getLogger(__name__)
WORKDIR = "/workspace"


class DeepCADOpenCodeConfig(DeepCADResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    workspace_sandbox_provider: str = "agent_sandbox"
    workspace_sandbox_config: SandboxConfig = Field(default_factory=lambda: SandboxConfig(ttl_s=2400))
    workspace_image: str = "deepcad-opencode:1.17.11"
    workspace_network_access: Literal["inherit", "model_only"] = "inherit"
    workspace_model_server: ModelServerRef | None = None
    max_input_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1)


class WorkspaceVerifyRequest(DeepCADVerifyRequest):
    """Server-collected source; never accepted from an external verify caller."""

    final_code: str


class WorkspaceSeedRequest(BaseSeedSessionRequest):
    verifier_metadata: TaskData
    target_images: list[str] = Field(min_length=1, max_length=3)


class DeepCADOpenCodeVerifyResponse(DeepCADVerifyResponse):
    final_file_status: str


class DeepCADOpenCodeResourcesServer(DeepCADResourcesServer):
    config: DeepCADOpenCodeConfig
    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = WorkspaceSeedRequest
    _workspaces: LentSandboxes = PrivateAttr()
    _accesses: dict[str, SandboxAccess] = PrivateAttr(default_factory=dict)

    def multiple_workers_unsupported_reason(self) -> str:
        return "lent CAD workspaces live in one worker's memory"

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.workspace_sandbox_config.ttl_s is None:
            raise ValueError("CAD workspaces require ttl_s for abandoned episodes")
        self._workspaces = LentSandboxes(
            self.config.workspace_sandbox_provider,
            self.config.workspace_sandbox_config,
            image=self.config.workspace_image,
            workdir=WORKDIR,
            network_access=self.config.workspace_network_access,
            model_server=self.config.workspace_model_server.name if self.config.workspace_model_server else None,
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
                await self._workspaces.release_all()

        app.router.lifespan_context = lifespan
        return app

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._accesses

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        identity = self.session_identity(session_id)
        if identity is None or not isinstance(body, WorkspaceSeedRequest):
            raise ResourcesSeedRejectedError("CAD workspaces require an Environment Server's episode seed")
        try:
            metadata = body.verifier_metadata
            files: dict[str, str | bytes] = {}
            for url in body.target_images:
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("Materialize target images as inline data URLs before seeding")
                encoded = url.split(",", 1)[1]
                if len(encoded) > 4 * ((self.config.max_input_image_bytes + 2) // 3):
                    raise ValueError("Target image exceeds max_input_image_bytes")
                data = base64.b64decode(encoded, validate=True)
                if len(data) > self.config.max_input_image_bytes:
                    raise ValueError("Target image exceeds max_input_image_bytes")
                with Image.open(io.BytesIO(data)) as image:
                    output = io.BytesIO()
                    image.convert("RGB").save(output, format="PNG")
                files[f"{WORKDIR}/target-{len(files)}.png"] = output.getvalue()
            if not files:
                raise ValueError("CAD workspaces require target images")
            files[f"{WORKDIR}/.cad/settings.json"] = json.dumps(
                {"view_mode": metadata.view_mode, "color": metadata.foreground_color or "teal"}
            )
        except (KeyError, ValueError, OSError, Image.DecompressionBombError) as error:
            raise ResourcesSeedRejectedError(str(error)) from error
        self._accesses[session_id] = await self._workspaces.lend(session_id, identity[1], files=files)

    async def release_session_state(self, session_id: str) -> None:
        await self._workspaces.release(session_id)
        self._accesses.pop(session_id, None)

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        response = await super().seed_session(body, request=request)
        if isinstance(response, ResourcesSeedSessionResponse):
            response.sandbox_access = self._accesses.get(response.resources_session_id)
        return response

    async def _collect_code(self, session_id: str) -> tuple[str, str]:
        sandbox = self._workspaces.get(session_id)
        if sandbox is None:
            raise RuntimeError("CAD workspace was lost before final-file collection")
        check = await sandbox.exec(
            'python -c \'import os,stat; p="/workspace/model.py"; s=os.lstat(p) if os.path.lexists(p) else None; '
            f"print(bool(s and stat.S_ISREG(s.st_mode) and 0 < s.st_size <= {self.config.max_code_bytes}))'",
            timeout_s=30,
        )
        if check.error_type or check.return_code:
            raise RuntimeError(f"Final workspace inspection failed: {check.error_type or check.stderr}")
        if (check.stdout or "").strip() != "True":
            return "", "missing_or_invalid_model_file"
        with tempfile.TemporaryDirectory(prefix="deepcad-final-") as directory:
            path = Path(directory) / "model.py"
            await sandbox.download(f"{WORKDIR}/model.py", path)
            if path.stat().st_size > self.config.max_code_bytes:
                return "", "oversized_model_file"
            try:
                return path.read_text(encoding="utf-8"), "collected"
            except UnicodeError:
                return "", "invalid_code_encoding"

    def code_from_response(self, body: DeepCADVerifyRequest) -> str:
        return body.final_code if isinstance(body, WorkspaceVerifyRequest) else ""

    async def verify(self, request: Request, body: DeepCADVerifyRequest) -> DeepCADOpenCodeVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        weights = (
            self.config.mixed_iou.weights.model_dump()
            if self.config.verification_metric == "mixed_iou"
            else self.config.reward_weights.model_dump()
        )
        async with self.verifying(session_id) as verification:
            if verification.is_lost:
                return DeepCADOpenCodeVerifyResponse(
                    **(
                        body.model_dump() | lost_session_verification("CAD workspace or retained final source is lost")
                    ),
                    reward_components=dict.fromkeys(weights, 0.0),
                    evaluation_completed=False,
                    final_file_status="workspace_lost",
                )
            if verification.is_live:
                try:
                    frozen = (
                        await self._collect_code(session_id)
                        if body.response.status == "completed" and not body.response.incomplete_details
                        else ("", "agent_incomplete")
                    )
                except (OSError, RuntimeError) as error:
                    return DeepCADOpenCodeVerifyResponse(
                        **body.model_dump(),
                        reward=0.0,
                        mask_sample=True,
                        evaluation_completed=False,
                        failure_kind=VERIFIER_ERROR,
                        failure_reason=str(error),
                        reward_components=dict.fromkeys(weights, 0.0),
                        final_file_status="collection_failed",
                    )
                # Keep the source before any cleanup or scoring await. A verify retry never re-reads model.py.
                verification.retain(frozen)
            else:
                frozen = verification.retained
            if self.has_session_state(session_id):
                try:
                    await self.release_session_state(session_id)
                except Exception as error:
                    LOG.warning("Could not stop CAD workspace before grading; cleanup will retry", exc_info=True)
                    return DeepCADOpenCodeVerifyResponse(
                        **body.model_dump(),
                        reward=0.0,
                        mask_sample=True,
                        evaluation_completed=False,
                        failure_kind=VERIFIER_ERROR,
                        failure_reason=str(error),
                        reward_components=dict.fromkeys(weights, 0.0),
                        final_file_status="cleanup_failed",
                    )
            code, status = frozen
            result = await super().verify(WorkspaceVerifyRequest(**body.model_dump(), final_code=code))
            if status == "agent_incomplete":
                result.failure_kind = "deepcad:agent_incomplete"
            return DeepCADOpenCodeVerifyResponse(**result.model_dump(), final_file_status=status)


if __name__ == "__main__":
    DeepCADOpenCodeResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepCADOpenCodeResourcesServer.run_webserver()  # noqa: F401
