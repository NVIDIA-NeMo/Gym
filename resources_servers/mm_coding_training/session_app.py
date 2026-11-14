# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lend a task sandbox through public lifecycle hooks and grade only its final source."""

import hashlib
import json
import shlex
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import VERIFIER_ERROR, NgFailureEvent
from nemo_gym.sandbox import SandboxAccess, SandboxConfig
from nemo_gym.server_runtime import (
    LentSandboxes,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    observe_programmatic_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.mm_coding_training.app import VERIFIER_FIXTURE as VERIFIER_FIXTURE
from resources_servers.mm_coding_training.app import (
    CodingConfig,
    CodingRequest,
    CodingResponse,
    CodingSeedRequest,
    CodingTrainingServer,
    log_verification_failure,
)
from resources_servers.mm_coding_training.remote_renderer import LANGUAGES


def request_fingerprint(body: CodingRequest) -> str:
    """Bind retained source to the task, original policy reply and model request."""
    data = body.model_dump(mode="json", include={"task", "response", "responses_create_params"})
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_binding(body: CodingRequest, source_sha256: str) -> str:
    return hashlib.sha256(f"{request_fingerprint(body)}:{source_sha256}".encode()).hexdigest()


@dataclass
class RetainedAnswer:
    fingerprint: str
    source: str | None
    verdict: CodingResponse | None = None


class SessionCodingConfig(CodingConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    sandbox_provider: str
    sandbox_config: SandboxConfig
    sandbox_model_server: str = "policy_model"
    sandbox_runtime_files: dict[str, Path] = Field(default_factory=dict)


class SessionCodingTrainingServer(CodingTrainingServer):
    """Use the shared hooks, retained verification receipts and public sandbox lender."""

    config: SessionCodingConfig
    checkpoint_mode = "restart_only"
    checkpoint_verify = "wait"
    _lender: LentSandboxes = PrivateAttr()
    _access: dict[str, SandboxAccess] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        self._lender = LentSandboxes(
            self.config.sandbox_provider,
            self.config.sandbox_config,
            image=self.config.sandbox_image,
            workdir="/workspace",
            network_access="model_only",
            model_server=self.config.sandbox_model_server,
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
                await self._lender.release_all()

        app.router.lifespan_context = lifespan
        return app

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._tasks or self._lender.get(session_id) is not None

    async def initialize_session_state(self, session_id: str, body: CodingSeedRequest) -> None:
        identity = self.session_identity(session_id)
        if identity is None:
            raise ResourcesSeedRejectedError("A task sandbox requires an Environment Server episode seed")
        if identity[1].task_id != f"{body.task.benchmark}:{body.task.task_id}":
            raise ResourcesSeedRejectedError("TaskId does not match the coding task")
        try:
            reference = self._reference(body.task)
        except (OSError, ValueError) as error:
            raise ResourcesSeedRejectedError(
                "Coding reference is unavailable or differs from its declared SHA256"
            ) from error
        files = {"/opt/mmrl/render.py": Path(__file__).with_name("renderer.py").read_text()}
        # The owner stages offline dependencies before the restricted task starts.
        # The stock harness verifies its private copies against the configured SHA256 pins.
        for destination, source in self.config.sandbox_runtime_files.items():
            path = PurePosixPath(destination)
            if str(path) != destination or not path.is_relative_to("/opt/mm-coding") or ".." in path.parts:
                raise ResourcesSeedRejectedError("Runtime files must stay inside /opt/mm-coding")
            files[destination] = source.read_bytes()
        if reference is not None:
            files["/workspace/reference" + Path(body.task.reference_image).suffix.lower()] = reference
        self._access[session_id] = await self._lender.lend(session_id, identity[1], files=files)
        await super().initialize_session_state(session_id, body)

    async def release_session_state(self, session_id: str) -> None:
        await self._lender.release(session_id)
        self._access.pop(session_id, None)
        await super().release_session_state(session_id)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        response = await super().seed_session(body, request=request)
        return response.model_copy(update={"sandbox_access": self._access[body.resources_session_id]})

    async def _capture(self, session_id: str, body: CodingRequest) -> str | None:
        sandbox = self._lender.get(session_id)
        if sandbox is None:
            raise RuntimeError("Task sandbox is no longer available")
        path = shlex.quote("/workspace/" + LANGUAGES[body.task.benchmark][1])
        probe = await sandbox.exec(
            f"if [ ! -e {path} ]; then exit 44; fi; "
            f"test -f {path} && test ! -L {path} && test $(wc -c < {path}) -le {self.config.max_source_bytes}",
            timeout_s=30,
        )
        if probe.error_type is not None:
            raise RuntimeError("Final artifact inspection was inconclusive")
        if probe.return_code in {1, 44}:
            return None
        if probe.return_code != 0:
            raise RuntimeError("Final artifact inspection failed")
        with tempfile.TemporaryDirectory(prefix="mm-coding-artifact-") as directory:
            output = Path(directory) / "answer"
            await sandbox.download("/workspace/" + LANGUAGES[body.task.benchmark][1], output)
            raw = output.read_bytes()
        if len(raw) > self.config.max_source_bytes or not raw.strip():
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _replayed_answer(self, body: CodingRequest) -> RetainedAnswer:
        source = body.source_artifact
        if source is None or len(source.encode()) > self.config.max_source_bytes:
            raise HTTPException(409, "Reverification needs a bounded retained source artifact")
        sha = hashlib.sha256(source.encode()).hexdigest()
        if sha != body.source_sha256 or source_binding(body, sha) != body.source_binding_sha256:
            raise HTTPException(409, "Retained source does not match the task and policy-response binding")
        # An empty, SHA-bound source is the receipt for a measured absent/invalid final file.
        # None means no receipt was saved, so a lost session still cannot be graded.
        return RetainedAnswer(request_fingerprint(body), source or None)

    async def verify(self, request: Request, body: CodingRequest) -> CodingResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.verifying(session_id, allow_missing=body.source_artifact is not None) as verification:
            if verification.is_lost:
                return self.lost_result(body)
            if verification.is_live:
                seeded = self._tasks.get(session_id)
                if seeded is None or seeded.task != body.task:
                    raise HTTPException(409, "Verification does not match the seeded task")
                if not self.response_is_measurable(body):
                    result = self.new_result(body)
                    result.mask_sample, result.failure_kind = True, VERIFIER_ERROR
                    result.failure_reason = "Harness did not return a measurable response"
                    return result
                try:
                    answer = RetainedAnswer(request_fingerprint(body), await self._capture(session_id, body))
                except (OSError, ValueError, RuntimeError, NgFailureEvent) as error:
                    log_verification_failure(error, stage="artifact_capture")
                    result = self.new_result(body)
                    result.mask_sample, result.failure_kind = True, VERIFIER_ERROR
                    result.failure_reason = "Final artifact capture failed; no model-quality measurement"
                    return result
            elif isinstance(verification.retained, RetainedAnswer):
                answer = verification.retained
            else:
                answer = self._replayed_answer(body)
            if answer.fingerprint != request_fingerprint(body):
                raise HTTPException(409, "Verification retry changed the original request")
            verification.retain(answer)
            if answer.verdict is not None:
                return answer.verdict.model_copy(deep=True)
            result = self.new_result(body)
            if answer.source is None:
                # Preserve this measured zero for full reverification after process restart. Reuse the source
                # receipt envelope (including its task/response binding), with an explicitly empty artifact.
                result.source_artifact = ""
                result.source_sha256 = hashlib.sha256(b"").hexdigest()
                result.source_binding_sha256 = source_binding(body, result.source_sha256)
                result.failure_kind = "mm_coding_training:invalid_artifact"
                result.evaluation_completed = True
                result = observe_programmatic_grading(lambda: result)
            else:
                sha = hashlib.sha256(answer.source.encode()).hexdigest()
                result.answer_sha256 = sha
                result.source_binding_sha256 = source_binding(body, sha)
                body.source_binding_sha256 = result.source_binding_sha256
                result = await self.grade_source(body, answer.source, result)
            if not result.mask_sample:
                verification.retain(RetainedAnswer(answer.fingerprint, answer.source, result.model_copy(deep=True)))
            return result


if __name__ == "__main__":
    SessionCodingTrainingServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SessionCodingTrainingServer.run_webserver()
