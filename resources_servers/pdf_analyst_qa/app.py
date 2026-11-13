# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted PDF task state, selected document staging, and answer verification."""

import asyncio
import copy
import logging
import shlex
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import ClassVar

from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr, ValidationError

from nemo_gym.agents import restricted_network_policy, sandbox_server_url
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.core import SESSION_LOST, VERIFIER_ERROR
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxResources,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verification import VerifierFixture
from resources_servers.pdf_analyst_qa.documents import (
    DOCUMENT_MANIFEST_PATH,
    NODE_CACHE_ROOT,
    SANDBOX_SCRIPTS_ROOT,
    MountedDocumentsConfig,
    mounted_document_manifest,
    node_document_manifest,
    sandbox_script_files,
)
from resources_servers.pdf_analyst_qa.task_data import PdfAnalystQaSeedRequest, PdfAnalystQaTask
from resources_servers.pdf_analyst_qa.verifier_fixture import (
    FIXTURE_CASES_PATH,
    create_fixture_server,
    fixture_request_model,
    invoke_fixture,
)
from resources_servers.pdf_analyst_qa.verifiers.verifier import grade_task_answer


LOGGER = logging.getLogger(__name__)
DOCS_ROOT = "/workspace/docs"
TASK_ROOT = "/workspace/task"
SANDBOX_ENV = {
    "HOME": TASK_ROOT,
    "LANG": "C.UTF-8",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "PATH": "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONUNBUFFERED": "1",
    "OPENCODE_CONFIG": "/opt/cli/opencode-model.json",
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_MODELS_FETCH": "1",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
}


class PdfAnalystQaResourcesServerConfig(BaseResourcesServerConfig):
    """Dataset location and sandbox settings."""

    dataset_root: str
    document_image: str
    sandbox_provider: str = "sandbox"
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    model_server: ModelServerRef
    mounted_documents: MountedDocumentsConfig
    sandbox_cpu: float = Field(default=2, gt=0)
    sandbox_memory_mib: int = Field(default=8192, gt=0)
    sandbox_disk_gib: int = Field(default=10, gt=0)
    sandbox_request_cpu: float = Field(default=0.5, gt=0)
    sandbox_request_memory_mib: int = Field(default=1024, gt=0)
    sandbox_ready_timeout_s: float = Field(default=180, gt=0)
    command_timeout_s: float = Field(default=600, gt=0)
    rollout_timeout_s: float = Field(default=1800, gt=0)
    cleanup_timeout_s: float = Field(default=60, gt=0)
    sandbox_ttl_s: float = Field(default=4500, gt=0)
    max_active_sandboxes: int = Field(default=32, ge=1)
    sandbox_queue_timeout_s: float = Field(default=120, gt=0)
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED


class PdfAnalystQaVerifyRequest(BaseVerifyRequest, PdfAnalystQaSeedRequest):
    model_config = ConfigDict(extra="allow")


class PdfAnalystQaVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    evaluation_completed: bool
    task_id: str
    workspace_id: str
    prompt_version: str
    final_answer: str
    semantic_correct: bool | None = None
    format_compliant: bool | None = None
    elapsed_s: float = 0


@dataclass
class _SessionState:
    task: PdfAnalystQaTask
    prompt_version: str
    started_at: float = field(default_factory=monotonic)
    sandbox: AsyncSandbox | None = None
    owns_slot: bool = False


@dataclass(frozen=True)
class _GradeEvidence:
    task: PdfAnalystQaTask
    prompt_version: str
    started_at: float


def _last_assistant_text(body: PdfAnalystQaVerifyRequest) -> str:
    for item in reversed(body.response.output):
        if item.type == "message" and getattr(item, "role", None) == "assistant":
            return "".join(part.text for part in item.content if part.type == "output_text").strip()
    return ""


class PdfAnalystQaResourcesServer(SimpleResourcesServer):
    """Manage task sandboxes and give OpenCode access to the selected PDFs."""

    config: PdfAnalystQaResourcesServerConfig
    seed_request_model: ClassVar[type[PdfAnalystQaSeedRequest]] = PdfAnalystQaSeedRequest
    _sessions: dict[str, _SessionState] = PrivateAttr(default_factory=dict)
    _sandbox_slots: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._sandbox_slots = asyncio.Semaphore(self.config.max_active_sandboxes)
        if self.config.sandbox_request_cpu > self.config.sandbox_cpu:
            raise ValueError("sandbox CPU request must not exceed its limit")
        if self.config.sandbox_request_memory_mib > self.config.sandbox_memory_mib:
            raise ValueError("sandbox memory request must not exceed its limit")
        if (
            self.config.sandbox_ttl_s
            <= self.config.sandbox_ready_timeout_s
            + self.config.command_timeout_s
            + min(self.config.command_timeout_s, 60)
            + self.config.rollout_timeout_s
            + self.config.cleanup_timeout_s
        ):
            raise ValueError("sandbox TTL must cover startup, staging, preflight, rollout and cleanup")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(inner_app: FastAPI):
            try:
                async with parent_lifespan(inner_app) as value:
                    yield value
            finally:
                results = await asyncio.gather(
                    *(self.release_session_state(key) for key in list(self._sessions)), return_exceptions=True
                )
                failures = [result for result in results if isinstance(result, BaseException)]
                if failures:
                    raise BaseExceptionGroup("PDF document sandbox shutdown failed", failures)

        app.router.lifespan_context = lifespan
        return app

    def _load_task(self, data: PdfAnalystQaSeedRequest) -> PdfAnalystQaTask:
        root = Path(self.config.dataset_root).expanduser().resolve()
        try:
            task = PdfAnalystQaTask.model_validate_json((root / "tasks" / f"{data.task_id}.json").read_bytes())
        except (OSError, ValidationError) as error:
            raise ResourcesSeedRejectedError("trusted PDF task is unavailable or invalid") from error
        if task.task_id != data.task_id or task.workspace_id != data.workspace_id:
            raise ResourcesSeedRejectedError("hidden task and requested workspace do not match")
        try:
            grade = grade_task_answer(task, task.final_answer)
        except (ValueError, KeyError) as error:
            raise ResourcesSeedRejectedError("reference answer is invalid") from error
        if not grade["semantic_correct"] or not grade["format_compliant"]:
            raise ResourcesSeedRejectedError("reference answer violates its requested answer format")
        return task

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def initialize_session_state(self, session_id: str, body: PdfAnalystQaSeedRequest) -> None:
        identity = self.session_identity(session_id)
        if identity is None or identity[1].task_id != body.task_id:
            raise ResourcesSeedRejectedError("TaskId differs from the trusted PDF task")
        task = await asyncio.to_thread(self._load_task, body)
        state = _SessionState(task, body.prompt_version)
        self._sessions[session_id] = state  # publish before the first allocation await
        async with asyncio.timeout(self.config.sandbox_queue_timeout_s):
            await self._sandbox_slots.acquire()
        state.owns_slot = True
        await self._start_sandbox(state, body.workspace_id)
        result = await state.sandbox.exec(
            shlex.join(
                [
                    "/opt/venv/bin/python",
                    f"{SANDBOX_SCRIPTS_ROOT}/preflight.py",
                    self.config.mounted_documents.mount_root,
                    sandbox_server_url(self.config.model_server.name, require_reachable=True),
                ]
            ),
            cwd=TASK_ROOT,
            timeout_s=min(self.config.command_timeout_s, 60),
        )
        if result.error_type or result.return_code != 0:
            raise RuntimeError(f"document sandbox preflight failed: {result.error_type or result.return_code}")
        state.started_at = monotonic()

    async def _start_sandbox(self, state: _SessionState, workspace_id: str) -> None:
        config = self.config
        root = Path(config.dataset_root).expanduser().resolve()
        node_cache = config.mounted_documents.node_cache_host_path
        manifest = await asyncio.to_thread(
            node_document_manifest if node_cache else mounted_document_manifest,
            root,
            workspace_id,
            config.mounted_documents,
        )
        global_config = self.server_client.global_config_dict
        provider_config = resolve_provider_config(config.sandbox_provider, global_config)
        if set(provider_config) != {"opensandbox"}:
            raise ValueError("PDF document views require the OpenSandbox provider")
        provider_config = copy.deepcopy(provider_config)
        options = copy.deepcopy(config.sandbox_config.get("provider_options", {}))
        if "network_policy" in options:
            raise ValueError("PDF document sandbox network_policy is managed by the resources server")
        if "resource_requests" in options:
            raise ValueError("PDF document sandbox resource_requests are managed by the resources server")
        options["network_policy"] = restricted_network_policy(
            "opensandbox", [sandbox_server_url(config.model_server.name, require_reachable=True)]
        )
        options["resource_requests"] = {
            "cpu": config.sandbox_request_cpu,
            "memory_mib": config.sandbox_request_memory_mib,
            "disk_gib": config.sandbox_disk_gib,
        }
        if node_cache:
            options["volumes"] = [
                {
                    "name": "pdf-analyst-qa-node-cache",
                    "host": {"path": node_cache},
                    "mountPath": NODE_CACHE_ROOT,
                    "readOnly": False,
                },
                {
                    "name": "pdf-analyst-qa-documents",
                    "host": {"path": f"{node_cache}/views/{workspace_id}"},
                    "mountPath": DOCS_ROOT,
                    "readOnly": True,
                },
            ]
        metadata = (
            resolve_provider_metadata(config.sandbox_provider, global_config)
            | dict(config.sandbox_config.get("metadata", {}))
            | {"workspace-id": workspace_id[:63]}
        )
        spec = SandboxSpec(
            image=config.document_image,
            workdir=TASK_ROOT,
            entrypoint=["/bin/bash", "-c", "exec sleep infinity"],
            env=SANDBOX_ENV,
            files={DOCUMENT_MANIFEST_PATH: manifest, **sandbox_script_files()},
            ttl_s=config.sandbox_ttl_s,
            ready_timeout_s=config.sandbox_ready_timeout_s,
            resources=SandboxResources(
                cpu=config.sandbox_cpu, memory_mib=config.sandbox_memory_mib, disk_gib=config.sandbox_disk_gib, gpu=0
            ),
            provider_options=options,
            metadata=metadata,
        )
        state.sandbox = AsyncSandbox(create_provider(provider_config))
        await state.sandbox.start(spec)
        result = await state.sandbox.exec(
            shlex.join(
                [
                    "/opt/venv/bin/python",
                    f"{SANDBOX_SCRIPTS_ROOT}/documents.py",
                    DOCUMENT_MANIFEST_PATH,
                    config.mounted_documents.mount_root,
                    *([workspace_id] if node_cache else []),
                ]
            ),
            cwd="/",
            user=0,
            timeout_s=config.command_timeout_s,
        )
        if result.error_type or result.return_code != 0:
            LOGGER.error(
                "Document staging failed: return_code=%s error_type=%s stdout=%s stderr=%s",
                result.return_code,
                result.error_type,
                result.stdout,
                result.stderr,
            )
            raise RuntimeError("could not stage selected PDFs from the shared bucket")
        # Keep the supervisor privileged for trusted initialization, and carry
        # the task UID through serialize/connect so every harness command runs
        # without access to the protected backing cache.
        state.sandbox.set_default_user(65534)

    async def release_session_state(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            return
        if state.sandbox is not None:
            async with asyncio.timeout(self.config.cleanup_timeout_s):
                await state.sandbox.stop()
            state.sandbox = None
        if state.owns_slot:
            self._sandbox_slots.release()
            state.owns_slot = False
        self._sessions.pop(session_id, None)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        response = await super().seed_session(body, request=request)
        state = self._sessions[body.resources_session_id]
        assert state.sandbox is not None
        return ResourcesSeedSessionResponse(
            resources_session_id=response.resources_session_id,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider,
                    descriptor=await state.sandbox.serialize(scope="operate"),
                ),
                workdir=TASK_ROOT,
                user=65534,
            ),
        )

    async def verify(self, request: Request, body: PdfAnalystQaVerifyRequest) -> PdfAnalystQaVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY, "")
        answer = _last_assistant_text(body)
        semantic = compliant = None
        failure = None
        reward = 0.0
        elapsed = 0.0
        completed = False
        async with self.verifying(session_id) as verification:
            if verification.is_lost:
                failure = SESSION_LOST
            else:
                if verification.is_live:
                    state = self._sessions[session_id]
                    evidence = _GradeEvidence(state.task, state.prompt_version, state.started_at)
                    verification.retain(evidence)
                else:
                    evidence = verification.retained
                elapsed = monotonic() - evidence.started_at
                if (evidence.task.task_id, evidence.task.workspace_id, evidence.prompt_version) != (
                    body.task_id,
                    body.workspace_id,
                    body.prompt_version,
                ):
                    failure = VERIFIER_ERROR
                elif body.response.status == "failed" or body.response.error is not None:
                    failure = "pdf_analyst_qa:agent_response_failed"
                else:
                    try:
                        if answer and body.response.status != "incomplete" and not body.response.incomplete_details:
                            grade = grade_task_answer(evidence.task, answer)
                            semantic, compliant = grade["semantic_correct"], grade["format_compliant"]
                            reward = float(semantic and compliant)
                        completed = True
                    except Exception:
                        LOGGER.exception("Document answer verification failed")
                        failure = VERIFIER_ERROR
        return PdfAnalystQaVerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": reward,
                "final_answer": answer,
                "semantic_correct": semantic,
                "format_compliant": compliant,
                "mask_sample": not completed,
                "evaluation_completed": completed,
                "failure_kind": failure,
                "failure_reason": failure,
                "elapsed_s": elapsed,
            }
        )

    def compute_metrics(self, tasks: list[list[dict[str, JsonValue]]]) -> dict[str, JsonValue]:
        rows = [row for task in tasks for row in task]
        measured = [row for row in rows if row.get("evaluation_completed") and not row.get("mask_sample")]
        result: dict[str, JsonValue] = {
            "pdf_analyst_qa/scored_rollouts": len(measured),
            "pdf_analyst_qa/accuracy": sum(float(row["reward"]) for row in measured) / len(measured)
            if measured
            else None,
        }
        for field in ("semantic_correct", "format_compliant"):
            result[f"pdf_analyst_qa/{field}"] = (
                sum(row.get(field) is True for row in measured) / len(measured) if measured else None
            )
        return result

    def get_key_metrics(self, agent_metrics: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {key: value for key, value in agent_metrics.items() if key.startswith("pdf_analyst_qa/")}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=fixture_request_model(),
    cases_path=FIXTURE_CASES_PATH,
    invoke=invoke_fixture,
)


if __name__ == "__main__":
    PdfAnalystQaResourcesServer.run_webserver()
