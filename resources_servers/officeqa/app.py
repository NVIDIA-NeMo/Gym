# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OfficeQA document sandbox and trusted official verification using Gym contracts."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Literal, NamedTuple

from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field, JsonValue, ValidationError

from nemo_gym.agents import restricted_network_policy, sandbox_server_url
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.metrics import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxResources,
    SandboxSpec,
    create_provider,
)
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verification import VerifierFixture
from resources_servers.officeqa.task_data import TaskData
from resources_servers.officeqa.verifier_fixture import create_fixture_server, invoke_fixture
from resources_servers.officeqa.verifiers.officeqa_pro import (
    extract_final_answer,
    extract_final_answer_from_xml,
    score_answer,
)
from resources_servers.officeqa.verifiers.officeqa_pro_task import OfficeQAProTask


TOLERANCES: dict[str, float] = {
    "correct_tol_0": 0.0,
    "correct_tol_0p1": 0.001,
    "correct_tol_1": 0.01,
    "correct_tol_5": 0.05,
}


class OfficeQAResourcesServerConfig(BaseResourcesServerConfig):
    """Trusted host-side task bundle; it must never be mounted in the agent sandbox."""

    workspace_root: str
    numeric_tolerance: float = Field(default=0.0, ge=0, le=1)
    sandbox_provider: str = "sandbox"
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    model_server: ModelServerRef | None = None
    network_access: Literal["model_only", "inherit"] = "model_only"
    sandbox_stop_timeout: float = Field(default=60, gt=0, allow_inf_nan=False)
    REVERIFY_MODE = ReverifyMode.STATELESS


class OfficeQARunRequest(BaseRunRequest, TaskData):
    pass


class OfficeQAVerifyRequest(BaseVerifyRequest, OfficeQARunRequest):
    model_config = ConfigDict(extra="allow")


class OfficeQAVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    evaluation_completed: bool
    task_id: str
    uid: str | None = None
    final_answer: str
    has_final_answer: bool = False
    correct_tol_0: float | None = None
    correct_tol_0p1: float | None = None
    correct_tol_1: float | None = None
    correct_tol_5: float | None = None


class OfficeQAFixtureRequest(OfficeQAVerifyRequest):
    """A synthetic expected answer provided only by the trusted fixture runner."""

    expected_answer: str = Field(min_length=1, exclude=True)
    missing_reference: bool = Field(default=False, exclude=True)


class _DocumentSession(NamedTuple):
    request: ResourcesSeedSessionRequest
    sandbox: AsyncSandbox
    ready: bool = False


class OfficeQAResourcesServer(SimpleResourcesServer):
    """Own PDF-only sandboxes; existing Gym agents borrow them and return answers."""

    config: OfficeQAResourcesServerConfig

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("OfficeQA process-local sandboxes require num_workers=1")
        self._sessions: dict[str, _DocumentSession] = {}
        self._locks = ResourcesSessionLocks()
        self._fences = ResourcesSessionFences(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    def _task(self, body: TaskData) -> OfficeQAProTask:
        root = Path(self.config.workspace_root).expanduser().resolve()
        task = OfficeQAProTask.model_validate_json((root / "tasks" / (body.task_id + ".json")).read_text())
        if task.task_id != body.task_id or task.workspace_id != body.workspace_id:
            raise ResourcesSeedRejectedError("Task identity differs from the trusted OfficeQA bundle")
        return task

    async def _create_sandbox(self) -> tuple[AsyncSandbox, SandboxSpec]:
        global_config = self.server_client.global_config_dict
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config)
        values = deepcopy(self.config.sandbox_config)
        values["resources"] = SandboxResources.from_mapping(values.get("resources", {}))
        values["metadata"] = resolve_provider_metadata(self.config.sandbox_provider, global_config) | values.get(
            "metadata", {}
        )
        if self.config.network_access == "model_only":
            if self.config.model_server is None:
                raise ValueError("OfficeQA model_only networking requires model_server")
            options = values.setdefault("provider_options", {})
            options["network_policy"] = restricted_network_policy(
                next(iter(provider_config)),
                [sandbox_server_url(self.config.model_server.name, require_reachable=True)],
            )
        spec = SandboxSpec(**values)
        if spec.image is None or not spec.ttl_s or spec.workdir != "/workspace/task":
            raise ValueError("OfficeQA needs a document image, a finite sandbox TTL and /workspace/task workdir")
        return AsyncSandbox(create_provider(provider_config)), spec

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        """Seed the existing resource-session contract, returning borrower access."""
        try:
            task = TaskData.model_validate(body.task_data)
        except ValidationError as error:
            raise ResourcesSeedRejectedError(f"Invalid OfficeQA task_data: {error}") from error
        self._task(task)
        if body.task_id.task_id != task.task_id:
            raise ResourcesSeedRejectedError("TaskId differs from the trusted OfficeQA task")
        session_id = body.resources_session_id
        request.session[SESSION_ID_KEY] = session_id
        async with self._locks.locked(session_id):
            self._fences.raise_if_fenced(session_id)
            existing = self._sessions.get(session_id)
            if existing is not None:
                if existing.request != body:
                    raise ResourcesSessionConflictError(
                        "OfficeQA resources session is bound to another episode or task"
                    )
                if not existing.ready:
                    raise RuntimeError("OfficeQA resources session is not ready; close it before retrying setup")
                sandbox = existing.sandbox
            else:
                sandbox, spec = await self._create_sandbox()
                # Retain ownership before start: a failed release remains retryable by /close_session.
                self._sessions[session_id] = _DocumentSession(body.model_copy(deep=True), sandbox)
                try:
                    await sandbox.start(spec)
                except BaseException:
                    await self._stop_session(session_id)
                    raise
                self._sessions[session_id] = self._sessions[session_id]._replace(ready=True)
            return ResourcesSeedSessionResponse(
                resources_session_id=session_id,
                sandbox_access=SandboxAccess(
                    connection=DirectSandboxConnection(
                        provider_config_ref=self.config.sandbox_provider, descriptor=await sandbox.serialize()
                    ),
                    workdir="/workspace/task",
                ),
            )

    async def _stop_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            async with asyncio.timeout(self.config.sandbox_stop_timeout):
                await session.sandbox.stop()
            self._sessions.pop(session_id)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Close idempotently, fencing late seed requests through Gym's existing helper."""
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            closed = self._fences.get(session_id)
            session = self._sessions.get(session_id)
            if (closed is not None and closed != body.episode_id) or (
                session is not None and session.request.episode_id != body.episode_id
            ):
                raise ResourcesSessionConflictError("EpisodeId differs from the OfficeQA resources session")
            await self._stop_session(session_id)
            self._fences.fence(session_id, body.episode_id)
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def shutdown(self) -> None:
        """Stop every remaining sandbox when the server exits."""
        results = await asyncio.gather(
            *(self._stop_session(session_id) for session_id in list(self._sessions)), return_exceptions=True
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("OfficeQA sandbox shutdown failed", failures)

    async def verify(self, request: Request, body: OfficeQAVerifyRequest) -> OfficeQAVerifyResponse:
        """Grade the final assistant message using only the host-side expected answer."""
        try:
            task = self._task(body)
        except (OSError, ValueError):
            return OfficeQAVerifyResponse.model_validate(
                body.model_dump(exclude={"uid"})
                | {
                    "reward": 0.0,
                    "mask_sample": True,
                    "evaluation_completed": False,
                    "failure_kind": "officeqa:reference_unavailable",
                    "failure_reason": "The trusted task reference is unavailable or invalid.",
                    "final_answer": "",
                    "has_final_answer": False,
                    **dict.fromkeys(TOLERANCES),
                }
            )
        answer = ""
        for item in reversed(body.response.output):
            if item.type == "message" and getattr(item, "role", None) == "assistant":
                answer = "".join(part.text for part in item.content if part.type == "output_text").strip()
                break
        # Chat clients can retain Gym's reasoning inline; grade only the answer text.
        answer = ChatCompletionsConverter.THINK_TAG_PATTERN.sub("", answer).strip()
        complete = body.response.status in (None, "completed") and not body.response.incomplete_details
        reward = (
            score_answer(task.expected_answer, answer, self.config.numeric_tolerance) if answer and complete else 0.0
        )
        final_answer = extract_final_answer(answer)
        # Without a FINAL_ANSWER tag, the scorer grades the whole text. no_answer counts only tagged answers.
        tagged_answer, reasoning = extract_final_answer_from_xml(answer)
        tagged = bool(answer) and (reasoning is not None or tagged_answer != answer)
        scores = {
            name: score_answer(task.expected_answer, answer, tolerance) if answer and complete else 0.0
            for name, tolerance in TOLERANCES.items()
        }
        return OfficeQAVerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": reward,
                "evaluation_completed": True,
                "task_id": task.task_id,
                "uid": task.uid,
                "final_answer": final_answer,
                "has_final_answer": complete and tagged and bool(tagged_answer.strip()),
                **scores,
            }
        )

    def compute_metrics(self, tasks: list[list[dict[str, JsonValue]]]) -> dict[str, JsonValue]:
        """Report mean accuracy, plus the tolerance and no_answer metrics where rows have them."""
        score_name = "accuracy_no_net" if self.config.network_access == "model_only" else "accuracy"

        def scores(result: dict[str, JsonValue]) -> dict[str, JsonValue]:
            # Older saved rows have no diagnostics; omit them instead of inventing zeros.
            values = {score_name: result["reward"]}
            values.update({name: result[name] for name in TOLERANCES if result.get(name) is not None})
            if "has_final_answer" in result:
                values["no_answer"] = float(not result["has_final_answer"])
            return values

        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=scores)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Select mean attempt accuracy at the available repeat count as the headline."""
        key = {
            name: agent_metrics[name] for name in ("mean/input_tokens", "mean/output_tokens") if name in agent_metrics
        }
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy", "accuracy_no_net"]))
        return key


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=OfficeQAFixtureRequest,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
    invoke=invoke_fixture,
)


if __name__ == "__main__":
    OfficeQAResourcesServer.run_webserver()
