# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agentic IF Resources Server: it owns each episode's task sandbox, lends it to the harness and grades it.

Seed borrows the task sandbox from the row's task backend (Scale-SWE or Terminal-Bench 2.1, through their typed
resources sessions), prepares the actor's workspace, stages the IF OpenCode plugins and lends the sandbox to the agent.
Verify reads what OpenCode and the plugins recorded, has the task backend verify the task, persists the episode and
grades instruction following against the row's checks. Close releases the task backend's session, which stops the
sandbox.
"""

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Literal
from uuid import uuid4

from aiohttp import ClientResponseError
from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr, ValidationError, field_validator

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.failure_kinds import SESSION_LOST
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.agentic_if.episode import bundle
from resources_servers.agentic_if.episode.agent_user import is_root_agent_user
from resources_servers.agentic_if.episode.artifacts import (
    EPISODE_RECORD,
    RESULT_RECORD,
    BackendVerdict,
    EpisodeRecord,
    PersistedVerdict,
    load_episode_record,
    save_json,
)
from resources_servers.agentic_if.episode.judging import (
    JudgeCallSummary,
    JudgeLedger,
    JudgeUsage,
    judge_prompt,
    judge_usage,
)
from resources_servers.agentic_if.episode.opencode_replay import restore_prefix_workspace
from resources_servers.agentic_if.episode.outcome import ExecutionStatus, Verdict, classify, execution_status
from resources_servers.agentic_if.episode.resources import SandboxTaskSession
from resources_servers.agentic_if.episode.session_evidence import (
    collection_failed,
    evidence_from_bundle,
    row_prompt_texts,
)
from resources_servers.agentic_if.episode.verifier_fixture import create_fixture_server, invoke_fixture
from resources_servers.agentic_if.episode.workspace import (
    snapshot_workspace,
    stage_workspace_files,
    verify_materialized_cwd,
)
from resources_servers.agentic_if.if_grading.contracts import IFResult, ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.engine import grade
from resources_servers.agentic_if.if_grading.judge import Judge
from resources_servers.agentic_if.task_variants.materialization import sealed_task_input
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, validate_runtime, validate_seed_runtime


LOG = logging.getLogger(__name__)

TaskBackend = Literal["swe", "tb21"]


class AgenticIFResourcesServerConfig(BaseResourcesServerConfig):
    """The task backends, the judge, and where each episode's evidence is kept."""

    # A session-less verify re-grades instruction following from the evidence a live verify persisted, and keeps the
    # task backend's persisted verdict: it never needs the sandbox again.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # The model server that answers judge questions. It must serve the model `if_grading/data/judge.yaml` names: the
    # judge's prompts and model settings are sent as that file sets them.
    judge_model_server: ModelServerRef
    max_concurrent_grades: int = Field(default=8, gt=0)
    # The Resources Servers that own the task sandboxes, by the row's `agentic_if.task_backend`.
    task_resources: dict[TaskBackend, ResourcesServerRef] = Field(min_length=1)
    # Per-episode evidence, receipts and verdicts: rollout data, keep it private. It backs re-verification.
    artifacts_dir: Path
    # Caps each command seed runs in the task sandbox (workspace staging, baseline, prefix replay).
    sandbox_timeout_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    # Host files seed copies into every task sandbox's bundle directory, for the agent to read by in-sandbox path;
    # relative paths are under the Gym root. The benchmark stages OpenCode's ripgrep and plugin dependencies this way
    # for sandboxes without internet access, and opencode_agent checks each copy against its pinned digest.
    opencode_runtime_files: list[Path] = Field(default_factory=list)

    @field_validator("opencode_runtime_files")
    @classmethod
    def resolve_opencode_runtime_files(cls, value: list[Path]) -> list[Path]:
        """Resolve relative paths under the Gym root; the files land in one directory, so their names must differ."""
        paths = [path if path.is_absolute() else PARENT_DIR / path for path in value]
        names = [path.name for path in paths]
        if len(set(names)) != len(names):
            raise ValueError(f"opencode_runtime_files must have distinct file names, got {names}")
        return paths

    @field_validator("artifacts_dir")
    @classmethod
    def require_absolute_artifacts_dir(cls, value: Path) -> Path:
        """Every worker and re-verification must find the same directory."""
        if not value.is_absolute():
            raise ValueError(f"artifacts_dir must be an absolute path, got {str(value)!r}")
        return value


class AgenticIFVerifyRequest(BaseVerifyRequest):
    """The flat verify body: the sealed row's fields, the actor's request and the harness's response.

    The task backend's own fields vary by backend, so they are kept as they arrive; a live verify checks them
    against the seeded task.
    """

    model_config = ConfigDict(extra="allow")

    # The row's content digest, as collation keeps it beside the task fields.
    task_id: str | None = None
    agentic_if: dict[str, JsonValue]
    # The episode's persisted evidence, as a live verify reports it; a verify without a live session re-grades it.
    deliverables_dir: str | None = None


class AgenticIFVerifyResponse(BaseVerifyResponse):
    """The task backend's verdict and fields, with the instruction-following result and the episode's evidence."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    agentic_if: dict[str, JsonValue]
    evaluation_completed: bool
    if_result: IFResult | None = None
    execution_status: ExecutionStatus | None = None
    evidence: SessionEvidence | None = None
    deliverables_dir: str | None = None
    # The calls of the grade behind this result, and their totals. The judge is asked outside the rollout's context,
    # so its calls are not in the rollout's model-call capture; the episode record keeps them with their questions.
    judge_calls: list[JudgeCallSummary] = Field(default_factory=list)
    judge_usage: JudgeUsage = Field(default_factory=JudgeUsage)


@dataclass(eq=False)
class _Episode:
    """One resources session: the seeded task, its borrowed sandbox and what verify recorded."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    request: ResourcesSeedSessionRequest | None = None
    response: ResourcesSeedSessionResponse | None = None
    # The sealed task without its identity field and the actor's request; the task backend sees it minus agentic_if.
    content: dict[str, JsonValue] | None = None
    task: SandboxTaskSession | None = None
    # What seed observed and staged: working directory, baseline diff, replay and bundle receipts.
    seed: dict[str, JsonValue] = field(default_factory=dict)
    profile_sha256: str | None = None
    # Set by close before it waits for the lock, so a seed still in progress releases what it allocated.
    closing: bool = False
    verification_request: dict[str, JsonValue] | None = None
    record: EpisodeRecord | None = None
    # Only a measured result: one without a measurement is not the session's verification result.
    verification: "AgenticIFVerifyResponse | None" = None

    async def release(self) -> None:
        """Close the task backend's session, which stops the sandbox, and disconnect."""
        if self.task is not None:
            await self.task.close()


def _seed_failure(error: Exception) -> HTTPException:
    """An invalid task is terminal (4xx); an infrastructure failure is retryable (5xx)."""
    if isinstance(error, HTTPException):
        return error
    if isinstance(error, ClientResponseError):
        # Gym's raise_for_status keeps the backend's reply on the error; its detail names the cause.
        reply = (getattr(error, "response_content", None) or b"").decode(errors="replace")[:1000] or error.message
        if 400 <= error.status < 500 and error.status not in (408, 425, 429):
            return HTTPException(422, f"The task backend rejected the task (HTTP {error.status}): {reply}")
        return HTTPException(503, f"The task backend could not seed the task (HTTP {error.status}): {reply}")
    if isinstance(error, ValueError):
        return HTTPException(422, f"Invalid agentic IF task: {error}")
    return HTTPException(503, f"Could not prepare the task sandbox: {type(error).__name__}: {error}")


def _closing(episode: _Episode) -> None:
    if episode.closing:
        raise HTTPException(409, "Resources session closed during setup")


class AgenticIFResourcesServer(SimpleResourcesServer):
    """Seed lends the task sandbox and stages the IF plugins; verify grades the episode; close releases it."""

    config: AgenticIFResourcesServerConfig
    # Sessions hold sandbox connections in this process.
    ray_enabled: ClassVar[bool] = False
    _episodes: dict[str, _Episode] = PrivateAttr(default_factory=dict)
    _closed: ClosedResourcesSessionRecords[EpisodeId] = PrivateAttr()
    _grading_slots: asyncio.Semaphore = PrivateAttr()
    _grade_jobs: set[asyncio.Task[IFResult]] = PrivateAttr(default_factory=set)
    _shutting_down: bool = PrivateAttr(default=False)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("agentic IF sessions hold sandbox connections in one process: num_workers must be 1")
        if missing := [str(path) for path in self.config.opencode_runtime_files if not path.is_file()]:
            raise ValueError(
                f"opencode_runtime_files are missing: {missing}; for the benchmark, run "
                "`gym eval prepare --benchmark agentic_if`"
            )
        self._closed = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)
        self._grading_slots = asyncio.Semaphore(self.config.max_concurrent_grades)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self._shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def _shutdown(self) -> None:
        self._shutting_down = True
        episodes = list(self._episodes.items())
        for _, episode in episodes:
            episode.closing = True
        results = await asyncio.gather(*(episode.release() for _, episode in episodes), return_exceptions=True)
        for (session_id, _), result in zip(episodes, results):
            if isinstance(result, BaseException):
                LOG.error("Could not release agentic IF session %s on shutdown", session_id, exc_info=result)

    # ----------------------------------------------------------------------------------------------------- seed

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        """Borrow the task backend's sandbox, prepare the actor's workspace and the IF plugins, and lend the sandbox.

        Repeating the same seed returns the existing session. An invalid row is a 422 (terminal); an infrastructure
        failure is a 503, which the Environment Server retries with another session. Either way nothing stays allocated.
        """
        session_id = body.resources_session_id
        if self._shutting_down:
            raise HTTPException(503, "The agentic IF server is shutting down")
        if session_id in self._closed:
            raise HTTPException(409, "Resources session is already closed")
        episode = self._episodes.setdefault(session_id, _Episode())
        async with episode.lock:
            if self._episodes.get(session_id) is not episode or episode.closing:
                raise HTTPException(409, "Resources session is already closed")
            if episode.request is not None:
                if episode.request != body:
                    raise HTTPException(409, "resources_session_id is already bound to a different request")
                if episode.response is None:
                    raise HTTPException(409, "Resources session failed to seed and is no longer available")
                request.session[SESSION_ID_KEY] = session_id
                return episode.response
            try:
                content, spec, backend = self._validated_task(body)
            except HTTPException:
                self._episodes.pop(session_id, None)
                raise
            episode.request = body.model_copy(deep=True)
            episode.content = content
            try:
                episode.response = await self._prepare(episode, body, spec, backend)
            except BaseException as error:
                await self._abandon(session_id, episode)
                if isinstance(error, Exception):
                    failure = _seed_failure(error)
                    LOG.warning(
                        "Seed of agentic IF session %s failed (HTTP %s): %s",
                        session_id,
                        failure.status_code,
                        failure.detail,
                        exc_info=error,
                    )
                    raise failure from error
                raise
            request.session[SESSION_ID_KEY] = session_id
            return episode.response

    def _validated_task(
        self, body: ResourcesSeedSessionRequest
    ) -> tuple[dict[str, JsonValue], RuntimeSpec, ResourcesServerRef]:
        """The sealed task (its identity field removed), its configuration and its task backend; 422 if invalid."""
        try:
            content = dict(
                sealed_task_input({"task_id": body.task_id.model_dump(mode="json"), "task_input": body.task_data})
            )
            spec = validate_seed_runtime(content)
        except ValueError as error:
            detail = error.errors(include_input=False) if isinstance(error, ValidationError) else str(error)
            raise HTTPException(status_code=422, detail=detail) from error
        backend = self.config.task_resources.get(spec.task_backend)
        if backend is None:
            raise HTTPException(422, f"Task backend {spec.task_backend!r} is not enabled on this server")
        return content, spec, backend

    async def _prepare(
        self,
        episode: _Episode,
        body: ResourcesSeedSessionRequest,
        spec: RuntimeSpec,
        backend: ResourcesServerRef,
    ) -> ResourcesSeedSessionResponse:
        """Seed the task backend, then prepare the borrowed sandbox for the actor, in the episode's order."""
        task = episode.task = SandboxTaskSession(
            client=self.server_client,
            server_name=backend.name,
            row=episode.content,
            client_session_id=f"agentic-if-{episode.run_id}",
            timeout_s=self.config.sandbox_timeout_seconds,
            episode_id=body.episode_id,
            task_id=body.task_id,
        )
        # Typed backend seed, connection, and the agent_user gate.
        await task.prepare()
        _closing(episode)
        sandbox = task.sandbox
        # As the actor: the materialized prompt's working directory, its initial files, and the baseline diff.
        await verify_materialized_cwd(sandbox, spec.provenance.get("materialized_cwd"))
        await stage_workspace_files(sandbox, spec.workspace_files)
        episode.seed["workspace_initial_diff"] = await snapshot_workspace(sandbox)
        _closing(episode)
        if spec.continuation is not None:
            episode.seed["replay"] = await restore_prefix_workspace(sandbox, spec)
            _closing(episode)
        agent_user = None if is_root_agent_user(sandbox.user) else sandbox.user
        staged = await bundle.stage(sandbox.transport, spec=spec, agent_user=agent_user)
        episode.profile_sha256 = staged.profile_sha256
        episode.seed["bundle"] = asdict(staged)
        if self.config.opencode_runtime_files:
            # After staging, which creates the bundle directory.
            episode.seed["runtime_files"] = await bundle.stage_runtime_files(
                sandbox.transport, self.config.opencode_runtime_files
            )
        if spec.continuation is not None:
            # After staging, so the bundle's own setup cannot remove it.
            episode.seed["session_import"] = await bundle.stage_session_import(
                sandbox.transport, spec.continuation.prefix_export
            )
        _closing(episode)
        episode.seed["workdir"] = sandbox.workdir
        return ResourcesSeedSessionResponse(
            resources_session_id=body.resources_session_id,
            sandbox_access=SandboxAccess(
                connection=task.access.connection,
                workdir=sandbox.workdir,
                user=agent_user,
                env=sandbox.execution_env,
            ),
        )

    async def _abandon(self, session_id: str, episode: _Episode) -> None:
        """Release what a failed seed allocated; if that fails, keep the session so close can retry it."""
        try:
            await episode.release()
        except Exception:
            LOG.exception("Could not release the task backend session of failed seed %s; close will retry", session_id)
            episode.response = None
            return
        if episode.closing and episode.request is not None:
            self._closed.remember(session_id, episode.request.episode_id)
        self._episodes.pop(session_id, None)

    # ----------------------------------------------------------------------------------------------------- close

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Release the task backend's session, which stops the sandbox, and refuse a later seed of the same ID.

        Closing a closed or never-seeded session succeeds; a failed release raises so the caller can retry it.
        """
        session_id = body.resources_session_id
        closed = self._closed.get(session_id)
        if closed is not None:
            if closed != body.episode_id:
                raise HTTPException(409, "episode_id does not match the resources session")
            return ResourcesCloseSessionResponse(resources_session_id=session_id)
        episode = self._episodes.get(session_id)
        if episode is None:
            self._closed.remember(session_id, body.episode_id)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)
        if episode.request is not None and episode.request.episode_id != body.episode_id:
            raise HTTPException(409, "episode_id does not match the resources session")
        # Fence a seed still in progress before waiting for it: it releases what it allocated itself.
        episode.closing = True
        async with episode.lock:
            if self._episodes.get(session_id) is episode:
                await episode.release()
                self._episodes.pop(session_id, None)
            self._closed.remember(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    # ---------------------------------------------------------------------------------------------------- verify

    async def verify(self, request: Request, body: AgenticIFVerifyRequest) -> AgenticIFVerifyResponse:
        """Verify one episode: read its evidence, have the task backend verify the task, persist it, grade IF.

        Without a live session, re-grade the evidence an earlier verify persisted (``deliverables_dir``). With
        neither, the session was lost: the row is masked as ``session_lost``.
        """
        session_id = request.session.get(SESSION_ID_KEY)
        episode = self._episodes.get(session_id) if session_id is not None else None
        if episode is not None:
            return await self._verify_live(session_id, episode, body)
        if session_id in self._closed:
            raise HTTPException(409, "Resources session is closed")
        if body.deliverables_dir is not None:
            return await self._verify_persisted(body)
        return self._lost(body)

    async def _verify_live(
        self, session_id: str, episode: _Episode, body: AgenticIFVerifyRequest
    ) -> AgenticIFVerifyResponse:
        """Verify the episode once; an identical retry replays a measured result.

        A result without a measurement (``evaluation_completed: false``: the bundle or the task backend failed, or the
        evidence is incomplete) is not kept. The first attempt has already used the agent's sandbox, so a retry gets
        a 503 and the episode is retried instead. A retry after a judge outage grades the persisted episode again.
        """
        async with episode.lock:
            if episode.closing or episode.response is None or self._episodes.get(session_id) is not episode:
                raise HTTPException(409, "Resources session is not available for verification")
            canonical = body.model_dump(mode="json", exclude={"deliverables_dir"})
            if episode.verification_request is not None:
                if episode.verification_request != canonical:
                    raise HTTPException(409, "Verification request does not match the first request")
                if episode.verification is not None:
                    # Replay a lost reply without verifying the task twice.
                    return episode.verification.model_copy(deep=True)
                if episode.record is None or not episode.record.verdict.evaluation_completed:
                    raise HTTPException(503, "Previous verification did not complete; retry the episode")
            spec = self._verified_runtime(body, episode.content, task_id=episode.request.task_id.task_id)
            if episode.record is None:
                # Recorded first: an attempt that ends without a verdict has already used the agent's sandbox.
                episode.verification_request = canonical
                episode.record = await self._observe(episode, body, spec)
            directory = self.config.artifacts_dir / episode.record.run_id
            # judge_failsafe answers with this request's own fields, so its row must name the persisted evidence too.
            body.deliverables_dir = str(directory)
            ledger = JudgeLedger()
            try:
                result = await self._grade_record(episode.record, body, spec, directory, ledger)
            finally:
                await self._keep_judge_calls(episode.record, directory, ledger)
            if result.evaluation_completed:
                episode.verification = result
            await asyncio.to_thread(save_json, directory / RESULT_RECORD, result.model_dump(mode="json"))
            return result.model_copy(deep=True)

    async def _keep_judge_calls(self, record: EpisodeRecord, directory: Path, ledger: JudgeLedger) -> None:
        """Add a grade's judge calls to the episode record, and persist it again."""
        calls = ledger.calls()
        if not calls:
            return
        record.judge_calls = record.judge_calls + calls
        try:
            await asyncio.to_thread(save_json, directory / EPISODE_RECORD, record.model_dump(mode="json"))
        except OSError:
            # This also runs after a grade that raised: a failed write must not replace its JudgeError.
            LOG.exception("Could not persist the judge calls of agentic IF run %s", record.run_id)

    def _verified_runtime(
        self, body: AgenticIFVerifyRequest, content: Mapping[str, JsonValue], *, task_id: str | None
    ) -> RuntimeSpec:
        """The row's configuration, every receipt checked now that the actor's request is here.

        The task fields must be the seeded ones (409); a receipt that does not hold is an invalid row (422).
        """
        fields = (body.model_extra or {}) | {"agentic_if": body.agentic_if}
        if any(key not in fields or fields[key] != value for key, value in content.items()) or (
            task_id is not None and body.task_id not in (None, task_id)
        ):
            raise HTTPException(409, "Verification task does not match the seeded task")
        row = dict(content) | {
            "responses_create_params": body.responses_create_params.model_dump(mode="json", exclude_unset=True)
        }
        try:
            return validate_runtime(row)
        except ValueError as error:
            detail = error.errors(include_input=False) if isinstance(error, ValidationError) else str(error)
            raise HTTPException(status_code=422, detail=detail) from error

    async def _observe(self, episode: _Episode, body: AgenticIFVerifyRequest, spec: RuntimeSpec) -> EpisodeRecord:
        """Read the bundle before the task backend touches the sandbox, then verify the task and persist both."""
        directory = self.config.artifacts_dir / episode.run_id
        task = episode.task
        collected, collection_error, root_export = None, None, None
        system, task_text = row_prompt_texts(
            body.responses_create_params.model_dump(mode="json", exclude_unset=True)["input"]
        )
        try:
            collected = await bundle.collect(task.sandbox.transport, expected=episode.seed["bundle"])
        # Deliberate boundary: an unreadable sandbox is this episode's recorded failure (session_lost), not a crash.
        except Exception as error:
            LOG.exception("Could not read the plugin bundle of agentic IF run %s", episode.run_id)
            collection_error = f"Could not read the plugin bundle: {type(error).__name__}: {error}"
            evidence = collection_failed(collection_error, episode.seed)
        else:
            evidence, root_export = await asyncio.to_thread(
                evidence_from_bundle,
                spec,
                collected,
                system=system,
                profile_sha256=episode.profile_sha256,
                seed=episode.seed,
                directory=directory,
                task_text=task_text,
            )
        backend = BackendVerdict(server=task.server_name)
        try:
            backend.result = await task.finish(
                body.response, params=body.responses_create_params.model_dump(mode="json", exclude_unset=True)
            )
        except Exception as error:
            LOG.exception("Task verification of agentic IF run %s failed", episode.run_id)
            backend.error = f"{type(error).__name__}: {error}"
        execution = execution_status(body.response, root_export, evidence)
        verdict = classify(
            response=body.response,
            root_export=root_export,
            evidence=evidence,
            execution=execution,
            backend_result=backend.result,
            backend_error=backend.error,
            collection_error=collection_error,
        )
        record = EpisodeRecord(
            run_id=episode.run_id,
            row=episode.content,
            evidence=evidence,
            execution_status=execution,
            backend=backend,
            verdict=PersistedVerdict.model_validate(asdict(verdict)),
            seed=episode.seed | {"profile_sha256": episode.profile_sha256},
            collection={"error": collection_error, "gaps": collected.gaps if collected is not None else []},
        )
        await asyncio.to_thread(save_json, directory / EPISODE_RECORD, record.model_dump(mode="json"))
        return record

    async def _verify_persisted(self, body: AgenticIFVerifyRequest) -> AgenticIFVerifyResponse:
        """Re-grade instruction following from a persisted episode; the task backend's verdict is kept."""
        root = self.config.artifacts_dir.resolve()
        directory = Path(body.deliverables_dir)
        directory = (directory if directory.is_absolute() else root / directory).resolve()
        if not directory.is_relative_to(root):
            raise HTTPException(422, "deliverables_dir must name an episode inside artifacts_dir")
        try:
            record = await asyncio.to_thread(load_episode_record, directory)
        except FileNotFoundError as error:
            raise HTTPException(404, f"No persisted agentic IF episode at {directory}") from error
        except (OSError, ValueError) as error:
            raise HTTPException(422, f"Unreadable agentic IF episode at {directory}: {error}") from error
        spec = self._verified_runtime(body, record.row, task_id=None)
        body.deliverables_dir = str(directory)
        return await self._grade_record(record, body, spec, directory, JudgeLedger())

    def _lost(self, body: AgenticIFVerifyRequest) -> AgenticIFVerifyResponse:
        """A verify with no live session and no persisted episode: nothing was measured."""
        content = {key: value for key, value in (body.model_extra or {}).items() if not key.startswith("_ng_")}
        spec = self._verified_runtime(body, content | {"agentic_if": body.agentic_if}, task_id=None)
        verdict = Verdict(
            0.0,
            True,
            False,
            SESSION_LOST,
            "No live agentic IF session for this verify and no persisted episode (deliverables_dir) to re-grade",
        )
        return AgenticIFVerifyResponse.model_validate(
            body.model_dump(mode="json") | {"agentic_if": spec.model_dump(mode="json")} | verdict.fields()
        )

    async def _grade_record(
        self,
        record: EpisodeRecord,
        body: AgenticIFVerifyRequest,
        spec: RuntimeSpec,
        directory: Path,
        ledger: JudgeLedger,
    ) -> AgenticIFVerifyResponse:
        """Grade the persisted evidence, and assemble the result around the persisted task verdict.

        ``ledger`` records the grade's judge calls, for the caller to keep even when grading raises.
        """
        if_result, judge_hash = await self._grade(spec.checks, record.evidence, ledger)
        outage = ledger.outage(if_result, judge_hash)
        if outage is not None and not record.verdict.mask_sample:
            # No attempt reached the judge: judge_failsafe records `judge_failed`, recoverable by re-verification.
            raise JudgeError(f"The judge could not be reached: {outage}")
        calls = ledger.calls()
        payload = body.model_dump(mode="json")
        payload |= {
            key: value
            for key, value in (record.backend.result or {}).items()
            if key not in ("responses_create_params", "response")
        }
        payload |= {
            "agentic_if": spec.model_dump(mode="json"),
            "if_result": if_result.model_dump(mode="json"),
            "execution_status": record.execution_status,
            "evidence": record.evidence.model_dump(mode="json"),
            "deliverables_dir": str(directory),
            "judge_calls": [call.summary().model_dump(mode="json") for call in calls],
            "judge_usage": judge_usage(calls).model_dump(mode="json"),
        }
        payload |= Verdict(**record.verdict.model_dump()).fields()
        return AgenticIFVerifyResponse.model_validate(payload)

    # ------------------------------------------------------------------------------------------------------ grading

    async def _grade(
        self, checks: list[ResolvedCheck], evidence: SessionEvidence, ledger: JudgeLedger
    ) -> tuple[IFResult, str]:
        judge = Judge(partial(self._ask_judge, asyncio.get_running_loop(), ledger))
        grading = asyncio.create_task(self._grade_in_slot(checks, evidence, judge))
        self._grade_jobs.add(grading)
        grading.add_done_callback(self._graded)
        # Cancelling the caller does not cancel grading: the job owns its slot until the judge finishes.
        return await asyncio.shield(grading), judge.hash

    async def _grade_in_slot(self, checks: list[ResolvedCheck], evidence: SessionEvidence, judge: Judge) -> IFResult:
        # The job, rather than its HTTP waiter, owns this slot. Cancelling a waiter cannot free capacity while
        # the judge is still busy.
        async with self._grading_slots:
            return await asyncio.to_thread(grade, checks, evidence, judge=judge)

    def _ask_judge(
        self, loop: asyncio.AbstractEventLoop, ledger: JudgeLedger, body: dict[str, JsonValue]
    ) -> NeMoGymResponse:
        # Grading runs on a worker thread; the request runs on the server's event loop, where Gym's HTTP client lives.
        prompt = judge_prompt(body)
        request = call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=body,
            response_model=NeMoGymResponse,
        )
        try:
            # The judge is the verifier's model, not the agent's. Out of the rollout's context, its calls stay out of
            # the rollout's model-call capture, where every call needs an agent invocation owner.
            with rollout_context(None):
                reply = asyncio.run_coroutine_threadsafe(request, loop).result()
        except JudgeError as error:
            ledger.failed(prompt, error)
            raise
        ledger.replied(prompt, reply)
        return reply

    def _graded(self, task: asyncio.Task[IFResult]) -> None:
        self._grade_jobs.discard(task)
        if not task.cancelled() and task.exception() is not None:
            LOG.error("IF grading job failed", exc_info=task.exception())

    # ------------------------------------------------------------------------------------------------------ metrics

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Instruction-following rates over scored rollouts, beside the task reward Gym already reports.

        A rollout whose applicable checks all passed counts as an IF pass. Rollouts with no applicable check, or
        with a check that could not be graded, are counted separately and excluded from the rates.
        """
        rollouts = [rollout for task in tasks for rollout in task]
        graded, not_applicable, errored = [], 0, 0
        for rollout in rollouts:
            if_result = rollout.get("if_result") or {}
            statuses = [check.get("status") for check in if_result.get("checks") or []]
            if if_result.get("if_all") is not None:
                graded.append(rollout)
            elif "error" in statuses:
                errored += 1
            else:
                not_applicable += 1
        metrics: dict[str, Any] = {
            "if/graded": len(graded),
            "if/not_applicable": not_applicable,
            "if/error": errored,
        }
        if graded:
            metrics["mean/if_all"] = sum(rollout["if_result"]["if_all"] for rollout in graded) / len(graded)
            metrics["mean/if_frac"] = sum(rollout["if_result"]["if_frac"] for rollout in graded) / len(graded)
            metrics["mean/task_and_if"] = sum(
                1.0 for rollout in graded if rollout["if_result"]["if_all"] == 1 and rollout.get("reward", 0) >= 1
            ) / len(graded)
        return metrics


# Full, zero and malformed cases on the session-less path, from fixed persisted episodes; no services start.
VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=AgenticIFVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_fixture" / "cases.jsonl",
    invoke=invoke_fixture,
)


if __name__ == "__main__":
    AgenticIFResourcesServer.run_webserver()
