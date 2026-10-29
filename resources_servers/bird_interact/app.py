# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD-Interact episode state: the database sandbox, the user simulator, the grader and the budgets.

One session holds one task. Its database lives in a sandbox that the session starts at seed and stops at close; the
agent's SQL and the grading run there (``grading.py``). In c-Interact the Environment Server calls the routes under
``/c_interact/``. In a-Interact the agent calls the same actions as tools, at the server's root or over MCP, and each
one costs BIRD-COIN. The agent never gets the sandbox itself. Each mode's routes refuse a session of the other mode,
and an MCP token lists the tools only for an a-Interact session.
"""

import asyncio
import json
import logging
import posixpath
import shlex
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, PrivateAttr

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from resources_servers.bird_interact.grading import MAX_PREVIEW_ROWS
from resources_servers.bird_interact.schemas import (
    A_INTERACT_TOOLS,
    AskRequest,
    AskResponse,
    BirdInteractSeedSessionResponse,
    BirdInteractVerifyRequest,
    BirdInteractVerifyResponse,
    ColumnMeaningRequest,
    EpisodeContext,
    InteractionMode,
    KnowledgeRequest,
    PhaseResult,
    SQLRequest,
    SubmitRequest,
    SubmitResponse,
    ToolObservation,
)
from resources_servers.bird_interact.simulator import (
    UNANSWERABLE_REPLY,
    classifier_prompt,
    parse_action,
    responder_prompt,
)
from resources_servers.bird_interact.task_data import BirdInteractTask


if TYPE_CHECKING:
    from nemo_gym.mcp_auto_exposure import MCPTool


LOGGER = logging.getLogger(__name__)
USER_SIMULATOR_FAILED = "bird_interact:user_simulator_failed"
SANDBOX_FAILED = "bird_interact:sandbox_failed"
# Runs inside the session sandbox; see its module docstring.
_SANDBOX_RUNNER = Path(__file__).with_name("grading.py")
# Time for a sandbox command beyond its two queries: the scratch copies and the round trip.
_SANDBOX_COMMAND_SLACK_SECONDS = 60.0
# BIRD-Interact's a-Interact action costs, in BIRD-COIN.
DEFAULT_ACTION_COSTS = {
    "ask": 2.0,
    "submit": 3.0,
    "execute": 1.0,
    "get_schema": 1.0,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
}
# The a-Interact budget before the per-ambiguity and patience parts: three for exploration, three for one submit.
_BASE_COIN_BUDGET = 6.0


class PhaseReward(BaseModel):
    first_attempt: float
    after_debug: float


class BirdInteractSandboxConfig(BaseModel):
    """The session sandbox. The image needs ``python3``; the database and the runner are uploaded at seed."""

    image: str = "python:3.12-slim"
    # The provider reclaims a sandbox that a crashed server never stopped.
    ttl_s: PositiveFloat | None = 7200.0
    ready_timeout_s: PositiveFloat | None = None
    resources: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)


class SandboxCommandError(HTTPException, RuntimeError):
    """The session sandbox could not run a command: an infrastructure failure, not the agent's mistake.

    FastAPI answers it with HTTP 503, which an Environment Server treats as retryable.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(503, detail[:2000])


class BirdInteractResourcesServerConfig(BaseResourcesServerConfig):
    """Data location, simulator model and the protocol's budgets and rewards."""

    database_dir: str = Field(
        description="Holds <db>/<db>.sqlite (or <db>.sql), <db>_schema.txt, <db>_column_meaning_base.json, <db>_kb.jsonl."
    )
    user_model: ModelServerRef
    sandbox_provider: str | dict[str, Any] = Field(
        description="A sandbox provider block name, such as `sandbox`, or an inline mapping like `{local: {}}`."
    )
    sandbox_config: BirdInteractSandboxConfig = Field(default_factory=BirdInteractSandboxConfig)
    # A command that fails for infrastructure reasons runs again on a fresh sandbox this many times.
    inconclusive_verification_retries: NonNegativeInt = 1
    sql_timeout_seconds: PositiveFloat = 30.0
    # Sandbox commands (queries and grading) running at once across sessions.
    max_concurrent_queries: PositiveInt = 16
    # c-Interact: clarifying questions beyond one for each ambiguity.
    question_patience: NonNegativeInt = 3
    # a-Interact: BIRD-COIN beyond the base budget and the cost of one question for each ambiguity.
    coin_patience: NonNegativeFloat = 6.0
    action_costs: dict[str, NonNegativeFloat] = Field(default_factory=lambda: dict(DEFAULT_ACTION_COSTS))
    # One entry for each phase. A follow-up question is asked only when the list has a second entry.
    phase_rewards: list[PhaseReward] = Field(
        default_factory=lambda: [
            PhaseReward(first_attempt=0.7, after_debug=0.5),
            PhaseReward(first_attempt=0.3, after_debug=0.2),
        ],
        min_length=1,
    )


@dataclass
class _DatabaseFiles:
    schema_text: str
    column_meanings: dict[str, str]
    knowledge: dict[int, dict]


@dataclass
class _SessionSandbox:
    sandbox: AsyncSandbox
    # The directory in the sandbox that holds the runner, the database source and the episode database.
    root: str = ""

    def path(self, name: str) -> str:
        return posixpath.join(self.root, name)


@dataclass
class _Session:
    episode_id: EpisodeId
    task_id: TaskId
    task: BirdInteractTask
    mode: InteractionMode
    sandbox: _SessionSandbox
    files: _DatabaseFiles
    coin_budget: float | None
    phase: int = 1
    attempts: int = 0
    questions_asked: int = 0
    coins_spent: float = 0.0
    phases: list[PhaseResult] = field(default_factory=list)
    finished: bool = False
    # Set when an infrastructure failure disrupted an a-Interact episode, so verify masks the sample.
    failure_kind: str | None = None
    failure_reason: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def coins_left(self) -> float:
        return (self.coin_budget or 0.0) - self.coins_spent

    def visible_knowledge(self) -> list[dict]:
        hidden = {point.deleted_knowledge for point in self.task.knowledge_ambiguity}
        return [entry for key, entry in self.files.knowledge.items() if key not in hidden]


class BirdInteractResourcesServer(SimpleResourcesServer):
    config: BirdInteractResourcesServerConfig
    _sessions: dict[str, _Session] = PrivateAttr(default_factory=dict)
    _files: dict[str, _DatabaseFiles] = PrivateAttr(default_factory=dict)
    _fence: ResourcesSessionFence = PrivateAttr()
    _queries: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)
        self._queries = asyncio.Semaphore(self.config.max_concurrent_queries)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self._stop_all_sandboxes()

        app.router.lifespan_context = lifespan
        app.post("/c_interact/ask")(self.ask)
        app.post("/c_interact/submit")(self.submit)
        tools = {
            "ask": self.tool_ask,
            "submit": self.tool_submit,
            "execute": self.tool_execute,
            "get_schema": self.tool_get_schema,
            "get_all_column_meanings": self.tool_get_all_column_meanings,
            "get_column_meaning": self.tool_get_column_meaning,
            "get_all_external_knowledge_names": self.tool_get_all_external_knowledge_names,
            "get_knowledge_definition": self.tool_get_knowledge_definition,
            "get_all_knowledge_definitions": self.tool_get_all_knowledge_definitions,
        }
        # MCP auto-exposure describes each tool with its route's description.
        for name, (description, _) in A_INTERACT_TOOLS.items():
            app.post(f"/{name}", description=description)(tools[name])
        return app

    def mcp_tools(self, harvested: list["MCPTool"], catchall: Any) -> list["MCPTool"]:
        """The a-Interact tools only; the ``/c_interact/`` routes are the Environment Server's."""
        return [tool for tool in harvested if tool.name in A_INTERACT_TOOLS]

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> list[str]:
        """Every tool for an a-Interact session; none for a c-Interact session, whose agent has no tools."""
        return list(A_INTERACT_TOOLS) if seed_body.get("interaction_mode") == "a_interact" else []

    # ----- session lifecycle ----- #

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> BirdInteractSeedSessionResponse:
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            existing = self._sessions.get(session_id)
            if existing is not None:
                if (existing.episode_id, existing.task_id) != (body.episode_id, body.task_id):
                    raise ResourcesSessionConflictError(
                        "resources_session_id is already bound to another episode or task"
                    )
            else:
                self._sessions[session_id] = await self._open_session(body)
            request.session[SESSION_ID_KEY] = session_id
            context = self._episode_context(self._sessions[session_id])
        return BirdInteractSeedSessionResponse(resources_session_id=session_id, context=context)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            closed_episode = self._fence.closed_episode(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            session = self._sessions.get(session_id)
            if session is not None and session.episode_id != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            if session is not None:
                # A failed stop keeps the session, so the next close retries it.
                await session.sandbox.sandbox.stop()
                self._sessions.pop(session_id, None)
            self._fence.mark_closed(session_id, body.episode_id)
            request.session.pop(SESSION_ID_KEY, None)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _open_session(self, body: ResourcesSeedSessionRequest) -> _Session:
        task_data = dict(body.task_data)
        mode = task_data.pop("interaction_mode", None)
        if mode not in ("c_interact", "a_interact"):
            raise ResourcesSeedRejectedError("task_data.interaction_mode must be 'c_interact' or 'a_interact'")
        try:
            task = BirdInteractTask.model_validate(task_data)
        except ValueError as error:
            raise ResourcesSeedRejectedError(f"task_data is not a BIRD-Interact task: {error}") from error
        if not task.sol_sql:
            raise ResourcesSeedRejectedError(
                f"Task {task.instance_id} has no sol_sql. The public BIRD-Interact releases withhold the ground truth; "
                "merge it into the rows before running."
            )
        if task.test_cases:
            raise ResourcesSeedRejectedError(
                f"Task {task.instance_id} has custom test cases, which this server does not run yet."
            )
        files = self._database_files(task.selected_database)
        self._database_source(task.selected_database)
        try:
            sandbox = await self._start_sandbox(task)
        except Exception as error:
            raise SandboxCommandError(
                f"Could not start the session sandbox: {type(error).__name__}: {error}"
            ) from error
        coin_budget = None
        if mode == "a_interact":
            coin_budget = (
                _BASE_COIN_BUDGET + self.config.action_costs["ask"] * task.ambiguity_count + self.config.coin_patience
            )
        return _Session(
            episode_id=body.episode_id,
            task_id=body.task_id,
            task=task,
            mode=mode,
            sandbox=sandbox,
            files=files,
            coin_budget=coin_budget,
        )

    def _database_source(self, name: str) -> Path:
        """The database file, ``<db>.sqlite``, or else the script ``<db>.sql``."""
        directory = self._database_root() / name
        for source in (directory / f"{name}.sqlite", directory / f"{name}.sql"):
            if source.exists():
                return source
        raise ResourcesSeedRejectedError(f"No database for {name!r} in {directory}")

    # ----- session sandbox ----- #

    async def _start_sandbox(self, task: BirdInteractTask) -> _SessionSandbox:
        """Start a sandbox holding the task's database with ``preprocess_sql`` applied. A failed setup stops it."""
        global_config = self.server_client.global_config_dict
        sandbox_config = self.config.sandbox_config
        spec = SandboxSpec(
            image=sandbox_config.image,
            ttl_s=sandbox_config.ttl_s,
            ready_timeout_s=sandbox_config.ready_timeout_s,
            metadata=resolve_provider_metadata(self.config.sandbox_provider, global_config)
            | sandbox_config.metadata
            | {"nemo_gym_resources": self.config.name[:63], "instance_id": task.instance_id[:63]},
            resources=sandbox_config.resources,
            provider_options=dict(sandbox_config.provider_options),
        )
        source = self._database_source(task.selected_database)
        handle = _SessionSandbox(AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, global_config)))

        async def setup(sandbox: AsyncSandbox) -> None:
            # The sandbox's working directory: the container's own, or a fresh host directory for the local provider.
            result = await sandbox.exec("pwd", timeout_s=60)
            if result.error_type is not None or result.return_code != 0 or not (result.stdout or "").strip():
                raise SandboxCommandError(f"pwd failed in the sandbox: {result.error_type or result.return_code}")
            handle.root = posixpath.join((result.stdout or "").strip(), "bird_interact")
            await sandbox.upload(_SANDBOX_RUNNER, handle.path(_SANDBOX_RUNNER.name))
            await sandbox.upload(source, handle.path(source.name))
            await self._run_command(
                handle, "setup", {"source": handle.path(source.name), "preprocess_sql": task.preprocess_sql}
            )

        await handle.sandbox.start_with_setup(spec, setup)
        return handle

    async def _run_command(self, handle: _SessionSandbox, command: str, request: dict[str, Any]) -> dict[str, Any]:
        """Run ``grading.py <command>`` in the sandbox and return its JSON result."""
        name = f"request-{uuid4().hex}.json"
        try:
            with tempfile.TemporaryDirectory(prefix="bird-interact-") as scratch:
                local = Path(scratch) / name
                # Text goes in a file, not in the command line.
                local.write_text(json.dumps(request | {"database": handle.path("episode.sqlite")}))
                await handle.sandbox.upload(local, handle.path(name))
            result = await handle.sandbox.exec(
                shlex.join(["python3", handle.path(_SANDBOX_RUNNER.name), command, handle.path(name)]),
                timeout_s=2 * self.config.sql_timeout_seconds + _SANDBOX_COMMAND_SLACK_SECONDS,
            )
        except Exception as error:
            raise SandboxCommandError(f"grading.py {command}: {type(error).__name__}: {error}") from error
        if result.error_type is not None or result.return_code != 0:
            outcome = result.error_type or f"exit code {result.return_code}"
            raise SandboxCommandError(f"grading.py {command} failed in the sandbox ({outcome}): {result.stderr or ''}")
        try:
            return json.loads(result.stdout or "")
        except json.JSONDecodeError as error:
            raise SandboxCommandError(f"grading.py {command} printed no JSON result: {result.stdout!r}") from error

    async def _in_sandbox(self, session: _Session, command: str, request: dict[str, Any]) -> dict[str, Any]:
        """Run a command in the session sandbox, retrying an infrastructure failure on a fresh sandbox.

        Only setup writes the session database, so a fresh sandbox holds the same state.
        """
        async with self._queries:
            for _ in range(self.config.inconclusive_verification_retries):
                try:
                    return await self._run_command(session.sandbox, command, request)
                except SandboxCommandError as error:
                    LOGGER.warning(f"Retrying {command} on a fresh sandbox for {session.task.instance_id}: {error}")
                    await self._replace_sandbox(session)
            return await self._run_command(session.sandbox, command, request)

    async def _replace_sandbox(self, session: _Session) -> None:
        try:
            fresh = await self._start_sandbox(session.task)
        except Exception as error:
            raise SandboxCommandError(f"Could not start a fresh sandbox: {type(error).__name__}: {error}") from error
        stale, session.sandbox = session.sandbox, fresh
        try:
            await stale.sandbox.stop()
        except Exception:
            LOGGER.exception("Could not stop a failed BIRD-Interact sandbox; the provider's ttl_s reclaims it")

    async def _stop_all_sandboxes(self) -> None:
        sessions = list(self._sessions.values())
        results = await asyncio.gather(
            *(session.sandbox.sandbox.stop() for session in sessions), return_exceptions=True
        )
        for session, result in zip(sessions, results):
            if isinstance(result, BaseException):
                LOGGER.error(f"Could not stop the sandbox of {session.task.instance_id} at shutdown: {result!r}")

    def _database_files(self, name: str) -> _DatabaseFiles:
        if name not in self._files:
            directory = self._database_root() / name
            try:
                knowledge_lines = (directory / f"{name}_kb.jsonl").read_text().splitlines()
                self._files[name] = _DatabaseFiles(
                    schema_text=(directory / f"{name}_schema.txt").read_text(),
                    column_meanings=json.loads((directory / f"{name}_column_meaning_base.json").read_text()),
                    knowledge={entry["id"]: entry for entry in map(json.loads, filter(str.strip, knowledge_lines))},
                )
            except FileNotFoundError as error:
                raise ResourcesSeedRejectedError(f"Missing database file for {name!r}: {error.filename}") from error
        return self._files[name]

    def _database_root(self) -> Path:
        """``database_dir``; a relative path is relative to the repository root, not the server's directory."""
        root = Path(self.config.database_dir)
        return root if root.is_absolute() else PARENT_DIR / root

    def _session(self, request: Request) -> _Session:
        session = self._sessions.get(request.session.get(SESSION_ID_KEY))
        if session is None:
            raise ResourcesSessionLostError("No BIRD-Interact session; seed_session must be called first")
        return session

    def _c_interact_session(self, request: Request) -> _Session:
        session = self._session(request)
        if session.mode != "c_interact":
            raise ResourcesSessionConflictError("c-Interact routes are not available in an a-Interact session")
        return session

    def _episode_context(self, session: _Session) -> EpisodeContext:
        if session.mode == "a_interact":
            return EpisodeContext(
                database=session.task.selected_database,
                question=session.task.amb_user_query,
                coin_budget=session.coin_budget,
            )
        return EpisodeContext(
            database=session.task.selected_database,
            question=session.task.amb_user_query,
            schema_text=session.files.schema_text,
            column_meanings=json.dumps(session.files.column_meanings, indent=1),
            knowledge="\n".join(json.dumps(entry) for entry in session.visible_knowledge()),
            question_budget=session.task.ambiguity_count + self.config.question_patience,
        )

    # ----- routes for the Environment Server ----- #

    async def ask(self, request: Request, body: AskRequest) -> AskResponse:
        """Answer a c-Interact clarifying question: in phase 1, before the first submission, within the budget."""
        session = self._c_interact_session(request)
        async with session.lock:
            budget = session.task.ambiguity_count + self.config.question_patience
            # As in BIRD-Interact, the debug turn after a failed submission is one more SQL attempt, not a question.
            if session.finished or session.phase != 1 or session.attempts or session.questions_asked >= budget:
                return AskResponse(reply="No more questions, please. Send me your SQL.", answered=False)
            session.questions_asked += 1
            reply = await self._simulate(session, body.question)
            questions_left = budget - session.questions_asked
            if questions_left == 0:
                reply += "\n\nThat was your last question. Send me your SQL next."
            return AskResponse(reply=reply, questions_left=questions_left)

    async def submit(self, request: Request, body: SubmitRequest) -> SubmitResponse:
        session = self._c_interact_session(request)
        async with session.lock:
            return await self._submit(session, body.sql)

    async def verify(self, request: Request, body: BirdInteractVerifyRequest) -> BirdInteractVerifyResponse:
        session = self._sessions.get(request.session.get(SESSION_ID_KEY))
        if session is None:
            return BirdInteractVerifyResponse(
                **(body.model_dump() | lost_session_verification("No BIRD-Interact session to verify")),
                phases=[],
                questions_asked=0,
                coins_spent=0.0,
                finished=False,
            )
        phases = list(session.phases)
        # The current phase has submissions but no result yet, or the budget ended the task after a failed one.
        if session.attempts and all(phase.phase != session.phase for phase in phases):
            phases.append(PhaseResult(phase=session.phase, passed=False, attempts=session.attempts, reward=0.0))
        fields = {}
        if session.failure_kind is not None:
            fields = {"mask_sample": True, "failure_kind": session.failure_kind}
            fields["failure_reason"] = session.failure_reason
        return BirdInteractVerifyResponse(
            **body.model_dump(),
            reward=sum(phase.reward for phase in phases),
            phases=phases,
            questions_asked=session.questions_asked,
            coins_spent=session.coins_spent,
            finished=session.finished,
            **fields,
        )

    # ----- a-Interact tools ----- #

    async def tool_ask(self, request: Request, body: AskRequest) -> ToolObservation:
        async def act(session: _Session) -> str:
            session.questions_asked += 1
            try:
                return await self._simulate(session, body.question)
            except Exception as error:
                _record_infrastructure_failure(session, USER_SIMULATOR_FAILED, "User simulator failed", error)
                raise

        return await self._tool(request, "ask", act)

    async def tool_submit(self, request: Request, body: SQLRequest) -> ToolObservation:
        async def act(session: _Session) -> str:
            try:
                result = await self._submit(session, body.sql)
            except SandboxCommandError as error:
                _record_infrastructure_failure(session, SANDBOX_FAILED, "Sandbox failed", error)
                raise
            return result.feedback

        return await self._tool(request, "submit", act)

    async def tool_execute(self, request: Request, body: SQLRequest) -> ToolObservation:
        async def act(session: _Session) -> str:
            request = {"sql": body.sql, "timeout_seconds": self.config.sql_timeout_seconds}
            try:
                result = await self._in_sandbox(session, "execute", request)
            except SandboxCommandError as error:
                _record_infrastructure_failure(session, SANDBOX_FAILED, "Sandbox failed", error)
                raise
            if result["error"] is not None:
                return f"Error: {result['error']}"
            count = result["row_count"]
            shown = "\n".join(result["preview"])
            more = f"\n... {count - MAX_PREVIEW_ROWS} more rows" if count > MAX_PREVIEW_ROWS else ""
            return f"{count} rows\n{shown}{more}"

        return await self._tool(request, "execute", act)

    async def tool_get_schema(self, request: Request) -> ToolObservation:
        async def act(session: _Session) -> str:
            return session.files.schema_text

        return await self._tool(request, "get_schema", act)

    async def tool_get_all_column_meanings(self, request: Request) -> ToolObservation:
        async def act(session: _Session) -> str:
            return json.dumps(session.files.column_meanings, indent=1)

        return await self._tool(request, "get_all_column_meanings", act)

    async def tool_get_column_meaning(self, request: Request, body: ColumnMeaningRequest) -> ToolObservation:
        async def act(session: _Session) -> str:
            wanted = f"{session.task.selected_database}|{body.table_name}|{body.column_name}".casefold()
            meaning = next(
                (text for key, text in session.files.column_meanings.items() if key.casefold() == wanted), None
            )
            return meaning or f"No column {body.column_name!r} in table {body.table_name!r}."

        return await self._tool(request, "get_column_meaning", act)

    async def tool_get_all_external_knowledge_names(self, request: Request) -> ToolObservation:
        async def act(session: _Session) -> str:
            return json.dumps([entry["knowledge"] for entry in session.visible_knowledge()])

        return await self._tool(request, "get_all_external_knowledge_names", act)

    async def tool_get_knowledge_definition(self, request: Request, body: KnowledgeRequest) -> ToolObservation:
        async def act(session: _Session) -> str:
            wanted = body.knowledge_name.casefold()
            entry = next((e for e in session.visible_knowledge() if e["knowledge"].casefold() == wanted), None)
            return json.dumps(entry) if entry else f"No knowledge named {body.knowledge_name!r}."

        return await self._tool(request, "get_knowledge_definition", act)

    async def tool_get_all_knowledge_definitions(self, request: Request) -> ToolObservation:
        async def act(session: _Session) -> str:
            return "\n".join(json.dumps(entry) for entry in session.visible_knowledge())

        return await self._tool(request, "get_all_knowledge_definitions", act)

    async def _tool(self, request: Request, action: str, act: Callable[[_Session], Awaitable[str]]) -> ToolObservation:
        """Charge the action's cost and run it; once the budget is spent, only ``submit`` runs."""
        session = self._session(request)
        async with session.lock:
            if session.mode != "a_interact":
                raise ResourcesSessionConflictError("a-Interact tools are not available in a c-Interact session")
            if session.finished:
                return ToolObservation(observation="The task is finished.", coins_left=session.coins_left)
            if session.coins_left <= 0 and action != "submit":
                return ToolObservation(
                    observation="Your budget is spent. Only submit is allowed now.", coins_left=session.coins_left
                )
            session.coins_spent += self.config.action_costs[action]
            observation = await act(session)
            if session.coins_left <= 0 and action == "submit" and not session.finished:
                session.finished = True
                observation += "\nYour budget is spent, so the task ends here."
            return ToolObservation(observation=observation, coins_left=session.coins_left)

    # ----- shared actions ----- #

    async def _simulate(self, session: _Session, question: str) -> str:
        classification = await self._user_model(classifier_prompt(session.task, question, phase=session.phase))
        action = parse_action(classification, session.task)
        if action.kind == "unanswerable":
            return UNANSWERABLE_REPLY
        prompt = responder_prompt(
            session.task,
            question,
            action,
            phase=session.phase,
            schema_text=session.files.schema_text,
            knowledge=session.files.knowledge,
        )
        return (await self._user_model(prompt)).strip()

    async def _user_model(self, prompt: str) -> str:
        response = await self.server_client.post(
            server_name=self.config.user_model.name,
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": prompt}]),
        )
        await raise_for_status(response)
        return NeMoGymResponse.model_validate(await get_response_json(response)).output_text

    async def _submit(self, session: _Session, sql: str) -> SubmitResponse:
        if session.finished:
            return SubmitResponse(passed=False, feedback="The task is finished.", finished=True, phase=session.phase)
        task, phase = session.task, session.phase
        solution, conditions = (
            (task.sol_sql, task.conditions) if phase == 1 else (task.follow_up.sol_sql, task.follow_up.conditions)
        )
        session.attempts += 1
        result = await self._in_sandbox(
            session,
            "grade",
            {
                "predicted_sql": sql,
                "solution_sql": ";\n".join(solution),
                "ordered": conditions.order,
                "timeout_seconds": self.config.sql_timeout_seconds,
            },
        )
        if "solution_error" in result:
            raise RuntimeError(f"Task {task.instance_id}: {result['solution_error']}")
        rewards = self.config.phase_rewards[phase - 1]
        if result["passed"]:
            debugged = session.mode == "c_interact" and session.attempts > 1
            reward = rewards.after_debug if debugged else rewards.first_attempt
            session.phases.append(PhaseResult(phase=phase, passed=True, attempts=session.attempts, reward=reward))
            if phase == 1 and task.follow_up is not None and len(self.config.phase_rewards) > 1:
                session.phase, session.attempts = 2, 0
                feedback = f"That's right, thank you. Here is my next question: {task.follow_up.query}"
                return SubmitResponse(passed=True, feedback=feedback, finished=False, phase=phase)
            session.finished = True
            return SubmitResponse(
                passed=True, feedback="That's right, thank you. That is all I needed.", finished=True, phase=phase
            )
        if session.mode == "c_interact" and session.attempts >= 2:
            session.phases.append(PhaseResult(phase=phase, passed=False, attempts=session.attempts, reward=0.0))
            session.finished = True
            return SubmitResponse(
                passed=False, feedback="That is still not right. Let's stop here.", finished=True, phase=phase
            )
        if result["execution_error"] is not None:
            feedback = f"Your SQL failed with this error: {result['execution_error']}"
        else:
            feedback = "Your SQL runs, but the result is not what I need."
        feedback += (
            " Send your corrected SQL; you have one more attempt."
            if session.mode == "c_interact"
            else " You can try again."
        )
        return SubmitResponse(passed=False, feedback=feedback, finished=False, phase=phase)


def _record_infrastructure_failure(session: _Session, kind: str, what: str, error: BaseException) -> None:
    """An infrastructure failure must not read as the agent's mistake, so verify masks the sample."""
    session.failure_kind = kind
    session.failure_reason = f"{what}: {type(error).__name__}: {error}"[:2000]


if __name__ == "__main__":
    BirdInteractResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BirdInteractResourcesServer.run_webserver()  # noqa: F401
