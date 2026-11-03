# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD-Interact in one Environment Server: the database sandbox, the user simulator, the grader and the turns.

Each episode's database lives in its own sandbox, which the episode starts and stops. The agent's SQL and the grading
run there (``grading.py``); the agent never gets the sandbox itself. The agent reaches the user and the grader through
tools that this server serves for the episode, over direct HTTP, MCP or both.

c-Interact follows the BIRD-Interact ADK: the agent has ``ask_user`` and ``submit_sql``, and the server opens each of
its turns as the ADK's orchestrator does: phase 1, a debug turn when phase 1 fails, the follow-up question as phase 2,
and a debug turn when phase 2 fails. Each turn ends at its submission. a-Interact gives the agent one turn and nine
tools, each charged in BIRD-COIN.
"""

import asyncio
import json
import logging
import posixpath
import shlex
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    ValidationError,
    model_validator,
)
from typing_extensions import Self

from environment_servers.bird_interact.adk_prompts import (
    ASK_USER_DESCRIPTION,
    C_INTERACT_INSTRUCTION,
    CLARIFICATIONS_LEFT_NOTE,
    DEBUG_EXECUTION_ERROR_MESSAGE,
    DEBUG_WRONG_RESULT_MESSAGE,
    EXECUTION_ERROR,
    EXECUTION_ERROR_FLAG,
    EXECUTION_TIMEOUT,
    FOLLOW_UP_MESSAGE,
    KNOWLEDGE_VISIBLE_FIELDS,
    PHASE_1_MESSAGE,
    PHASE_1_PASSED,
    PHASE_1_PASSED_WITH_FOLLOW_UP,
    PHASE_2_PASSED,
    PHASE_FAILED,
    SUBMISSION_BUDGET_LINE,
    SUBMISSION_FOLLOW_UP_LINE,
    SUBMISSION_REWARD_LINE,
    SUBMIT_SQL_DESCRIPTION,
    SUBMITTED_REPLY,
    TASK_COMPLETED_REPLY,
    WRONG_RESULT,
)
from environment_servers.bird_interact.agent_session import AgentSession, TurnDeadlineExpired, TurnRecord
from environment_servers.bird_interact.grading import MAX_PREVIEW_ROWS
from environment_servers.bird_interact.schemas import (
    AskArguments,
    ColumnMeaningArguments,
    InteractionMode,
    KnowledgeArguments,
    NoArguments,
    PhaseResult,
    SQLArguments,
    ToolObservation,
)
from environment_servers.bird_interact.simulator import (
    ACTION_PARSER_MAX_TOKENS,
    RESPONSE_GENERATOR_MAX_TOKENS,
    action_parser_prompt,
    read_action,
    read_reply,
    response_generator_prompt,
)
from environment_servers.bird_interact.task_data import BirdInteractTask, custom_test_cases
from nemo_gym import PARENT_DIR
from nemo_gym.agents import AgentOutcomePolicyConfig, OutcomeAction, agent_outcome_policy
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AgentServerRef, AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.core import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
    FailureStage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
)
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.server_runtime import (
    BaseEnvironmentServerWithTools,
    BaseEnvironmentServerWithToolsConfig,
    CleanupContext,
    EpisodeToolset,
    ToolArgumentsError,
    ToolCallRecord,
    ToolResult,
    ToolSpec,
    TypedToolset,
    dependency_failure,
    episode_tool,
    handled_episode_failure,
    is_handled_episode_failure,
)
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


LOGGER = logging.getLogger(__name__)
USER_SIMULATOR_FAILED = "bird_interact:user_simulator_failed"
SANDBOX_FAILED = "bird_interact:sandbox_failed"
# Runs inside the episode's sandbox; see its module docstring.
_SANDBOX_RUNNER = Path(__file__).with_name("grading.py")
# Time for a sandbox command beyond its queries: the scratch copies and the round trip.
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
# The ADK's c-Interact agent has no BIRD-COIN budget, so the reply to its submission reads 0.
_C_INTERACT_BUDGET = 0
_SQLITE_IMAGE = "python:3.12-slim"
DatabaseEngine = Literal["sqlite", "postgresql"]

A_INTERACT_INSTRUCTIONS = """You are a data analyst who writes {dialect} queries for a user. The user's request can be \
ambiguous, and part of the knowledge you need can be missing from the knowledge base.

Use the tools to read the schema, column meanings and knowledge base, to run exploratory queries, to ask the user \
clarifying questions, and to submit your final SQL. Each tool call costs BIRD-COIN; your budget is {coin_budget}. \
Each tool result shows the coins you have left. When they run out, you can only submit.

The submit tool tells you whether your SQL is right. If the user then asks a follow-up question, solve it and submit \
again. Stop calling tools when the task is finished."""


class PhaseReward(BaseModel):
    first_attempt: float
    after_debug: float


class BirdInteractSandboxConfig(BaseModel):
    """The episode's sandbox. The runner is uploaded at the start.

    For SQLite the image needs only ``python3`` (default ``python:3.12-slim``), and the database is uploaded too. For
    PostgreSQL the image holds the release's databases as ``<db>_template``; ``benchmarks/bird_interact`` builds it.
    """

    image: str | None = None
    # The provider reclaims a sandbox that a crashed server never stopped.
    ttl_s: PositiveFloat | None = 7200.0
    ready_timeout_s: PositiveFloat | None = None
    resources: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)


class BirdInteractEnvironmentServerConfig(BaseEnvironmentServerWithToolsConfig):
    """The agent under test, the interaction mode, the data location, the simulator model, and the protocol's
    budgets and rewards."""

    agent_server: AgentServerRef
    turn_timeout_seconds: PositiveFloat
    interaction_timeout_seconds: PositiveFloat | None = Field(
        default=None,
        description="Bound on all turns together. When it expires, the agent is closed and the episode is graded.",
    )
    agent_outcome_policy: AgentOutcomePolicyConfig = Field(default_factory=AgentOutcomePolicyConfig)
    mode: InteractionMode
    database_engine: DatabaseEngine = Field(
        default="sqlite",
        description=(
            "sqlite: a task's database is <db>/<db>.sqlite (or <db>.sql) under database_dir, as in Mini-Interact. "
            "postgresql: it is <db>_template in the server of the sandbox image, as in Lite and Full."
        ),
    )
    database_dir: str = Field(
        description=(
            "Holds <db>/<db>_schema.txt, <db>_column_meaning_base.json and <db>_kb.jsonl for each database, and for "
            "sqlite the database itself."
        )
    )
    user_model: ModelServerRef
    sandbox_config: BirdInteractSandboxConfig = Field(default_factory=BirdInteractSandboxConfig)
    # A command that fails for infrastructure reasons runs again on a fresh sandbox this many times.
    inconclusive_verification_retries: NonNegativeInt = 1
    # Bounds each statement, as the ADK's 60 seconds do.
    sql_timeout_seconds: PositiveFloat = 60.0
    # Sandbox commands (queries and grading) running at once across episodes.
    max_concurrent_queries: PositiveInt = 16
    # c-Interact: clarifying questions beyond one for each critical and knowledge ambiguity, the ADK's patience.
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

    @model_validator(mode="after")
    def postgresql_names_its_image(self) -> Self:
        if self.database_engine == "postgresql" and self.sandbox_config.image is None:
            raise ValueError(
                "database_engine postgresql needs sandbox_config.image: an image that holds the release's "
                "databases, such as one built from benchmarks/bird_interact/Dockerfile"
            )
        return self


class BirdInteractRequest(BaseEpisodeRequest[BirdInteractTask]):
    pass


class BirdInteractResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float
    turns: list[TurnRecord]
    stop_reason: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    ng_agent_observations: AgentObservationBundle | None = None
    ng_trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)
    phases: list[PhaseResult]
    questions_asked: int
    coins_spent: float
    finished: bool
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None


class BirdInteractFailure(EpisodeFailure):
    """A failed episode, with the turns and tool calls completed before the failure and the agent's evidence."""

    turns: list[TurnRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    agent_observations: AgentObservationBundle | None = None
    trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


class BirdInteractResponse(BaseEpisodeResponse[BirdInteractResult]):
    failure: BirdInteractFailure | None = None


class SandboxCommandError(RuntimeError):
    """The episode's sandbox could not run a command: an infrastructure failure, not the agent's mistake."""


@dataclass
class _DatabaseFiles:
    schema_text: str
    column_meanings: dict[str, str]
    # Keyed by name, as the ADK keys it.
    knowledge: dict[str, dict]


@dataclass
class _EpisodeSandbox:
    sandbox: AsyncSandbox
    # The directory in the sandbox that holds the runner and, for SQLite, the database source and the episode's database.
    root: str = ""
    # The episode's database: a SQLite file in root, or a PostgreSQL database in the image's server.
    database: str = ""

    def path(self, name: str) -> str:
        return posixpath.join(self.root, name)


@dataclass(frozen=True)
class _Stop:
    """End the conversation. ``reason`` becomes the episode's ``stop_reason``."""

    reason: str


@dataclass(frozen=True)
class _Submission:
    """A graded submission, as the ADK's grader reports it."""

    passed: bool
    # Starts with the execution-error flag when the SQL did not run.
    message: str
    reward: float = 0.0
    # Set when phase 1 passed and the follow-up question is next.
    follow_up_query: str | None = None


@dataclass
class BirdInteractEpisode:
    """One task's state: its database sandbox, the phase, the budgets and the results so far."""

    request: BirdInteractRequest
    # The episode's cleanup, which stops each sandbox the episode starts.
    cleanup: CleanupContext
    task: BirdInteractTask
    files: _DatabaseFiles
    sandbox: _EpisodeSandbox
    # c-Interact's clarification turns (the ADK's max_turn), or a-Interact's BIRD-COIN.
    question_budget: int | None
    coin_budget: float | None
    phase: int = 1
    # Submissions in the current phase.
    attempts: int = 0
    # c-Interact: the agent's turns in the current phase, and whether the current turn has submitted.
    phase_turns: int = 0
    submitted_this_turn: bool = False
    # The grader's message for the last submission, which picks the debug message.
    last_submission_message: str = ""
    # Phase 1's passing SQL, applied to the database for phase 2.
    applied_sql: list[str] = field(default_factory=list)
    questions_asked: int = 0
    coins_spent: float = 0.0
    phases: list[PhaseResult] = field(default_factory=list)
    finished: bool = False
    # Set when an infrastructure failure disrupted the episode, so its sample is masked.
    failure_kind: str | None = None
    failure_reason: str | None = None
    # Set when the task's own solution SQL failed, which fails the episode.
    solution_error: str | None = None
    toolset: EpisodeToolset | None = None

    @property
    def coins_left(self) -> float:
        return (self.coin_budget or 0.0) - self.coins_spent

    def visible_knowledge(self) -> list[dict]:
        hidden = {point.deleted_knowledge for point in self.task.knowledge_ambiguity}
        return [entry for entry in self.files.knowledge.values() if entry.get("id") not in hidden]

    def record_infrastructure_failure(self, kind: str, what: str, error: BaseException) -> None:
        """An infrastructure failure must not read as the agent's mistake, so the sample is masked."""
        self.failure_kind = kind
        self.failure_reason = f"{what}: {type(error).__name__}: {error}"[:2000]


def _submission_reply(submission: _Submission, *, budget: float | None) -> str:
    """What the agent reads after a submission, as the ADK's ``submit_sql`` writes it."""
    lines = [submission.message.replace(f"{EXECUTION_ERROR_FLAG} ", "")]
    if submission.reward > 0:
        lines.append(SUBMISSION_REWARD_LINE.format(reward=submission.reward))
    if submission.follow_up_query is not None:
        lines.append(SUBMISSION_FOLLOW_UP_LINE.format(follow_up_query=submission.follow_up_query))
    if budget is not None:
        lines.append(SUBMISSION_BUDGET_LINE.format(budget=budget))
    return "\n".join(lines)


class CInteractTools:
    """The ADK's two c-Interact tools of one episode, ``ask_user`` and ``submit_sql``, with its descriptions.

    As in the ADK, a turn ends at its submission: a later call in the turn gets the reply that the ADK's agent gets
    instead of another model call, and so does a call once the task is done. Questions are not capped; each answer
    says how many of the clarification turns remain.
    """

    def __init__(self, server: "BirdInteractEnvironmentServer", episode: BirdInteractEpisode) -> None:
        self._server = server
        self._episode = episode
        self._tools: dict[str, tuple[type[BaseModel], str, Callable[[Any], Awaitable[str]]]] = {
            "ask_user": (AskArguments, ASK_USER_DESCRIPTION, self._ask_user),
            "submit_sql": (SQLArguments, server.in_dialect(SUBMIT_SQL_DESCRIPTION), self._submit_sql),
        }

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=name, description=description, parameters=arguments.model_json_schema())
            for name, (arguments, description, _) in self._tools.items()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        model, _, run = self._tools[name]
        try:
            parsed = model.model_validate(arguments)
        except ValidationError as error:
            raise ToolArgumentsError(str(error)) from error
        if self._episode.finished:
            return ToolResult(output=TASK_COMPLETED_REPLY)
        if self._episode.submitted_this_turn:
            return ToolResult(output=SUBMITTED_REPLY)
        return ToolResult(output=await run(parsed))

    async def _ask_user(self, arguments: AskArguments) -> str:
        episode = self._episode
        episode.questions_asked += 1
        try:
            answer = await self._server.simulate(episode, arguments.question)
        except Exception as error:
            episode.record_infrastructure_failure(USER_SIMULATOR_FAILED, "User simulator failed", error)
            raise
        budget = episode.question_budget
        return answer + CLARIFICATIONS_LEFT_NOTE.format(
            remaining=max(0, budget - episode.questions_asked), max_turn=budget
        )

    async def _submit_sql(self, arguments: SQLArguments) -> str:
        episode = self._episode
        episode.submitted_this_turn = True
        try:
            submission = await self._server.submit(episode, arguments.sql)
        except SandboxCommandError as error:
            episode.record_infrastructure_failure(SANDBOX_FAILED, "Sandbox failed", error)
            raise
        return _submission_reply(submission, budget=_C_INTERACT_BUDGET)


class AInteractTools:
    """The nine a-Interact tools of one episode, each charged in BIRD-COIN.

    Once the budget is spent, only ``submit`` runs. The server runs one call at a time for an episode.
    """

    def __init__(self, server: "BirdInteractEnvironmentServer", episode: BirdInteractEpisode) -> None:
        self._server = server
        self._episode = episode

    @episode_tool
    async def ask(self, arguments: AskArguments) -> ToolObservation:
        """Ask the user one clarifying question."""

        async def act(episode: BirdInteractEpisode) -> str:
            episode.questions_asked += 1
            try:
                return await self._server.simulate(episode, arguments.question)
            except Exception as error:
                episode.record_infrastructure_failure(USER_SIMULATOR_FAILED, "User simulator failed", error)
                raise

        return await self._charge("ask", act)

    @episode_tool
    async def submit(self, arguments: SQLArguments) -> ToolObservation:
        """Submit your final SQL for the current question."""

        async def act(episode: BirdInteractEpisode) -> str:
            try:
                return _submission_reply(await self._server.submit(episode, arguments.sql), budget=None)
            except SandboxCommandError as error:
                episode.record_infrastructure_failure(SANDBOX_FAILED, "Sandbox failed", error)
                raise

        return await self._charge("submit", act)

    @episode_tool
    async def execute(self, arguments: SQLArguments) -> ToolObservation:
        """Run an exploratory SQL query on the database and see up to 20 rows."""

        async def act(episode: BirdInteractEpisode) -> str:
            request = {"sql": arguments.sql, "timeout_seconds": self._server.config.sql_timeout_seconds}
            try:
                result = await self._server.in_sandbox(episode, "execute", request)
            except SandboxCommandError as error:
                episode.record_infrastructure_failure(SANDBOX_FAILED, "Sandbox failed", error)
                raise
            if result["error"] is not None:
                return f"Error: {result['error']}"
            count = result["row_count"]
            shown = "\n".join(result["preview"])
            more = f"\n... {count - MAX_PREVIEW_ROWS} more rows" if count > MAX_PREVIEW_ROWS else ""
            return f"{count} rows\n{shown}{more}"

        return await self._charge("execute", act)

    @episode_tool
    async def get_schema(self, arguments: NoArguments) -> ToolObservation:
        """Read the database schema, with sample rows."""

        async def act(episode: BirdInteractEpisode) -> str:
            return episode.files.schema_text

        return await self._charge("get_schema", act)

    @episode_tool
    async def get_all_column_meanings(self, arguments: NoArguments) -> ToolObservation:
        """Read the meaning of every column."""

        async def act(episode: BirdInteractEpisode) -> str:
            return json.dumps(episode.files.column_meanings, indent=1)

        return await self._charge("get_all_column_meanings", act)

    @episode_tool
    async def get_column_meaning(self, arguments: ColumnMeaningArguments) -> ToolObservation:
        """Read the meaning of one column."""

        async def act(episode: BirdInteractEpisode) -> str:
            wanted = f"{episode.task.selected_database}|{arguments.table_name}|{arguments.column_name}".casefold()
            meaning = next(
                (text for key, text in episode.files.column_meanings.items() if key.casefold() == wanted), None
            )
            return meaning or f"No column {arguments.column_name!r} in table {arguments.table_name!r}."

        return await self._charge("get_column_meaning", act)

    @episode_tool
    async def get_all_external_knowledge_names(self, arguments: NoArguments) -> ToolObservation:
        """List the names of the knowledge-base entries."""

        async def act(episode: BirdInteractEpisode) -> str:
            return json.dumps([entry["knowledge"] for entry in episode.visible_knowledge()])

        return await self._charge("get_all_external_knowledge_names", act)

    @episode_tool
    async def get_knowledge_definition(self, arguments: KnowledgeArguments) -> ToolObservation:
        """Read one knowledge-base entry."""

        async def act(episode: BirdInteractEpisode) -> str:
            wanted = arguments.knowledge_name.casefold()
            entry = next((e for e in episode.visible_knowledge() if e["knowledge"].casefold() == wanted), None)
            return json.dumps(entry) if entry else f"No knowledge named {arguments.knowledge_name!r}."

        return await self._charge("get_knowledge_definition", act)

    @episode_tool
    async def get_all_knowledge_definitions(self, arguments: NoArguments) -> ToolObservation:
        """Read every knowledge-base entry."""

        async def act(episode: BirdInteractEpisode) -> str:
            return "\n".join(json.dumps(entry) for entry in episode.visible_knowledge())

        return await self._charge("get_all_knowledge_definitions", act)

    async def _charge(self, action: str, act: Callable[[BirdInteractEpisode], Awaitable[str]]) -> ToolObservation:
        episode = self._episode
        if episode.finished:
            return ToolObservation(observation="The task is finished.", coins_left=episode.coins_left)
        if episode.coins_left <= 0 and action != "submit":
            return ToolObservation(
                observation="Your budget is spent. Only submit is allowed now.", coins_left=episode.coins_left
            )
        episode.coins_spent += self._server.config.action_costs[action]
        observation = await act(episode)
        if episode.coins_left <= 0 and action == "submit" and not episode.finished:
            episode.finished = True
            observation += "\nYour budget is spent, so the task ends here."
        return ToolObservation(observation=observation, coins_left=episode.coins_left)


def function_tool(spec: ToolSpec) -> dict[str, Any]:
    """Declare a served tool to the model, for an agent that calls it over direct HTTP.

    Every argument of a BIRD-Interact tool is a required string, so the declaration is strict.
    """
    properties = {
        name: {key: value for key, value in schema.items() if key != "title"}
        for name, schema in spec.parameters.get("properties", {}).items()
    }
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


class BirdInteractEnvironmentServer(BaseEnvironmentServerWithTools[BirdInteractRequest, BirdInteractResponse]):
    config: BirdInteractEnvironmentServerConfig
    request_model = BirdInteractRequest
    response_model = BirdInteractResponse
    _files: dict[str, _DatabaseFiles] = PrivateAttr(default_factory=dict)
    _queries: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._queries = asyncio.Semaphore(self.config.max_concurrent_queries)

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        return compute_aggregate_metrics(body.verify_responses)

    def in_dialect(self, text: str) -> str:
        """The ADK's text, which names PostgreSQL, for this server's database engine."""
        return text if self.config.database_engine == "postgresql" else text.replace("PostgreSQL", "SQLite")

    # ----- the episode ----- #

    async def run(self, request: BirdInteractRequest, cleanup: CleanupContext) -> BirdInteractResponse:
        episode = await self._start_episode(request, cleanup)
        served = await self.serve_tools(request, episode.toolset, cleanup)
        agent = AgentSession(self.server_client, self.config.agent_server, request)
        try:
            # c-Interact's later turns send only the new message, so the agent must resume its session.
            await agent.seed(
                cleanup, tool_accesses=served.tool_accesses, continuation=self.config.mode == "c_interact"
            )
            stop_reason = await self._converse(episode, agent)
            # The agent stops before grading, so grading reads the state the agent left.
            await agent.close()
        except NgFailureEvent as error:
            if not is_handled_episode_failure(error):
                raise
            # A failure keeps the completed turns, the tool calls and the agent's evidence.
            try:
                await agent.close()  # A no-op when the agent is already closed.
            except Exception:
                pass  # The episode's cleanup retries the close and logs its failure.
            error.episode_evidence.update(
                turns=list(agent.turns),
                tool_calls=list(served.calls),
                agent_observations=agent.closed.agent_observations if agent.closed else None,
                trajectory=agent.closed.trajectory if agent.closed else None,
            )
            raise
        return self._response(episode, agent, stop_reason, served.calls)

    async def _converse(self, episode: BirdInteractEpisode, agent: AgentSession) -> str:
        """Run the agent's turns until the protocol stops them or the interaction deadline does; return the reason.

        A turn over ``turn_timeout_seconds`` fails the episode for a retry. The interaction deadline stops the turns,
        and the episode is still graded.
        """
        loop = asyncio.get_running_loop()
        timeout = self.config.interaction_timeout_seconds
        interaction_deadline = loop.time() + timeout if timeout else None
        params = self._first_request(episode)
        while True:
            turn = len(agent.turns) + 1
            now = loop.time()
            if interaction_deadline is not None and now >= interaction_deadline:
                return "interaction_timeout"
            turn_deadline = now + self.config.turn_timeout_seconds
            deadline = turn_deadline if interaction_deadline is None else min(turn_deadline, interaction_deadline)
            episode.phase_turns += 1
            episode.submitted_this_turn = False
            try:
                record = await agent.activate(params, deadline=deadline)
            except TurnDeadlineExpired as error:
                if deadline != turn_deadline:
                    return "interaction_timeout"
                raise handled_episode_failure(
                    stage="agent",
                    operation="episode",
                    failure_reason=f"Agent turn {turn} timed out after {self.config.turn_timeout_seconds} seconds",
                    terminal=False,
                ) from error
            action = agent_outcome_policy(record.response, config=self.config.agent_outcome_policy)
            if action != OutcomeAction.VERIFY:
                error = record.response.error
                raise handled_episode_failure(
                    stage="agent",
                    operation="episode",
                    failure_reason=(error.message if error else f"Agent reported failure in turn {turn}")[:2000],
                    terminal=action == OutcomeAction.FAIL,
                )
            if episode.solution_error is not None:
                # Another attempt would grade against the same broken solution.
                raise handled_episode_failure(
                    stage="verification", operation="episode", failure_reason=episode.solution_error, terminal=True
                )
            message = self._next_message(episode)
            if isinstance(message, _Stop):
                return message.reason
            params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": message}])

    async def _start_episode(self, request: BirdInteractRequest, cleanup: CleanupContext) -> BirdInteractEpisode:
        task = request.task.task_input
        if not task.sol_sql:
            raise _refused(
                f"Task {task.instance_id} has no sol_sql. The public BIRD-Interact releases withhold the ground truth; "
                "merge it into the rows before running."
            )
        if self.config.database_engine == "sqlite" and task.needs_custom_test_cases():
            raise _refused(
                f"Task {task.instance_id} is graded by its custom test cases (a phase of a category other than "
                "Query), which this server runs on PostgreSQL only."
            )
        files = self._database_files(task.selected_database)
        source = self._sandbox_source(task)
        try:
            sandbox = await self._start_sandbox(request, cleanup, task, source)
        except Exception as error:
            raise _failure(error, stage="seed", terminal=False, name_error_type=True) from error
        question_budget, coin_budget = None, None
        if self.config.mode == "c_interact":
            question_budget = task.ambiguity_count + self.config.question_patience
        else:
            coin_budget = (
                _BASE_COIN_BUDGET + self.config.action_costs["ask"] * task.ambiguity_count + self.config.coin_patience
            )
        episode = BirdInteractEpisode(
            request=request,
            cleanup=cleanup,
            task=task,
            files=files,
            sandbox=sandbox,
            question_budget=question_budget,
            coin_budget=coin_budget,
        )
        if self.config.mode == "c_interact":
            episode.toolset = CInteractTools(self, episode)
        else:
            episode.toolset = TypedToolset(AInteractTools(self, episode))
        return episode

    def _first_request(self, episode: BirdInteractEpisode) -> NeMoGymResponseCreateParamsNonStreaming:
        task = episode.task
        if self.config.mode == "c_interact":
            # The ADK's instruction and first message. Its instruction shows the schema and the visible knowledge.
            knowledge = [
                {key: entry[key] for key in KNOWLEDGE_VISIBLE_FIELDS if key in entry}
                for entry in episode.visible_knowledge()
            ]
            instructions = C_INTERACT_INSTRUCTION.format(
                db_name=task.selected_database,
                db_schema=episode.files.schema_text,
                external_kg=json.dumps(knowledge, indent=2),
                max_turn=episode.question_budget,
            )
            message = PHASE_1_MESSAGE.format(amb_user_query=task.amb_user_query, max_turn=episode.question_budget)
            instructions, message = self.in_dialect(instructions), self.in_dialect(message)
        else:
            # The schema, column meanings and knowledge base cost BIRD-COIN, so the request leaves them out.
            dialect = "PostgreSQL" if self.config.database_engine == "postgresql" else "SQLite"
            instructions = A_INTERACT_INSTRUCTIONS.format(dialect=dialect, coin_budget=episode.coin_budget)
            message = f"Database: {task.selected_database}\n\nMy request:\n{task.amb_user_query}"
        # An MCP agent lists the tools from the server, and some harnesses refuse a request that declares tools.
        declared = "direct_http" in self.config.tool_transports
        tools = [function_tool(spec) for spec in episode.toolset.specs()] if declared else []
        return NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "system", "content": instructions}, {"role": "user", "content": message}], tools=tools
        )

    def _next_message(self, episode: BirdInteractEpisode) -> str | _Stop:
        """The message that opens the agent's next turn, as the ADK's orchestrator sends it; or a stop."""
        if self.config.mode == "a_interact":
            return _Stop("agent_finished")
        if episode.failure_kind is not None:
            # The sample is masked, so its later turns would be wasted.
            return _Stop(episode.failure_kind)
        if episode.finished:
            return _Stop("solved")
        if episode.phase_turns == 0:
            # Phase 1 passed in the last turn, and the follow-up question is phase 2.
            return self.in_dialect(FOLLOW_UP_MESSAGE.format(follow_up_query=episode.task.follow_up.query))
        if episode.phase_turns == 1:
            # The phase failed, or the agent did not submit: one more turn. The last submission picks the message.
            last = episode.last_submission_message
            if EXECUTION_ERROR_FLAG in last:
                return DEBUG_EXECUTION_ERROR_MESSAGE.format(error=last.split(f"{EXECUTION_ERROR_FLAG} ")[-1])
            return DEBUG_WRONG_RESULT_MESSAGE
        episode.finished = True
        return _Stop("failed" if episode.attempts else "no_sql_submitted")

    def _response(
        self,
        episode: BirdInteractEpisode,
        agent: AgentSession,
        stop_reason: str,
        tool_calls: list[ToolCallRecord],
    ) -> BirdInteractResponse:
        phases = list(episode.phases)
        # The current phase has submissions but no result yet: it failed.
        if episode.attempts and all(phase.phase != episode.phase for phase in phases):
            phases.append(PhaseResult(phase=episode.phase, passed=False, attempts=episode.attempts, reward=0.0))
        return BirdInteractResponse(
            episode_id=episode.request.episode_id,
            task_id=episode.request.task.task_id,
            result=BirdInteractResult(
                reward=sum(phase.reward for phase in phases),
                turns=list(agent.turns),
                stop_reason=stop_reason,
                tool_calls=list(tool_calls),
                ng_agent_observations=agent.closed.agent_observations if agent.closed else None,
                ng_trajectory=agent.closed.trajectory if agent.closed else None,
                phases=phases,
                questions_asked=episode.questions_asked,
                coins_spent=episode.coins_spent,
                finished=episode.finished,
                mask_sample=episode.failure_kind is not None,
                failure_kind=episode.failure_kind,
                failure_reason=episode.failure_reason,
            ),
        )

    # ----- protocol actions ----- #

    async def simulate(self, episode: BirdInteractEpisode, question: str) -> str:
        """The simulated user's reply: the action parser picks an action, and the response generator answers by it."""
        task, phase = episode.task, episode.phase
        parsed = await self._user_model(
            action_parser_prompt(task, question, phase=phase), max_output_tokens=ACTION_PARSER_MAX_TOKENS
        )
        prompt = response_generator_prompt(
            task, question, read_action(parsed), phase=phase, schema_text=episode.files.schema_text
        )
        return read_reply(await self._user_model(prompt, max_output_tokens=RESPONSE_GENERATOR_MAX_TOKENS))

    async def submit(self, episode: BirdInteractEpisode, sql: str) -> _Submission:
        """Grade a submission for the current phase and move the protocol on, as the ADK's grader does.

        In c-Interact a pass on the phase's first submission earns its ``first_attempt`` reward, and a later one its
        ``after_debug`` reward; a-Interact always earns ``first_attempt``. Raises ``SandboxCommandError`` when the
        sandbox cannot grade it, and ``ValueError`` when the task's own solution fails, which also records it on the
        episode.
        """
        task, phase = episode.task, episode.phase
        current = task if phase == 1 else task.follow_up
        follows = phase == 1 and task.follow_up is not None and len(self.config.phase_rewards) > 1
        episode.attempts += 1
        result = await self.in_sandbox(
            episode,
            "grade",
            {
                "predicted_sql": sql,
                "solution_sql": current.sol_sql,
                "ordered": current.conditions.order,
                # A management phase runs its own; any other phase is graded by the default test case.
                "test_cases": custom_test_cases(current),
                "timeout_seconds": self.config.sql_timeout_seconds,
                # Phase 2 sees what phase 1 changed.
                "apply_if_passed": follows,
            },
        )
        if "solution_error" in result:
            episode.solution_error = f"Task {task.instance_id}: {result['solution_error']}"[:2000]
            raise ValueError(episode.solution_error)
        if not result["passed"]:
            if result["timed_out"]:
                message = EXECUTION_TIMEOUT
            elif result["execution_error"] is not None:
                message = EXECUTION_ERROR.format(error=result["execution_error"])
            else:
                message = WRONG_RESULT
            submission = _Submission(passed=False, message=PHASE_FAILED.format(phase=phase, message=message))
        else:
            rewards = self.config.phase_rewards[phase - 1]
            debugged = self.config.mode == "c_interact" and episode.attempts > 1
            reward = rewards.after_debug if debugged else rewards.first_attempt
            episode.phases.append(PhaseResult(phase=phase, passed=True, attempts=episode.attempts, reward=reward))
            if follows:
                episode.applied_sql = [sql]
                episode.phase, episode.attempts, episode.phase_turns = 2, 0, 0
                message = PHASE_1_PASSED_WITH_FOLLOW_UP.format(reward=reward)
                submission = _Submission(True, message, reward, follow_up_query=task.follow_up.query)
            else:
                episode.finished = True
                message = (PHASE_1_PASSED if phase == 1 else PHASE_2_PASSED).format(reward=reward)
                submission = _Submission(True, message, reward)
        episode.last_submission_message = submission.message
        return submission

    async def _user_model(self, prompt: str, *, max_output_tokens: int) -> str:
        response = await self.server_client.post(
            server_name=self.config.user_model.name,
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": prompt}], temperature=0.0, max_output_tokens=max_output_tokens
            ),
        )
        await raise_for_status(response)
        return NeMoGymResponse.model_validate(await get_response_json(response)).output_text

    # ----- the episode's sandbox ----- #

    async def in_sandbox(self, episode: BirdInteractEpisode, command: str, request: dict[str, Any]) -> dict[str, Any]:
        """Run a command in the episode's sandbox, retrying an infrastructure failure on a fresh sandbox.

        Only setup and a passing phase 1 write the episode's database, and a fresh sandbox's setup applies both, so
        it holds the same state.
        """
        async with self._queries:
            for _ in range(self.config.inconclusive_verification_retries):
                try:
                    return await self._run_command(episode.sandbox, command, request)
                except SandboxCommandError as error:
                    LOGGER.warning(f"Retrying {command} on a fresh sandbox for {episode.task.instance_id}: {error}")
                    await self._replace_sandbox(episode)
            return await self._run_command(episode.sandbox, command, request)

    async def _start_sandbox(
        self,
        request: BirdInteractRequest,
        cleanup: CleanupContext,
        task: BirdInteractTask,
        source: Path | None,
        applied_sql: Sequence[str] = (),
    ) -> _EpisodeSandbox:
        """Start a sandbox holding the task's database with ``preprocess_sql`` and ``applied_sql`` applied.

        ``source`` is the SQLite database to upload; PostgreSQL's database is in the image. The episode's cleanup
        stops the sandbox, and a failed setup stops it at once (see ``start_episode_sandbox``).
        """
        sandbox_config = self.config.sandbox_config
        spec = SandboxSpec(
            image=sandbox_config.image or _SQLITE_IMAGE,
            ttl_s=sandbox_config.ttl_s,
            ready_timeout_s=sandbox_config.ready_timeout_s,
            metadata=sandbox_config.metadata | {"instance_id": task.instance_id[:63]},
            resources=sandbox_config.resources,
            provider_options=dict(sandbox_config.provider_options),
        )
        handle: _EpisodeSandbox | None = None

        async def setup(sandbox: AsyncSandbox) -> None:
            nonlocal handle
            handle = _EpisodeSandbox(sandbox)
            # The sandbox's working directory: the container's own, or a fresh host directory for the local provider.
            result = await sandbox.exec("pwd", timeout_s=60)
            if result.error_type is not None or result.return_code != 0 or not (result.stdout or "").strip():
                raise SandboxCommandError(f"pwd failed in the sandbox: {result.error_type or result.return_code}")
            handle.root = posixpath.join((result.stdout or "").strip(), "bird_interact")
            await sandbox.upload(_SANDBOX_RUNNER, handle.path(_SANDBOX_RUNNER.name))
            setup_request: dict[str, Any] = {
                "preprocess_sql": task.preprocess_sql,
                "applied_sql": list(applied_sql),
                "timeout_seconds": self.config.sql_timeout_seconds,
            }
            if source is None:
                handle.database = f"{task.selected_database}_template"
            else:
                handle.database = handle.path("episode.sqlite")
                await sandbox.upload(source, handle.path(source.name))
                setup_request["source"] = handle.path(source.name)
            await self._run_command(handle, "setup", setup_request)

        await self.start_episode_sandbox(request, cleanup, spec, setup, name="database sandbox")
        assert handle is not None  # setup ran
        return handle

    async def _run_command(self, handle: _EpisodeSandbox, command: str, request: dict[str, Any]) -> dict[str, Any]:
        """Run ``grading.py <command>`` in the sandbox and return its JSON result."""
        name = f"request-{uuid4().hex}.json"
        try:
            with tempfile.TemporaryDirectory(prefix="bird-interact-") as scratch:
                local = Path(scratch) / name
                # Text goes in a file, not in the command line.
                local.write_text(
                    json.dumps(request | {"engine": self.config.database_engine, "database": handle.database})
                )
                await handle.sandbox.upload(local, handle.path(name))
            # Two queries, or the submission and two for each test case, which mostly runs the submission's and the
            # solution's queries again.
            test_cases = request.get("test_cases") or []
            queries = 1 + 2 * len(test_cases) if test_cases else 2
            result = await handle.sandbox.exec(
                shlex.join(["python3", handle.path(_SANDBOX_RUNNER.name), command, handle.path(name)]),
                timeout_s=queries * self.config.sql_timeout_seconds + _SANDBOX_COMMAND_SLACK_SECONDS,
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

    async def _replace_sandbox(self, episode: BirdInteractEpisode) -> None:
        try:
            fresh = await self._start_sandbox(
                episode.request,
                episode.cleanup,
                episode.task,
                self._sandbox_source(episode.task),
                episode.applied_sql,
            )
        except Exception as error:
            raise SandboxCommandError(f"Could not start a fresh sandbox: {type(error).__name__}: {error}") from error
        stale, episode.sandbox = episode.sandbox, fresh
        try:
            await stale.sandbox.stop()
        except Exception:
            LOGGER.exception("Could not stop a failed BIRD-Interact sandbox; the episode's cleanup tries again")

    # ----- the task's database files ----- #

    def _database_files(self, name: str) -> _DatabaseFiles:
        if name not in self._files:
            directory = self._database_root() / name
            try:
                knowledge_lines = (directory / f"{name}_kb.jsonl").read_text().splitlines()
                self._files[name] = _DatabaseFiles(
                    schema_text=(directory / f"{name}_schema.txt").read_text(),
                    column_meanings=json.loads((directory / f"{name}_column_meaning_base.json").read_text()),
                    knowledge={
                        entry["knowledge"]: entry for entry in map(json.loads, filter(str.strip, knowledge_lines))
                    },
                )
            except FileNotFoundError as error:
                raise _refused(f"Missing database file for {name!r}: {error.filename}") from error
        return self._files[name]

    def _sandbox_source(self, task: BirdInteractTask) -> Path | None:
        """The SQLite database to upload into the episode's sandbox; PostgreSQL's is in the sandbox image."""
        return self._database_source(task.selected_database) if self.config.database_engine == "sqlite" else None

    def _database_source(self, name: str) -> Path:
        """The database file, ``<db>.sqlite``, or else the script ``<db>.sql``."""
        directory = self._database_root() / name
        for source in (directory / f"{name}.sqlite", directory / f"{name}.sql"):
            if source.exists():
                return source
        raise _refused(f"No database for {name!r} in {directory}")

    def _database_root(self) -> Path:
        """``database_dir``; a relative path is relative to the repository root, not the server's directory."""
        root = Path(self.config.database_dir)
        return root if root.is_absolute() else PARENT_DIR / root


def _failure(error: BaseException, *, stage: FailureStage, **fields: Any) -> NgFailureEvent:
    """Map a failed call to another server to the episode's failure (see ``dependency_failure``)."""
    return dependency_failure(error, stage=stage, **fields)


def _refused(reason: str) -> NgFailureEvent:
    """A task this server cannot run; another attempt would fail the same way."""
    return handled_episode_failure(stage="seed", operation="episode", failure_reason=reason[:2000], terminal=True)


if __name__ == "__main__":
    BirdInteractEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BirdInteractEnvironmentServer.run_webserver()  # noqa: F401
