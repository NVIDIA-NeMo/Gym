# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD-Interact in one Environment Server: the database sandbox, the user simulator, the grader and the turns.

Each episode's database lives in its own sandbox, which the episode starts and stops. The agent's SQL and the grading
run there (``grading.py``); the agent never gets the sandbox itself.

c-Interact is environment-driven: each agent reply is a clarifying question or an SQL submission, and the next user
message is the simulator's answer, the debug prompt or the follow-up question. a-Interact is agent-driven: the agent
gets one turn and nine tools that this server serves for the episode, over direct HTTP, MCP or both. Each tool call
costs BIRD-COIN.
"""

import asyncio
import json
import logging
import posixpath
import re
import shlex
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self

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
    UNANSWERABLE_REPLY,
    classifier_prompt,
    parse_action,
    responder_prompt,
)
from environment_servers.bird_interact.task_data import BirdInteractTask
from nemo_gym import PARENT_DIR
from nemo_gym.base_environment_server import CleanupContext, HandledEpisodeError
from nemo_gym.base_user_simulation_environment_server import (
    BaseUserSimulationEnvironmentServer,
    Stop,
    Transcript,
    TurnRecord,
    UserMessage,
    UserSimulationEnvironmentServerConfig,
    UserSimulationFailure,
    UserSimulationResult,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.episode_tools import ToolSpec, TypedToolset, episode_tool
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


LOGGER = logging.getLogger(__name__)
USER_SIMULATOR_FAILED = "bird_interact:user_simulator_failed"
SANDBOX_FAILED = "bird_interact:sandbox_failed"
# Runs inside the episode's sandbox; see its module docstring.
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
# The last fenced block marked sql or sqlite.
_SQL_BLOCK = re.compile(r"```(?:sqlite|sql)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

C_INTERACT_INSTRUCTIONS = """You are a data analyst who writes SQLite queries for a user. The user's request can be \
ambiguous, and part of the knowledge you need can be missing from what you see.

You may ask the user clarifying questions, one in each reply, at most {question_budget} in total. Each reply must be \
exactly one of these:
- one clarifying question in plain text, with no SQL;
- your final SQL in one ```sqlite fenced code block.

Ask your questions before you send SQL. After you send SQL, the user tells you whether it is right. If it is not, \
you get one more attempt, which must be SQL. The user can then ask a follow-up question, which you answer with SQL."""

A_INTERACT_INSTRUCTIONS = """You are a data analyst who writes SQLite queries for a user. The user's request can be \
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
    """The episode's sandbox. The image needs ``python3``; the database and the runner are uploaded at the start."""

    image: str = "python:3.12-slim"
    # The provider reclaims a sandbox that a crashed server never stopped.
    ttl_s: PositiveFloat | None = 7200.0
    ready_timeout_s: PositiveFloat | None = None
    resources: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)


class BirdInteractEnvironmentServerConfig(UserSimulationEnvironmentServerConfig):
    """The interaction mode, the data location, the simulator model and the protocol's budgets and rewards."""

    mode: InteractionMode
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
    # Sandbox commands (queries and grading) running at once across episodes.
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

    @model_validator(mode="after")
    def a_interact_has_one_turn(self) -> Self:
        if self.mode == "a_interact" and self.max_turns != 1:
            raise ValueError("a_interact runs in one agent turn; set max_turns to 1")
        return self


class BirdInteractRequest(BaseEpisodeRequest[BirdInteractTask]):
    pass


class BirdInteractResult(UserSimulationResult):
    phases: list[PhaseResult]
    questions_asked: int
    coins_spent: float
    finished: bool
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None


class BirdInteractResponse(BaseEpisodeResponse[BirdInteractResult]):
    failure: UserSimulationFailure | None = None


class SandboxCommandError(RuntimeError):
    """The episode's sandbox could not run a command: an infrastructure failure, not the agent's mistake."""


@dataclass
class _DatabaseFiles:
    schema_text: str
    column_meanings: dict[str, str]
    knowledge: dict[int, dict]


@dataclass
class _EpisodeSandbox:
    sandbox: AsyncSandbox
    # The directory in the sandbox that holds the runner, the database source and the episode's database.
    root: str = ""

    def path(self, name: str) -> str:
        return posixpath.join(self.root, name)


@dataclass
class _Submission:
    passed: bool
    # What the user says next: the debug prompt, the follow-up question, or the end of the task.
    feedback: str
    finished: bool


@dataclass
class BirdInteractEpisode:
    """One task's state: its database sandbox, the phase, the budgets and the results so far."""

    request: BirdInteractRequest
    task: BirdInteractTask
    files: _DatabaseFiles
    sandbox: _EpisodeSandbox
    coin_budget: float | None
    phase: int = 1
    attempts: int = 0
    questions_asked: int = 0
    coins_spent: float = 0.0
    phases: list[PhaseResult] = field(default_factory=list)
    finished: bool = False
    # Set when an infrastructure failure disrupted an a-Interact episode, so its sample is masked.
    failure_kind: str | None = None
    failure_reason: str | None = None
    # a-Interact only.
    toolset: TypedToolset | None = None

    @property
    def coins_left(self) -> float:
        return (self.coin_budget or 0.0) - self.coins_spent

    def visible_knowledge(self) -> list[dict]:
        hidden = {point.deleted_knowledge for point in self.task.knowledge_ambiguity}
        return [entry for key, entry in self.files.knowledge.items() if key not in hidden]

    def record_infrastructure_failure(self, kind: str, what: str, error: BaseException) -> None:
        """An infrastructure failure must not read as the agent's mistake, so the sample is masked."""
        self.failure_kind = kind
        self.failure_reason = f"{what}: {type(error).__name__}: {error}"[:2000]


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
                return (await self._server.submit(episode, arguments.sql)).feedback
            except SandboxCommandError as error:
                episode.record_infrastructure_failure(SANDBOX_FAILED, "Sandbox failed", error)
                raise

        return await self._charge("submit", act)

    @episode_tool
    async def execute(self, arguments: SQLArguments) -> ToolObservation:
        """Run an exploratory SQLite query and see up to 20 rows."""

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


def extract_sql(text: str) -> str | None:
    """Return the last ```sql or ```sqlite block of a reply, or None for a reply without one."""
    blocks = _SQL_BLOCK.findall(text)
    return blocks[-1].strip() if blocks and blocks[-1].strip() else None


def last_message_text(response: NeMoGymResponse) -> str:
    """The text of a turn's last message: the agent's reply to the user.

    A harness with its own tools, such as a shell, can take several steps in one turn, each with a message. Only the
    last one answers the user; ``output_text`` would join them all.
    """
    for item in reversed(response.output):
        if item.type == "message":
            return "".join(part.text for part in item.content if part.type == "output_text")
    return ""


class BirdInteractEnvironmentServer(
    BaseUserSimulationEnvironmentServer[BirdInteractRequest, BirdInteractResponse, BirdInteractEpisode]
):
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

    # ----- user-simulation hooks ----- #

    async def start_episode(self, request: BirdInteractRequest, cleanup: CleanupContext) -> BirdInteractEpisode:
        task = request.task.task_input
        if not task.sol_sql:
            raise _refused(
                f"Task {task.instance_id} has no sol_sql. The public BIRD-Interact releases withhold the ground truth; "
                "merge it into the rows before running."
            )
        if task.test_cases:
            raise _refused(f"Task {task.instance_id} has custom test cases, which this server does not run yet.")
        files = self._database_files(task.selected_database)
        source = self._database_source(task.selected_database)
        try:
            sandbox = await self._start_sandbox(task, source)
        except Exception as error:
            raise self.failure(error, stage="seed", terminal=False, name_error_type=True) from error
        coin_budget = None
        if self.config.mode == "a_interact":
            coin_budget = (
                _BASE_COIN_BUDGET + self.config.action_costs["ask"] * task.ambiguity_count + self.config.coin_patience
            )
        episode = BirdInteractEpisode(
            request=request, task=task, files=files, sandbox=sandbox, coin_budget=coin_budget
        )
        # Stops whichever sandbox the episode holds last: a failed command moves it onto a fresh one.
        cleanup.register_cleanup("database sandbox", lambda: episode.sandbox.sandbox.stop())
        if self.config.mode == "a_interact":
            episode.toolset = TypedToolset(AInteractTools(self, episode))
        return episode

    async def episode_tools(self, episode: BirdInteractEpisode) -> TypedToolset | None:
        return episode.toolset

    async def responses_create_params(self, episode: BirdInteractEpisode) -> NeMoGymResponseCreateParamsNonStreaming:
        task = episode.task
        if self.config.mode == "c_interact":
            question_budget = task.ambiguity_count + self.config.question_patience
            instructions = C_INTERACT_INSTRUCTIONS.format(question_budget=question_budget)
            knowledge = "\n".join(json.dumps(entry) for entry in episode.visible_knowledge())
            message = (
                f"Database: {task.selected_database}\n\nSchema:\n{episode.files.schema_text}\n\n"
                f"Column meanings:\n{json.dumps(episode.files.column_meanings, indent=1)}\n\n"
                f"Knowledge base:\n{knowledge}\n\nMy request:\n{task.amb_user_query}"
            )
            tools = []
        else:
            # The schema, column meanings and knowledge base cost BIRD-COIN, so the request leaves them out.
            instructions = A_INTERACT_INSTRUCTIONS.format(coin_budget=episode.coin_budget)
            message = f"Database: {task.selected_database}\n\nMy request:\n{task.amb_user_query}"
            # An MCP agent lists the tools from the server, and some harnesses refuse a request that declares tools.
            declared = "direct_http" in self.config.tool_transports
            tools = [function_tool(spec) for spec in episode.toolset.specs()] if declared else []
        return NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "system", "content": instructions}, {"role": "user", "content": message}], tools=tools
        )

    async def next_user_turn(self, episode: BirdInteractEpisode, turn: TurnRecord) -> UserMessage | Stop:
        if self.config.mode == "a_interact":
            return Stop(reason="agent_finished")
        reply = last_message_text(turn.response)
        sql = extract_sql(reply)
        if sql is None:
            answer = await self._answer_question(episode, reply)
            # A question the protocol does not allow ends the episode without a submission.
            return UserMessage(content=answer) if answer is not None else Stop(reason="no_sql_submitted")
        try:
            result = await self.submit(episode, sql)
        except SandboxCommandError as error:
            # The sandbox kept failing on fresh sandboxes, so another attempt of the episode may succeed.
            raise self.failure(error, stage="verification", terminal=False) from error
        except ValueError as error:
            raise self.failure(error, stage="verification") from error
        if result.finished:
            return Stop(reason="solved" if result.passed else "failed")
        return UserMessage(content=result.feedback)

    async def finish_episode(self, episode: BirdInteractEpisode, transcript: Transcript) -> BirdInteractResponse:
        phases = list(episode.phases)
        # The current phase has submissions but no result yet, or the budget ended the task after a failed one.
        if episode.attempts and all(phase.phase != episode.phase for phase in phases):
            phases.append(PhaseResult(phase=episode.phase, passed=False, attempts=episode.attempts, reward=0.0))
        return BirdInteractResponse(
            episode_id=episode.request.episode_id,
            task_id=episode.request.task.task_id,
            result=BirdInteractResult(
                reward=sum(phase.reward for phase in phases),
                **transcript.result_fields(),
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

    async def _answer_question(self, episode: BirdInteractEpisode, question: str) -> str | None:
        """Answer a c-Interact clarifying question: in phase 1, before the first submission, within the budget."""
        budget = episode.task.ambiguity_count + self.config.question_patience
        # As in BIRD-Interact, the debug turn after a failed submission is one more SQL attempt, not a question.
        if episode.finished or episode.phase != 1 or episode.attempts or episode.questions_asked >= budget:
            return None
        episode.questions_asked += 1
        try:
            reply = await self.simulate(episode, question)
        except Exception as error:
            raise self.failure(error, stage="agent", name_error_type=True) from error
        if episode.questions_asked == budget:
            reply += "\n\nThat was your last question. Send me your SQL next."
        return reply

    async def simulate(self, episode: BirdInteractEpisode, question: str) -> str:
        """The simulated user's reply: classify the question, then answer from that action alone."""
        classification = await self._user_model(classifier_prompt(episode.task, question, phase=episode.phase))
        action = parse_action(classification, episode.task)
        if action.kind == "unanswerable":
            return UNANSWERABLE_REPLY
        prompt = responder_prompt(
            episode.task,
            question,
            action,
            phase=episode.phase,
            schema_text=episode.files.schema_text,
            knowledge=episode.files.knowledge,
        )
        return (await self._user_model(prompt)).strip()

    async def submit(self, episode: BirdInteractEpisode, sql: str) -> _Submission:
        """Grade a submission for the current phase and move the protocol on.

        Raises ``SandboxCommandError`` when the sandbox cannot grade it, and ``ValueError`` when the task's own
        solution fails.
        """
        if episode.finished:
            return _Submission(passed=False, feedback="The task is finished.", finished=True)
        task, phase = episode.task, episode.phase
        solution, conditions = (
            (task.sol_sql, task.conditions) if phase == 1 else (task.follow_up.sol_sql, task.follow_up.conditions)
        )
        episode.attempts += 1
        result = await self.in_sandbox(
            episode,
            "grade",
            {
                "predicted_sql": sql,
                "solution_sql": ";\n".join(solution),
                "ordered": conditions.order,
                "timeout_seconds": self.config.sql_timeout_seconds,
            },
        )
        if "solution_error" in result:
            raise ValueError(f"Task {task.instance_id}: {result['solution_error']}")
        c_interact = self.config.mode == "c_interact"
        rewards = self.config.phase_rewards[phase - 1]
        if result["passed"]:
            debugged = c_interact and episode.attempts > 1
            reward = rewards.after_debug if debugged else rewards.first_attempt
            episode.phases.append(PhaseResult(phase=phase, passed=True, attempts=episode.attempts, reward=reward))
            if phase == 1 and task.follow_up is not None and len(self.config.phase_rewards) > 1:
                episode.phase, episode.attempts = 2, 0
                feedback = f"That's right, thank you. Here is my next question: {task.follow_up.query}"
                return _Submission(passed=True, feedback=feedback, finished=False)
            episode.finished = True
            return _Submission(passed=True, feedback="That's right, thank you. That is all I needed.", finished=True)
        if c_interact and episode.attempts >= 2:
            episode.phases.append(PhaseResult(phase=phase, passed=False, attempts=episode.attempts, reward=0.0))
            episode.finished = True
            return _Submission(passed=False, feedback="That is still not right. Let's stop here.", finished=True)
        if result["execution_error"] is not None:
            feedback = f"Your SQL failed with this error: {result['execution_error']}"
        else:
            feedback = "Your SQL runs, but the result is not what I need."
        feedback += " Send your corrected SQL; you have one more attempt." if c_interact else " You can try again."
        return _Submission(passed=False, feedback=feedback, finished=False)

    async def _user_model(self, prompt: str) -> str:
        response = await self.server_client.post(
            server_name=self.config.user_model.name,
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": prompt}]),
        )
        await raise_for_status(response)
        return NeMoGymResponse.model_validate(await get_response_json(response)).output_text

    # ----- the episode's sandbox ----- #

    async def in_sandbox(self, episode: BirdInteractEpisode, command: str, request: dict[str, Any]) -> dict[str, Any]:
        """Run a command in the episode's sandbox, retrying an infrastructure failure on a fresh sandbox.

        Only setup writes the episode's database, so a fresh sandbox holds the same state.
        """
        async with self._queries:
            for _ in range(self.config.inconclusive_verification_retries):
                try:
                    return await self._run_command(episode.sandbox, command, request)
                except SandboxCommandError as error:
                    LOGGER.warning(f"Retrying {command} on a fresh sandbox for {episode.task.instance_id}: {error}")
                    await self._replace_sandbox(episode)
            return await self._run_command(episode.sandbox, command, request)

    async def _start_sandbox(self, task: BirdInteractTask, source: Path) -> _EpisodeSandbox:
        """Start a sandbox holding the task's database with ``preprocess_sql`` applied. A failed setup stops it."""
        global_config = self.server_client.global_config_dict
        sandbox_config = self.config.sandbox_config
        spec = SandboxSpec(
            image=sandbox_config.image,
            ttl_s=sandbox_config.ttl_s,
            ready_timeout_s=sandbox_config.ready_timeout_s,
            metadata=resolve_provider_metadata(self.config.sandbox_provider, global_config)
            | sandbox_config.metadata
            | {"nemo_gym_environment": self.config.name[:63], "instance_id": task.instance_id[:63]},
            resources=sandbox_config.resources,
            provider_options=dict(sandbox_config.provider_options),
        )
        handle = _EpisodeSandbox(AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, global_config)))

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

    async def _run_command(self, handle: _EpisodeSandbox, command: str, request: dict[str, Any]) -> dict[str, Any]:
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

    async def _replace_sandbox(self, episode: BirdInteractEpisode) -> None:
        try:
            fresh = await self._start_sandbox(episode.task, self._database_source(episode.task.selected_database))
        except Exception as error:
            raise SandboxCommandError(f"Could not start a fresh sandbox: {type(error).__name__}: {error}") from error
        stale, episode.sandbox = episode.sandbox, fresh
        try:
            await stale.sandbox.stop()
        except Exception:
            LOGGER.exception("Could not stop a failed BIRD-Interact sandbox; the provider's ttl_s reclaims it")

    # ----- the task's database files ----- #

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
                raise _refused(f"Missing database file for {name!r}: {error.filename}") from error
        return self._files[name]

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


def _refused(reason: str) -> HandledEpisodeError:
    """A task this server cannot run; another attempt would fail the same way."""
    return HandledEpisodeError(UserSimulationFailure(stage="seed", failure_reason=reason[:2000], terminal=True))


if __name__ == "__main__":
    BirdInteractEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BirdInteractEnvironmentServer.run_webserver()  # noqa: F401
