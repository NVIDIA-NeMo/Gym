# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD-Interact on the user-simulation base class.

c-Interact is environment-driven: each agent reply is a clarifying question or an SQL submission, and the next user
message is the simulator's answer, the debug prompt or the follow-up question. a-Interact is agent-driven: the agent
gets one turn and reaches the user, the database and the grader through tools on the Resources Server, over direct
HTTP or MCP.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from nemo_gym.base_environment_server import CleanupContext
from nemo_gym.base_resources_server import (
    MCPServerMetadata,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
)
from nemo_gym.base_user_simulation_environment_server import (
    AgentAccess,
    BaseUserSimulationEnvironmentServer,
    Stop,
    Transcript,
    TurnRecord,
    UserMessage,
    UserSimulationEnvironmentServerConfig,
    UserSimulationFailure,
    UserSimulationResult,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse
from nemo_gym.failure_kinds import FailureStage
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess, ToolAccess
from resources_servers.bird_interact.schemas import (
    A_INTERACT_TOOLS,
    AskRequest,
    AskResponse,
    BirdInteractSeedSessionResponse,
    BirdInteractVerifyRequest,
    BirdInteractVerifyResponse,
    EpisodeContext,
    InteractionMode,
    PhaseResult,
    SubmitRequest,
    SubmitResponse,
)
from resources_servers.bird_interact.task_data import BirdInteractTask


ModelT = TypeVar("ModelT", bound=BaseModel)
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


def _function(name: str, description: str, request_model: type[BaseModel] | None) -> dict[str, Any]:
    """A function tool for a direct HTTP grant. Over MCP the Resources Server describes the same tool."""
    fields = request_model.model_fields if request_model is not None else {}
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": "string", "description": info.description} for key, info in fields.items()},
            "required": list(fields),
            "additionalProperties": False,
        },
        "strict": True,
    }


A_INTERACT_FUNCTIONS = [_function(name, *tool) for name, tool in A_INTERACT_TOOLS.items()]


class BirdInteractEnvironmentServerConfig(UserSimulationEnvironmentServerConfig):
    """Bind the BIRD-Interact Resources Server and choose the interaction mode."""

    resources_server: ResourcesServerRef
    mode: InteractionMode
    # a-Interact: how the agent reaches the tools. `mcp` needs `expose_tools_over_mcp` on the Resources Server.
    resources_tool_transports: list[Literal["direct_http", "mcp"]] = Field(default_factory=lambda: ["direct_http"])

    @model_validator(mode="after")
    def a_interact_has_one_turn_and_tools(self) -> Self:
        if self.mode == "a_interact" and self.max_turns != 1:
            raise ValueError("a_interact runs in one agent turn; set max_turns to 1")
        if self.mode == "a_interact" and not self.resources_tool_transports:
            raise ValueError("a_interact needs a tool transport; set resources_tool_transports")
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


@dataclass
class _Episode:
    request: BirdInteractRequest
    resources_session_id: str = field(default_factory=lambda: f"resources-session-{uuid4().hex}")
    resources_cookies: dict[str, str] = field(default_factory=dict)
    context: EpisodeContext | None = None
    # The MCP endpoint and session token from the seed reply, when the Resources Server exposes its tools over MCP.
    resources_tools: MCPServerMetadata | None = None
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None


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
    BaseUserSimulationEnvironmentServer[BirdInteractRequest, BirdInteractResponse, _Episode]
):
    config: BirdInteractEnvironmentServerConfig
    request_model = BirdInteractRequest
    response_model = BirdInteractResponse

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        return compute_aggregate_metrics(body.verify_responses)

    async def start_episode(self, request: BirdInteractRequest, cleanup: CleanupContext) -> _Episode:
        episode = _Episode(request=request)
        # Registered before the seed, so a cancellation releases a session whose seed reply was lost.
        cleanup.register_cleanup("resources session", lambda: self._close_resources(episode))
        task_data = request.task.task_input.model_dump(mode="json") | {"interaction_mode": self.config.mode}
        seed = await self._call_resources(
            episode,
            "/seed_session",
            ResourcesSeedSessionRequest(
                resources_session_id=episode.resources_session_id,
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                task_data=task_data,
            ),
            BirdInteractSeedSessionResponse,
            stage="seed",
        )
        episode.context, episode.resources_tools = seed.context, seed.resources_tools
        if self._grants_mcp() and seed.resources_tools is None:
            # The same configuration fails every episode, so the failure is terminal.
            raise self.failure(
                ValueError("The Resources Server returned no MCP access; set expose_tools_over_mcp on it"),
                stage="seed",
            )
        return episode

    async def agent_access(self, episode: _Episode) -> AgentAccess:
        if self.config.mode == "c_interact":
            return AgentAccess()
        base_url = self.server_client._resolve_base_url(self.config.resources_server.name).rstrip("/")
        tool_accesses: list[ToolAccess] = []
        if "direct_http" in self.config.resources_tool_transports:
            tool_accesses.append(
                DirectHTTPToolAccess(
                    name=f"{self.config.resources_server.name}.direct_http",
                    required=True,
                    base_url=base_url,
                    cookies=episode.resources_cookies,
                )
            )
        if self._grants_mcp():
            tools = episode.resources_tools
            tool_accesses.append(
                MCPToolAccess(
                    name=tools.server_name,
                    required=True,
                    connection=MCPStreamableHTTPConnection(
                        url=f"{base_url}/{tools.url_path.lstrip('/')}", headers=tools.headers
                    ),
                )
            )
        return AgentAccess(tool_accesses=tool_accesses)

    async def responses_create_params(self, episode: _Episode) -> NeMoGymResponseCreateParamsNonStreaming:
        context = episode.context
        if self.config.mode == "c_interact":
            instructions = C_INTERACT_INSTRUCTIONS.format(question_budget=context.question_budget)
            message = (
                f"Database: {context.database}\n\nSchema:\n{context.schema_text}\n\n"
                f"Column meanings:\n{context.column_meanings}\n\nKnowledge base:\n{context.knowledge}\n\n"
                f"My request:\n{context.question}"
            )
            tools = []
        else:
            instructions = A_INTERACT_INSTRUCTIONS.format(coin_budget=context.coin_budget)
            message = f"Database: {context.database}\n\nMy request:\n{context.question}"
            # An MCP agent lists the tools from the server, and some harnesses refuse a request that declares tools.
            tools = A_INTERACT_FUNCTIONS if "direct_http" in self.config.resources_tool_transports else []
        episode.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "system", "content": instructions}, {"role": "user", "content": message}], tools=tools
        )
        return episode.responses_create_params

    async def next_user_turn(self, episode: _Episode, turn: TurnRecord) -> UserMessage | Stop:
        if self.config.mode == "a_interact":
            return Stop(reason="agent_finished")
        reply = last_message_text(turn.response)
        sql = extract_sql(reply)
        if sql is None:
            answer = await self._call_resources(
                episode, "/c_interact/ask", AskRequest(question=reply), AskResponse, stage="agent"
            )
            # A question the protocol does not allow ends the episode without a submission.
            return UserMessage(content=answer.reply) if answer.answered else Stop(reason="no_sql_submitted")
        result = await self._call_resources(
            episode, "/c_interact/submit", SubmitRequest(sql=sql), SubmitResponse, stage="agent"
        )
        if result.finished:
            return Stop(reason="solved" if result.passed else "failed")
        return UserMessage(content=result.feedback)

    async def finish_episode(self, episode: _Episode, transcript: Transcript) -> BirdInteractResponse:
        last_response = transcript.turns[-1].response if transcript.turns else _empty_response()
        verification = await self._call_resources(
            episode,
            "/verify",
            BirdInteractVerifyRequest(responses_create_params=episode.responses_create_params, response=last_response),
            BirdInteractVerifyResponse,
            stage="verification",
        )
        return BirdInteractResponse(
            episode_id=episode.request.episode_id,
            task_id=episode.request.task.task_id,
            result=BirdInteractResult(
                reward=verification.reward,
                **transcript.result_fields(),
                phases=verification.phases,
                questions_asked=verification.questions_asked,
                coins_spent=verification.coins_spent,
                finished=verification.finished,
                mask_sample=verification.mask_sample,
                failure_kind=verification.failure_kind,
                failure_reason=verification.failure_reason,
            ),
        )

    def _grants_mcp(self) -> bool:
        return self.config.mode == "a_interact" and "mcp" in self.config.resources_tool_transports

    async def _call_resources(
        self, episode: _Episode, path: str, body: BaseModel, model: type[ModelT], *, stage: FailureStage
    ) -> ModelT:
        try:
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=path,
                json=body,
                cookies=episode.resources_cookies,
            )
            await raise_for_status(response)
            episode.resources_cookies.update({name: cookie.value for name, cookie in response.cookies.items()})
            return model.model_validate(await get_response_json(response))
        except Exception as error:
            raise self.failure(error, stage=stage) from error

    async def _close_resources(self, episode: _Episode) -> None:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id=episode.resources_session_id, episode_id=episode.request.episode_id
            ),
            cookies=episode.resources_cookies,
        )
        await raise_for_status(response)


def _empty_response() -> NeMoGymResponse:
    """The verify body's response for an episode whose first turn never finished."""
    return NeMoGymResponse(
        id="no-response",
        created_at=0,
        model="none",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=False,
        tools=[],
    )


if __name__ == "__main__":
    BirdInteractEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BirdInteractEnvironmentServer.run_webserver()  # noqa: F401
