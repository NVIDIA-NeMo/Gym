# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD-Interact on the user-simulation base class.

c-Interact is environment-driven: each agent reply is a clarifying question or an SQL submission, and the next user
message is the simulator's answer, the debug prompt or the follow-up question. a-Interact is agent-driven: the agent
gets one turn and reaches the user, the database and the grader through tools on the Resources Server.
"""

import re
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from nemo_gym.base_environment_server import CleanupContext
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
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
from nemo_gym.tool_access import DirectHTTPToolAccess
from resources_servers.bird_interact.schemas import (
    AskRequest,
    AskResponse,
    BirdInteractSeedSessionResponse,
    BirdInteractTask,
    BirdInteractVerifyRequest,
    BirdInteractVerifyResponse,
    EpisodeContext,
    InteractionMode,
    PhaseResult,
    SubmitRequest,
    SubmitResponse,
)


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


def _function(name: str, description: str, properties: dict[str, str] | None = None) -> dict[str, Any]:
    properties = properties or {}
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": "string", "description": text} for key, text in properties.items()},
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


A_INTERACT_TOOLS = [
    _function("ask", "Ask the user one clarifying question.", {"question": "The question."}),
    _function("submit", "Submit your final SQL for the current question.", {"sql": "The SQLite query."}),
    _function("execute", "Run an exploratory SQLite query and see up to 20 rows.", {"sql": "The SQLite query."}),
    _function("get_schema", "Read the database schema, with sample rows."),
    _function("get_all_column_meanings", "Read the meaning of every column."),
    _function(
        "get_column_meaning",
        "Read the meaning of one column.",
        {"table_name": "The table.", "column_name": "The column."},
    ),
    _function("get_all_external_knowledge_names", "List the names of the knowledge-base entries."),
    _function("get_knowledge_definition", "Read one knowledge-base entry.", {"knowledge_name": "The entry's name."}),
    _function("get_all_knowledge_definitions", "Read every knowledge-base entry."),
]


class BirdInteractEnvironmentServerConfig(UserSimulationEnvironmentServerConfig):
    """Bind the BIRD-Interact Resources Server and choose the interaction mode."""

    resources_server: ResourcesServerRef
    mode: InteractionMode

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


@dataclass
class _Episode:
    request: BirdInteractRequest
    resources_session_id: str = field(default_factory=lambda: f"resources-session-{uuid4().hex}")
    resources_cookies: dict[str, str] = field(default_factory=dict)
    context: EpisodeContext | None = None
    opening: NeMoGymResponseCreateParamsNonStreaming | None = None


def extract_sql(text: str) -> str | None:
    """Return the last ```sql or ```sqlite block of a reply, or None for a reply without one."""
    blocks = _SQL_BLOCK.findall(text)
    return blocks[-1].strip() if blocks and blocks[-1].strip() else None


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
        episode.context = seed.context
        return episode

    async def agent_access(self, episode: _Episode) -> AgentAccess:
        if self.config.mode == "c_interact":
            return AgentAccess()
        base_url = self.server_client._resolve_base_url(self.config.resources_server.name).rstrip("/")
        return AgentAccess(
            tool_accesses=[
                DirectHTTPToolAccess(
                    name=f"{self.config.resources_server.name}.a_interact",
                    required=True,
                    base_url=f"{base_url}/a_interact",
                    cookies=episode.resources_cookies,
                )
            ]
        )

    async def opening_request(self, episode: _Episode) -> NeMoGymResponseCreateParamsNonStreaming:
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
            tools = A_INTERACT_TOOLS
        episode.opening = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "system", "content": instructions}, {"role": "user", "content": message}], tools=tools
        )
        return episode.opening

    async def next_user_turn(self, episode: _Episode, turn: TurnRecord) -> UserMessage | Stop:
        if self.config.mode == "a_interact":
            return Stop(reason="agent_finished")
        reply = turn.response.output_text
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
            BirdInteractVerifyRequest(responses_create_params=episode.opening, response=last_response),
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
