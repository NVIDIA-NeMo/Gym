# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Auto Ontology Agent Server: Auto Ontology's ReAct text-to-SQL agent, run as a Gym agent session.

Auto Ontology (https://github.com/NVIDIA/auto-ontology) answers a question with one ReAct episode: the model searches
Auto Ontology's catalog (``retrieve``), runs SQL on the source database (``run_sql``) and submits a query (``submit``).
The Environment Server seeds one agent session for each task, with the question, its evidence and its database in the
seed's agent context, and activates it once. The agent runs the episode and answers with the submitted query in a
fenced ``sql`` block, which the benchmark's verifier grades. See README.md.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from time import perf_counter, time
from typing import Any, Callable, Optional
from uuid import uuid4

import httpx
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, PrivateAttr, field_validator

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_runtime import BaseRunRequest
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.auto_ontology_agent.model_calls import ModelCallRoute, ModelCallTransport, model_call_route
from responses_api_agents.auto_ontology_agent.runtime import AutoOntologyRuntime, Episode


LOG = logging.getLogger(__name__)

# Auto Ontology's model client needs a key; a Gym Model Server holds the provider's key itself.
_MODEL_SERVER_API_KEY = "gym-model-server"
# Model names with these prefixes select Auto Ontology's ChatOpenAI client, the one that can send through Gym.
_CHAT_OPENAI_PREFIXES = ("openai/", "azure/", "aws/")


class AutoOntologyTask(BaseModel):
    """The task, from the seed's agent context: what Auto Ontology answers, and on which database."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    # The source database's name, which fills ``source_database_url``.
    database: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    # BIRD-style evidence, which Auto Ontology shows the model apart from the question.
    evidence: str = ""


class AutoOntologyCatalogConfig(BaseModel):
    """The PostgreSQL database that holds Auto Ontology's catalog and vector stores (its ``POSTGRES_*`` settings)."""

    model_config = ConfigDict(extra="forbid")

    host: str
    port: PositiveInt = 5432
    user: str
    password: str
    database: str


class AutoOntologyAgentConfig(BaseResponsesAPIAgentConfig):
    """Settings of the Auto Ontology agent. Auto Ontology reads them when the first episode starts; see README.md."""

    # The Gym Model Server of the model the ReAct episode runs on: the policy.
    model_server: ModelServerRef
    # The name Auto Ontology sends. It needs an ``openai/``, ``azure/`` or ``aws/`` prefix, which selects Auto
    # Ontology's OpenAI-compatible client. Gym's Model Servers send their own model name, so ``openai/<any name>``
    # works.
    model: str
    # Auto Ontology's embedding endpoint, for retrieval. The catalog must have been ingested with the same model.
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    catalog: AutoOntologyCatalogConfig
    # A connection string with a ``{database}`` field, for the source databases that ``run_sql`` queries. The catalog
    # names each database as its ingest's connection string did, so use the same strings.
    source_database_url: str
    # Environment variables that Auto Ontology reads at import, such as the ReAct agent's limits and tools
    # (``TEXT_TO_SQL_REACT_MAX_TOOL_CALLS``, ``TEXT_TO_SQL_REACT_SUBMIT_TOOL``, ``TEXT_TO_SQL_REACT_GREP``).
    feature_flags: dict[str, str] = {}
    # Auto Ontology's numeric type alignment, which auto-ontology-eval leaves off.
    numeric_type_alignment: bool = False
    # Episodes run in threads; this bounds how many run at once across sessions.
    max_concurrent_episodes: PositiveInt = 8

    @field_validator("source_database_url")
    @classmethod
    def names_the_database(cls, value: str) -> str:
        if "{database}" not in value:
            raise ValueError("source_database_url must contain a {database} field")
        return value

    @field_validator("model")
    @classmethod
    def selects_chat_openai(cls, value: str) -> str:
        if not value.startswith(_CHAT_OPENAI_PREFIXES):
            raise ValueError(
                f"{value!r} must start with {', '.join(_CHAT_OPENAI_PREFIXES)}: only Auto Ontology's "
                "OpenAI-compatible client sends through Gym. Gym's Model Servers send their own model name, so "
                "openai/<name> works"
            )
        return value


@dataclass
class AutoOntologySessionState(AgentSessionState):
    """One task's session: the task, its activation's evidence and how its episode ended."""

    task: Optional[AutoOntologyTask] = None
    # The activation's rollout route, which its model calls also take; empty without a rollout.
    model_path_prefix: str = ""
    # Calls made outside any activation, across the process, when the session was seeded.
    unrouted_calls_at_seed: int = 0
    activation_without_rollout: bool = False
    usage_unavailable: bool = False
    invocations: list[AgentInvocation] = field(default_factory=list)
    # The reply of the session's one episode, given again to an activation that is sent again.
    reply: Optional[NeMoGymResponse] = None


class AutoOntologyAgent(SimpleResponsesAPIAgent):
    config: AutoOntologyAgentConfig
    agent_context_model = AutoOntologyTask

    _runtime: Optional[AutoOntologyRuntime] = PrivateAttr(default=None)
    _model_transport: Optional[ModelCallTransport] = PrivateAttr(default=None)
    _episodes: Optional[asyncio.Semaphore] = PrivateAttr(default=None)

    def runtime(self) -> AutoOntologyRuntime:
        """The Auto Ontology of this process, set up with this agent's settings on first use, on the event loop."""
        if self._runtime is None:
            config = self.config
            model_url = self.resolve_model_base_url(config.model_server.name, trace_context=False)
            self._model_transport = ModelCallTransport(
                self.server_client, {model_url: config.model_server.name}, loop=asyncio.get_running_loop()
            )
            environment = {
                **config.feature_flags,
                "REASONING_ENDPOINT": model_url,
                "REASONING_API_KEY": _MODEL_SERVER_API_KEY,
                "REASONING_MODEL": config.model,
                "EMBED_ENDPOINT": config.embedding_base_url,
                "EMBED_API_KEY": config.embedding_api_key,
                "EMBED_MODEL": config.embedding_model,
                "POSTGRES_HOST": config.catalog.host,
                "POSTGRES_PORT": str(config.catalog.port),
                "POSTGRES_USER": config.catalog.user,
                "POSTGRES_PASSWORD": config.catalog.password,
                "POSTGRES_DATABASE": config.catalog.database,
            }
            self._runtime = AutoOntologyRuntime(
                environment,
                config.source_database_url,
                http_client=httpx.Client(transport=self._model_transport),
                numeric_type_alignment=config.numeric_type_alignment,
            )
        return self._runtime

    def _unrouted_calls(self) -> int:
        return self._model_transport.unrouted_calls if self._model_transport is not None else 0

    async def _in_thread(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous Auto Ontology call in a thread, at most ``max_concurrent_episodes`` at a time."""
        if self._episodes is None:
            self._episodes = asyncio.Semaphore(self.config.max_concurrent_episodes)
        async with self._episodes:
            return await asyncio.to_thread(function, *args, **kwargs)

    # ----- agent session ----- #

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AutoOntologySessionState:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("Auto Ontology runs in the Agent Server's process and uses no task sandbox")
        # Auto Ontology's tools are its own; it calls no episode tools.
        if required := [grant.name for grant in self.effective_tool_accesses(body) if grant.required]:
            raise AgentSeedRejectedError(
                f"Auto Ontology uses its own tools and cannot use the required grants {required}"
            )
        # The base class refuses a context that does not satisfy AutoOntologyTask, a missing one included.
        task = self.parse_agent_context(body)
        return AutoOntologySessionState(request=body, task=task, unrouted_calls_at_seed=self._unrouted_calls())

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, AutoOntologySessionState):
            raise TypeError("Expected Auto Ontology session state")
        gaps = self._model_call_gaps(state)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(
                source="auto_ontology", records=list(state.invocations), gaps=gaps
            ),
            trajectory=TrajectoryRecord(
                task_id=state.request.task_id.task_id,
                rollout_id=state.request.episode_id.capture_key,
                invocations=list(state.invocations),
                gaps=list(gaps),
            ),
        )

    def _model_call_gaps(self, state: AutoOntologySessionState) -> list[ObservationGap]:
        """The model-call evidence the session lacks.

        Gym gives each captured call to its invocation by its ``x-session-id``, so no gap means every call has both.
        """
        gaps = []
        if state.activation_without_rollout:
            gaps.append(
                ObservationGap(
                    code="model_call_ownership_unavailable",
                    detail="The activation came without a rollout route, so its model calls carry no rollout",
                )
            )
        elif self._unrouted_calls() > state.unrouted_calls_at_seed:
            # The transport cannot tell whose call it was, so each session open at the time declares the gap.
            gaps.append(
                ObservationGap(
                    code="model_call_ownership_unavailable",
                    detail="Auto Ontology made a model call outside an activation while this session was open",
                )
            )
        if state.usage_unavailable:
            gaps.append(
                ObservationGap(code="model_call_usage_unavailable", detail="A model call's reply had no usage")
            )
        return gaps

    async def _run_episode(
        self, state: AutoOntologySessionState, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """Run the task's ReAct episode, and answer with its query.

        The activation's input, the benchmark's own prompt, is not used: Auto Ontology writes its prompts from the
        task in the agent context.
        """
        if state.task is None:
            raise TypeError("Expected a seeded Auto Ontology session state")
        # A failed episode that the Environment Server sends again runs again, under its own invocation.
        invocation_id = "root" if not state.invocations else f"activation-{len(state.invocations) + 1}"
        began = perf_counter()
        status, error_type = "failed", None
        route = ModelCallRoute(path_prefix=state.model_path_prefix, invocation_id=invocation_id)
        state.activation_without_rollout |= not route.path_prefix
        task = state.task
        try:
            # asyncio.to_thread copies the route, so each model call of the episode takes it.
            with model_call_route(route):
                episode = await self._in_thread(
                    self.runtime().run_episode, question=task.question, evidence=task.evidence, database=task.database
                )
            status = "completed"
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            state.invocations.append(
                AgentInvocation(
                    invocation_id=invocation_id,
                    activation_id=1,
                    in_session=True,
                    status=status,
                    duration_ms=(perf_counter() - began) * 1000,
                    error_type=error_type,
                )
            )
        usage = route.usage()
        state.usage_unavailable |= usage is None
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=[_answer(episode)],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=False,
            usage=usage,
            metadata={
                **(body.metadata or {}),
                "auto_ontology_end_reason": episode.reason,
                "auto_ontology_tool_calls": str(episode.tool_calls),
            },
        )

    # ----- routes ----- #

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            raise HTTPException(409, "Auto Ontology runs only in agent sessions; use an Environment Server")
        # One activation at a time: one sent again while the episode runs waits for its reply.
        async with self._locked_agent_session(agent_session_id):
            state = self._require_agent_session(agent_session_id)
            if not isinstance(state, AutoOntologySessionState):
                raise TypeError("Expected Auto Ontology session state")
            rollout_id = request.path_params.get("rollout_id")
            if rollout_id is not None and rollout_id != state.request.episode_id.capture_key:
                raise AgentSessionConflictError("Activation rollout does not match the seeded session")
            if state.reply is not None:
                # The session's one episode has run; an activation sent again gets its reply.
                return state.reply
            # The Environment Server activates the session on its rollout's route, with the training-capture segment
            # when training capture is on. Auto Ontology's model calls take the same route.
            state.model_path_prefix = self.url_path_for_request("", request)
            state.reply = await self._run_episode(state, body)
            return state.reply

    async def run(self, request: Request, body: BaseRunRequest = Body()) -> None:
        raise HTTPException(409, "Auto Ontology runs only in agent sessions; use an Environment Server")


def _answer(episode: Episode) -> NeMoGymResponseOutputMessage:
    """The agent's answer: the submitted query in a fenced ``sql`` block, or why the episode has none."""
    if episode.sql is not None:
        text = f"```sql\n{episode.sql.strip()}\n```"
    else:
        text = f"Auto Ontology submitted no query ({episode.reason})."
    return NeMoGymResponseOutputMessage(
        id=f"msg_{uuid4().hex}",
        role="assistant",
        status="completed",
        type="message",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
    )


if __name__ == "__main__":
    AutoOntologyAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AutoOntologyAgent.run_webserver()  # noqa: F401
