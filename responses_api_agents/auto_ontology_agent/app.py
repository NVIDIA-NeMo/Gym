# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Auto Ontology Agent Server: Auto Ontology's text-to-SQL agent, run as a Gym agent session, in two modes.

Auto Ontology (https://github.com/NVIDIA/auto-ontology) is a Python library.

- **One question** (BIRD dev): Auto Ontology answers with one ReAct episode: the model searches Auto Ontology's catalog
  (``retrieve``), runs SQL on the source database (``run_sql``) and submits a query (``submit``). The Environment
  Server seeds the session with the question, its evidence and its database in the seed's agent context, and
  activates it once. The agent answers with the submitted query in a fenced ``sql`` block, which the benchmark's
  verifier grades.
- **c-Interact** (BIRD-Interact): Auto Ontology's interactive mode advances a session one action at a time: ask the
  user, or submit SQL. The ``bird_interact`` Environment Server seeds the session with continuation and a grant for
  the episode's ``ask_user`` and ``submit_sql`` tools, and activates it once for each turn. In a turn, the agent steps
  Auto Ontology until it submits, and runs each action as an episode tool.

See README.md.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, time
from typing import Any, Callable, Optional
from uuid import uuid4

import httpx
from fastapi import HTTPException, Request, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
)

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentContinuationCapabilities,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.client import request as http_request
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_runtime import BaseRunRequest, DirectHTTPToolAccess
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
    TrajectoryToolCall,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.auto_ontology_agent import c_interact
from responses_api_agents.auto_ontology_agent.model_calls import ModelCallRoute, ModelCallTransport, model_call_route
from responses_api_agents.auto_ontology_agent.runtime import AskUser, AutoOntologyRuntime, Episode, TurnKind


LOG = logging.getLogger(__name__)

# Auto Ontology's model client needs a key; a Gym Model Server holds the provider's key itself.
_MODEL_SERVER_API_KEY = "gym-model-server"
# Model names with these prefixes select Auto Ontology's ChatOpenAI client, the one that can send through Gym.
_CHAT_OPENAI_PREFIXES = ("openai/", "azure/", "aws/")
C_INTERACT_TOOLS = ("ask_user", "submit_sql")
_DATABASE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


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

    # The Gym Model Server of the model that decides what to search, ask and submit: the policy. Auto Ontology calls it
    # its reasoning model.
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
    # Auto Ontology's numeric type alignment. auto-ontology-eval leaves it off for BIRD dev and turns it on for
    # BIRD-Interact.
    numeric_type_alignment: bool = False
    # Episodes and c-Interact steps run in threads; this bounds how many run at once across sessions.
    max_concurrent_episodes: PositiveInt = 8

    # c-Interact only. The interactive mode also makes helper calls (merging the user's answers into the question,
    # evidence checks, structured extraction) on Auto Ontology's fast model, usually on another Model Server.
    fast_model_server: Optional[ModelServerRef] = None
    fast_model: Optional[str] = None
    # A directory with ``<database>/<database>_kb.jsonl``, to read each knowledge entry's ``children_knowledge``,
    # which the ADK hides from agents. Off by default: then the agent has only the access the protocol gives.
    knowledge_children_dir: Optional[str] = None
    # An argument of Auto Ontology's create_session, as auto-ontology-eval sets it for BIRD-Interact.
    skip_evidence_refinement: bool = True
    # Steps a turn may take beyond the question budget before the agent stops it without a submission.
    step_margin: NonNegativeInt = 5
    tool_call_timeout_seconds: PositiveFloat = 600
    continuation_close_timeout_seconds: PositiveFloat = 30

    @field_validator("source_database_url")
    @classmethod
    def names_the_database(cls, value: str) -> str:
        if "{database}" not in value:
            raise ValueError("source_database_url must contain a {database} field")
        return value

    @field_validator("model", "fast_model")
    @classmethod
    def selects_chat_openai(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.startswith(_CHAT_OPENAI_PREFIXES):
            raise ValueError(
                f"{value!r} must start with {', '.join(_CHAT_OPENAI_PREFIXES)}: only Auto Ontology's "
                "OpenAI-compatible client sends through Gym. Gym's Model Servers send their own model name, so "
                "openai/<name> works"
            )
        return value


@dataclass
class AutoOntologySessionState(AgentSessionState):
    """One task's session: its task or its c-Interact grant, and its evidence.

    A session with ``task`` runs one ReAct episode; a session with ``tool_access`` runs c-Interact.
    """

    task: Optional[AutoOntologyTask] = None
    # c-Interact: the grant for the episode's tools, Auto Ontology's session and the question budget.
    tool_access: Optional[DirectHTTPToolAccess] = None
    session: Any = None
    question_budget: int = 0
    tool_calls: list[TrajectoryToolCall] = field(default_factory=list)
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
    _knowledge_children: dict[str, dict[int, list[int]]] = PrivateAttr(default_factory=dict)

    def runtime(self) -> AutoOntologyRuntime:
        """The Auto Ontology of this process, set up with this agent's settings on first use, on the event loop."""
        if self._runtime is None:
            config = self.config
            model_url = self.resolve_model_base_url(config.model_server.name, trace_context=False)
            model_servers = {model_url: config.model_server.name}
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
            if config.fast_model_server is not None and config.fast_model is not None:
                fast_url = self.resolve_model_base_url(config.fast_model_server.name, trace_context=False)
                model_servers[fast_url] = config.fast_model_server.name
                environment |= {
                    "NON_REASONING_ENDPOINT": fast_url,
                    "NON_REASONING_API_KEY": _MODEL_SERVER_API_KEY,
                    "NON_REASONING_MODEL": config.fast_model,
                }
            self._model_transport = ModelCallTransport(
                self.server_client, model_servers, loop=asyncio.get_running_loop()
            )
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

    def parse_agent_context(self, request: AgentSeedSessionRequest) -> Optional[BaseModel]:
        """The task for one question; none for a c-Interact seed, whose task comes in its turns."""
        if request.continuation is not None and request.agent_context is None:
            return None
        return super().parse_agent_context(request)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AutoOntologySessionState:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("Auto Ontology runs in the Agent Server's process and uses no task sandbox")
        if body.continuation is not None:
            return self._seed_c_interact(body)
        # One question: Auto Ontology's tools are its own; it calls no episode tools.
        if required := [grant.name for grant in self.effective_tool_accesses(body) if grant.required]:
            raise AgentSeedRejectedError(
                f"Auto Ontology uses its own tools and cannot use the required grants {required}"
            )
        # The base class refuses a context that does not satisfy AutoOntologyTask, a missing one included.
        task = self.parse_agent_context(body)
        return AutoOntologySessionState(request=body, task=task, unrouted_calls_at_seed=self._unrouted_calls())

    def _seed_c_interact(self, body: AgentSeedSessionRequest) -> AutoOntologySessionState:
        """A c-Interact session: continuation, and one direct HTTP grant for the episode's tools."""
        if self.config.fast_model_server is None or self.config.fast_model is None:
            raise AgentSeedRejectedError(
                "c-Interact needs Auto Ontology's fast model: set fast_model_server and fast_model"
            )
        if body.agent_context is not None:
            raise AgentSeedRejectedError(
                "A c-Interact session reads its task from its turns, not from an agent context"
            )
        grants = self.effective_tool_accesses(body)
        direct = [grant for grant in grants if isinstance(grant, DirectHTTPToolAccess)]
        unusable = [grant.name for grant in grants if not isinstance(grant, DirectHTTPToolAccess) and grant.required]
        if unusable:
            raise AgentSeedRejectedError(f"Auto Ontology calls tools over direct HTTP only; it cannot use {unusable}")
        if len(direct) != 1:
            raise AgentSeedRejectedError(
                f"Auto Ontology needs one direct HTTP grant for the episode's {' and '.join(C_INTERACT_TOOLS)} tools, "
                f"got {len(direct)}"
            )
        return AutoOntologySessionState(
            request=body, tool_access=direct[0], unrouted_calls_at_seed=self._unrouted_calls()
        )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, AutoOntologySessionState):
            raise TypeError("Expected Auto Ontology session state")
        tool_observations = [
            ToolCallObservation.model_validate(call.model_dump(exclude={"output"})) for call in state.tool_calls
        ]
        gaps = self._model_call_gaps(state)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(
                source="auto_ontology", records=[*state.invocations, *tool_observations], gaps=gaps
            ),
            trajectory=TrajectoryRecord(
                task_id=state.request.task_id.task_id,
                rollout_id=state.request.episode_id.capture_key,
                invocations=list(state.invocations),
                tool_calls=list(state.tool_calls),
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

    # ----- c-Interact ----- #

    def _agent_continuation_capabilities(self) -> AgentContinuationCapabilities:
        return AgentContinuationCapabilities(observations=["trajectory", "timing"])

    def _continuation_close_timeout(self) -> float:
        return self.config.continuation_close_timeout_seconds

    def _validate_agent_activation(
        self, state: AgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> None:
        if not isinstance(state, AutoOntologySessionState) or state.tool_access is None:
            raise HTTPException(409, "Only a c-Interact session takes turns")
        if not body.input:
            raise HTTPException(422, "An activation needs the message that opens its turn")

    async def _activate_agent_session_state(
        self, state: AgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming, *, activation_id: int
    ) -> NeMoGymResponse:
        """Run one c-Interact turn: step Auto Ontology until it submits, running each action as an episode tool."""
        if not isinstance(state, AutoOntologySessionState):
            raise TypeError("Expected Auto Ontology session state")
        invocation_id = "root" if activation_id == 1 else f"activation-{activation_id}"
        began = perf_counter()
        output: list[Any] = []
        status, error_type = "failed", None
        route = ModelCallRoute(path_prefix=state.model_path_prefix, invocation_id=invocation_id)
        state.activation_without_rollout |= not route.path_prefix
        try:
            # Auto Ontology's threads copy the route, so each model call of this turn takes it.
            with model_call_route(route):
                if state.session is None:
                    turn, text = await self._open(state, body)
                else:
                    next_turn = c_interact.read_next_turn(_message_text(body.input, ("user",)))
                    turn, text = next_turn.kind, next_turn.text
                await self._run_turn(state, turn, text, invocation_id, output)
            status = "completed"
        except c_interact.ProtocolError as error:
            error_type = type(error).__name__
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            state.invocations.append(
                AgentInvocation(
                    invocation_id=invocation_id,
                    activation_id=activation_id,
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
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=False,
            usage=usage,
        )

    async def _open(
        self, state: AutoOntologySessionState, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> tuple[TurnKind, str]:
        """Open the Auto Ontology session from the first activation: the ADK's instruction and phase 1 message."""
        declared = {tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None) for tool in body.tools}
        if missing := [name for name in C_INTERACT_TOOLS if name not in declared]:
            raise HTTPException(
                422, f"Auto Ontology runs c-Interact, but the first activation does not declare {missing}"
            )
        instructions = body.instructions or _message_text(body.input, ("system", "developer"))
        context = c_interact.read_instructions(instructions)
        question = c_interact.read_question(_message_text(body.input, ("user",)))
        knowledge, children = c_interact.format_knowledge(context.knowledge, self._children_of(context.database))
        state.question_budget = context.question_budget
        state.session = await self._in_thread(
            self.runtime().create_session,
            session_id=state.request.agent_session_id,
            task_id=state.request.task_id.task_id,
            database=context.database,
            schema=context.schema,
            knowledge=knowledge,
            knowledge_children=children,
            question_budget=context.question_budget,
            skip_evidence_refinement=self.config.skip_evidence_refinement,
        )
        return "initial", question

    async def _run_turn(
        self,
        state: AutoOntologySessionState,
        turn: TurnKind,
        text: Optional[str],
        invocation_id: str,
        output: list[Any],
    ) -> None:
        runtime = self.runtime()
        # Auto Ontology stops asking at the question budget, so a healthy turn ends well within this bound.
        for _ in range(state.question_budget + self.config.step_margin + 1):
            action = await self._in_thread(runtime.step, state.session, turn=turn, text=text)
            if isinstance(action, AskUser):
                reply = await self._call_tool(state, "ask_user", {"question": action.question}, invocation_id, output)
                await self._in_thread(runtime.apply_user_answer, state.session, c_interact.user_answer(reply))
                continue
            sql = c_interact.strip_catalog_qualifiers(action.sql)
            reply = await self._call_tool(state, "submit_sql", {"sql": sql}, invocation_id, output)
            await self._in_thread(runtime.apply_submit_result, state.session, c_interact.submit_result(reply))
            return
        LOG.warning(
            "Auto Ontology did not submit within %d steps: agent_session_id=%s",
            state.question_budget + self.config.step_margin + 1,
            state.request.agent_session_id,
        )

    async def _call_tool(
        self,
        state: AutoOntologySessionState,
        name: str,
        arguments: dict[str, str],
        invocation_id: str,
        output: list[Any],
    ) -> str:
        """Call one of the episode's tools and return its reply; record the call in the response and the trajectory."""
        if state.tool_access is None:
            raise TypeError("Expected a c-Interact session")
        call_id = f"call_{uuid4().hex}"
        output.append(
            NeMoGymResponseFunctionToolCall(
                type="function_call", id=call_id, call_id=call_id, name=name, arguments=json.dumps(arguments)
            )
        )
        started, began = time(), perf_counter()
        reply: Optional[str] = None
        error_type: Optional[str] = None
        try:
            async with asyncio.timeout(self.config.tool_call_timeout_seconds):
                # Each call has an effect on the episode, a question or a graded submission, so a call that may have
                # reached the server is not sent again.
                response = await http_request(
                    method="POST",
                    url=f"{str(state.tool_access.base_url).rstrip('/')}/{name}",
                    json=arguments,
                    headers=dict(state.tool_access.headers),
                    cookies=dict(state.tool_access.cookies),
                    _internal=True,
                    _resend_after_disconnect=False,
                    _retry_other_errors=False,
                )
                await raise_for_status(response)
                result = await get_response_json(response)
            reply = result if isinstance(result, str) else json.dumps(result)
            output.append(NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=reply))
            return reply
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            state.tool_calls.append(
                TrajectoryToolCall(
                    invocation_id=invocation_id,
                    tool_call_id=call_id,
                    tool_name=name,
                    started_at=started,
                    completed_at=max(time(), started),
                    duration_ms=(perf_counter() - began) * 1000,
                    timing_source="executor",
                    status="completed" if reply is not None else "failed",
                    error_type=error_type,
                    output=reply,
                )
            )

    def _children_of(self, database: str) -> Optional[dict[int, list[int]]]:
        """Each knowledge entry's ``children_knowledge``, when ``knowledge_children_dir`` is set."""
        if self.config.knowledge_children_dir is None:
            return None
        if not _DATABASE_NAME.match(database):
            raise c_interact.ProtocolError(f"Unexpected database name {database!r}")
        if database not in self._knowledge_children:
            path = Path(self.config.knowledge_children_dir) / database / f"{database}_kb.jsonl"
            children: dict[int, list[int]] = {}
            for line in path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("id") is not None and isinstance(entry.get("children_knowledge"), list):
                        children[entry["id"]] = entry["children_knowledge"]
            self._knowledge_children[database] = children
        return self._knowledge_children[database]

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
        state = self._require_agent_session(agent_session_id)
        if isinstance(state, AutoOntologySessionState) and state.tool_access is not None:
            # c-Interact: one activation for each turn, which the base class orders, locks and replays. The
            # Environment Server activates the session on its rollout's route, which the turn's model calls take.
            state.model_path_prefix = self.url_path_for_request("", request)
            return await self.activate_agent_session(request, body)
        # One question: one activation at a time; one sent again while the episode runs waits for its reply.
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


def _message_text(items: Any, roles: tuple[str, ...]) -> str:
    """The text of the input's messages with one of ``roles``, joined by blank lines."""
    if isinstance(items, str):
        return items if "user" in roles else ""
    texts = []
    for item in items:
        item = item if isinstance(item, dict) else item.model_dump()
        if item.get("type", "message") != "message" or item.get("role") not in roles:
            continue
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.extend(part.get("text", "") for part in content or [] if isinstance(part, dict))
    return "\n\n".join(texts)


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
