# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Auto Ontology Agent Server: Auto Ontology's ReAct text-to-SQL agent, one sandbox per episode.

Auto Ontology (https://github.com/NVIDIA/auto-ontology) answers a question with one ReAct episode: its model searches
Auto Ontology's catalog (``retrieve``), runs SQL (``run_sql``) and submits a query (``submit``). This server runs that
episode as Auto Ontology's own ``ReactSQLAgent`` does, with Auto Ontology's prompts, tools and loop, on the policy.

An Environment Server seeds one agent session per task, with the question, its evidence and its database in the seed's
agent context, and activates it once. At the seed this server starts a sandbox from the catalog image (Auto Ontology,
the databases and their catalog; ``benchmarks/birdbench/auto_ontology``) with the episode runner
(``episode_runner.py``) in it, and starts the catalog's Postgres. At the activation it runs the episode there and
answers with the submitted query in a fenced ``sql`` block, which the benchmark's Resources Server grades. At the
close it stops the sandbox. The episode's model calls go from the sandbox to the policy's Model Server, on the
rollout's route, so Gym serves and captures each of them.

BIRD-Interact's c-Interact (``c_interact.py``) is a multi-turn session instead: the seed carries no task but grants the
Environment Server's ``ask_user`` and ``submit_sql``, and each turn's request holds the ADK's texts. Each turn runs one
episode in the same sandbox, with ``ask_user`` added to Auto Ontology's tools, and sends the query Auto Ontology
accepts to ``submit_sql``. The agent keeps the user's answers and its submissions across the turns.

This process imports only NeMo Gym: Auto Ontology runs in the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, time
from typing import Any, Optional, cast
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentContinuationCapabilities,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SandboxCommand,
    SandboxNetworkAccess,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    restrict_sandbox_network,
)
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxSpec,
    create_provider,
    read_text,
    resolve_provider_config,
    upload_text,
)
from nemo_gym.server_runtime import DirectHTTPToolAccess
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.auto_ontology_agent import c_interact


LOG = logging.getLogger(__name__)

RUNNER_SOURCE = Path(__file__).with_name("episode_runner.py")
# Where the sandbox gets the episode runner, through SandboxSpec.files, and the episode's files.
SANDBOX_DIR = "/tmp/nemo-gym-auto-ontology"
SANDBOX_RUNNER = f"{SANDBOX_DIR}/episode_runner.py"


class AutoOntologyTask(BaseModel):
    """The task, from the seed's agent context."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    # The source database's name, as the catalog knows it (BIRD's db_id).
    database: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    evidence: str = ""


class AutoOntologyAgentConfig(BaseResponsesAPIAgentConfig):
    """Settings of the Auto Ontology agent. See README.md."""

    # The Gym Model Server the episode runs on: the policy.
    model_server: ModelServerRef
    # The name Auto Ontology sends. Its openai/ prefix selects Auto Ontology's
    # OpenAI-compatible client; the Model Server sends its own model name.
    model: str = Field(default="openai/policy", pattern=r"^(openai|azure|aws)/")
    # A sandbox provider block of the global config, and the sandbox to start
    # with it: at least the catalog image.
    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    network_access: SandboxNetworkAccess = "inherit"
    # In the image: the Python that has Auto Ontology, the folder whose .env
    # names the source databases and the embedding endpoint, and the command
    # that starts the catalog's Postgres on $POSTGRES_PORT.
    catalog_python: str = "/opt/auto-ontology/.venv/bin/python"
    catalog_workdir: str = "/opt/bird-catalog"
    catalog_start_command: str = "/opt/bird-catalog/start-catalog"
    # Environment variables for Auto Ontology in the sandbox, which it reads at
    # import: the key of its embedding and reranking endpoints
    # (DEFAULT_MODELS_API_KEY), and the ReAct agent's limits
    # (TEXT_TO_SQL_REACT_MAX_TOOL_CALLS, ...). Its model is set apart: the policy.
    environment: dict[str, str] = Field(default_factory=dict)
    setup_timeout_seconds: float = Field(default=300, gt=0)
    episode_timeout_seconds: float = Field(default=1800, gt=0)
    close_timeout_seconds: float = Field(default=60, gt=0)
    # Episodes at once in this server: each holds a sandbox with a Postgres.
    max_concurrent_episodes: int = Field(default=8, gt=0)
    # The loopback ports a sandbox's catalog may take. Sandboxes on the host
    # network share loopback, so each takes its own, at random from this range.
    catalog_ports: tuple[int, int] = (20000, 60000)
    catalog_port_attempts: int = Field(default=3, gt=0)


@dataclass(kw_only=True)
class AutoOntologySessionState(AgentSessionState):
    """One task's session: its task, its sandbox and its one activation, or its turns in c-Interact."""

    # None in c-Interact, whose turns carry the task.
    task: Optional[AutoOntologyTask]
    session: SandboxSession[dict[str, Any]]
    catalog_port: int
    activation: Optional[asyncio.Task[NeMoGymResponse]] = None
    activation_request: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    episode: Optional[dict[str, Any]] = None
    # c-Interact: the grant of the Environment Server's tools, the ask_user tool as Auto Ontology offers it, what the
    # agent knows across turns, and a record of each turn.
    tool_access: Optional[DirectHTTPToolAccess] = None
    ask_tool: Optional[dict[str, Any]] = None
    dialogue: Optional[c_interact.Dialogue] = None
    invocations: list[AgentInvocation] = field(default_factory=list)


class AutoOntologyAgent(SimpleResponsesAPIAgent):
    config: AutoOntologyAgentConfig
    agent_context_model = AutoOntologyTask

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._episodes = asyncio.Semaphore(self.config.max_concurrent_episodes)

    # ----- agent session ----- #

    def parse_agent_context(self, request: AgentSeedSessionRequest) -> BaseModel | None:
        # A multi-turn session (c-Interact) reads its task from each turn's request.
        if request.continuation is not None:
            if request.agent_context is not None and request.agent_context.fields():
                raise AgentSeedRejectedError("A multi-turn Auto Ontology session reads its task from each turn")
            return None
        return super().parse_agent_context(request)

    def _agent_continuation_capabilities(self) -> AgentContinuationCapabilities | None:
        return AgentContinuationCapabilities(observations=["trajectory", "timing"])

    def _continuation_close_timeout(self) -> float:
        return self.config.close_timeout_seconds

    def _c_interact_tool_access(self, body: AgentSeedSessionRequest) -> DirectHTTPToolAccess:
        """The grant of the Environment Server's ask_user and submit_sql, which the sandbox calls over HTTP."""
        accesses = self.effective_tool_accesses(body)
        if unusable := [a.name for a in accesses if a.required and not isinstance(a, DirectHTTPToolAccess)]:
            raise AgentSeedRejectedError(
                f"Auto Ontology calls an episode's tools over direct HTTP only, not {unusable}"
            )
        direct = [access for access in accesses if isinstance(access, DirectHTTPToolAccess)]
        if len(direct) != 1:
            raise AgentSeedRejectedError(
                f"A multi-turn Auto Ontology session needs one direct HTTP grant, of {c_interact.ASK_TOOL} and "
                f"{c_interact.SUBMIT_TOOL}; it was granted {len(direct)}"
            )
        return direct[0]

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AutoOntologySessionState:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError(
                "Auto Ontology does not support sandbox access: it runs in its own sandbox, from the Auto Ontology image"
            )
        tool_access = None
        if body.continuation is not None:
            tool_access = self._c_interact_tool_access(body)
        elif required := [grant.name for grant in self.effective_tool_accesses(body) if grant.required]:
            raise AgentSeedRejectedError(f"Auto Ontology uses its own tools and cannot use the grants {required}")
        # The base class refuses a context that does not satisfy AutoOntologyTask, a missing one included.
        task = cast(Optional[AutoOntologyTask], self.parse_agent_context(body))

        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        provider_name = next(iter(provider_config), None)
        model_base_url = self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key)
        # The sandbox calls the Model Server, and in c-Interact the Environment Server's tools.
        tool_urls = [str(tool_access.base_url)] if tool_access is not None else []
        try:
            for url in [model_base_url, *tool_urls]:
                assert_model_url_reachable_from_sandbox(url, provider_name=provider_name)
        except ValueError as error:
            raise AgentSeedRejectedError(str(error)) from error

        provider = create_provider(provider_config)
        try:
            sandbox_config = dict(self.config.sandbox_config)
            turns = c_interact.MAX_TURNS if body.continuation is not None else 1
            sandbox_config.setdefault(
                "ttl_s",
                self.config.setup_timeout_seconds
                + turns * self.config.episode_timeout_seconds
                + 2 * self.config.close_timeout_seconds,
            )
            files = {SANDBOX_RUNNER: RUNNER_SOURCE.read_text()}
            spec = restrict_sandbox_network(
                SandboxSpec(**sandbox_config, files=files),
                network_access=self.config.network_access,
                provider=provider,
                model_base_url=model_base_url,
                tool_urls=tool_urls,
                required_tool_urls=tool_urls,
            )
            sandbox = AsyncSandbox(provider)
            await sandbox.start(spec)
        except BaseException:
            await provider.aclose()
            raise

        session = SandboxSession[dict[str, Any]](
            sandbox=sandbox,
            workdir=None,
            session_dir=f"{SANDBOX_DIR}/sessions/{uuid4().hex}",
            owns_sandbox=True,
            harness="Auto Ontology",
        )
        try:
            catalog_port = await self._start_catalog(sandbox)
        except BaseException:
            await session.close(timeout=self.config.close_timeout_seconds)
            raise
        return AutoOntologySessionState(
            request=body, task=task, session=session, catalog_port=catalog_port, tool_access=tool_access
        )

    async def _start_catalog(self, sandbox: AsyncSandbox) -> int:
        """Start the sandbox's catalog on a free loopback port, and return the port."""
        low, high = self.config.catalog_ports
        failures = []
        for _ in range(self.config.catalog_port_attempts):
            port = random.randrange(low, high)
            started = await sandbox.exec(
                self.config.catalog_start_command,
                env={"POSTGRES_PORT": str(port)},
                timeout_s=self.config.setup_timeout_seconds,
            )
            if started.error_type is None and started.return_code == 0:
                return port
            failures.append(
                f"port {port}: {started.error_type or f'exit {started.return_code}'}: "
                f"{(started.stderr or started.stdout or '')[-1000:]}"
            )
        raise RuntimeError("The catalog did not start; " + "; ".join(failures))

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, AutoOntologySessionState):
            raise TypeError("Expected Auto Ontology session state")
        if state.activation is not None and not state.activation.done():
            state.activation.cancel()
        await state.session.close(timeout=self.config.close_timeout_seconds)
        gaps = [
            # The episode's model calls are captured per rollout by the Model
            # Server; this server does not see them, so it reports no usage.
            ObservationGap(
                code="model_call_usage_unavailable",
                detail="Usage is in the Model Server's capture",
            ),
        ]
        if state.episode is None:
            gaps.append(ObservationGap(code="observation_capture_failed"))
        trajectory = None
        if state.request.continuation is not None:
            # Each turn's input and answer. The model calls are in the Model Server's capture; the trajectory
            # builder reports their absence here.
            trajectory = TrajectoryRecord(
                task_id=state.request.task_id.task_id,
                rollout_id=state.request.episode_id.capture_key,
                invocations=state.invocations,
            )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(source="auto_ontology", records=state.invocations, gaps=gaps),
            cleanup_confirmed=state.session.cleanup_confirmed,
            trajectory=trajectory,
        )

    # ----- the episode ----- #

    async def _run_episode(
        self,
        state: AutoOntologySessionState,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponse:
        assert state.task is not None
        state.episode = await self._run_in_sandbox(state, state.task.model_dump())
        return self._answer(state.episode, body)

    async def _run_in_sandbox(
        self, state: AutoOntologySessionState, task: dict[str, Any], *, turn: Optional[int] = None
    ) -> dict[str, Any]:
        """Run one episode of ``task`` in the session's sandbox, with ``turn``'s own directory in c-Interact."""
        session = state.session
        if turn is not None:
            await session.begin_activation(
                session_dir=f"{session.root_dir}/turn-{turn}", timeout=self.config.close_timeout_seconds
            )
        task_path = f"{session.session_dir}/task.json"
        result_path = f"{session.session_dir}/result.json"
        payload = {
            **task,
            "model": self.config.model,
            "env_file": f"{self.config.catalog_workdir}/.env",
            # The sandbox calls the Model Server itself, on the rollout's route.
            "model_base_url": self.resolve_model_base_url(
                self.config.model_server.name, state.request.episode_id.capture_key
            ),
            # The sandbox's own catalog port wins over the configured settings.
            "environment": {
                **self.config.environment,
                "POSTGRES_PORT": str(state.catalog_port),
            },
        }

        async def stage() -> SandboxCommand:
            await upload_text(session.sandbox, path=task_path, text=json.dumps(payload))
            return SandboxCommand(argv=[self.config.catalog_python, SANDBOX_RUNNER, task_path, result_path])

        async def collect() -> dict[str, Any]:
            return json.loads(await read_text(session.sandbox, path=result_path))

        async with self._episodes:
            try:
                return await session.execute(
                    stage_activation=stage,
                    collect=collect,
                    timeout=self.config.episode_timeout_seconds,
                    close_timeout=self.config.close_timeout_seconds,
                )
            except Exception:
                log = await session.read_output_log()
                LOG.error(
                    "Auto Ontology episode failed for %s:\n%s",
                    state.request.task_id.task_id,
                    log[-4000:],
                )
                raise

    def _answer(self, episode: dict[str, Any], body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=[answer_message(episode)],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=False,
            metadata={
                **(body.metadata or {}),
                "auto_ontology_end_reason": str(episode.get("reason")),
                "auto_ontology_tool_calls": str(episode.get("tool_calls")),
            },
        )

    # ----- c-Interact's turns ----- #

    def _validate_agent_activation(
        self, state: AgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> None:
        if not isinstance(state, AutoOntologySessionState):
            raise TypeError("Expected Auto Ontology session state")
        system, message = request_texts(body)
        try:
            if state.dialogue is None:
                c_interact.read_task(system)
                ask_tool_definition(body)
            c_interact.read_turn(message)
        except c_interact.CInteractTextError as error:
            raise HTTPException(422, str(error)) from error

    async def _activate_agent_session_state(
        self, state: AgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming, *, activation_id: int
    ) -> NeMoGymResponse:
        """Run one c-Interact turn: one Auto Ontology episode, which may ask the user and submits through the server."""
        if not isinstance(state, AutoOntologySessionState) or state.tool_access is None:
            raise TypeError("Expected a c-Interact Auto Ontology session")
        system, message = request_texts(body)
        if state.dialogue is None:
            state.dialogue = c_interact.Dialogue(task=c_interact.read_task(system))
            state.ask_tool = ask_tool_definition(body)
        dialogue = state.dialogue
        turn = c_interact.read_turn(message)
        question = c_interact.advance(dialogue, turn)
        task = {
            "question": question,
            "evidence": c_interact.evidence(dialogue, turn),
            "database": dialogue.task.database,
            "custom_prompts": c_interact.asking_prompt(dialogue),
            "tools": [{"definition": state.ask_tool, **tool_endpoint(state.tool_access, c_interact.ASK_TOOL)}],
            "submit": tool_endpoint(state.tool_access, c_interact.SUBMIT_TOOL),
        }
        invocation = AgentInvocation(
            invocation_id="root" if activation_id == 1 else f"activation-{activation_id}",
            activation_id=activation_id,
            in_session=True,
            conversation=[NeMoGymEasyInputMessage(role="user", content=message)],
        )
        started = perf_counter()
        try:
            state.episode = episode = await self._run_in_sandbox(state, task, turn=activation_id)
        except BaseException as error:
            state.invocations.append(
                invocation.model_copy(
                    update={
                        "status": "failed" if isinstance(error, Exception) else "incomplete",
                        "error_type": type(error).__name__,
                        "duration_ms": (perf_counter() - started) * 1000,
                    }
                )
            )
            raise
        for call in episode.get("environment_tool_calls") or []:
            if call.get("name") == c_interact.ASK_TOOL:
                dialogue.exchanges.append(
                    c_interact.Exchange.from_reply(str(call["arguments"].get("question", "")), call["reply"])
                )
        if episode.get("sql") and episode.get("submit_reply") is not None:
            dialogue.submissions.append(
                c_interact.Submission(phase=dialogue.phase, sql=episode["sql"], reply=episode["submit_reply"])
            )
        response = self._answer(episode, body)
        state.invocations.append(
            invocation.model_copy(
                update={
                    "status": "completed",
                    "stop_reason": str(episode.get("reason")),
                    "duration_ms": (perf_counter() - started) * 1000,
                    "conversation": [*invocation.conversation, *response.output],
                }
            )
        )
        return response

    # ----- routes ----- #

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            raise AgentSessionConflictError("Auto Ontology runs only in agent sessions; use an Environment Server")
        state = self._require_agent_session(agent_session_id)
        if not isinstance(state, AutoOntologySessionState):
            raise TypeError("Expected Auto Ontology session state")
        if state.request.continuation is not None:
            return await self.activate_agent_session(request, body)
        rollout_id = request.path_params.get("rollout_id")
        if rollout_id != state.request.episode_id.capture_key:
            raise AgentSessionConflictError("The activation's rollout does not match the seeded session")
        if state.activation is None:
            # No await before the activation is installed: a retry joins it.
            state.activation_request = body.model_copy(deep=True)
            state.activation = asyncio.create_task(self._run_episode(state, body))
        elif body != state.activation_request:
            raise AgentSessionConflictError("An Auto Ontology session has one activation; retry the same request")
        # A waiter that disconnects must not cancel the episode; close does.
        return (await asyncio.shield(state.activation)).model_copy(deep=True)


def answer_message(episode: dict[str, Any]) -> NeMoGymResponseOutputMessage:
    """The agent's answer: the submitted query in a fenced ``sql`` block, or why the episode has none.

    In c-Interact the Environment Server's reply to the submission follows.
    """
    sql = episode.get("sql")
    if sql:
        text = f"```sql\n{str(sql).strip()}\n```"
    else:
        text = f"Auto Ontology submitted no query ({episode.get('reason')})."
    if episode.get("submit_reply"):
        text += f"\n\n{episode['submit_reply']}"
    return NeMoGymResponseOutputMessage(
        id=f"msg_{uuid4().hex}",
        role="assistant",
        status="completed",
        type="message",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def request_texts(body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
    """A c-Interact request's instructions (its system message) and the turn's message (its last user message)."""
    if isinstance(body.input, str):
        return body.instructions or "", body.input
    items = [item if isinstance(item, dict) else item.model_dump() for item in body.input]
    system = [_content_text(item.get("content")) for item in items if item.get("role") in ("system", "developer")]
    users = [_content_text(item.get("content")) for item in items if item.get("role") == "user"]
    if not users:
        raise HTTPException(422, "A c-Interact turn needs a user message")
    return "\n\n".join([*([body.instructions] if body.instructions else []), *system]), users[-1]


def ask_tool_definition(body: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
    """The first request's ask_user, as Auto Ontology's tools are written."""
    for tool in body.tools or []:
        tool = tool if isinstance(tool, dict) else tool.model_dump()
        if tool.get("type") == "function" and tool.get("name") == c_interact.ASK_TOOL:
            function = {key: tool[key] for key in ("name", "description", "parameters") if tool.get(key) is not None}
            return {"type": "function", "function": function}
    raise c_interact.CInteractTextError(f"The first c-Interact request declares no {c_interact.ASK_TOOL} tool")


def tool_endpoint(access: DirectHTTPToolAccess, name: str) -> dict[str, Any]:
    """Where the sandbox calls the granted tool ``name``."""
    return {
        "url": f"{str(access.base_url).rstrip('/')}/{name}",
        "headers": dict(access.headers),
        "cookies": dict(access.cookies),
    }


if __name__ == "__main__":
    AutoOntologyAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AutoOntologyAgent.run_webserver()  # noqa: F401
