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

This process imports only NeMo Gym: Auto Ontology runs in the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Optional, cast
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionSetupError,
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
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
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
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


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
    # Episodes this server runs at once. It bounds the episodes, not the sandboxes: each session holds its sandbox
    # and Postgres from seed to close, so the run's concurrency bounds those.
    max_concurrent_episodes: int = Field(default=8, gt=0)
    # The loopback ports a sandbox's catalog may take. Sandboxes on the host
    # network share loopback, so each takes its own, at random from this range.
    catalog_ports: tuple[int, int] = (20000, 60000)
    catalog_port_attempts: int = Field(default=3, gt=0)


@dataclass(kw_only=True)
class AutoOntologySessionState(AgentSessionState):
    """One task's session: its task, its sandbox and its one activation."""

    task: AutoOntologyTask
    session: SandboxSession[dict[str, Any]]
    catalog_port: int
    activation: Optional[asyncio.Task[NeMoGymResponse]] = None
    activation_request: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    episode: Optional[dict[str, Any]] = None


class AutoOntologyAgent(SimpleResponsesAPIAgent):
    config: AutoOntologyAgentConfig
    agent_context_model = AutoOntologyTask

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._episodes = asyncio.Semaphore(self.config.max_concurrent_episodes)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """The model call that ended the episode, from the runner's ``model_failure``."""
        status = native_result.get("status_code")
        if isinstance(status, int) and 400 <= status <= 599:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)
        if native_result.get("no_reply"):
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        return AgentOutcome(failure_kind="unknown")

    # ----- agent session ----- #

    def parse_agent_context(self, request: AgentSeedSessionRequest) -> BaseModel | None:
        # The base class reads the context before it calls _seed_agent_session_state: refuse a lent sandbox here, so
        # that the refusal gives this reason whatever the context holds.
        if request.sandbox_access is not None:
            raise AgentSeedRejectedError(
                "Auto Ontology does not support sandbox access: it runs in its own sandbox, from the Auto Ontology image"
            )
        return super().parse_agent_context(request)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AutoOntologySessionState:
        if required := [grant.name for grant in self.effective_tool_accesses(body) if grant.required]:
            raise AgentSeedRejectedError(f"Auto Ontology uses its own tools and cannot use the grants {required}")
        # The base class refuses a context that does not satisfy AutoOntologyTask, a missing one included.
        task = cast(AutoOntologyTask, self.parse_agent_context(body))

        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        model_base_url = self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key)
        try:
            assert_model_url_reachable_from_sandbox(model_base_url, provider_name=next(iter(provider_config), None))
        except ValueError as error:
            raise AgentSeedRejectedError(str(error)) from error

        provider = create_provider(provider_config)
        sandbox = AsyncSandbox(provider)
        session = SandboxSession[dict[str, Any]](
            sandbox=sandbox,
            workdir=None,
            session_dir=f"{SANDBOX_DIR}/sessions/{uuid4().hex}",
            owns_sandbox=True,
            harness="Auto Ontology",
        )
        try:
            sandbox_config = dict(self.config.sandbox_config)
            sandbox_config.setdefault(
                "ttl_s",
                self.config.setup_timeout_seconds
                + self.config.episode_timeout_seconds
                + 2 * self.config.close_timeout_seconds,
            )
            files = {SANDBOX_RUNNER: RUNNER_SOURCE.read_text()}
            spec = restrict_sandbox_network(
                SandboxSpec(**sandbox_config, files=files),
                network_access=self.config.network_access,
                provider=provider,
                model_base_url=model_base_url,
                tool_urls=[],
                required_tool_urls=[],
            )
            await sandbox.start(spec)
            catalog_port = await self._start_catalog(sandbox)
        except BaseException as error:
            # A sandbox that start() created stays started when start() could not deliver its files and then could
            # not stop it either. Like one whose catalog did not start, it is stopped through the session.
            if not sandbox.started:
                await provider.aclose()
                raise
            try:
                await session.close(timeout=self.config.close_timeout_seconds)
            except BaseException:
                # The sandbox may still run: keep it in the session, so that close tries to stop it again.
                state = AutoOntologySessionState(request=body, task=task, session=session, catalog_port=0)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return AutoOntologySessionState(request=body, task=task, session=session, catalog_port=catalog_port)

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
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(source="auto_ontology", gaps=gaps),
            cleanup_confirmed=state.session.cleanup_confirmed,
        )

    # ----- the episode ----- #

    async def _run_episode(
        self,
        state: AutoOntologySessionState,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponse:
        session = state.session
        task_path = f"{session.session_dir}/task.json"
        result_path = f"{session.session_dir}/result.json"
        payload = {
            **state.task.model_dump(),
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
                state.episode = await session.execute(
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
        # A model call that Auto Ontology's loop gave up on ended the episode with no answer. The Environment Server's
        # outcome policy decides whether to grade it, run it again or fail it, by the call's HTTP status: it is not a
        # task the agent did not answer.
        failure = state.episode.get("model_failure")
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=[answer_message(state.episode)],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=False,
            metadata={
                **(body.metadata or {}),
                "auto_ontology_end_reason": str(state.episode.get("reason")),
                "auto_ontology_tool_calls": str(state.episode.get("tool_calls")),
            },
            **(
                {"status": "failed", "error": {"code": "server_error", "message": failure["error"]}} if failure else {}
            ),
        )
        return self.report_outcome(response, failure) if failure else response

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
        rollout_id = request.path_params.get("rollout_id")
        if rollout_id != state.request.episode_id.capture_key:
            raise AgentSessionConflictError("The activation's rollout does not match the seeded session")
        refuse_unsupported_request_fields(body, model=self.config.model, honored={"input"})
        if state.activation is None:
            # No await before the activation is installed: a retry joins it.
            state.activation_request = body.model_copy(deep=True)
            state.activation = asyncio.create_task(self._run_episode(state, body))
        elif body != state.activation_request:
            raise AgentSessionConflictError("An Auto Ontology session has one activation; retry the same request")
        # A waiter that disconnects must not cancel the episode; close does.
        return (await asyncio.shield(state.activation)).model_copy(deep=True)


def refuse_unsupported_request_fields(
    body: NeMoGymResponseCreateParamsNonStreaming, *, model: str, honored: set[str]
) -> None:
    """Refuse an activation that sets a field the episode does not honor, rather than drop it.

    Auto Ontology's loop sets its own sampling, limits and instructions. ``honored`` names the fields the episode reads;
    ``model`` may also name the configured model, and ``metadata`` is echoed in the response.
    """
    unsupported = []
    for name, field in type(body).model_fields.items():
        value = getattr(body, name)
        if name in honored or name == "metadata" or (name == "model" and value in (None, model)):
            continue
        if name in ("stream", "background") and value is False:
            continue
        if value != field.get_default(call_default_factory=True):
            unsupported.append(name)
    if unsupported:
        raise HTTPException(
            422,
            f"Auto Ontology does not support the request fields {unsupported}: its loop sets its own sampling, "
            "limits and instructions",
        )


def answer_message(episode: dict[str, Any]) -> NeMoGymResponseOutputMessage:
    """The agent's answer: the submitted query in a fenced ``sql`` block, or why the episode has none."""
    sql = episode.get("sql")
    if sql:
        text = f"```sql\n{str(sql).strip()}\n```"
    else:
        text = f"Auto Ontology submitted no query ({episode.get('reason')})."
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
