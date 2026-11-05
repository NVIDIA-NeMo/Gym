# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-tree agent-server for wrapping any LangChain `deepagents` graph as a native Gym `responses_api_agents` server.

Generic base for wrapping any LangChain `deepagents` graph as a native Gym `responses_api_agents` server
(model calls go through Gym's own `model_server`, on-policy, instead of the agent bringing its own model).
Concrete instances subclass `DeepAgentsAgent` and implement `build_agent(model)` — see
`reasoning_search_agent.py` for the one used by this repo's reasoning_gym/tavily_search comparison.

See `responses_langchain_bridge.py` for the LangChain <-> Gym Responses API bridge (`GymResponsesChatModel`
and the message-conversion helpers) and why it can't just be `langchain_openai.ChatOpenAI`.

The compiled deepagents graph is built once, in `__init__`, and stored as `self.agent` — not per-request,
and named to match how `deepagents`' own docs/examples name the object `create_deep_agent()` returns
(`agent = create_deep_agent(...)`), rather than `langgraph_agent`'s `self.graph`. An agent session that borrows
the Resources Server's task sandbox gets a graph of its own, built at seed with a backend that runs the graph's
file and shell tools in that sandbox (see `sandbox_backend.py`).
"""

import asyncio
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from fastapi import Body, HTTPException, Request, Response
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
)
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap, SandboxObservation
from responses_api_agents.langchain_deepagents_agent.responses_langchain_bridge import (
    GymResponsesChatModel,
    to_langchain,
    to_responses,
)
from responses_api_agents.langchain_deepagents_agent.sandbox_backend import task_sandbox_backend
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentSessionState


OBSERVATION_SOURCE = "langchain_deepagents_agent"


@dataclass
class DeepAgentsSessionState(SimpleAgentSessionState):
    """One agent session: its single activation and, when the Resources Server lent one, the task sandbox.

    The first activation request starts `activation`. An identical retry joins it, a different request is refused,
    and close cancels it before releasing the sandbox.
    """

    sandbox: AsyncSandbox | None = None
    # The graph whose tools run in `sandbox`; None runs the agent's in-process graph.
    agent: Any = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    activation: asyncio.Task[NeMoGymResponse] | None = None


class DeepAgentsAgentConfig(SimpleAgentConfig):
    """No tool-specific fields — those belong on a concrete subclass's config (see reasoning_search_agent.py).

    `max_steps` (inherited from SimpleAgentConfig) is rejected, not merely unused: deepagents runs its own
    internal tool loop, so there's no per-model-call step counter here for it to bound (SimpleAgent.max_steps
    counts internal model round-trips; LangChain's own `recursion_limit` counts LangGraph super-steps —
    neither is what a caller setting `max_steps` on this agent would actually get). Silently accepting and
    ignoring it would let a caller believe it's enforced when it isn't.
    """

    @model_validator(mode="after")
    def reject_max_steps(self) -> "DeepAgentsAgentConfig":
        if self.max_steps is not None:
            raise ValueError(
                "max_steps is not supported by DeepAgentsAgent: deepagents runs its own internal tool loop, "
                "with no per-model-call step counter for this to bound. Remove max_steps from this agent's "
                "config."
            )
        return self

    @model_validator(mode="after")
    def reject_execute_tools_false(self) -> "DeepAgentsAgentConfig":
        # SimpleAgent can hand unresolved tool calls back to the Environment Server; this graph cannot,
        # because deepagents always runs its own tools inside ainvoke().
        if not self.execute_tools:
            raise ValueError(
                "execute_tools=false is not supported by DeepAgentsAgent: the deepagents graph always runs its "
                "own tools. Remove execute_tools from this agent's config."
            )
        return self


class DeepAgentsAgent(SimpleAgent):
    """Inherits SimpleAgent.run() unchanged (seed_session -> self-POST /v1/responses -> verify) and SimpleAgent's
    agent-session endpoints. Overrides responses() with everything true of any deepagents-based agent wired into
    Gym, and narrows the session seed to what the graph actually uses.

    Does not build a TrajectoryRecord (per-tool-call/model-call observability) — see README.md's "Known
    limitation" section. That's deliberately deferred, not an oversight; a session reports it as a gap."""

    # The graph's state lives inside one ainvoke() call, which a checkpoint cannot export. Checkpointing
    # therefore restarts an unfinished rollout from its input instead of resuming it (GYMINT-357).
    checkpoint_mode = "restart_only"

    config: DeepAgentsAgentConfig
    agent: Any = Field(default=None, exclude=True)
    _chat_model: GymResponsesChatModel = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chat_model = GymResponsesChatModel(agent=self)
        self.agent = self.build_agent(self._chat_model)

    @abstractmethod
    def build_agent(self, model: BaseChatModel, backend: BackendProtocol | None = None) -> Any:
        """Return a compiled deepagents graph, e.g. create_deep_agent(model=model, backend=backend, tools=[...]).

        Pass `backend` through to `create_deep_agent()`. `__init__` calls this once without one, for the in-process
        graph whose files live in graph state. A session seeded with a task sandbox calls it again with a backend
        that runs the graph's file tools and its `execute` tool in that sandbox.
        """
        raise NotImplementedError

    def _new_session_state(self, body: AgentSeedSessionRequest) -> DeepAgentsSessionState:
        """Accept a session only for what the graph uses: its own tools, in process or in a lent task sandbox.

        The graph never calls a granted Resources tool, so a required grant is refused rather than silently
        dropped. An optional grant is skipped. The session holds no Resources cookie jar, so close leaves the
        Environment Server's jar for /verify unchanged.
        """
        if body.sandbox_access is not None:
            # Files reach the sandbox through the provider's upload, which does not take a user.
            if body.sandbox_access.user is not None:
                raise AgentSeedRejectedError("DeepAgents agent does not support a sandbox user override")
            if not PurePosixPath(body.sandbox_access.workdir).is_absolute():
                raise AgentSeedRejectedError("DeepAgents agent requires an absolute sandbox workdir")
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            raise AgentSeedRejectedError(
                f"DeepAgents agent does not support required tool access: {', '.join(required)}"
            )
        return DeepAgentsSessionState(
            request=body,
            tool_access=None,
            resources_cookies=None,
            observations=AgentObservationBundle(
                source=OBSERVATION_SOURCE,
                gaps=[
                    ObservationGap(
                        code="agent_transcript_unavailable",
                        detail="the deepagents graph reports no invocation or tool-call records",
                    ),
                    ObservationGap(code="model_call_ownership_unavailable"),
                ],
            ),
        )

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> DeepAgentsSessionState:
        """Connect to a lent task sandbox, as a borrower, and build the session's graph on it."""
        state = self._new_session_state(body)
        access = body.sandbox_access
        if access is None:
            return state
        connection = access.connection
        provider = create_provider(
            resolve_provider_config(connection.provider_config_ref, self.server_client.global_config_dict)
        )
        try:
            state.sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            # No sandbox owns the provider yet.
            await provider.aclose()
            raise
        try:
            state.agent = self.build_agent(
                self._chat_model, backend=task_sandbox_backend(state.sandbox, workdir=access.workdir, env=access.env)
            )
        except BaseException:
            await state.sandbox.disconnect()
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Stop the activation, then release a borrowed sandbox without stopping it.

        The Resources Server that lent the sandbox stops it after /verify. A close whose activation does not stop in
        time raises, which keeps the session for the Environment Server's close retry.
        """
        if not isinstance(state, DeepAgentsSessionState):
            raise TypeError("Expected a DeepAgents session state")
        activation = state.activation
        if activation is not None and not activation.done():
            activation.cancel()
            await asyncio.wait({activation}, timeout=self.config.session_close_timeout_seconds)
            if not activation.done():
                raise TimeoutError("DeepAgents activation did not stop within session_close_timeout_seconds")
        if state.sandbox is not None:
            observation = self._sandbox_observation(state)
            await state.sandbox.disconnect()
            state.sandbox = None
            state.observations.records.append(observation)
        return await super()._close_agent_session_state(state)

    @staticmethod
    def _sandbox_observation(state: DeepAgentsSessionState) -> SandboxObservation:
        """How the graph's run in the lent sandbox ended; a session closed before a run finished is cancelled."""
        activation = state.activation
        outcome, error_type = "cancelled", None
        if activation is not None and not activation.cancelled():
            error = activation.exception()
            outcome, error_type = ("failed", type(error).__name__) if error is not None else ("completed", None)
        return SandboxObservation(
            role="agent",
            provider=state.sandbox.provider_name,
            sandbox_id=state.sandbox.sandbox_id,
            outcome=outcome,
            error_type=error_type,
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        # Same extraction SimpleAgent.responses() does, including the "unscoped" fallback.
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        model_url_path = self.url_path_for_request("/v1/responses", request)
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            # Legacy self-dispatch from run(). Different servers have different session-middleware keys
            # (get_session_middleware_key() is f"{class_name}___{server_name}"), so a cookie from the resources
            # server's /seed_session does not automatically ride through to /verify unless re-set here.
            for key, value in request.cookies.items():
                response.set_cookie(key, value)
            return await self._run_graph(self.agent, body, model_url_path=model_url_path, rollout_id=rollout_id)

        # An activation of a seeded session: the Environment Server holds the Resources cookies and sends them to
        # /verify itself, so nothing is mirrored back. An unknown or closing session answers 409.
        state = self._require_agent_session(agent_session_id)
        if not isinstance(state, DeepAgentsSessionState):
            raise TypeError("Expected a DeepAgents session state")
        if rollout_id != state.request.episode_id.capture_key:
            raise HTTPException(409, "Activation rollout does not match the seeded agent session")
        if state.activation is None:
            # Bound before the first await, so a retry that arrives meanwhile joins this run.
            state.activation_request = body.model_copy(deep=True)
            state.activation = asyncio.create_task(
                self._run_graph(state.agent or self.agent, body, model_url_path=model_url_path, rollout_id=rollout_id)
            )
        elif body != state.activation_request:
            raise HTTPException(409, "DeepAgents sessions run one activation; retry the same request")
        # A disconnected HTTP client must not cancel the shared run; only close cancels it.
        return (await asyncio.shield(state.activation)).model_copy(deep=True)

    async def _run_graph(
        self,
        agent: Any,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        rollout_id: str | None,
    ) -> NeMoGymResponse:
        if isinstance(body.input, str):
            input_items = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        else:
            input_items = body.input
        input_messages = to_langchain(input_items)

        # `model_cookies` starts empty, not from `request.cookies` (those are resources-server session
        # cookies chained from /seed_session — see the cookie-mirroring loop in responses() — and are unrelated
        # to the model server's own session). `GymResponsesChatModel._agenerate()` fills this holder in from
        # each model response and reads it back on the next internal call within this rollout; see
        # responses_langchain_bridge.py. `model_usage` similarly accumulates usage from every internal
        # model call so the final agent response reports the whole rollout.
        model_usage = {"usage": None}
        run_config: RunnableConfig = {
            "configurable": {
                "model_url_path": model_url_path,
                "model_cookies": {"cookies": None},
                "model_usage": model_usage,
                # deepagents' SummarizationMiddleware offloads evicted history to
                # `/conversation_history/{thread_id}.md`, appending to that same path on each later
                # summarization so a rollout accumulates one running log. Without this key it falls back to
                # a *freshly generated* `session_<uuid>` per summarization event: every eviction lands in a
                # new file, and the pointer it hands the model — under a prompt claiming the full history
                # was saved there — only ever covers the most recent one, orphaning everything older.
                # Safe to share the "unscoped" fallback across rollouts even though `thread_id` is
                # LangGraph's checkpointing key: the graph is compiled without a checkpointer, and
                # deepagents' StateBackend keeps that file in per-`ainvoke` graph state (a sandbox session's
                # graph routes it there too; see sandbox_backend.py).
                "thread_id": rollout_id or "unscoped",
                # Forwarded verbatim to every internal model call (see GymResponsesChatModel._agenerate()),
                # so a caller requesting e.g. {"summary": "auto"} gets reasoning summaries back on every
                # turn — unlike SimpleAgent, which forwards this for free via body.model_copy(), this agent
                # rebuilds each internal request from scratch and has to thread it through explicitly.
                # `body.reasoning` is already a plain dict at runtime (its field type, Reasoning, is a
                # TypedDict, not a BaseModel — verified via NeMoGymResponseCreateParamsNonStreaming; no
                # .model_dump() needed, and calling one would raise AttributeError on a plain dict).
                "model_reasoning": body.reasoning,
            }
        }
        final_state = await agent.ainvoke({"messages": input_messages}, config=run_config)

        new_messages = final_state["messages"][len(input_messages) :]
        response_body = to_responses(new_messages, self.config.model_server.name)
        response_body["usage"] = model_usage["usage"]
        return NeMoGymResponse.model_validate(response_body)
