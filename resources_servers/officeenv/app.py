# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OfficeEnv resources server: connected SaaS apps simulated per rollout.

The environment is the ``officeenv`` package at the release requirements.txt pins. This server
routes Gym's endpoints to OfficeEnv's framework-neutral episode service
(``officeenv.service.EpisodeService``) and adds only Gym's own response fields.

A row (one JSONL line, written by ``officeenv emit`` or ``officeenv.emit.gym_row``):

    {"responses_create_params": {"input": [...], "tools": [...]},
     "officeenv": {<TaskSpec>}, "officeenv_version": "1.7.0", "task_id": "..."}

An Agent's ``/run`` (the simple agent) POSTs the row to ``/seed_session``, each function call to
``POST /<tool_name>`` with the call's arguments as the body, and the row plus the model's response
to ``/verify``. An Environment Server seeds with a ``ResourcesSeedSessionRequest`` whose
``task_data`` holds the row's other fields, so the session cookie names its
``resources_session_id``; its agent reaches the same tool routes (or the MCP endpoint below) with
that cookie, ``/verify`` grades the episode as for an Agent, and ``/close_session`` ends it.

With ``expose_tools_over_mcp: true``, Gym's MCP endpoint serves every tool of OfficeEnv's
``tools`` exposure under its tool name, and each episode's token admits only that episode's tools.
MCP clients read the tool definitions from that endpoint rather than from the row, so a seed whose
tools differ from those definitions (a row in the ``api_search``, ``cli`` or ``mcp`` exposure, or
with another tool-name separator) is refused.

For training, ``length_penalty`` (off by default) deducts AutomationBench's training length
penalty from the reward and reports the two terms as ``reward_components``, so a trainer can
schedule the penalty on its own (NeMo RL ``grpo.phasic_reward_components``).
"""

import contextvars
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

from fastapi import FastAPI, HTTPException, Request, Response
from officeenv.apps import available, get_app_class
from officeenv.exposure import Catalog, ToolsExposure
from officeenv.service import EpisodeService, Reply
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.failure_kinds import SESSION_LOST
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY


if TYPE_CHECKING:
    from nemo_gym.mcp_auto_exposure import MCPTool


class LengthPenaltyConfig(BaseModel):
    """Optional training reward shaping, AutomationBench's (benchmarks/automationbench): a
    rollout whose task reward is above ``reward_threshold`` loses
    ``min(max_penalty, turns / turn_scale + tokens / token_scale)``, where turns are the model's
    responses and tokens the completion tokens it generated over all of them (reasoning
    included). Off by default; turn it on for training only, so evaluation rewards stay
    OfficeEnv's."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    reward_threshold: float = Field(default=0.5, ge=0)
    max_penalty: float = Field(default=0.5, ge=0)
    turn_scale: float = Field(default=200.0, gt=0)
    token_scale: float = Field(default=200000.0, gt=0)


class OfficeEnvResourcesServerConfig(BaseResourcesServerConfig):
    # the reward depends on the session's state, which verify consumes
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    length_penalty: LengthPenaltyConfig = Field(default_factory=LengthPenaltyConfig)


class OfficeEnvSeedSessionRequest(BaseSeedSessionRequest):
    """An Agent's seed: the row."""

    model_config = ConfigDict(extra="allow")

    officeenv: dict[str, Any] = Field(description="The TaskSpec for this episode")


class OfficeEnvSeedSessionResponse(BaseSeedSessionResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    tools: list[str]
    apps: list[str]
    officeenv_server_version: str
    tool_drift: list[str] = Field(
        default_factory=list,
        description="Tools the row shows the model differently from what this version serves",
    )


class OfficeEnvVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class OfficeEnvVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class OfficeEnvLengthPenaltyVerifyResponse(BaseMultiRewardVerifyResponse):
    """The verify response with the length penalty on: ``reward_components`` holds the reward's
    terms, which sum to ``reward``: ``task_reward`` (OfficeEnv's) and ``length_penalty`` (minus the
    deduction)."""

    model_config = ConfigDict(extra="allow")


_MODEL_OUTPUT_TYPES = frozenset({"reasoning", "message", "function_call"})


def model_turns(response: NeMoGymResponse) -> int:
    """The model's responses in an episode: the runs of model output items in its output,
    which the outputs of the tools each response called separate."""
    turns, in_turn = 0, False
    for item in response.output:
        own = item.type in _MODEL_OUTPUT_TYPES and getattr(item, "role", "assistant") == "assistant"
        turns += own and not in_turn
        in_turn = own
    return turns


def length_penalty_fields(reward: float, response: NeMoGymResponse, config: LengthPenaltyConfig) -> dict[str, Any]:
    """The verify fields the length penalty sets: the penalized ``reward``, its terms
    (``reward_components``), and as metrics the raw ``task_reward``, the ``length_penalty``
    deducted, ``model_turns`` and ``generated_tokens`` (the response's accumulated usage, which
    the penalty needs whenever the task reward is above the threshold)."""
    turns = model_turns(response)
    tokens = None if response.usage is None else response.usage.output_tokens
    penalty = 0.0
    if reward > config.reward_threshold:
        if tokens is None:
            raise ValueError("The length penalty needs the response's usage (its generated tokens)")
        penalty = min(config.max_penalty, turns / config.turn_scale + tokens / config.token_scale)
    fields: dict[str, Any] = {
        "reward": reward - penalty,
        "reward_components": {"task_reward": reward, "length_penalty": -penalty},
        "task_reward": reward,
        "length_penalty": penalty,
        "model_turns": float(turns),
    }
    if tokens is not None:
        fields["generated_tokens"] = float(tokens)
    return fields


@cache
def tools_exposure_catalog() -> dict[str, dict[str, Any]]:
    """Every tool of OfficeEnv's ``tools`` exposure with all of its apps connected, by name: the
    tool definitions the MCP endpoint serves."""
    apps = {name: get_app_class(name)() for name in available()}
    return {tool["name"]: tool for tool in ToolsExposure(Catalog(apps), {}).tools()}


# The tools of the session the current /seed_session request seeded. Gym mints the episode's MCP
# token after the seed handler returns, and reads them then (mcp_allowed_tools_for_session).
_SEEDED_TOOLS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "officeenv_seeded_tools", default=None
)


@dataclass
class _Episode:
    """An Environment Server episode, kept by its ``resources_session_id`` until it is closed."""

    episode_id: EpisodeId
    task_id: TaskId
    tools: list[str]


class OfficeEnvResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: OfficeEnvResourcesServerConfig
    # Episodes by Gym's session ID, which an Environment Server's seed sets to its resources_session_id.
    service: EpisodeService = Field(default_factory=EpisodeService)
    _episodes: dict[str, _Episode] = PrivateAttr(default_factory=dict)
    _closed: ClosedResourcesSessionRecords[EpisodeId] = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._closed = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/officeenv/apps")(self.list_apps)
        app.post("/{tool_name}")(self.tool_call)  # registered last: fixed routes win
        return app

    async def list_apps(self) -> dict[str, Any]:
        return self.service.catalog()

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | OfficeEnvSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | OfficeEnvSeedSessionResponse:
        if isinstance(body, ResourcesSeedSessionRequest):
            return self._seed_episode(request, body)
        reply = self._seed(request.session[SESSION_ID_KEY], body.model_dump())
        _SEEDED_TOOLS.set(reply.body["tools"])
        return OfficeEnvSeedSessionResponse(**reply.body)

    def _seed_episode(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        """Seed an Environment Server episode under its ``resources_session_id``. Seeding the same ID
        for the same episode again answers as the first seed did; a closed ID is never seeded again."""
        session_id = body.resources_session_id
        # Nothing below awaits, so no other request runs between the checks and the updates.
        if session_id in self._closed:
            raise HTTPException(status_code=409, detail=f"Resources session {session_id} is already closed")
        episode = self._episodes.get(session_id)
        if episode is None:
            reply = self._seed(session_id, body.task_data)
            episode = self._episodes[session_id] = _Episode(body.episode_id, body.task_id, reply.body["tools"])
        elif (episode.episode_id, episode.task_id) != (body.episode_id, body.task_id):
            raise HTTPException(
                status_code=409, detail="resources_session_id is already bound to another episode or task"
            )
        # Gym's session routing names the cookie session after the ID too; this keeps the two the same without it.
        request.session[SESSION_ID_KEY] = session_id
        _SEEDED_TOOLS.set(episode.tools)
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    def _seed(self, session_id: str, row: dict[str, Any]) -> Reply:
        """Build the session ``row`` describes. A malformed task, an unknown app or a row from
        another OfficeEnv major version fails the seed with OfficeEnv's status and error code;
        nothing is stored, so the seed can be retried."""
        reply = self.service.seed(session_id, row)
        if reply.status != 200:
            raise HTTPException(status_code=reply.status, detail=reply.body)
        if self.config.expose_tools_over_mcp:
            catalog = tools_exposure_catalog()
            tools = self.service.store.get(session_id).tools()
            unserved = sorted(tool["name"] for tool in tools if catalog.get(tool["name"]) != tool)
            if unserved:
                self._discard(session_id)
                message = (
                    f"The MCP endpoint serves the tools exposure's tools, not {unserved[:5]}: "
                    "emit the row with exposure mode 'tools' and the default separator"
                )
                raise HTTPException(
                    status_code=400, detail={"error": {"code": "unsupported_exposure", "message": message}}
                )
        return reply

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Free the episode's world unless verification already has; a retried close, or the
        close of a session that was never seeded, succeeds."""
        session_id = body.resources_session_id
        # Nothing below awaits, so no other request runs between the checks and the updates.
        closed = self._closed.get(session_id)
        if closed is None:
            episode = self._episodes.get(session_id)
            if episode is not None and episode.episode_id != body.episode_id:
                raise HTTPException(status_code=409, detail="episode_id does not match the seeded resources session")
            self._episodes.pop(session_id, None)
            self._discard(session_id)
            self._closed.remember(session_id, body.episode_id)
        elif closed != body.episode_id:
            raise HTTPException(status_code=409, detail="episode_id does not match the closed resources session")
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def retire_session_state(self, session_id: str) -> None:
        self._discard(session_id)

    def _discard(self, session_id: str) -> None:
        """Free a session's world without grading it."""
        self.service.store.sessions.pop(session_id, None)
        self.service.notes.pop(session_id, None)

    async def tool_call(self, request: Request, tool_name: str) -> Response:
        # app-level errors are tool outputs the model sees (200); only a body that is no JSON
        # object (400) and a call without a session (409) are integration failures
        reply = self.service.call(request.session[SESSION_ID_KEY], tool_name, await request.body())
        return Response(content=reply.text(), media_type="application/json", status_code=reply.status)

    async def verify(
        self, request: Request, body: OfficeEnvVerifyRequest
    ) -> OfficeEnvVerifyResponse | OfficeEnvLengthPenaltyVerifyResponse:
        result = self.service.verify(request.session[SESSION_ID_KEY])
        if result.get("officeenv_failure") == "no_session":
            # the rollout has no session to grade (verified twice, or the server restarted):
            # the reward measures the environment, not the policy
            result = {**result, "mask_sample": True, "failure_kind": SESSION_LOST}
        if not self.config.length_penalty.enabled:
            return OfficeEnvVerifyResponse(**{**body.model_dump(), **result})
        penalty = length_penalty_fields(result["reward"], body.response, self.config.length_penalty)
        return OfficeEnvLengthPenaltyVerifyResponse(**{**body.model_dump(), **result, **penalty})

    def mcp_tools(self, harvested: list["MCPTool"], catchall: Any | None) -> list["MCPTool"] | None:
        """Every tool of the ``tools`` exposure, dispatched through the tool route; no other route is a tool."""
        if catchall is None:
            return []
        catalog = tools_exposure_catalog()
        return [catchall.tool(name, tool["parameters"], tool["description"]) for name, tool in catalog.items()]

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> list[str] | None:
        """The tools of the episode this request seeded: its MCP token admits only those."""
        tools = _SEEDED_TOOLS.get()
        if tools is None:
            raise ValueError("No OfficeEnv session was seeded in this request, so no MCP tools can be granted")
        return tools


if __name__ == "__main__":
    OfficeEnvResourcesServer.run_webserver()
