# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OfficeEnv resources server: connected SaaS apps simulated per rollout.

The environment is the ``officeenv`` package, which installs apart from this server's locked
dependencies until it is published (README.md, "The OfficeEnv version"). This server routes Gym's
endpoints to OfficeEnv's framework-neutral episode service (``officeenv.service.EpisodeService``)
and adds only Gym's own response fields.

A row (one JSONL line, written by ``officeenv emit`` or ``officeenv.emit.gym_row``):

    {"responses_create_params": {"input": [...], "tools": [...]},
     "officeenv": {<TaskSpec>}, "officeenv_version": "1.14.0", "task_id": "..."}

An Agent's ``/run`` (the simple agent) POSTs the row to ``/seed_session``, each function call to
``POST /<tool_name>`` with the call's arguments as the body, and the row plus the model's response
to ``/verify``. An Environment Server seeds with a ``ResourcesSeedSessionRequest`` whose
``task_data`` holds the row's other fields, so the session cookie names its
``resources_session_id``; its agent reaches the same tool routes (or the MCP endpoint below) with
that cookie, ``/verify`` grades the episode as for an Agent, and ``/close_session`` ends it.
``GET /task_state`` (Gym's ``ResourcesTaskState``) says whether the episode's task has ended
(``complete_task`` succeeded), so an environment server can end the episode there: ``single_agent_turn``
drives the agent one step at a time and reads it between steps (``end_on_task_state``). With
``close_tools_on_task_end``, a tool call after the call that ended the task is refused.

With ``expose_tools_over_mcp: true``, Gym's MCP endpoint serves every tool of OfficeEnv's
``tools`` exposure under its tool name, and each episode's token admits only that episode's tools.
MCP clients read the tool definitions from that endpoint rather than from the row, so a seed whose
tools differ from those definitions (a row in the ``api_search``, ``cli`` or ``mcp`` exposure, or
with another tool-name separator) is refused.

A row may name its TaskSpec's large strings by SHA-256 instead of carrying them (OfficeEnv's
artifact references, ``officeenv.artifacts``). The stores in ``artifacts_dirs`` (such as a
corpus's ``artifacts/`` folder beside its rows file), searched in order, go to the episode
service, which reads the references back at seed, each checked against its hash, and refuses a
row whose artifacts cannot be read (400).

For training, ``length_penalty`` (off by default) deducts AutomationBench's training length
penalty from the reward and reports the two terms as ``reward_components``, so a trainer can
schedule the penalty on its own (NeMo RL ``grpo.phasic_reward_components``).
"""

import contextvars
import importlib.util
import os
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import (
    RESOURCES_TASK_STATE_PATH,
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLostError,
    ResourcesTaskState,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


# OfficeEnv installs apart from this server's locked dependencies until it is published (README.md, "The OfficeEnv
# version"): without it the server stops here, naming the step that installs it.
if importlib.util.find_spec("officeenv") is None:
    raise ImportError(
        "OfficeEnv is not installed in this server's venv. Until OfficeEnv is published it installs apart from the "
        "server's locked dependencies: see 'The OfficeEnv version' in resources_servers/officeenv/README.md "
        "(OFFICEENV_GIT_URL)."
    )
# The tool catalog this server serves over MCP (EpisodeService.catalog_tools) is OfficeEnv 1.13.0's, as are workspace
# mode and the file call (officeenv.files) that a subclass lending sandboxes runs: an older release stops the server
# here too, naming both versions.
OFFICEENV_NEEDED = "1.13.0"
if importlib.util.find_spec("officeenv.files") is None:
    from officeenv import __version__ as installed_officeenv

    raise ImportError(
        f"This server needs OfficeEnv {OFFICEENV_NEEDED} or later (officeenv.files), and its venv has OfficeEnv "
        f"{installed_officeenv}: install the release 'The OfficeEnv version' in resources_servers/officeenv/README.md "
        "names (OFFICEENV_GIT_URL)."
    )

from officeenv.artifacts import ArtifactStore
from officeenv.files import Workspace
from officeenv.service import EpisodeService, Reply


if TYPE_CHECKING:
    from nemo_gym.server_runtime import MCPTool


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
    artifacts_dirs: list[str] = Field(
        default_factory=list,
        description=(
            "Stores of the artifacts a row's TaskSpec names by SHA-256 instead of carrying them "
            "(officeenv.artifacts), searched in order: absolute paths, such as a corpus's artifacts/ folder "
            "beside its rows file. A row that names artifacts no store holds is refused (400)."
        ),
    )
    close_tools_on_task_end: bool = Field(
        default=False,
        description=(
            "Refuse a tool call (409) once a call has returned with the episode's task ended (GET /task_state), so "
            "the calls a turn makes after the one that ended it change nothing; the environment server that ends "
            "episodes there sets it"
        ),
    )

    @field_validator("artifacts_dirs")
    @classmethod
    def _stores_are_absolute(cls, stores: list[str]) -> list[str]:
        # the server runs in its own directory, not where its config was written
        relative = [store for store in stores if not os.path.isabs(store)]
        if relative:
            raise ValueError(f"artifact stores must be absolute paths: {relative}")
        return stores


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


class OfficeEnvTaskState(ResourcesTaskState):
    """``GET /task_state``: whether the episode's task has ended (``complete_task`` succeeded, either status)."""


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
def tools_exposure_catalog(workspace: Workspace | None = None) -> dict[str, dict[str, Any]]:
    """Every tool of OfficeEnv's ``tools`` exposure with all of its apps connected, by name, as workspace mode
    describes them when ``workspace`` is given (``EpisodeService.catalog_tools``): the tool definitions the MCP
    endpoint serves, which an episode's own are checked against at its seed."""
    return {tool["name"]: tool for tool in EpisodeService.catalog_tools(workspace=workspace)}


# The tools of the session the current /seed_session request seeded. Gym mints the episode's MCP
# token after the seed handler returns, and reads them then (mcp_allowed_tools_for_session).
_SEEDED_TOOLS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "officeenv_seeded_tools", default=None
)


@dataclass
class _Episode:
    """An episode session, kept by its ``resources_session_id`` until it is closed, with the tools its MCP token
    admits."""

    episode_id: EpisodeId
    task_id: TaskId
    grant: list[str] = field(default_factory=list)


class OfficeEnvResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: OfficeEnvResourcesServerConfig
    # Episodes by Gym's session ID, which an environment server's seed sets to its resources_session_id.
    service: EpisodeService = Field(default_factory=EpisodeService)
    _episodes: dict[str, _Episode] = PrivateAttr(default_factory=dict)
    # Whether each session's task had ended when its last tool call returned (GET /task_state,
    # close_tools_on_task_end). Published once a whole call is over, so no read sees a state the same call reverses.
    _task_ends: dict[str, bool] = PrivateAttr(default_factory=dict)
    _fences: ResourcesSessionFences[EpisodeId] = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._fences = ResourcesSessionFences(self.config.session_close_retry_window_seconds)
        if self.config.artifacts_dirs:  # the service reads a row's artifact references back from them at seed
            self.service = EpisodeService(artifacts=ArtifactStore(self.config.artifacts_dirs))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/officeenv/apps")(self.list_apps)
        app.get(RESOURCES_TASK_STATE_PATH)(self.task_state)
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
        """Seed an episode session under its ``resources_session_id``. Seeding the same ID
        for the same episode again answers as the first seed did; a fenced ID is not seeded again."""
        session_id = body.resources_session_id
        # Nothing below awaits, so no other request runs between the checks and the updates.
        if session_id in self._fences:
            raise ResourcesSessionConflictError(f"Resources session {session_id} is already closed")
        episode = self._episodes.get(session_id)
        if episode is None:
            episode = self._episodes[session_id] = self._new_episode(session_id, body)
        elif (episode.episode_id, episode.task_id) != (body.episode_id, body.task_id):
            raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
        # Gym's session routing names the cookie session after the ID too; this keeps the two the same without it.
        request.session[SESSION_ID_KEY] = session_id
        _SEEDED_TOOLS.set(episode.grant)
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    def _new_episode(self, session_id: str, body: ResourcesSeedSessionRequest) -> _Episode:
        """A new episode: its world, built from its row (``_seed``), and the tools its MCP token admits."""
        reply = self._seed(session_id, body.task_data)
        return _Episode(body.episode_id, body.task_id, grant=list(reply.body["tools"]))

    def _seed(self, session_id: str, row: dict[str, Any]) -> Reply:
        """Build the session ``row`` describes. A malformed task, an unknown app or a row from
        another OfficeEnv major version fails the seed with OfficeEnv's status and error code, and so
        does a row whose tools the MCP endpoint does not serve (``_unreachable``); nothing is stored,
        so the seed can be retried."""
        reply = self.service.seed(session_id, row)
        if reply.status != 200:
            raise HTTPException(status_code=reply.status, detail=reply.body)
        self._refuse_unreachable(session_id, self._unreachable(session_id))
        return reply

    def _unreachable(self, session_id: str, workspace: Workspace | None = None) -> str | None:
        """Why the MCP endpoint cannot serve a seeded session's tools, if it cannot: it serves the ``tools``
        exposure's tools under their own names, as workspace mode describes them for a session that runs in it
        (``workspace``)."""
        if not self.config.expose_tools_over_mcp:
            return None
        catalog = tools_exposure_catalog(workspace)
        tools = self.service.store.get(session_id).tools()
        unserved = sorted(tool["name"] for tool in tools if catalog.get(tool["name"]) != tool)
        if not unserved:
            return None
        return (
            f"The MCP endpoint serves the tools exposure's tools, not {unserved[:5]}: "
            "emit the row with exposure mode 'tools' and the default separator"
        )

    def _refuse_unreachable(self, session_id: str, message: str | None) -> None:
        """Fail a seed whose session's apps cannot be reached (``message`` says why), freeing its world."""
        if message is not None:
            self._discard(session_id)
            raise HTTPException(
                status_code=400, detail={"error": {"code": "unsupported_exposure", "message": message}}
            )

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Release the episode's world unless verification already has; a retried close, or the
        close of a session that was never seeded, succeeds."""
        return self._close(body)

    def _close(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        # Nothing below awaits, so no other request runs between the checks and the updates.
        fenced_episode = self._fences.get(session_id)
        if fenced_episode is None:
            episode = self._episodes.get(session_id)
            if episode is not None and episode.episode_id != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            self._episodes.pop(session_id, None)
            self._discard(session_id)
            self._fences.fence(session_id, body.episode_id)
        elif fenced_episode != body.episode_id:
            raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def retire_session_state(self, session_id: str) -> None:
        # A retired attempt is never closed, and its replacement may seed the same session ID with its own episode.
        self._episodes.pop(session_id, None)
        self._discard(session_id)

    def _discard(self, session_id: str) -> None:
        """Free a session's world without grading it."""
        self.service.store.sessions.pop(session_id, None)
        self.service.notes.pop(session_id, None)
        self._task_ends.pop(session_id, None)

    async def tool_call(self, request: Request, tool_name: str) -> Response:
        # app-level errors are tool outputs the model sees (200); only a body that is no JSON
        # object (400) and a call without a session (404) are integration failures
        session_id = request.session[SESSION_ID_KEY]
        if self.service.store.get(session_id) is None:
            raise ResourcesSessionLostError("This session has ended.")
        body = await request.body()
        reply = await self._episode_call(session_id, self._episodes.get(session_id), tool_name, body)
        return Response(content=reply.text(), media_type="application/json", status_code=reply.status)

    async def task_state(self, request: Request) -> OfficeEnvTaskState:
        """Whether the session's task has ended, for an environment server that ends its episode there."""
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id is None or self.service.store.get(session_id) is None:
            raise ResourcesSessionLostError("This session has ended.")
        return OfficeEnvTaskState(task_ended=self._task_ends.get(session_id, False))

    def _task_ended(self, session_id: str) -> bool:
        """Whether the episode's task has ended: the user app's task row left ``pending`` (``complete_task``
        succeeded, either status). The row is read rather than the tool name, because every exposure reaches the op
        its own way (``user_complete_task``, ``api_call``, ``run_command``, ``call_tool``)."""
        session = self.service.store.get(session_id)
        if session is None or "user" not in session.apps:
            return False
        rows = session.world.collection("user", "tasks").rows.values()
        return any(row.get("status", "pending") != "pending" for row in rows)

    async def _episode_call(
        self, session_id: str, episode: _Episode | None, tool_name: str, body: bytes | str
    ) -> Reply:
        """Call ``tool_name`` with ``body``. The task state the call leaves is published once the whole call is over
        (``_task_ends``), and with ``close_tools_on_task_end`` a call after a published end is refused."""
        try:
            self._refuse_after_task_end(session_id)
            return self.service.call(session_id, tool_name, body)
        finally:
            self._publish_task_state(session_id)

    def _refuse_after_task_end(self, session_id: str) -> None:
        """With ``close_tools_on_task_end``, refuse a call once one has returned with the episode's task ended."""
        if self.config.close_tools_on_task_end and self._task_ends.get(session_id, False):
            raise HTTPException(status_code=409, detail="The task has ended: its tools are closed")

    def _publish_task_state(self, session_id: str) -> None:
        """Publish the task state a whole call left (``GET /task_state``), while the session holds its world."""
        if self.service.store.get(session_id) is not None:
            self._task_ends[session_id] = self._task_ended(session_id)

    async def verify(
        self, request: Request, body: OfficeEnvVerifyRequest
    ) -> OfficeEnvVerifyResponse | OfficeEnvLengthPenaltyVerifyResponse:
        result = await self._verification(request.session[SESSION_ID_KEY])
        if result.get("officeenv_failure") == "no_session":
            # Never seeded, verified already, closed, or gone with a restart: the world went with
            # it, so the reward would measure the environment, not the policy.
            result = result | lost_session_verification(result["failure_reason"])
        if not self.config.length_penalty.enabled:
            return OfficeEnvVerifyResponse(**{**body.model_dump(), **result})
        penalty = length_penalty_fields(result["reward"], body.response, self.config.length_penalty)
        return OfficeEnvLengthPenaltyVerifyResponse(**{**body.model_dump(), **result, **penalty})

    async def _verification(self, session_id: str) -> dict[str, Any]:
        """OfficeEnv's verification of the session, which grades its world and releases it."""
        result = self.service.verify(session_id)
        self._task_ends.pop(session_id, None)  # verification releases the world, and its tools are closed
        return result

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
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`; Gym's
    # session routing sends each episode's requests to the worker that seeded it.
    app = OfficeEnvResourcesServer.run_webserver()  # noqa: F401
