# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Resources Server whose tools are a toolset over each session's state, so the tools can run apart from it.

A Resources Server does two jobs: its tools change the world of a session, and its verifier computes the reward from
that world. ``ToolsetResourcesServer`` keeps the two apart in code, joined only by the session's state:

- ``session_state_model``: a Pydantic model of one session's state. The seed creates it, the tools change it and the
  verifier reads it. ``None`` when the tools keep no state.
- ``toolset``: a ``ResourcesToolset`` subclass whose ``@episode_tool`` methods are the tools. A toolset sees the
  server's config and one session's state, and nothing else: no request, no session and no other server.
- The server: ``seed_state`` builds a session's start state from its task, and ``verify`` reads the state with
  ``session_state``.

The server serves each tool at ``POST /<tool name>``, with the same request and reply as a tool route registered by
hand, so datasets and agents do not change. Because a toolset needs only the config and the state, it also runs in
another process, such as the episode's tool sandbox (``resources_tool_sandbox``): ``/export_session_state`` and
``/restore_session_state`` move the state out and back.

A server with a ``session_state_model`` keeps its state through the base's session hooks, so typed seeds, closes,
fences and checkpoints come from the base, and its ``checkpoint_mode`` is ``exported``. A server without one is
``stateless``.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import ClassVar

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, JsonValue, PrivateAttr, ValidationError

from nemo_gym.client.server_client import SESSION_ID_KEY
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    ResourcesSessionLostError,
    SimpleResourcesServer,
)
from nemo_gym.server_runtime.episode_tools import as_tool_result, episode_tool_arguments
from nemo_gym.server_runtime.tool_access import END_ACTIVATION_HEADER


class ResourcesSessionState(BaseModel):
    """One session's state as JSON, as ``/export_session_state`` returns it; ``None`` when the tools keep none."""

    model_config = ConfigDict(extra="forbid")

    state: JsonValue = None


class ResourcesToolset:
    """The tools of a ``ToolsetResourcesServer``: its ``@episode_tool`` methods, over one session's state.

    Each method takes one Pydantic model of arguments and returns a Pydantic model, a ``ToolResult`` or a JSON value.
    Its name is the tool's name and its docstring the tool's description. A subclass narrows the types of ``config``
    and ``state``. The toolset is built again for each call, so everything a tool changes belongs in ``state``.
    """

    config: BaseResourcesServerConfig
    state: BaseModel | None

    def __init__(self, *, config: BaseResourcesServerConfig, state: BaseModel | None) -> None:
        self.config = config
        self.state = state


OpenToolset = Callable[[Request], AbstractAsyncContextManager[ResourcesToolset]]


def register_tool_routes(app: FastAPI, toolset: type[ResourcesToolset], open_toolset: OpenToolset) -> None:
    """Serve each tool of ``toolset`` at ``POST /<tool name>``, on the toolset ``open_toolset`` opens for a request.

    The body is the tool's arguments model, so a bad argument answers 422 as on any FastAPI route. The reply is the
    tool's output as JSON, with ``END_ACTIVATION_HEADER`` when the tool ends the agent's activation.
    """
    for name, arguments_model in episode_tool_arguments(toolset).items():
        app.post(f"/{name}", name=name)(_tool_endpoint(name, arguments_model, open_toolset))


def _tool_endpoint(name: str, arguments_model: type[BaseModel], open_toolset: OpenToolset) -> Callable:
    # FastAPI reads the body's model from this annotation, so this module must not postpone annotations.
    async def call_tool(request: Request, arguments: arguments_model) -> JSONResponse:  # type: ignore[valid-type]
        async with open_toolset(request) as tools:
            result = as_tool_result(await getattr(tools, name)(arguments))
        headers = {END_ACTIVATION_HEADER: "true"} if result.ends_activation else None
        return JSONResponse(result.output, headers=headers)

    return call_tool


class ToolsetResourcesServer(SimpleResourcesServer):
    """A Resources Server whose tools are a ``ResourcesToolset`` over each session's state (see the module docstring).

    Set ``toolset``, and, when the tools keep state, ``session_state_model`` and ``seed_state``. Unless the class
    declares ``checkpoint_mode`` itself, it follows from ``session_state_model``.
    """

    toolset: ClassVar[type[ResourcesToolset]]
    session_state_model: ClassVar[type[BaseModel] | None] = None
    _session_states: dict[str, BaseModel] = PrivateAttr(default_factory=dict)
    _unclaimed: set[str] = PrivateAttr(default_factory=set)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # A class may still declare its own, such as "restart_only" for a test of restarts.
        if "checkpoint_mode" not in vars(cls):
            cls.checkpoint_mode = "exported" if cls.session_state_model is not None else "stateless"

    async def seed_state(self, body: BaseSeedSessionRequest) -> BaseModel:
        """Return a session's start state, an instance of ``session_state_model``, from its task as
        ``seed_request_model`` validated it.

        Raise ``ResourcesSeedRejectedError`` for a task that no retry can seed. Called only when the server has a
        ``session_state_model``.
        """
        raise NotImplementedError(f"{type(self).__name__} sets session_state_model, so it implements seed_state")

    def session_state(self, session_id: str) -> BaseModel | None:
        """The session's state for ``verify``, or ``None`` when this server does not hold the session."""
        return self._session_states.get(session_id)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        register_tool_routes(app, self.toolset, self._open_toolset)
        app.post("/export_session_state")(self.export_session_state)
        app.post("/restore_session_state")(self.restore_session_state)
        return app

    @asynccontextmanager
    async def _open_toolset(self, request: Request) -> AsyncIterator[ResourcesToolset]:
        if self.session_state_model is None:
            yield self.toolset(config=self.config, state=None)
            return
        session_id = request.session[SESSION_ID_KEY]
        async with self.session_lock(session_id):
            state = self._session_states.get(session_id)
            if state is None:
                raise ResourcesSessionLostError(f"{type(self).__name__} holds no state for this tool call's session")
            yield self.toolset(config=self.config, state=state)

    async def export_session_state(self, request: Request) -> ResourcesSessionState:
        """Return the state of the cookie's session, for its tools to continue it in another process."""
        if self.session_state_model is None:
            return ResourcesSessionState()
        session_id = request.session[SESSION_ID_KEY]
        async with self.session_lock(session_id):
            state = self._session_states.get(session_id)
            if state is None:
                raise ResourcesSessionLostError(f"{type(self).__name__} holds no state for this session")
            return ResourcesSessionState(state=state.model_dump(mode="json"))

    async def restore_session_state(self, request: Request, body: ResourcesSessionState) -> ResourcesSessionState:
        """Replace the cookie's session state with the state its tools left in another process."""
        if self.session_state_model is None:
            return ResourcesSessionState()
        try:
            state = self.session_state_model.model_validate(body.state)
        except ValidationError as error:
            raise HTTPException(
                status_code=422, detail=f"The state is not a valid {self.session_state_model.__name__}: {error}"
            ) from error
        session_id = request.session[SESSION_ID_KEY]
        async with self.session_lock(session_id):
            # A closed or lost session stays gone: the state of its last episode does not bring it back.
            if session_id not in self._session_states:
                raise ResourcesSessionLostError(f"{type(self).__name__} holds no state for this session")
            self._session_states[session_id] = state
        return ResourcesSessionState()

    # The base's session hooks, over ``_session_states``. Without a ``session_state_model`` the base manages no state.
    # A session a checkpoint restored is unclaimed until the seed of its replacement attempt, cookie or typed, claims
    # it: that seed continues the restored state instead of seeding the session again.

    def _manages_session_state(self) -> bool:
        return self.session_state_model is not None

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._session_states and session_id not in self._unclaimed

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        if session_id in self._unclaimed:
            self._unclaimed.discard(session_id)
            return
        self._session_states[session_id] = await self.seed_state(body)

    async def release_session_state(self, session_id: str) -> None:
        self._session_states.pop(session_id, None)
        self._unclaimed.discard(session_id)

    async def export_session_states(self, session_ids: list[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        return {
            session_id: self._session_states[session_id].model_dump(mode="json")
            for session_id in session_ids
            if session_id in self._session_states
        }

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        # Validate every state before installing any.
        restored = {session_id: self.session_state_model.model_validate(state) for session_id, state in states.items()}
        self._session_states.update(restored)
        self._unclaimed.update(restored)
