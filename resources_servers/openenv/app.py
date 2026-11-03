# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generic NeMo-Gym resource server adapter for OpenEnv environments.

Wraps any OpenEnv Environment class and exposes it through NeMo-Gym's
three-server architecture. MCP environments get per-tool POST endpoints
discovered at startup. Non-MCP environments get a single POST /step endpoint.
The YAML config selects which environment to load — no Python code needed
to add a new environment.
"""

import importlib
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, create_model

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
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
    ResourcesSessionFence,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)


def _import_class(dotted_path: str) -> type:
    """Import a class from a dotted path like 'package.module.ClassName'."""
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to import '{dotted_path}': {e}. Ensure the environment's dependencies are installed."
        ) from e


class OpenEnvResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the OpenEnv adapter resource server."""

    env_class: str  # Dotted import path to the Environment class
    action_class: str  # Dotted import path to the Action class
    reset_kwargs: dict = {}
    is_mcp: bool = False


class SessionState(BaseModel):
    """Per-session state tracking an OpenEnv environment instance."""

    model_config = {"arbitrary_types_allowed": True}

    env: Any  # The OpenEnv Environment instance
    accumulated_reward: float = 0.0
    step_count: int = 0
    done: bool = False
    last_observation: Optional[dict] = None


class OpenEnvResourcesServer(SimpleResourcesServer):
    """Generic adapter that wraps any OpenEnv environment as a NeMo-Gym resource server."""

    config: OpenEnvResourcesServerConfig
    # Keyed by the session cookie's SESSION_ID_KEY, which a typed seed binds to its resources_session_id.
    _sessions: Dict[str, SessionState] = {}
    # The episode and task each Environment Server session was seeded for, keyed by resources_session_id.
    _session_episodes: Dict[str, tuple[EpisodeId, TaskId]] = {}
    _fence: Optional[ResourcesSessionFence] = None
    _env_class: Any = None
    _action_class: Any = None
    _is_mcp: bool = False

    def model_post_init(self, context: Any) -> None:
        """Initialize private attributes: session stores and dynamically imported classes."""
        self._sessions = {}
        self._session_episodes = {}
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)
        self._env_class = _import_class(self.config.env_class)
        self._action_class = _import_class(self.config.action_class)

    def setup_webserver(self) -> FastAPI:
        """Set up FastAPI app with tool endpoints based on environment type.

        MCP environments get per-tool POST endpoints discovered at startup.
        Non-MCP environments get a single POST /step endpoint.
        """
        app = super().setup_webserver()

        self._is_mcp = self.config.is_mcp

        if self._is_mcp:
            self._register_mcp_endpoints(app)
        else:
            app.post("/step")(self._handle_step)

        return app

    def _register_mcp_endpoints(self, app: FastAPI) -> None:
        """Discover MCP tools and register each as a POST endpoint."""
        from openenv.core.env_server.mcp_types import ListToolsAction

        temp_env = self._env_class()
        temp_env.reset()
        obs = temp_env.step(ListToolsAction())
        tools = obs.tools
        if hasattr(temp_env, "close"):
            temp_env.close()

        self._mcp_tools = {tool.name: tool for tool in tools}

        for tool in tools:
            request_model = self._schema_to_pydantic(tool.name, tool.input_schema)
            handler = self._make_mcp_handler(tool.name, request_model)
            app.post(f"/{tool.name}")(handler)

    def _schema_to_pydantic(self, tool_name: str, schema: dict) -> type:
        """Convert a JSON schema dict to a Pydantic model class."""
        fields = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for name, prop in properties.items():
            python_type = self._json_type_to_python(prop.get("type", "string"))
            if name in required:
                fields[name] = (python_type, ...)
            else:
                fields[name] = (Optional[python_type], None)
        model_name = f"{tool_name.title().replace('_', '')}Request"
        return create_model(model_name, **fields)

    @staticmethod
    def _json_type_to_python(json_type: str) -> type:
        """Map JSON schema types to Python types."""
        mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        return mapping.get(json_type, str)

    def _make_mcp_handler(self, tool_name: str, request_model: type):
        """Create a POST handler for an MCP tool."""

        async def handler(request: Request, body: request_model) -> dict:
            session = self._open_session(request)
            if session.done:
                return {"error": "Episode has ended.", "result": None}

            try:
                action = self._action_class(tool_name=tool_name, arguments=body.model_dump())
                obs = session.env.step(action)
            except Exception as e:
                return {"error": str(e), "result": None}

            if obs.reward is not None:
                session.accumulated_reward += float(obs.reward)
            session.step_count += 1
            session.done = obs.done

            result = obs.result if hasattr(obs, "result") else None
            error = str(obs.error) if hasattr(obs, "error") and obs.error else None
            return {"result": result, "error": error}

        handler.__name__ = f"handle_{tool_name}"
        return handler

    async def _handle_step(self, request: Request, body: dict) -> dict:
        """Handle non-MCP step calls by constructing the Action and calling env.step()."""
        session = self._open_session(request)
        if session.done:
            return {"error": "Episode has ended.", "result": None}

        try:
            action = self._action_class(**body)
            obs = session.env.step(action)
        except Exception as e:
            return {"error": str(e), "result": None}

        if obs.reward is not None:
            session.accumulated_reward += float(obs.reward)
        session.step_count += 1
        session.done = obs.done
        session.last_observation = obs.model_dump() if hasattr(obs, "model_dump") else {}

        return obs.model_dump() if hasattr(obs, "model_dump") else {"result": str(obs)}

    def _open_session(self, request: Request) -> SessionState:
        """The session a tool call names, or a 404 if it was never seeded or has already ended."""
        session = self._sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            raise ResourcesSessionLostError("No active OpenEnv session for this request. Call /seed_session first.")
        return session

    def _new_session(self) -> SessionState:
        """Create an environment instance, call reset(), and wrap it in session state.

        Calls env.reset() directly with only the configured reset_kwargs, because
        some OpenEnv environments don't accept the extra seed/episode_id kwargs
        that reset_async() injects.
        """
        env = self._env_class()
        try:
            initial_obs = env.reset(**self.config.reset_kwargs)
        except BaseException:
            # The seed fails, so nothing would ever close this instance.
            if hasattr(env, "close"):
                env.close()
            raise
        initial_reward = float(initial_obs.reward) if initial_obs.reward is not None else 0.0
        return SessionState(env=env, accumulated_reward=initial_reward, done=initial_obs.done)

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Create and reset this session's environment instance.

        An Agent's /run seeds with its legacy body under the session its cookie names. An Environment Server seeds
        with a ResourcesSeedSessionRequest: the environment is keyed by resources_session_id, the cookie is bound
        to that ID so the agent's tool calls reach it, and a repeated seed for the same episode keeps the
        environment it already has. The rows carry no task fields, so there is no task_data to validate.
        """
        if not isinstance(body, ResourcesSeedSessionRequest):
            self._sessions[request.session[SESSION_ID_KEY]] = self._new_session()
            return BaseSeedSessionResponse()

        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            identity = self._session_episodes.get(session_id)
            if identity is None:
                self._sessions[session_id] = self._new_session()
                self._session_episodes[session_id] = (body.episode_id, body.task_id)
            elif identity != (body.episode_id, body.task_id):
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
            request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Close the session's environment instance, if verify has not already closed it.

        Keyed by resources_session_id, so a close without the cookie (the seed's reply was lost) still releases it.
        Closing an unknown or already-closed session succeeds, and the fence refuses a later seed of it.
        """
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            closed_episode = self._fence.closed_episode(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            identity = self._session_episodes.get(session_id)
            if identity is not None and identity[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            session = self._sessions.get(session_id)
            if session is not None and hasattr(session.env, "close"):
                # A failure raises before the session is forgotten or fenced, so the caller can retry the close.
                session.env.close()
            self._sessions.pop(session_id, None)
            self._session_episodes.pop(session_id, None)
            self._fence.mark_closed(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    # TODO(ahmadki): Reward accumulation strategy needs a revisit.
    #
    # NeMo-Gym expects a single reward per rollout from verify(). This adapter
    # sums obs.reward across all step() calls during the episode, which assumes
    # environments return **per-step deltas** (e.g., 0 + 0 + 1.0 = 1.0).
    #
    # This breaks if an environment returns **cumulative** rewards (e.g., step
    # returns 0.3, 0.6, 1.0 meaning total=1.0, but we'd sum to 1.9).
    #
    # Works correctly for: terminal-only rewards (chess, coding), zero rewards
    # (echo MCP), and true per-step deltas (maze).
    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Return accumulated reward, close the environment, and clean up session state.

        Verify releases the environment because an Agent's /run never closes its session. An Environment Server's
        close after this finds nothing left to release. A verify for a session this server no longer holds (never
        seeded, already verified or closed, or lost to a restart) has no reward to report, so its sample is masked.
        """
        session_id = request.session[SESSION_ID_KEY]
        session = self._sessions.get(session_id)
        if session is None:
            return BaseVerifyResponse(
                **(body.model_dump() | lost_session_verification("No active OpenEnv session to verify"))
            )

        reward = session.accumulated_reward
        if hasattr(session.env, "close"):
            session.env.close()
        del self._sessions[session_id]

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    OpenEnvResourcesServer.run_webserver()
