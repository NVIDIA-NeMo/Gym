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
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, create_model

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSessionLostError,
    StatefulResourcesServer,
    lost_session_verification,
)


logger = logging.getLogger(__name__)


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


class OpenEnvResourcesServer(StatefulResourcesServer[float]):
    """Generic adapter that wraps any OpenEnv environment as a NeMo-Gym resource server."""

    config: OpenEnvResourcesServerConfig
    # Keyed by the session cookie's SESSION_ID_KEY, which a typed seed binds to its resources_session_id.
    _sessions: Dict[str, SessionState] = {}
    # Failed resets whose cleanup also failed still belong to the session's eventual close.
    _pending_envs: Dict[str, Any] = {}
    _env_class: Any = None
    _action_class: Any = None
    _is_mcp: bool = False

    def model_post_init(self, context: Any) -> None:
        """Initialize private attributes: session stores and dynamically imported classes."""
        super().model_post_init(context)
        self._sessions = {}
        self._pending_envs = {}
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

    def _new_session(self, session_id: str) -> SessionState:
        """Create an environment instance, call reset(), and wrap it in session state.

        Calls env.reset() directly with only the configured reset_kwargs, because
        some OpenEnv environments don't accept the extra seed/episode_id kwargs
        that reset_async() injects.
        """
        self._close_pending_env(session_id)
        env = self._env_class()
        self._pending_envs[session_id] = env
        try:
            initial_obs = env.reset(**self.config.reset_kwargs)
            initial_reward = float(initial_obs.reward) if initial_obs.reward is not None else 0.0
            session = SessionState(env=env, accumulated_reward=initial_reward, done=initial_obs.done)
        except BaseException:
            try:
                self._close_pending_env(session_id)
            except Exception:
                # Preserve the reset error and ownership so the caller's close or seed retry can release it.
                logger.exception("Failed to close OpenEnv instance after reset failed for session %s", session_id)
            raise
        self._pending_envs.pop(session_id)
        return session

    def _close_pending_env(self, session_id: str) -> None:
        env = self._pending_envs.get(session_id)
        if env is not None and hasattr(env, "close"):
            env.close()
        self._pending_envs.pop(session_id, None)

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Reset an environment, closing a previous legacy run before replacing it."""
        previous = self._sessions.get(session_id)
        if previous is not None and hasattr(previous.env, "close"):
            previous.env.close()
        self._sessions.pop(session_id, None)
        self._sessions[session_id] = self._new_session(session_id)

    async def release_session_state(self, session_id: str) -> None:
        """Close pending and live environments, retaining them if cleanup fails."""
        self._close_pending_env(session_id)
        session = self._sessions.get(session_id)
        if session is not None and hasattr(session.env, "close"):
            session.env.close()
        self._sessions.pop(session_id, None)

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
        """Release the environment and retain only its reward for a bounded verification retry window."""
        session_id = request.session[SESSION_ID_KEY]
        verified = self._verified_sessions.get(session_id)
        if self._fence.closed_episode(session_id) is None:
            if verified is not None:
                return BaseVerifyResponse(**body.model_dump(), reward=verified[1])
            session = self._sessions.get(session_id)
            if session is not None:
                reward = session.accumulated_reward
                if hasattr(session.env, "close"):
                    session.env.close()
                self._sessions.pop(session_id)
                identity = self._session_episodes.pop(session_id, None)
                self._verified_sessions.remember(session_id, (identity, reward))
                return BaseVerifyResponse(**body.model_dump(), reward=reward)
        return BaseVerifyResponse(
            **(body.model_dump() | lost_session_verification("No active OpenEnv session to verify"))
        )


if __name__ == "__main__":
    OpenEnvResourcesServer.run_webserver()
