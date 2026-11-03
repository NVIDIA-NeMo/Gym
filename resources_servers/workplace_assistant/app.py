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
import asyncio
import math
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, ValidationError

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
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
    next_attempt,
)
from resources_servers.workplace_assistant.task_data import TaskData
from resources_servers.workplace_assistant.utils import get_tools, is_correct


_TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    # Column by column, so each cell becomes a plain Python value.
    # Float NaN (empty CSV cells) and None (values a tool call stored) both serialize as null,
    # so NaN positions are listed separately to restore them exactly.
    columns = []
    for name, series in frame.items():
        values = series.tolist()
        columns.append(
            {
                "name": name,
                "dtype": str(series.dtype),
                "values": [None if isinstance(value, float) and math.isnan(value) else value for value in values],
                "nan": [i for i, value in enumerate(values) if isinstance(value, float) and math.isnan(value)],
            }
        )
    return {"index": frame.index.tolist(), "columns": columns}


def _decode_frame(payload: dict[str, Any]) -> pd.DataFrame:
    index = payload["index"]
    data = {}
    for column in payload["columns"]:
        values = list(column["values"])
        if len(values) != len(index):
            raise ValueError(f"column {column['name']!r} has {len(values)} values for {len(index)} rows")
        for position in column["nan"]:
            values[position] = float("nan")
        data[column["name"]] = pd.Series(values, dtype=object).astype(column["dtype"])
    frame = pd.DataFrame(data)
    # Tools append rows by label (``.loc[len(frame)]``), so the index labels are part of the state.
    frame.index = pd.RangeIndex(len(index)) if index == list(range(len(index))) else pd.Index(index)
    return frame


def _export_tool_env(tool_env: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            attribute: _encode_frame(frame)
            for attribute, frame in vars(container).items()
            if isinstance(frame, pd.DataFrame)
        }
        for name, container in tool_env["containers"].items()
    }


def _restore_tool_env(state: dict[str, Any]) -> dict[str, Any]:
    tool_env = get_tools(_TOOLKITS)
    containers = tool_env["containers"]
    if set(state) != set(containers):
        raise ValueError(f"expected containers {sorted(containers)}, got {sorted(state)}")
    for name, frames in state.items():
        container = containers[name]
        expected = {attribute for attribute, value in vars(container).items() if isinstance(value, pd.DataFrame)}
        if set(frames) != expected:
            raise ValueError(f"expected {name} tables {sorted(expected)}, got {sorted(frames)}")
        for attribute, payload in frames.items():
            setattr(container, attribute, _decode_frame(payload))
    return tool_env


class WorkbenchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class WorkbenchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchVerifyRequest(BaseVerifyRequest):
    ground_truth: list[Dict[str, str]] | str
    id: int
    category: str
    environment_name: str


class WorkbenchVerifyResponse(BaseVerifyResponse):
    pass


class WorkbenchResourcesServer(SimpleResourcesServer):
    # Each session's tool environment is a set of in-memory tables,
    # exported whole at a checkpoint. /verify keeps the default "wait"
    # because it discards the tool environment of a session an Agent's /run seeded.
    checkpoint_mode = "exported"
    config: WorkbenchResourcesServerConfig
    session_id_to_tool_env: Dict[str, Any] = Field(default_factory=dict)
    # The episode and task each typed seed bound its resources_session_id to.
    # These sessions keep their tool environment until /close_session; an Agent's /run never closes, so /verify
    # releases the sessions it seeds.
    _typed_sessions: Dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    _fence: ResourcesSessionFence = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    # Register all 27 workplace tools as MCP tools via the catch-all route when expose_tools_over_mcp is enabled.
    def mcp_tools(self, harvested, catchall):
        specs = get_tools(_TOOLKITS)["schemas"]
        return harvested + [catchall.tool(s["name"], s["parameters"], s.get("description")) for s in specs]

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        if not isinstance(body, ResourcesSeedSessionRequest):
            # init session once for each sample.
            session_id = request.session[SESSION_ID_KEY]
            self.session_id_to_tool_env[session_id] = get_tools(_TOOLKITS)
            return BaseSeedSessionResponse()

        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            identity = self._typed_sessions.get(session_id)
            if identity is None:
                # The verify body is the task's fields, so a task that fails here would fail its verify too.
                try:
                    TaskData.model_validate(body.task_data)
                except ValidationError as error:
                    raise ResourcesSeedRejectedError(
                        f"task_data is not a workplace_assistant task: {error}"
                    ) from error
                # A session a checkpoint restored keeps the tables its tool calls changed. Build the tables before
                # binding, so a get_tools failure (a 500 the caller retries) leaves no binding without tables.
                if session_id not in self.session_id_to_tool_env:
                    self.session_id_to_tool_env[session_id] = get_tools(_TOOLKITS)
                self._typed_sessions[session_id] = (body.episode_id, body.task_id)
            elif identity != (body.episode_id, body.task_id):
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
        # Only on success: a refused seed must not hand the caller a cookie for a session another episode holds.
        request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        # Typed sessions are keyed by resources_session_id, so a close without the seed's cookie still finds them.
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            closed_episode = self._fence.closed_episode(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            identity = self._typed_sessions.get(session_id)
            if identity is not None and identity[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            self._release(session_id)
            self._fence.mark_closed(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    def _release(self, session_id: str) -> None:
        self.session_id_to_tool_env.pop(session_id, None)
        self._typed_sessions.pop(session_id, None)

    async def route_to_python_function(self, path: str, body: WorkbenchRequest, request: Request) -> WorkbenchResponse:
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_tool_env:
            raise ResourcesSessionLostError("No seeded workplace_assistant session for this tool call")

        tool_env = self.session_id_to_tool_env[session_id]
        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        try:
            function = tool_env["functions"][path]
            result = function(**args)
            return WorkbenchResponse(output=result)
        except Exception as e:
            return WorkbenchResponse(
                output=f"Error executing tool '{path}': {str(e)}"
            )  # return error to model so that it can correct itself

    async def verify(self, request: Request, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_tool_env:
            # Never seeded, closed, or gone with a restart: the agent's tool calls had no tables to act on,
            # so replaying them would not measure the policy.
            lost = lost_session_verification("No seeded workplace_assistant session for this verify")
            return WorkbenchVerifyResponse(**(body.model_dump() | lost))
        try:
            ground_truth = body.ground_truth
            response = body.response.output

            total_score = 0.0

            # Convert list of ResponseFunctionToolCall objects into list of dictionaries
            predicted_function_calls = []

            for message in response:
                if message.type == "function_call":
                    predicted_function_calls.append(message.model_dump())

            predicted_chat_content = []

            for message in response:
                if message.type == "output_text":
                    predicted_chat_content.append(message.model_dump())

            total_score += is_correct(predicted_function_calls, ground_truth, None) * 1.0
            return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)
        finally:
            if session_id not in self._typed_sessions:
                self.session_id_to_tool_env.pop(session_id, None)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        # A session verify already discarded, for example because it raised, is left out.
        tool_envs = {
            session_id: self.session_id_to_tool_env[session_id]
            for session_id in session_ids
            if session_id in self.session_id_to_tool_env
        }

        def encode() -> dict[str, JsonValue]:
            states = {}
            for session_id, tool_env in tool_envs.items():
                state = _export_tool_env(tool_env)
                identity = self._typed_sessions.get(session_id)
                if identity is not None:
                    episode_id, task_id = identity
                    state = {
                        "tool_env": state,
                        "episode_id": episode_id.model_dump(mode="json"),
                        "task_id": task_id.model_dump(mode="json"),
                    }
                states[session_id] = state
            return states

        # Encoding every table is CPU work; admission is closed during a commit, so no tool call mutates them.
        return await asyncio.to_thread(encode)

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        def rebuild() -> tuple[dict[str, Any], dict[str, tuple[EpisodeId, TaskId]]]:
            restored = {}
            typed_sessions = {}
            for session_id, state in states.items():
                try:
                    if "tool_env" in state:
                        # The checkpoint participant transfers a restored session to the replacement attempt.
                        typed_sessions[session_id] = (
                            next_attempt(EpisodeId.model_validate(state["episode_id"])),
                            TaskId.model_validate(state["task_id"]),
                        )
                        state = state["tool_env"]
                    # Existing snapshots and sessions seeded by /run contain only the tables.
                    restored[session_id] = _restore_tool_env(state)
                except (AttributeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid workplace assistant state for session {session_id}: {error}") from error
            return restored, typed_sessions

        # Rebuild every session before installing any, so an invalid state installs nothing.
        restored, typed_sessions = await asyncio.to_thread(rebuild)
        self.session_id_to_tool_env.update(restored)
        self._typed_sessions.update(typed_sessions)

    async def retire_session_state(self, session_id: str) -> None:
        self._release(session_id)


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
