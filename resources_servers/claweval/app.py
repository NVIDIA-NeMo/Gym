# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ClawEval owns task services, tools, simulated-user implementation and native grading."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, Tool
from pydantic import Field, PrivateAttr
from starlette.responses import JSONResponse, PlainTextResponse, Response

from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.server_runtime import (
    SESSION_ID_KEY,
    BaseResourcesServerConfig,
    MCPServerMetadata,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.claweval.process import TaskProcess, TaskSetupRejectedError
from resources_servers.claweval.task_data import TaskData, VerifyRequest, VerifyResponse


class ClawEvalResourcesConfig(BaseResourcesServerConfig):
    """Native task runtime; assistant policy execution belongs to a separate Agent Server."""

    num_workers: Literal[1] = 1
    expose_tools_over_mcp: Literal[False] = False
    claweval_root: str
    claweval_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    claweval_python: str
    claweval_config: str
    fixture_root: str | None = None
    sandbox_image: str
    sandbox_dependencies: str
    sandbox_port: int = Field(default=18080, ge=1024, le=65535)
    sandbox_ready_timeout: float = Field(default=180, gt=0)
    output_root: str = "outputs/claweval_sessions"
    operation_timeout_seconds: float = Field(default=900, gt=0)
    no_judge: bool = False
    model_overrides: dict[str, Any] = Field(default_factory=dict)
    judge_overrides: dict[str, Any] = Field(default_factory=dict)
    user_agent_model_overrides: dict[str, Any] = Field(default_factory=dict)


@dataclass
class TaskSession:
    worker: TaskProcess
    token: str = field(default_factory=lambda: uuid4().hex)
    info: dict = field(default_factory=dict)


_mcp_session: ContextVar[TaskSession] = ContextVar("claweval_mcp_session")


class ClawEvalResourcesServer(SimpleResourcesServer):
    config: ClawEvalResourcesConfig
    seed_request_model = TaskData
    grading_may_use_uninstrumented_model = True
    _sessions: dict[str, TaskSession] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/task_info")(self.task_info)
        app.post("/simulate_user")(self.simulate_user)
        app.post("/{tool_name}")(self.call_tool)
        server = Server("claweval")

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(name=s["name"], description=s["description"], inputSchema=s["input_schema"])
                for s in _mcp_session.get().info["tools"]
            ]

        # The SDK's global tool-name cache cannot validate task-scoped schemas with the same name.
        @server.call_tool(validate_input=False)
        async def call_tool(name: str, arguments: dict) -> CallToolResult:
            result = await self._tool(_mcp_session.get(), name, arguments)
            return CallToolResult.model_validate(result)

        manager = StreamableHTTPSessionManager(server, json_response=True, stateless=True)

        async def mcp_app(scope, receive, send):
            headers = dict(scope.get("headers", []))
            credential = headers.get(b"authorization", b"").decode()
            sessions = [state for state in self._sessions.values() if credential == f"Bearer {state.token}"]
            if not sessions:
                from starlette.responses import Response

                await Response(status_code=401)(scope, receive, send)
                return
            binding = _mcp_session.set(sessions[0])
            try:
                await manager.handle_request(scope, receive, send)
            finally:
                _mcp_session.reset(binding)

        app.mount("/mcp", mcp_app)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI):
            try:
                async with parent_lifespan(application), manager.run():
                    yield
            finally:
                results = await asyncio.gather(
                    *(self.release_session_state(identifier) for identifier in list(self._sessions)),
                    return_exceptions=True,
                )
                failures = [result for result in results if isinstance(result, BaseException)]
                if failures:
                    raise BaseExceptionGroup("ClawEval resource cleanup failed", failures)

        app.router.lifespan_context = lifespan
        return app

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def initialize_session_state(self, session_id: str, body: TaskData) -> None:
        data = TaskData.model_validate(body.model_dump())
        state = TaskSession(worker=TaskProcess())
        self._sessions[session_id] = state
        # Never use a caller-controlled session identifier as a filesystem path.
        output = Path(self.config.output_root).resolve() / uuid4().hex
        await state.worker.start(self.config.claweval_python, output)
        payload = self.config.model_dump()
        payload.update(task_data=data.model_dump(), output_dir=str(output), agent_name="Gym agent session")
        try:
            state.info = await state.worker.call("seed", payload, timeout=self.config.operation_timeout_seconds)
        except TaskSetupRejectedError as error:
            raise ResourcesSeedRejectedError(str(error)) from error

    async def release_session_state(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is not None:
            await state.worker.close()
            self._sessions.pop(session_id, None)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        result = await super().seed_session(body, request=request)
        state = self._sessions[body.resources_session_id]
        return result.model_copy(
            update={
                "resources_tools": MCPServerMetadata(
                    server_name="claweval", url_path="/mcp/", headers={"Authorization": f"Bearer {state.token}"}
                )
            }
        )

    def _state(self, request: Request) -> TaskSession:
        state = self._sessions.get(request.session.get(SESSION_ID_KEY))
        if state is None:
            raise HTTPException(409, "No active ClawEval resources session")
        return state

    async def task_info(self, request: Request) -> dict:
        return self._state(request).info

    async def simulate_user(self, request: Request, body: dict = Body()) -> dict:
        state = self._state(request)
        return await state.worker.call("simulate", body, timeout=self.config.operation_timeout_seconds)

    async def _tool(self, state: TaskSession, name: str, arguments: dict) -> dict:
        if state not in self._sessions.values():
            raise HTTPException(409, "ClawEval resources session was closed")
        return await state.worker.call(
            "tool", {"name": name, "arguments": arguments}, timeout=self.config.operation_timeout_seconds
        )

    async def call_tool(self, request: Request, tool_name: str, body: dict = Body()) -> Response:
        result = await self._tool(self._state(request), tool_name, body)
        text = "\n".join(part["text"] for part in result["content"] if part["type"] == "text")
        if result["isError"]:
            raise HTTPException(400, text)
        images = [part for part in result["content"] if part["type"] == "image"]
        if images:
            return JSONResponse(
                {
                    "output": text,
                    "tool_output_images": [
                        {"image_url": f"data:{part['mimeType']};base64,{part['data']}"} for part in images
                    ],
                }
            )
        return PlainTextResponse(text)

    async def verify(self, request: Request, body: VerifyRequest) -> VerifyResponse:
        identifier = request.session.get(SESSION_ID_KEY)
        async with self.verifying(identifier) as verification:
            if verification.is_lost:
                return VerifyResponse(
                    **lost_session_verification("Native ClawEval state is unavailable"), reward_components={}
                )
            if verification.retained is not None:
                return VerifyResponse.model_validate(verification.retained)
            result = await self._state(request).worker.call(
                "verify",
                body.verification_input.model_dump(mode="json"),
                timeout=self.config.operation_timeout_seconds,
            )
            verification.retain(result)
            return VerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


if __name__ == "__main__":
    ClawEvalResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ClawEvalResourcesServer.run_webserver()
