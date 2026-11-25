# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Resources Server whose tools keep per-session state that grading reads, served from a tool sidecar in tests.

Each tool call appends ``<tool>:<argument>`` to the session's ledger. The ledger is the tool state: the sidecar's copy
exports it, and the Resources host's copy imports it before ``/verify``, which rewards a ledger that holds a ``pay``
call. A catch-all tool route, registered last as AutomationBench's is, checks that no tool route answers for the tool
host's own routes.
"""

from typing import Any

from fastapi import Body, FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import JsonValue, PrivateAttr

from nemo_gym.client.server_client import SESSION_ID_KEY
from nemo_gym.environment.tool_sidecar import ToolSidecarResources, ToolSidecarResourcesConfig
from nemo_gym.server_runtime.base_resources_server import (
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class LedgerConfig(ToolSidecarResourcesConfig):
    pass


class LedgerSeed(BaseSeedSessionRequest):
    opening: str = ""


class LedgerResourcesServer(ToolSidecarResources, SimpleResourcesServer):
    config: LedgerConfig
    seed_request_model = LedgerSeed

    _ledgers: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{tool_name}")(self.call_tool)
        return app

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._ledgers

    async def initialize_session_state(self, session_id: str, body: LedgerSeed) -> None:
        self._ledgers[session_id] = [f"opening:{body.opening}"]

    async def release_session_state(self, session_id: str) -> None:
        self._ledgers.pop(session_id, None)

    async def call_tool(self, request: Request, tool_name: str, arguments: Any = Body(None)) -> PlainTextResponse:
        ledger = self._ledgers[request.session[SESSION_ID_KEY]]
        ledger.append(f"{tool_name}:{(arguments or {}).get('value')}")
        return PlainTextResponse(f"{tool_name} #{len(ledger) - 1}")

    def export_tool_state(self, session_id: str) -> JsonValue:
        return {"ledger": self._ledgers[session_id]}

    async def import_tool_state(self, session_id: str, state: JsonValue) -> None:
        self._ledgers[session_id] = list(state["ledger"])

    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        ledger = self._ledgers.get(request.session[SESSION_ID_KEY], [])
        return BaseVerifyResponse(**body.model_dump(), reward=float(any(entry.startswith("pay:") for entry in ledger)))
