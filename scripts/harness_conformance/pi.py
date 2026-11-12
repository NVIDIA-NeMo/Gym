# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script Pi's context-overflow recovery within a single agent invocation."""

from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .provider import Probe


@dataclass
class PiProbe(Probe):
    tool_output_chars: int = 20000

    def _tool(self, body: dict) -> tuple[str, dict, dict]:
        name, arguments, namespaces = super()._tool(body)
        # Pi retains 20k estimated tokens. Real output makes older history eligible
        # for summarization; reporting large usage alone cannot create that history.
        arguments["command"] = f"head -c {self.tool_output_chars} /dev/zero | tr '\\0' x; " + arguments["command"]
        return name, arguments, namespaces

    async def model(self, request: Request) -> JSONResponse | StreamingResponse:
        index = len(self.attempts)
        if index >= len(self.responses) or self.responses[index]["kind"] != "context_overflow":
            return await super().model(request)
        body = await request.json()
        self._observe_tool_results(body)
        payload = {
            "error": {
                "message": "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }
        self.attempts.append(
            {"attempt_id": f"attempt-{index}", "request": body, "response": payload, "status_code": 400}
        )
        self.save()
        return JSONResponse(payload, status_code=400)
