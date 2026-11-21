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

from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import Request
from pydantic import Field

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.browserbase_resources import (
    BrowserbaseDOMResourcesServer,
    BrowserbaseDOMResourcesServerConfig,
)
from nemo_gym.openai_utils import NeMoGymResponseOutputMessage
from nemo_gym.server_utils import SESSION_ID_KEY


class BrowserbaseWebVoyagerResourcesServerConfig(BrowserbaseDOMResourcesServerConfig):
    expected_answer_match_mode: str = Field(
        default="contains",
        description='How verify() compares the final assistant answer to expected_answer. Supported: "contains", "exact".',
    )


class BrowserbaseWebVoyagerVerifyRequest(BaseVerifyRequest):
    task_id: Optional[str | int] = None
    website: Optional[str] = None
    question: str
    start_url: str
    expected_answer: str


class BrowserbaseWebVoyagerVerifyResponse(BaseVerifyResponse):
    task_id: Optional[str | int] = None
    website: Optional[str] = None
    question: str
    start_url: str
    expected_answer: str
    final_answer: str
    stagehand_session_id: Optional[str] = None
    matched_expected_answer: bool
    tool_call_count: int


class BrowserbaseWebVoyagerResourcesServer(BrowserbaseDOMResourcesServer):
    config: BrowserbaseWebVoyagerResourcesServerConfig

    async def verify(
        self,
        request: Request,
        body: BrowserbaseWebVoyagerVerifyRequest,
    ) -> BrowserbaseWebVoyagerVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        runtime = self.session_id_to_runtime.get(session_id)
        stagehand_session_id = runtime.stagehand_session_id if runtime is not None else None

        try:
            final_answer = self._extract_final_answer(body.response.output)
            matched_expected_answer = self._matches_expected_answer(
                final_answer=final_answer,
                expected_answer=body.expected_answer,
            )
            tool_call_count = len([item for item in body.response.output if item.type == "function_call"])

            return BrowserbaseWebVoyagerVerifyResponse(
                **body.model_dump(),
                reward=float(matched_expected_answer),
                final_answer=final_answer,
                stagehand_session_id=stagehand_session_id,
                matched_expected_answer=matched_expected_answer,
                tool_call_count=tool_call_count,
            )
        finally:
            await self._close_session(session_id)

    def _extract_final_answer(self, output: list[Any]) -> str:
        for item in reversed(output):
            if isinstance(item, NeMoGymResponseOutputMessage):
                return self._render_message_text(item)
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
                return self._render_message_text(item)
        return ""

    def _render_message_text(self, message: Any) -> str:
        parts: list[str] = []
        for content_part in getattr(message, "content", []):
            text = getattr(content_part, "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts).strip()

    def _matches_expected_answer(self, *, final_answer: str, expected_answer: str) -> bool:
        normalized_final = self._normalize_text(final_answer)
        normalized_expected = self._normalize_text(expected_answer)
        if not normalized_expected:
            return False

        if self.config.expected_answer_match_mode == "exact":
            return normalized_final == normalized_expected
        if self.config.expected_answer_match_mode == "contains":
            return normalized_expected in normalized_final
        raise ValueError(f"Unsupported expected_answer_match_mode: {self.config.expected_answer_match_mode}")

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())


if __name__ == "__main__":
    BrowserbaseWebVoyagerResourcesServer.run_webserver()
