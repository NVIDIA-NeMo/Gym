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
from typing import Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


def _has_tool_call_tag(text: str) -> bool:
    tool_call_tags = ("<tool_call>", "</tool_call>")
    return any(tag in text for tag in tool_call_tags)


class MalformedThinkAndToolCallsVerifierResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MalformedThinkAndToolCallsVerifierRequest(BaseVerifyRequest):
    label: str


class MalformedThinkAndToolCallsVerifierResponse(BaseVerifyResponse):
    label: str
    input_problem_type: str
    response_error_type: Optional[str] = None


class MalformedThinkAndToolCallsVerifierResourcesServer(SimpleResourcesServer):
    config: MalformedThinkAndToolCallsVerifierResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(
        self, body: MalformedThinkAndToolCallsVerifierRequest
    ) -> MalformedThinkAndToolCallsVerifierResponse:
        """
        Rules:
        1) malformed_thinking:
            - message["type"] == "reasoning"
            - "<tool_call>" or "</tool_call>" appears in any summary[i]["text"]
        2) malformed_tool_call:
            - message["type"] == "message"
            - message["role"] == "assistant"
            - "<tool_call>" or "</tool_call>" appears in any content[i]["text"]
        """
        reward = 1.0
        input_problem_type = body.label
        response_error_type = None

        if body.response.incomplete_details is not None:
            reward = 0.0
            response_error_type = "incomplete"
            return MalformedThinkAndToolCallsVerifierResponse(
                **(
                    body.model_dump()
                    | {
                        "input_problem_type": input_problem_type,
                        "response_error_type": response_error_type,
                        "reward": reward,
                    }
                )
            )

        for message in body.response.output:
            if message.type == "reasoning":
                if any(_has_tool_call_tag(s.text) for s in message.summary):
                    reward = 0.0
                    response_error_type = "malformed_thinking"
                    break
            elif message.type == "message" and getattr(message, "role", None) == "assistant":
                for content_item in getattr(message, "content", []):
                    text = getattr(content_item, "text", None)
                    if text and _has_tool_call_tag(text):
                        reward = 0.0
                        response_error_type = "malformed_tool_call"
                        break
                if reward == 0.0:
                    break

        return MalformedThinkAndToolCallsVerifierResponse(
            **(
                body.model_dump()
                | {
                    "input_problem_type": input_problem_type,
                    "response_error_type": response_error_type,
                    "reward": reward,
                }
            )
        )


if __name__ == "__main__":
    MalformedThinkAndToolCallsVerifierResourcesServer.run_webserver()
