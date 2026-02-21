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

from typing import Any

from fastapi import FastAPI
from utils import grade_assistant_response

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class CalendarRunRequest(BaseRunRequest):
    exp_cal_state: dict[str, Any]


class CalendarVerifyRequest(CalendarRunRequest, BaseVerifyRequest):
    pass


class CalendarResourcesServerConfig(BaseResourcesServerConfig):
    pass


class CalendarResourcesServer(SimpleResourcesServer):
    config: CalendarResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: CalendarVerifyRequest) -> BaseVerifyResponse:
        # Extract the assistant's text response from the last output item.
        #
        # For reasoning models (e.g., with deepseek_r1 reasoning_parser), the output
        # structure is: [ReasoningItem, MessageItem] where:
        #   - ReasoningItem: has .reasoning attribute (thinking/CoT tokens)
        #   - MessageItem: has .content attribute (actual response text)
        #
        # The last item should be a MessageItem with .content, but if the model
        # hit the token limit while still thinking, the last item will be a
        # ReasoningItem without .content. In that case, we return reward=0.
        assistant_response = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                assistant_response = last_output.content[0].text

        # If no valid response (e.g., model only produced thinking tokens),
        # return zero reward
        if not assistant_response:
            return BaseVerifyResponse(**body.model_dump(), reward=0)

        exp_cal_state = body.exp_cal_state
        try:
            reward, reason = grade_assistant_response(assistant_response, exp_cal_state)
        except Exception:
            reward = 0

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    CalendarResourcesServer.run_webserver()
