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

"""A foreground keyboard-controlled agent server."""

import asyncio
from time import time
from uuid import uuid4

from fastapi import Body, FastAPI
from pydantic import PrivateAttr

from nemo_gym.config_types import BaseRunServerInstanceConfig
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import SimpleServer


class KeyboardAgentConfig(BaseRunServerInstanceConfig):
    agent_id: str
    display_name: str


class KeyboardAgent(SimpleServer):
    config: KeyboardAgentConfig
    _input_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        return app

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        observation = self._latest_user_text(body)
        prompt = (
            f"\n--- {self.config.display_name} ({self.config.agent_id}) ---\n"
            f"{observation}\n{self.config.display_name}> "
        )
        async with self._input_lock:
            action = await asyncio.to_thread(input, prompt)
        return NeMoGymResponse(
            id=f"keyboard-{uuid4()}",
            created_at=time(),
            model="keyboard-agent",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"message-{uuid4()}",
                    content=[NeMoGymResponseOutputText(annotations=[], text=action, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    @staticmethod
    def _latest_user_text(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
        if isinstance(body.input, str):
            return body.input
        for item in reversed(body.input):
            if getattr(item, "role", None) != "user":
                continue
            content = getattr(item, "content", "")
            if isinstance(content, str):
                return content
            texts = [getattr(part, "text", "") for part in content]
            if text := "\n".join(part for part in texts if part):
                return text
        return ""


if __name__ == "__main__":
    KeyboardAgent.run_webserver()
