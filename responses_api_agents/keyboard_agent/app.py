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

from fastapi import FastAPI, HTTPException
from pydantic import PrivateAttr

from nemo_gym.config_types import BaseRunServerInstanceConfig
from nemo_gym.multi_agent import AgentActRequest, AgentActResponse
from nemo_gym.server_utils import SimpleServer


class KeyboardAgentConfig(BaseRunServerInstanceConfig):
    agent_id: str
    display_name: str


class KeyboardAgent(SimpleServer):
    config: KeyboardAgentConfig
    _input_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/act")(self.act)
        return app

    async def act(self, body: AgentActRequest) -> AgentActResponse:
        if body.agent_id != self.config.agent_id:
            raise HTTPException(
                status_code=409,
                detail=f"This server controls {self.config.agent_id}, not {body.agent_id}.",
            )
        prompt = (
            f"\n--- {self.config.display_name} ({body.agent_id}) ---\n{body.observation}\n{self.config.display_name}> "
        )
        async with self._input_lock:
            action = await asyncio.to_thread(input, prompt)
        return AgentActResponse(action=action)


if __name__ == "__main__":
    KeyboardAgent.run_webserver()
