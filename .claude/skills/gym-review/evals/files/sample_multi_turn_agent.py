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
"""Sample multi-turn agent with intentional anti-patterns for eval testing."""

from pydantic import BaseModel
from starlette.requests import Request

from nemo_gym.servers.responses_api_agent import SimpleResponsesAPIAgent


class MultiTurnAgentConfig(BaseModel):
    max_turns: int = 3
    resources_server: dict = {}
    model_server: dict = {}
    name: str = "multi_turn_agent"


class MultiTurnAgent(SimpleResponsesAPIAgent):
    config: MultiTurnAgentConfig

    async def run(self, request: Request, body):
        current_input = body.model_dump()

        for turn in range(self.config.max_turns):
            # Model call - not forwarding session state
            gen_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=current_input,
            )

            model_response = await gen_resp.json()
            output_text = model_response.get("output_text", "")

            # Parsing output without stripping think blocks
            if "```" in output_text:
                code = output_text.split("```")[1]
            else:
                code = output_text

            # Verify call - not forwarding session state
            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.get("name", ""),
                url_path="/verify",
                json={"code": code, "verifier_metadata": body.get("verifier_metadata", {})},
            )

            verify_data = await verify_resp.json()
            if verify_data.get("reward", 0.0) == 1.0:
                break

            # Build next turn input (no token ID accumulation)
            current_input = {
                "input": [
                    {
                        "role": "user",
                        "content": f"Your code was wrong. Error: {verify_data.get('errors', '')}. Try again.",
                    }
                ]
            }

        return verify_data
