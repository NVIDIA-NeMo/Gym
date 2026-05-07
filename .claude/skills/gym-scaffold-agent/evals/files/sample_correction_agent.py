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
"""Multi-turn correction agent — gives model 3 attempts to solve a problem.

Intentional bugs:
- Missing cookie propagation on server_client.post calls
- Missing token ID accumulation across turns
"""

import asyncio

from pydantic import BaseModel
from starlette.requests import Request

from nemo_gym.server_utils import raise_for_status
from nemo_gym.servers.responses_api_agent import SimpleResponsesAPIAgent


class CorrectionAgentConfig(BaseModel):
    max_turns: int = 3
    resources_server: dict = {}
    model_server: dict = {}
    name: str = "correction_agent"


class CorrectionAgent(SimpleResponsesAPIAgent):
    config: CorrectionAgentConfig

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.semaphore = asyncio.Semaphore(32)

    async def run(self, request: Request, body):
        current_input = body.model_dump()

        for turn in range(self.config.max_turns):
            # Model call — not forwarding session state
            gen_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=current_input,
            )
            await raise_for_status(gen_resp)

            model_response = await gen_resp.json()
            output_text = model_response.get("output_text", "")

            # Strip think blocks before extraction
            if "</think>" in output_text:
                output_text = output_text.split("</think>")[-1].strip()

            # Verify — not forwarding session state
            async with self.semaphore:
                verify_resp = await self.server_client.post(
                    server_name=self.config.resources_server.get("name", ""),
                    url_path="/verify",
                    json={
                        "output_text": output_text,
                        "verifier_metadata": body.get("verifier_metadata", {}),
                    },
                )
            await raise_for_status(verify_resp)

            verify_data = await verify_resp.json()

            if verify_data.get("reward", 0.0) == 1.0:
                return verify_data

            # Build error feedback for next turn
            error_msg = verify_data.get("errors", "Incorrect.")
            current_input = {
                "input": current_input.get("input", [])
                + [
                    {"role": "assistant", "content": output_text},
                    {"role": "user", "content": f"That was wrong. {error_msg} Try again."},
                ]
            }

        return verify_data
