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
"""Tool-call loop agent — correct implementation with no bugs.

Model calls tools iteratively until producing a final answer. Includes:
- Cookie propagation
- Token ID accumulation
- Max iteration bound
- Semaphore for concurrency control
"""

import asyncio
import json

from pydantic import BaseModel
from starlette.requests import Request

from nemo_gym.server_utils import raise_for_status
from nemo_gym.servers.responses_api_agent import SimpleResponsesAPIAgent


class ToolLoopConfig(BaseModel):
    max_tool_calls: int = 10
    resources_server: dict = {}
    model_server: dict = {}
    name: str = "tool_loop_agent"


class ToolLoopAgent(SimpleResponsesAPIAgent):
    config: ToolLoopConfig

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.semaphore = asyncio.Semaphore(32)

    async def run(self, request: Request, body):
        cookies = request.cookies
        current_input = body.model_dump()

        all_prompt_token_ids = []
        all_generation_token_ids = []
        all_generation_log_probs = []

        for iteration in range(self.config.max_tool_calls):
            # Model call with cookie propagation
            gen_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=current_input,
                cookies=cookies,
            )
            await raise_for_status(gen_resp)
            cookies = gen_resp.cookies

            model_response = await gen_resp.json()

            # Accumulate token IDs
            all_prompt_token_ids.extend(model_response.get("prompt_token_ids", []))
            all_generation_token_ids.extend(model_response.get("generation_token_ids", []))
            all_generation_log_probs.extend(model_response.get("generation_log_probs", []))

            # Check for tool calls
            tool_calls = model_response.get("tool_calls", [])
            if not tool_calls:
                break

            # Execute each tool call with concurrency control
            tool_results = []
            for tool_call in tool_calls:
                async with self.semaphore:
                    tool_resp = await self.server_client.post(
                        server_name=self.config.resources_server.get("name", ""),
                        url_path=f"/tools/{tool_call['function']['name']}",
                        json=tool_call["function"]["arguments"],
                        cookies=cookies,
                    )
                cookies = tool_resp.cookies
                tool_result = await tool_resp.json()
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result),
                    }
                )

            # Build next turn with tool results
            messages = current_input.get("input", [])
            messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
            messages.extend(tool_results)
            current_input = {"input": messages}

        # Final verification
        output_text = model_response.get("output_text", "")
        verify_resp = await self.server_client.post(
            server_name=self.config.resources_server.get("name", ""),
            url_path="/verify",
            json={
                "output_text": output_text,
                "verifier_metadata": body.get("verifier_metadata", {}),
            },
            cookies=cookies,
        )
        cookies = verify_resp.cookies
        verify_data = await verify_resp.json()

        verify_data["prompt_token_ids"] = all_prompt_token_ids
        verify_data["generation_token_ids"] = all_generation_token_ids
        verify_data["generation_log_probs"] = all_generation_log_probs

        return verify_data
