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
"""External library wrapper agent.

Intentional bugs:
- Uses httpx directly instead of aiohttp adapter
- Missing Semaphore for concurrent library calls
"""

import httpx
from pydantic import BaseModel
from starlette.requests import Request

from nemo_gym.servers.responses_api_agent import SimpleResponsesAPIAgent


class ExternalLibConfig(BaseModel):
    api_url: str = "http://localhost:9000"
    resources_server: dict = {}
    model_server: dict = {}
    name: str = "external_wrapper"


class ExternalLibraryWrapper(SimpleResponsesAPIAgent):
    config: ExternalLibConfig

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.client = httpx.AsyncClient(base_url=self.config.api_url)

    async def run(self, request: Request, body):
        cookies = request.cookies

        # Model call
        gen_resp = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.model_dump(),
            cookies=cookies,
        )
        cookies = gen_resp.cookies

        model_response = await gen_resp.json()
        output_text = model_response.get("output_text", "")

        # Pre-process: Gym schema to library format
        library_input = {
            "code": output_text,
            "task_id": body.get("verifier_metadata", {}).get("task_id", ""),
            "test_cases": body.get("verifier_metadata", {}).get("test_cases", []),
        }

        # Call external library — no concurrency control
        response = await self.client.post(
            "/evaluate",
            content=str(library_input),
            timeout=60.0,
        )
        result = response.json()

        # Post-process: library output to Gym response
        return {
            "reward": 1.0 if result.get("passed", False) else 0.0,
            "output_text": output_text,
            "response": {"output_text": output_text},
        }
