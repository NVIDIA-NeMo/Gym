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
import logging
import re

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import raise_for_status, request


LOG = logging.getLogger(__name__)


class CritPtResourcesServerConfig(BaseResourcesServerConfig):
    api_url: str = "https://artificialanalysis.ai/api/v2/critpt/evaluate"
    api_key: str


class CritPtRunRequest(BaseRunRequest):
    problem_id: str


class CritPtVerifyRequest(CritPtRunRequest, BaseVerifyRequest):
    pass


class CritPtVerifyResponse(BaseVerifyResponse):
    problem_id: str
    accuracy: float
    timeout_rate: float


class CritPtResourcesServer(SimpleResourcesServer):
    config: CritPtResourcesServerConfig

    async def verify(self, body: CritPtVerifyRequest) -> CritPtVerifyResponse:
        output_text = _extract_output_text(body)
        code = _extract_code(output_text)

        if not code:
            return CritPtVerifyResponse(**body.model_dump(), reward=0.0, accuracy=0.0, timeout_rate=0.0)

        submission = {
            "problem_id": body.problem_id,
            "generated_code": f"```python\n{code}\n```",
            "model": "unknown",
            "generation_config": {},
        }

        result = await _call_api(self.config.api_url, self.config.api_key, [submission])
        accuracy = result["accuracy"]
        timeout_rate = result.get("timeout_rate", 0.0)
        reward = 1.0 if accuracy >= 1.0 else 0.0

        return CritPtVerifyResponse(
            **body.model_dump(),
            reward=reward,
            accuracy=accuracy,
            timeout_rate=timeout_rate,
        )


def _extract_output_text(body: CritPtVerifyRequest) -> str:
    parts = []
    for output_item in body.response.output:
        if output_item.type != "message":
            continue
        for content_item in output_item.content:
            if content_item.type != "output_text":
                continue
            parts.append(content_item.text)
    return "".join(parts)


def _extract_code(text: str) -> str:
    """Extract Python code from model output. Matches nemo-skills _extract_code_from_generation logic."""
    matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


async def _call_api(api_url: str, api_key: str, submissions: list[dict]) -> dict:
    payload = {
        "submissions": submissions,
        "batch_metadata": {},
    }
    response = await request(
        method="POST",
        url=api_url,
        json=payload,
        headers={"x-api-key": api_key},
    )
    await raise_for_status(response)
    return await response.json()


if __name__ == "__main__":
    CritPtResourcesServer.run_webserver()
