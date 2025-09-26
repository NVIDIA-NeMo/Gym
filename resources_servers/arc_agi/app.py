# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from typing import List, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ARCAGIResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ARCAGIRunRequest(BaseRunRequest):
    train: List[dict] = []
    test_input: List[List[int]] = []
    expected_output: List[List[int]] = []
    task_id: Optional[str] = None


class ARCAGIVerifyRequest(ARCAGIRunRequest, BaseVerifyRequest):
    pass


class ARCAGIVerifyResponse(BaseVerifyResponse):
    expected_output: List[List[int]]
    predicted_output: Optional[List[List[int]]] = None
    extraction_successful: bool = False


def _extract_assistant_text(body: BaseVerifyRequest) -> str:
    texts = []
    for output in body.response.output:
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            if isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def _parse_grid(text: str) -> Optional[List[List[int]]]:
    """expects format: \\boxed{[[1,2,3],[4,5,6]]}"""
    boxed_pattern = r'\\boxed\{(\[\s*\[[\d\s,\[\]]+\]\s*\])\}'
    boxed_matches = re.findall(boxed_pattern, text, re.DOTALL)

    if not boxed_matches:
        boxed_matches = re.findall(r'\[\s*\[[\d\s,\[\]]+\]\s*\]', text, re.DOTALL)

    for match in boxed_matches:
        try:
            cleaned = re.sub(r'\s+', '', match)
            grid = json.loads(cleaned)

            if (isinstance(grid, list) and
                all(isinstance(row, list) and all(isinstance(cell, int) for cell in row) for row in grid) and
                len(grid) > 0 and len(grid[0]) > 0):
                return grid
        except (json.JSONDecodeError, (IndexError, TypeError)):
            continue

    return None


class ARCAGIResourcesServer(SimpleResourcesServer):
    config: ARCAGIResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: ARCAGIVerifyRequest) -> ARCAGIVerifyResponse:
        assistant_text = _extract_assistant_text(body)
        predicted_grid = _parse_grid(assistant_text)

        extraction_successful = predicted_grid is not None
        reward = 1.0 if extraction_successful and predicted_grid == body.expected_output else 0.0

        return ARCAGIVerifyResponse(
            **body.model_dump(),
            reward=reward,
            predicted_output=predicted_grid,
            extraction_successful=extraction_successful,
        )


if __name__ == "__main__":
    ARCAGIResourcesServer.run_webserver()