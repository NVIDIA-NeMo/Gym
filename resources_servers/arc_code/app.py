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

"""Resources server for ARC-style grid puzzles solved with code.

``verify`` reads the grid in the final ``<answer>`` block of the last assistant message and gives reward 1.0 when it
equals ``expected_output`` cell for cell, else 0.0. It holds no per-task state: the code sandbox lives in the agent.
"""

from typing import List, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.arc_code.grid import normalized_grid, parse_answer


class ArcCodeResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ArcCodeRunRequest(BaseRunRequest):
    task_id: Optional[str] = None
    examples: List[dict] = []
    test_input: List[List[int]] = []
    expected_output: List[List[int]] = []


class ArcCodeVerifyRequest(ArcCodeRunRequest, BaseVerifyRequest):
    pass


class ArcCodeVerifyResponse(BaseVerifyResponse):
    expected_output: List[List[int]]
    predicted_output: Optional[List[List[int]]] = None
    extraction_successful: bool = False


def final_assistant_text(body: BaseVerifyRequest) -> str:
    """Text of the last assistant message in the response (the turn that holds the final answer)."""
    for item in reversed(body.response.output):
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            return content
        return "\n".join(t for t in (getattr(p, "text", None) for p in content or []) if isinstance(t, str))
    return ""


class ArcCodeResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: ArcCodeResourcesServerConfig

    async def verify(self, body: ArcCodeVerifyRequest) -> ArcCodeVerifyResponse:
        predicted = parse_answer(final_assistant_text(body))
        correct = predicted is not None and normalized_grid(predicted) == normalized_grid(body.expected_output)
        return ArcCodeVerifyResponse(
            **body.model_dump(),
            reward=1.0 if correct else 0.0,
            predicted_output=predicted,
            extraction_successful=predicted is not None,
        )


if __name__ == "__main__":
    ArcCodeResourcesServer.run_webserver()
