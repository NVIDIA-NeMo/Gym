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
import re
from typing import Any, Dict, List, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class CircleCountConfig(BaseResourcesServerConfig):
    pass


class CircleCountVerifyRequest(BaseVerifyRequest):
    circles: List[Dict[str, Any]] = []
    target_color: str = ""


class CircleCountVerifyResponse(BaseVerifyResponse):
    predicted_count: Optional[int] = None
    expected_count: int = 0
    correct: bool = False


class CircleCountResourcesServer(SimpleResourcesServer):
    config: CircleCountConfig

    async def verify(self, body: CircleCountVerifyRequest) -> CircleCountVerifyResponse:
        output_text = body.response.output_text or ""
        match = re.search(r"\\boxed\{(\d+)\}", output_text)
        predicted_count = int(match.group(1)) if match else None

        expected_count = sum(1 for c in body.circles if c["color"] == body.target_color)
        correct = predicted_count is not None and predicted_count == expected_count
        reward = 1.0 if correct else 0.0

        return CircleCountVerifyResponse(
            **body.model_dump(),
            reward=reward,
            predicted_count=predicted_count,
            expected_count=expected_count,
            correct=correct,
        )


if __name__ == "__main__":
    CircleCountResourcesServer.run_webserver()
