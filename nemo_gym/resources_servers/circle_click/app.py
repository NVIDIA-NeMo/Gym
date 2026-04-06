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
import json
import math
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class CircleClickConfig(BaseResourcesServerConfig):
    pass


class ClickRequest(BaseModel):
    x: Any
    y: Any


class ClickResponse(BaseModel):
    x: Any
    y: Any


class CircleClickVerifyRequest(BaseVerifyRequest):
    circles: List[Dict[str, Any]] = Field(default_factory=list)
    target_color: str = ""


class CircleClickVerifyResponse(BaseVerifyResponse):
    clicked_x: Optional[int] = None
    clicked_y: Optional[int] = None
    hit: bool = False


class CircleClickResourcesServer(SimpleResourcesServer):
    config: CircleClickConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/click")(self.click)
        return app

    async def click(self, body: ClickRequest) -> ClickResponse:
        return ClickResponse(x=body.x, y=body.y)

    @staticmethod
    def _point_in_circle(px: int, py: int, circle: Dict[str, Any]) -> bool:
        dx = px - circle["x"]
        dy = py - circle["y"]
        return math.sqrt(dx * dx + dy * dy) <= circle["radius"]

    async def verify(self, body: CircleClickVerifyRequest) -> CircleClickVerifyResponse:
        clicked_x = None
        clicked_y = None

        for output_item in body.response.output:
            if output_item.type == "function_call" and output_item.name == "click":
                try:
                    args = json.loads(output_item.arguments)
                    clicked_x = int(args["x"])
                    clicked_y = int(args["y"])
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    pass
                break

        hit = False
        reward = 0.0
        if clicked_x is not None and clicked_y is not None:
            for circle in body.circles:
                if circle["color"] == body.target_color:
                    if self._point_in_circle(clicked_x, clicked_y, circle):
                        hit = True
                        reward = 1.0
                    break

        return CircleClickVerifyResponse(
            **body.model_dump(),
            reward=reward,
            clicked_x=clicked_x,
            clicked_y=clicked_y,
            hit=hit,
        )


if __name__ == "__main__":
    CircleClickResourcesServer.run_webserver()
