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
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

# Import all functions from the tools file
from resources_servers.math_advanced_calculations.math_advanced_calculations_tools import (
    add,
    cos,
    divide,
    log,
    multiply,
    negate,
    pi,
    power,
    return_constant,
    sin,
    subtract,
)


class MultiVerseMathHardResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MultiVerseMathHardRequest(BaseModel):
    a: Optional[float] = None
    b: Optional[float] = None
    radians: Optional[float] = None
    base: Optional[float] = None


class MultiVerseMathHardResponse(BaseModel):
    solution: float


class MultiVerseMathHardVerifyRequest(BaseVerifyRequest):
    ground_truth: list[float] | str
    id: int
    depth: int
    breadth: int


class MultiVerseMathHardVerifyResponse(BaseVerifyResponse):
    pass


class MultiVerseMathHardResourcesServer(SimpleResourcesServer):
    config: MultiVerseMathHardResourcesServerConfig

    _function_map = {
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
        "sin": sin,
        "cos": cos,
        "power": power,
        "log": log,
        "pi": pi,
        "negate": negate,
        "return_constant": return_constant,
    }

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)

        return app

    async def route_to_python_function(self, path: str, body: MultiVerseMathHardRequest) -> MultiVerseMathHardResponse:
        func = self._function_map.get(path)

        if not func:
            raise HTTPException(status_code=404, detail="Function not found")

        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        try:
            result = func(**args)
            return MultiVerseMathHardResponse(solution=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def verify(self, body: MultiVerseMathHardVerifyRequest) -> MultiVerseMathHardVerifyResponse:
        ground_truth = json.loads(body.ground_truth)
        response = body.response.output

        predicted_tool_call_output = []
        for output in response:
            if output.type == "function_call_output":
                # Add try catch block to catch exceptions if there is a math error while calculation.
                try:
                    predicted_tool_call_output.append(float(json.loads(output.output)["solution"]))
                except Exception:
                    predicted_tool_call_output.append(None)

        reward = 1.0
        for gt in ground_truth:
            if gt not in predicted_tool_call_output:
                reward = 0.0
                break

        return MultiVerseMathHardVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    MultiVerseMathHardResourcesServer.run_webserver()
