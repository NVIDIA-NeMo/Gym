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
from typing import Callable, Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
    gym_tool,
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

    @gym_tool
    async def add(self, a: Optional[float] = None, b: Optional[float] = None) -> MultiVerseMathHardResponse:
        """Add two numbers; a + b."""
        return self._solve(add, a=a, b=b)

    @gym_tool
    async def subtract(self, a: Optional[float] = None, b: Optional[float] = None) -> MultiVerseMathHardResponse:
        """Subtract two numbers; a - b."""
        return self._solve(subtract, a=a, b=b)

    @gym_tool
    async def multiply(self, a: Optional[float] = None, b: Optional[float] = None) -> MultiVerseMathHardResponse:
        """Multiply two numbers; a * b."""
        return self._solve(multiply, a=a, b=b)

    @gym_tool
    async def divide(self, a: Optional[float] = None, b: Optional[float] = None) -> MultiVerseMathHardResponse:
        """Divide two numbers; a / b."""
        return self._solve(divide, a=a, b=b)

    @gym_tool
    async def sin(self, radians: Optional[float] = None) -> MultiVerseMathHardResponse:
        """The sine of an angle in radians."""
        return self._solve(sin, radians=radians)

    @gym_tool
    async def cos(self, radians: Optional[float] = None) -> MultiVerseMathHardResponse:
        """The cosine of an angle in radians."""
        return self._solve(cos, radians=radians)

    @gym_tool
    async def power(self, a: Optional[float] = None, b: Optional[float] = None) -> MultiVerseMathHardResponse:
        """Raise a number to a power; a ** b."""
        return self._solve(power, a=a, b=b)

    @gym_tool
    async def log(self, a: Optional[float] = None, base: Optional[float] = None) -> MultiVerseMathHardResponse:
        """Take the log of a number; log(a, base)."""
        return self._solve(log, a=a, base=base)

    @gym_tool
    async def pi(self) -> MultiVerseMathHardResponse:
        """Returns a precise value of PI for this alternate universe."""
        return self._solve(pi)

    @gym_tool
    async def negate(self, a: Optional[float] = None) -> MultiVerseMathHardResponse:
        """Negate a number; -a."""
        return self._solve(negate, a=a)

    @gym_tool
    async def return_constant(self, a: Optional[float] = None) -> MultiVerseMathHardResponse:
        """Return a constant number: a with no modifications"""
        return self._solve(return_constant, a=a)

    def _solve(self, func: Callable[..., float], **arguments: Optional[float]) -> MultiVerseMathHardResponse:
        # The pre-migration dispatcher contract, verbatim: null (or omitted) arguments are silently
        # dropped before the call, and any failure — missing arguments, math domain errors — becomes
        # a 500 whose detail is the bare exception string.
        args = {key: value for key, value in arguments.items() if value is not None}

        try:
            result = func(**args)
            return MultiVerseMathHardResponse(solution=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def handle_unknown_tool(self, tool_name: str, request: Request) -> None:
        # Preserve the pre-migration catch-all bytes: 404 {"detail": "Function not found"}.
        raise HTTPException(status_code=404, detail="Function not found")

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
