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
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse

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


# Final answers are often rounded, so a stated value matches when it agrees with the expected value to about
# three significant figures.
ANSWER_REL_TOL = 5e-3
ANSWER_ABS_TOL = 1e-6
# Plain, signed, comma-grouped (1,073,741,824.0) and scientific (1.2e-05) numbers.
NUMBER_PATTERN = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
LATEX_DELIMITER_PATTERN = re.compile(r"\\[()\[\]]")
LIST_MARKER_PATTERN = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.MULTILINE)
PARENTHESIZED_PATTERN = re.compile(r"\([^()]*\)")
# A reply states one answer per line, list item or clause, after restating the expression it answers.
CLAUSE_SEPARATOR_PATTERN = re.compile(r"\n|;|,\s|\.\s|\sand\s")


def extract_final_answer_values(response: NeMoGymResponse, count: int) -> list[float]:
    """Return the last ``count`` answers stated in the last assistant message, in order.

    The answer of each line, list item or clause is its last number, after dropping anything in parentheses
    (restated expressions such as "sin(5.0)" and remarks such as "(rounded)"). Taking the last ``count`` answers
    reads the reply in its last stated form, so values it quotes and then replaces do not count. Returns an empty
    list if there is no assistant message.
    """
    for output in reversed(response.output):
        if output.type == "message" and getattr(output, "role", None) == "assistant":
            text = "\n".join(part.text for part in output.content if part.type == "output_text")
            text = LATEX_DELIMITER_PATTERN.sub(" ", text.replace("\u2212", "-"))
            text = LIST_MARKER_PATTERN.sub("", text)
            previous = None
            while previous != text:
                previous, text = text, PARENTHESIZED_PATTERN.sub(" ", text)
            answers = []
            for clause in CLAUSE_SEPARATOR_PATTERN.split(text):
                numbers = NUMBER_PATTERN.findall(clause)
                if numbers:
                    answers.append(float(numbers[-1].replace(",", "")))
            return answers[-count:] if count else []
    return []


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

        # Each expected value must come from a tool call, and the final answer must state the values in order.
        answer_values = extract_final_answer_values(body.response, len(ground_truth))
        answer_matches = len(answer_values) == len(ground_truth) and all(
            math.isclose(value, gt, rel_tol=ANSWER_REL_TOL, abs_tol=ANSWER_ABS_TOL)
            for value, gt in zip(answer_values, ground_truth)
        )

        reward = 1.0 if answer_matches else 0.0
        for gt in ground_truth:
            if gt not in predicted_tool_call_output:
                reward = 0.0
                break

        return MultiVerseMathHardVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    MultiVerseMathHardResourcesServer.run_webserver()
