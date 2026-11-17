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
import ast
import json
import logging
import math
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

logger = logging.getLogger(__name__)


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
    id: int
    depth: int
    breadth: int
    # [CUSTOM-FRESH-GROUND-TRUTH 2026-11-06] REQUIRED: replaces the old pre-rounded
    # `ground_truth` (list[float] | str, derived from the upstream dataset's own
    # `simplified_values`) entirely -- see the comment block above _ALLOWED_FUNCTIONS below
    # for the full rationale. There is exactly one grading path now; data that doesn't carry
    # this field will fail request validation rather than silently falling back to the old,
    # known-incorrect-for-depth>=2 comparison.
    ground_truth_expression: str


class MultiVerseMathHardVerifyResponse(BaseVerifyResponse):
    pass


# [CUSTOM-FRESH-GROUND-TRUTH 2026-11-06] The upstream HF dataset
# (nvidia/Nemotron-RL-math-advanced_calculations) stores pre-computed ground truth
# (`simplified_values`) as round(true_value, N) per value (N varies 0-5 decimals), AND for
# multi-step rows the rounding COMPOUNDS: later steps' literal arguments reuse the
# PREVIOUSLY-ROUNDED result of an earlier step, not its true value (confirmed directly on
# real data: the dataset's own trace for sin(log(7.0,4.0)) literally computes
# sin(radians=1.14), using the rounded 1.14, not the true 1.1414649225389244). A prior fix
# attempted rounding the *predicted* value to each ground-truth value's own implied
# precision before comparing -- this fully fixed depth-1 (single-step) problems, but could
# not fix depth>=2: the dataset's own rounded-chain value can diverge from a model's
# genuinely-correct, full-precision tool-call chain by far more than one rounding digit, and
# this divergence compounds with depth. Confirmed empirically on 500-row/depth samples: no
# global or per-depth tolerance can separate "correct but diverged from the rounded-chain
# ground truth" from "actually wrong" starting at depth 2 -- the error distributions
# overlap (e.g. depth 2's 90th-percentile correct-but-diverged error already exceeds the
# 10th-percentile genuinely-wrong-answer error).
#
# The only real fix: stop trusting the upstream rounded/compounded ground truth entirely
# and recompute it fresh, in true float64 precision, from the raw nested expression (e.g.
# "sin(radians=log(a=7.0, base=4.0))", optionally ";"-separated for multi-answer rows) --
# this is the SAME expression format as the upstream HF dataset's own `ground_truth` column
# (not `simplified_values`/`simplifications`, which are exactly the corrupted columns
# above). Ground-truth expressions only ever call this fixed, known set of 11 functions
# with numeric-literal keyword arguments, so this is evaluated via a restricted AST walker
# (not raw eval()) that rejects anything outside that shape -- cheap to do properly, and
# this is externally-sourced data, not trusted Python.
#
# Comparison uses a tight float tolerance (not exact ==) because this is still floating-
# point arithmetic: the model's own tool-call sequence and this fresh recomputation can
# differ by last-ULP noise depending on operation order. rel_tol=abs_tol=1e-9 is many
# orders of magnitude tighter than any real wrong answer ever gets to the correct value
# (empirically, the CLOSEST synthetic wrong-answer variant found across thousands of
# sampled rows this session never landed closer than ~1.6% relative error) while
# comfortably absorbing float64 noise (~1e-15 per op, negligible even compounded over the
# deepest observed 12-call chain).
_ALLOWED_FUNCTIONS = frozenset(
    {"multiply", "divide", "add", "subtract", "power", "log", "sin", "cos", "pi", "negate", "return_constant"}
)
_GROUND_TRUTH_RTOL = 1e-9
_GROUND_TRUTH_ATOL = 1e-9


class GroundTruthExpressionError(ValueError):
    """Raised when a ground_truth_expression contains anything outside the restricted shape."""


def _safe_eval_expression(expr: str, function_map: dict) -> float:
    """Evaluate a single ground-truth expression via a restricted AST walker.

    Only permits: calls to names in _ALLOWED_FUNCTIONS, numeric (int/float) constant
    arguments, and keyword arguments -- matching the exact shape of every real
    ground_truth expression observed in the upstream dataset. Anything else (arbitrary
    names, operators, attribute access, non-numeric literals, positional args) raises
    GroundTruthExpressionError rather than being evaluated.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise GroundTruthExpressionError(f"Could not parse expression {expr!r}: {e}") from e

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise GroundTruthExpressionError(f"Disallowed function call in {expr!r}: {ast.dump(node)}")
            if node.args:
                raise GroundTruthExpressionError(f"Positional args not allowed in {expr!r}: {ast.dump(node)}")
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise GroundTruthExpressionError(f"**kwargs not allowed in {expr!r}")
                kwargs[kw.arg] = _eval_node(kw.value)
            func = function_map[node.func.id]
            return float(func(**kwargs))
        raise GroundTruthExpressionError(f"Disallowed node in {expr!r}: {ast.dump(node)}")

    return _eval_node(tree)


def _eval_ground_truth_expressions(expression: str, function_map: dict) -> list[float]:
    """Split a (possibly ";"-joined) ground_truth_expression and evaluate each sub-expression."""
    return [_safe_eval_expression(sub, function_map) for sub in expression.split(";") if sub.strip()]


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
        except (TypeError, ValueError, ArithmeticError) as e:
            # [CUSTOM-MATHCALC-ERRORHANDLING 2026-10-24] These are client-input errors, not
            # server bugs, for example: 
            # - TypeError when the model omits a required arg (e.g. calling /log with 
            #   only "a", no "base")
            # - ValueError/ArithmeticError (covers ZeroDivisionError, logs of negative 
            #   numbers, etc.)
            # Previously these all fell through the bare `except Exception` below and
            # came back as 500s, which is indistinguishable from an actual server fault.
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def verify(self, body: MultiVerseMathHardVerifyRequest) -> MultiVerseMathHardVerifyResponse:
        response = body.response.output

        predicted_tool_call_output = []
        for output in response:
            if output.type == "function_call_output":
                # Add try catch block to catch exceptions if there is a math error while calculation.
                try:
                    predicted_tool_call_output.append(float(json.loads(output.output)["solution"]))
                except Exception:
                    predicted_tool_call_output.append(None)

        # [CUSTOM-FRESH-GROUND-TRUTH 2026-11-06] Recompute true ground truth fresh from the
        # raw expression -- see the comment above _ALLOWED_FUNCTIONS for the full rationale.
        #
        # [CUSTOM-MATHCALC-GT-ERRORHANDLING 2026-11-09] This used to let TypeError/ValueError/
        # ArithmeticError/GroundTruthExpressionError propagate uncaught out of verify() on the
        # (rare, ~0.9% of sampled rows) ground_truth_expressions whose altered-tool evaluation
        # is itself undefined -- e.g. power()'s a ** (b + 2) going complex for a negative base
        # with a non-integer exponent (TypeError on the float() cast), or log()'s
        # math.log(a, abs(base + 1.5)) raising ValueError/OverflowError for a <= 0 or an
        # out-of-range result. Confirmed directly: this surfaces as a 500 from /verify, and
        # async_grpo_train's periodic validation pass has no try/except around it, so a single
        # such row took down an entire 65-node MOPD run (job 63861875). Unlike a malformed/
        # unparseable *expression* (a real data bug, still raised via GroundTruthExpressionError
        # from a parse failure), these are legitimate numeric domain errors in otherwise
        # well-formed expressions -- same class of error route_to_python_function already
        # treats as a client-input problem (line ~199), not a server fault. Mirror that here:
        # log the row id and score reward=0 instead of crashing the whole run.
        try:
            ground_truth_values = _eval_ground_truth_expressions(body.ground_truth_expression, self._function_map)
        except (TypeError, ValueError, ArithmeticError, GroundTruthExpressionError) as e:
            logger.error(
                f"math_advanced_calculations.verify() failed to evaluate ground_truth_expression "
                f"for id={body.id!r}: {type(e).__name__}: {e}. Scoring reward=0."
            )
            return MultiVerseMathHardVerifyResponse(**body.model_dump(), reward=0.0)

        reward = 1.0
        for gt in ground_truth_values:
            if not any(
                p is not None and math.isclose(p, gt, rel_tol=_GROUND_TRUTH_RTOL, abs_tol=_GROUND_TRUTH_ATOL)
                for p in predicted_tool_call_output
            ):
                reward = 0.0
                break

        return MultiVerseMathHardVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    MultiVerseMathHardResourcesServer.run_webserver()
