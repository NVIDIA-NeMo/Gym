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

import io
import re
import sys
from typing import Any, ClassVar, List, Optional, Pattern, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# ----------------------------
# Config
# ----------------------------
class CompCodingResourcesServerConfig(BaseResourcesServerConfig):
    pass


# ----------------------------
# Schemas
# ----------------------------
class UnitTests(BaseModel):
    inputs: List[str]
    outputs: List[str]


class CompCodingRunRequest(BaseRunRequest):
    uuid: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class CompCodingVerifyRequest(CompCodingRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[dict[str, Any]] = None


class CompCodingVerifyResponse(BaseVerifyResponse):
    reason: Optional[str] = None


# ------------ helpers ------------
CODE_BLOCK_RE: ClassVar[Pattern[str]] = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> Optional[str]:
    # We allow two kinds of responses:
    # 1. Code inside a fenced block (```python ... ``` or ``` ... ```)
    # 2. Raw code returned without any fences
    if not text:
        return None
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _parse_unit_tests(ut_dict: dict) -> UnitTests:
    return UnitTests.model_validate(ut_dict)


def _run_code_against_tests(code: str, tests: UnitTests) -> Tuple[bool, str]:
    """
    Executes `code` with in-process exec(), redirecting stdin/stdout per test.
    Assumes dataset pre-processing has already validated test shapes (non-empty,
    equal-length inputs/outputs).
    """
    for i, (test_input, expected_output) in enumerate(zip(tests.inputs, tests.outputs), start=1):
        # capture originals
        orig_stdin, orig_stdout = sys.stdin, sys.stdout
        try:
            sys.stdin = io.StringIO(test_input.replace("\\n", "\n"))
            sys.stdout = io.StringIO()

            exec_globals = {
                "__builtins__": __builtins__,
                "input": input,
                "print": print,
                "range": range,
                "len": len,
                "list": list,
                "int": int,
                "str": str,
                "float": float,
                "bool": bool,
                "enumerate": enumerate,
                "zip": zip,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "all": all,
                "any": any,
                "set": set,
                "dict": dict,
                "tuple": tuple,
                "map": map,
                "filter": filter,
            }

            exec(code, exec_globals)

            actual_output = sys.stdout.getvalue()
            exp = expected_output.replace("\\n", "\n")
            if actual_output.rstrip() != exp.rstrip():
                return (
                    False,
                    f"TEST_CASE_{i}_FAILED: Expected {repr(exp)} got {repr(actual_output)}",
                )
        except SystemExit as e:
            # Handle sys.exit() calls in the executed code
            return False, f"TEST_CASE_{i}_ERROR: Code called sys.exit({e.code})"
        except Exception as e:
            return False, f"TEST_CASE_{i}_ERROR: {e}"
        finally:
            sys.stdin, sys.stdout = orig_stdin, orig_stdout

    return True, f"SUCCESS: All {len(tests.inputs)} test cases passed"


def _extract_text_from_response(response_obj) -> Optional[str]:
    """
    Extract the assistant's output_text string from a NeMoGymResponse-like object:
    response.output -> list of messages
        message.content -> list of blocks with {"type": "output_text", "text": "..."}
    """
    try:
        if not response_obj:
            return None
        output_list = getattr(response_obj, "output", None)
        if not output_list:
            return None
        for msg in output_list:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if not content:
                continue

            # content may be a list of blocks or a bare string (be tolerant)
            if isinstance(content, str) and content.strip():
                return content

            for block in content:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if btype in ("output_text", "text"):
                    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text
        return None
    except Exception:
        return None


# ----------------------------
# Server
# ----------------------------
class CompCodingResourcesServer(SimpleResourcesServer):
    config: CompCodingResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # (optional) a simple health route
        @app.get("/health")
        async def health():
            return {"ok": True, "server": "comp_coding"}

        return app

    # ------------ verifier ------------
    async def verify(self, body: CompCodingVerifyRequest) -> CompCodingVerifyResponse:
        # Enforce a single source of truth for model output: the Responses API object
        response_obj = getattr(body, "response", None)
        if not response_obj:
            # Treat absence of a response as an input/contract error
            raise HTTPException(status_code=422, detail="Missing response")

        model_out = _extract_text_from_response(response_obj)
        if not model_out or not model_out.strip():
            # A response existed but had no usable text -> model failure
            return CompCodingVerifyResponse(**body.model_dump(), reward=0.0, reason="Empty model output")

        # 2) unit tests (must be present & valid BEFORE runtime; otherwise raise)
        if not body.verifier_metadata or "unit_tests" not in body.verifier_metadata:
            raise HTTPException(status_code=422, detail="Missing verifier_metadata.unit_tests")
        try:
            tests = _parse_unit_tests(body.verifier_metadata["unit_tests"])
        except Exception as e:
            # Treat bad inputs as an input error, not a model failure.
            raise HTTPException(status_code=422, detail=f"Invalid unit_tests: {e}")

        # 3) extract code (code fence or raw)
        code = _extract_code(model_out)
        if not code:
            return CompCodingVerifyResponse(**body.model_dump(), reward=0.0, reason="Could not extract code")

        # 4) run (no sandbox)
        ok, msg = _run_code_against_tests(code, tests)

        return CompCodingVerifyResponse(
            **body.model_dump(),
            reward=1.0 if ok else 0.0,
            reason=msg,  # always include the reason message
        )


if __name__ == "__main__":
    CompCodingResourcesServer.run_webserver()
