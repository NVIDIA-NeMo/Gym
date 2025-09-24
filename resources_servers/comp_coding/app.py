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

import sys
from asyncio import Semaphore, sleep
from contextlib import asynccontextmanager, redirect_stdout
from io import StringIO
from multiprocessing.pool import Pool
from time import time
from traceback import print_exc
from typing import Any, List, Optional

from fastapi import FastAPI
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
    num_processes: int
    unit_test_timeout_secs: float


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
    reason: str
    extracted_model_output: Optional[str]
    extracted_model_code: Optional[str]
    tests_time_taken: Optional[float]
    stdout: Optional[str]


# ------------ helpers ------------


def _extract_code(text: str) -> Optional[str]:
    # We allow two kinds of responses:
    # 1. Code inside a fenced block (```python ... ``` or ``` ... ```)
    # 2. Raw code returned without any fences
    if not text or "```" not in text:
        return text

    start_code_idx = text.rfind("```python\n")
    if start_code_idx == -1:
        return text
    start_code_idx += len("```python\n")

    end_code_idx = text.find("```", start_code_idx)
    if end_code_idx == -1:
        return text[start_code_idx:]

    return text[start_code_idx:end_code_idx]


class TestResult(BaseModel):
    ok: bool
    message: str
    tests_time_taken: float
    stdout: Optional[str]


def _run_code_against_tests(code: str, tests: UnitTests) -> TestResult:
    """
    Executes `code` with in-process exec(), redirecting stdin/stdout per test.
    Assumes dataset pre-processing has already validated test shapes (non-empty,
    equal-length inputs/outputs).
    """
    start_time = time()

    for i, (test_input, expected_output) in enumerate(zip(tests.inputs, tests.outputs), start=1):
        exec_globals = {
            "__builtins__": __builtins__,
        }

        orig_stdin = sys.stdin
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                # This is ONLY safe because we run this in a separate process.
                sys.stdin = StringIO(test_input)

                try:
                    exec(code, exec_globals)
                except Exception as e:
                    return TestResult(
                        ok=False,
                        message=f"TEST_CASE_{i}_ERROR: {e}",
                        tests_time_taken=time() - start_time,
                        stdout=captured_output.getvalue(),
                    )
                except:
                    # Handle Non-exception-based calls in the executed code
                    return TestResult(
                        ok=False,
                        message=f"TEST_CASE_{i}_ERROR: {print_exc()}",
                        tests_time_taken=time() - start_time,
                        stdout=captured_output.getvalue(),
                    )
                finally:
                    sys.stdin = orig_stdin

                actual_output = captured_output.getvalue()

        if actual_output.rstrip() != expected_output.rstrip():
            return TestResult(
                ok=False,
                message=f"TEST_CASE_{i}_FAILED: Expected {repr(expected_output)} got {repr(actual_output)}",
                tests_time_taken=time() - start_time,
                stdout=actual_output,
            )

    return TestResult(
        ok=True,
        message=f"SUCCESS: All {len(tests.inputs)} test cases passed",
        tests_time_taken=time() - start_time,
        stdout=actual_output,  # Just the last one is fine.
    )


# ----------------------------
# Server
# ----------------------------
class CompCodingResourcesServer(SimpleResourcesServer):
    config: CompCodingResourcesServerConfig

    def model_post_init(self, context):
        self._pool: Optional[Pool] = None
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)

    def setup_webserver(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            with Pool(self.config.num_processes) as pool:
                self._pool = pool
                yield

            print(f"Finished {self.__class__.__name__} {self.config.name} and shut down pool!")

        app = FastAPI(lifespan=lifespan)

        app.post("/seed_session")(self.seed_session)
        app.post("/verify")(self.verify)

        return app

    async def verify(self, body: CompCodingVerifyRequest) -> CompCodingVerifyResponse:
        model_out = body.response.output_text
        if not model_out or not model_out.strip():
            # A response existed but had no usable text -> model failure
            return CompCodingVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                reason="Empty model output",
                extracted_model_code=None,
                extracted_model_output=None,
                tests_time_taken=None,
            )

        tests = UnitTests.model_validate(body.verifier_metadata["unit_tests"])

        # 3) extract code (code fence or raw)
        code = _extract_code(model_out)
        if not code:
            return CompCodingVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                reason="Could not extract code",
                extracted_model_output=model_out,
                extracted_model_code=None,
                tests_time_taken=None,
            )

        # 4) run (no sandbox)
        # Use a semaphore here to guarantee that we are actually running this actively during the timeout.
        async with self._semaphore:
            result = self._pool.apply_async(_run_code_against_tests, (code, tests))
            start_time = time()
            await sleep(self.config.unit_test_timeout_secs)

            try:
                test_result = result.get()
            except:
                print_exc()
                print(
                    f"Comp coding verifier {self.config.name} hit an exception while retrieving unit test result. The traceback is shown above."
                )

                test_result = TestResult(
                    ok=False,
                    message="",
                    tests_time_taken=time() - start_time,
                    stdout=None,
                )

        return CompCodingVerifyResponse(
            **body.model_dump(),
            reward=1.0 if test_result.ok else 0.0,
            reason=test_result.message,
            extracted_model_output=model_out,
            extracted_model_code=code,
            tests_time_taken=test_result.tests_time_taken,
            stdout=test_result.stdout,
        )


if __name__ == "__main__":
    CompCodingResourcesServer.run_webserver()
