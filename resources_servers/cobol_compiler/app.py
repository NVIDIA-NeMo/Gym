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

"""COBOL compiler resource server.

Compiles COBOL code with GnuCOBOL and runs stdin/stdout test cases.
Follows the same pattern as resources_servers/code_gen/app.py.
"""

import logging
import os
from asyncio import Semaphore
from time import time
from typing import Any, Dict, List, Optional

from cobol_utils import compile_and_test_remote, extract_cobol_code
from setup_cobc import ensure_cobc

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


LOG = logging.getLogger(__name__)


# ----------------------------
# Config
# ----------------------------
class CobolCompilerConfig(BaseResourcesServerConfig):
    num_processes: int = 8
    timeout_secs: int = 30
    compiler_command: str = "cobc"
    compiler_flags: List[str] = ["-x", "-free", "-fdiagnostics-plain-output"]
    debug: bool = False
    soft_match: bool = False


# ----------------------------
# Schemas
# ----------------------------
class CobolRunRequest(BaseRunRequest):
    pass


class CobolVerifyRequest(CobolRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class CobolVerifyResponse(BaseVerifyResponse):
    extracted_code: Optional[str] = None
    compilation_success: Optional[bool] = None
    compilation_errors: Optional[List[str]] = None
    compilation_warnings: Optional[List[str]] = None
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None
    test_results: Optional[List[Dict[str, Any]]] = None
    compilation_time: Optional[float] = None


# ----------------------------
# Server
# ----------------------------
class CobolCompilerServer(SimpleResourcesServer):
    config: CobolCompilerConfig

    def model_post_init(self, context):
        ensure_cobc()
        self._ray_env_vars = {
            "PATH": os.environ.get("PATH", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        }
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)

    async def verify(self, body: CobolVerifyRequest) -> CobolVerifyResponse:
        model_out = body.response.output_text
        if not model_out or not model_out.strip():
            return CobolVerifyResponse(
                **body.model_dump(),
                reward=0.0,
            )

        test_cases = (body.verifier_metadata or {}).get("test_cases", [])
        if not test_cases:
            LOG.warning("No test cases in verifier_metadata")
            return CobolVerifyResponse(
                **body.model_dump(),
                reward=0.0,
            )

        # Extract COBOL code from model output
        code = extract_cobol_code(model_out)
        if not code:
            return CobolVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_code=None,
            )

        # Compile and run tests
        async with self._semaphore:
            start_time = time()

            future = compile_and_test_remote.options(
                runtime_env={"env_vars": self._ray_env_vars},
            ).remote(
                code,
                test_cases,
                self.config.compiler_command,
                self.config.compiler_flags,
                self.config.timeout_secs,
                self.config.debug,
                self.config.soft_match,
            )
            result = await future

            compilation_time = time() - start_time

        return CobolVerifyResponse(
            **body.model_dump(),
            reward=1.0 if result["all_passed"] else 0.0,
            extracted_code=code,
            compilation_success=result["compilation_success"],
            compilation_errors=result["compilation_errors"],
            compilation_warnings=result.get("compilation_warnings", []),
            tests_passed=result["tests_passed"],
            tests_total=result["tests_total"],
            test_results=result["test_results"],
            compilation_time=compilation_time,
        )


if __name__ == "__main__":
    CobolCompilerServer.run_webserver()
