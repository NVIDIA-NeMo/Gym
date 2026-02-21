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

from asyncio import Semaphore, get_running_loop
from time import time
from typing import Any, Dict, List, Optional, Union

import ray
from lcb_integration.compute_code_generation_metrics import check_correctness_remote
from lcb_integration.extraction_utils import LMStyle, extract_code
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
    unit_test_timeout_secs: int
    debug: bool


# ----------------------------
# Schemas
# ----------------------------


# This is LiveCodeBench format
class UnitTests(BaseModel):
    inputs: List[str]
    outputs: List[str]
    fn_name: Optional[str] = None


class CompCodingRunRequest(BaseRunRequest):
    pass


class CompCodingVerifyRequest(CompCodingRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class CompCodingVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    result: Optional[List[Union[int, bool]]] = None
    metadata: Optional[Dict[str, Any]] = None
    unit_tests_time_taken: Optional[float] = None


# ----------------------------
# Server
# ----------------------------
class CompCodingResourcesServer(SimpleResourcesServer):
    config: CompCodingResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)

    async def verify(self, body: CompCodingVerifyRequest) -> CompCodingVerifyResponse:
        model_out = body.response.output_text
        if not model_out or not model_out.strip():
            # A response existed but had no usable text -> model failure
            return CompCodingVerifyResponse(
                **body.model_dump(),
                reward=0.0,
            )

        tests = UnitTests.model_validate(body.verifier_metadata["unit_tests"])

        # 3) extract code (code fence or raw)
        code = extract_code(model_out, LMStyle.OpenAIChat)
        if not code:
            return CompCodingVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
            )

        # 4) run (no sandbox)
        async with self._semaphore:
            loop = get_running_loop()

            """
            Sample looks like this:
            ```json
            {
                "input_output": "{\"inputs\": [...], ...}",
            }
            ```
            `input_output` looks like this:
            ```json
            {
                "inputs": [
                    "6\n4 13 2 3 2 6",
                    ...
                ],
                "outputs": [
                    "4 30 2 13 2 13",
                    ...
                ],
                "fn_name": null
            }
            ```
            """

            # We can directly measure here since we are inside the semaphore.
            start_time = time()

            task_args = (
                {"input_output": tests.model_dump_json()},  # sample
                code,  # generation
                self.config.unit_test_timeout_secs,  # timeout
                self.config.debug,  # debug
            )

            future = check_correctness_remote.remote(*task_args)
            result, metadata = await loop.run_in_executor(None, ray.get, future)

            unit_tests_time_taken = time() - start_time

        return CompCodingVerifyResponse(
            **body.model_dump(),
            reward=1.0 if all(r == True for r in result) else 0.0,
            extracted_model_output=model_out,
            extracted_model_code=code,
            result=result,
            metadata=metadata,
            unit_tests_time_taken=unit_tests_time_taken,
        )


if __name__ == "__main__":
    CompCodingResourcesServer.run_webserver()
