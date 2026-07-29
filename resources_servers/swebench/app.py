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

from dataclasses import fields
from typing import Optional

from fastapi import Request
from swebench.harness.run_evaluation import TestSpec, run_instance

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


class SwebenchResourcesServerConfig(BaseResourcesServerConfig):
    evaluation_timeout: Optional[int] = None


class SWEBenchVerifyRequest(BaseVerifyRequest):
    # See https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
    # See swebench.harness.run_evaluation.TestSpec https://github.com/SWE-bench/SWE-bench/blob/f7bbbb2ccdf479001d6467c9e34af59e44a840f9/swebench/harness/test_spec/test_spec.py#L28
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    # These are JSON strings.
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str
    difficulty: str
    subset: str
    split: str


class SWEBenchVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool


class SwebenchResourcesServer(SimpleResourcesServer):
    config: SwebenchResourcesServerConfig

    async def verify(self, request: Request, body: SWEBenchVerifyRequest) -> SWEBenchVerifyResponse:
        # TODO: model_patch, client
        test_spec_field_names = set(f.name for f in fields(TestSpec))
        test_spec = TestSpec(**{k: v for k, v in body.model_dump() if k in test_spec_field_names})

        # Res has 2 keys: completed (whether evaluation completed or not), resolved (whether the issue is resolved)
        res = run_instance(
            test_spec=test_spec,
            pred={"instance_id": test_spec.instance_id, "model_patch": None},
            rm_image=False,
            force_rebuild=False,
            client=None,
            run_id=request.session[SESSION_ID_KEY],
            timeout=self.config.evaluation_timeout,
            rewrite_reports=False,
        )
        return SWEBenchVerifyResponse(
            **body.model_dump(),
            **res,
            reward=int(res["resolved"]),
        )


if __name__ == "__main__":
    SwebenchResourcesServer.run_webserver()
