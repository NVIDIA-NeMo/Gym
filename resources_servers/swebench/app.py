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

from typing import Optional

from fastapi import Request
from swebench.harness.run_evaluation import make_test_spec, run_instance
from swebench.harness.test_spec.test_spec import LATEST

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
        """
        Key requirements:
        1. Extract the model_patch from the input container
            Proposal
                1. Spinup a fresh container (need this anyways for running eval)
                2. pwd in the fresh container (defaults to WORKDIR)
                3. cd into WORKDIR in the input container
                4. extract the patch via git
            Notes
                1. DeepSWE expects the model to commit. That will go in the DeepSWE resources server and not this one.
        2. Docker Client - Make a mock client class here that wraps our sandbox client.
        3. Harnesses like OpenCode open a new terminal rather than reusing the existing one. Grab the environment variables and workdir from the outer terminal first, and then export/cd as appropriate in the new terminal
            Notes
                1. This is a harness-specific thing that the harness will handle across benchmarks.
        4. Interleaved thinking - Verify how is the harness behaving i.e. it has interleaved thinking or not and to force interleaved thinking unconditionally.
            Proposal
                1. For seeing what the harness is doing, use model call capture
                2. For forcing, we can add it in the Responses API model proxy i.e. save all the past requests/responses and populate as necessary.
        5. Restrict number of turns - same as interleaved thinking, we could add in Responses API model proxy
        """

        test_spec = make_test_spec(
            # This accepts a SWEbenchInstance which is identically our body.
            body.model_dump(),
            namespace="swebench",  # Dockerhub namespace
            instance_image_tag=LATEST,
            env_image_tag=LATEST,
        )

        # Res has 2 keys: completed (whether evaluation completed or not), resolved (whether the issue is resolved)
        res = run_instance(
            test_spec=test_spec,
            pred={
                "instance_id": test_spec.instance_id,
                "model_patch": None,  # TODO
            },
            rm_image=False,
            force_rebuild=False,
            client=None,  # TODO
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
