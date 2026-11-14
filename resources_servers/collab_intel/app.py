# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Collaborative Intelligence verify: the agent (collab_intel_agent) runs the upstream harness and its checkers; this
server turns the result into the reward. reward = the test run's pass (1/0); a row the policy failed alone has
alone_pass=false and reward 0. It is a measured row (not masked), and collab_intel/score leaves it out by filtering on
alone_pass, so mean/reward is the unconditional test pass rate, not the upstream score."""

from typing import Any, Optional

from pydantic import ConfigDict

from nemo_gym.core import FailureClass, classify_failure
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class CollabIntelConfig(BaseResourcesServerConfig):
    pass


class CollabIntelVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    collab_intel: Optional[dict[str, Any]] = None


class CollabIntelVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    alone_pass: bool = False
    test_pass: Optional[bool] = None
    unfinished: bool = False


class CollabIntelResourcesServer(SimpleResourcesServer):
    config: CollabIntelConfig

    async def verify(self, body: CollabIntelVerifyRequest) -> CollabIntelVerifyResponse:
        r = body.collab_intel or {}
        test = r.get("test") or {}
        # the last episode run decides the row. One that failed for the evaluation's sake (a harness crash, a judge
        # failure) is unfinished: test_pass None, left out of the score. A model-caused failure (agent_timeout,
        # model_context_limit) is a measured fail.
        last = test or r.get("alone") or {}
        kind = last.get("failure_kind")
        unfinished = kind is not None and classify_failure(failure_kind=kind) != FailureClass.MODEL
        test_pass = test.get("pass") if r.get("alone_pass") and not unfinished else None
        return CollabIntelVerifyResponse(
            **body.model_dump(),
            reward=float(bool(test_pass)),
            # mask only evaluation-side faults. A gate failure is a scoring rule, carried by alone_pass for the score.
            mask_sample=unfinished,
            alone_pass=bool(r.get("alone_pass")),
            test_pass=test_pass,
            unfinished=unfinished,
            failure_kind=kind,
            failure_reason=last.get("failure_reason"),
        )


if __name__ == "__main__":
    CollabIntelResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CollabIntelResourcesServer.run_webserver()
