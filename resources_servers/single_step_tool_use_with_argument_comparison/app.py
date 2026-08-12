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
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import extract_action
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ActionComparator,
    ExpectedAction,
    StepRewardCategory,
    ToolCallComparatorConfig,
)


class SingleStepToolUseArgumentComparisonResourcesServerConfig(BaseResourcesServerConfig):
    tool_call_comparator_config: ToolCallComparatorConfig


class SingleStepToolUseArgumentComparisonRunRequest(BaseRunRequest):
    expected_action: ExpectedAction


class SingleStepToolUseArgumentComparisonVerifyRequest(
    SingleStepToolUseArgumentComparisonRunRequest, BaseVerifyRequest
):
    pass


class SingleStepToolUseArgumentComparisonVerifyResponse(BaseVerifyResponse):
    expected_action: ExpectedAction
    category: StepRewardCategory


class SingleStepToolUseArgumentComparisonResourcesServer(SimpleResourcesServer):
    config: SingleStepToolUseArgumentComparisonResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(
        self, body: SingleStepToolUseArgumentComparisonVerifyRequest
    ) -> SingleStepToolUseArgumentComparisonVerifyResponse:
        actual_action = extract_action(body.response)
        if actual_action is None:
            return SingleStepToolUseArgumentComparisonVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                category=StepRewardCategory.NO_ACTION_FOUND,
            )

        action_comparator = ActionComparator(config=self.config.tool_call_comparator_config)
        result = action_comparator.compare_action(body.expected_action, actual_action)

        return SingleStepToolUseArgumentComparisonVerifyResponse(
            **body.model_dump(),
            reward=result.reward,
            category=result.category,
        )


if __name__ == "__main__":
    SingleStepToolUseArgumentComparisonResourcesServer.run_webserver()
