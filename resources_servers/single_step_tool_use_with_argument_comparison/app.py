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
from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import (
    extract_tool_call_or_text,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedAction,
    ListF1MatchDetail,
    StepRewardCategory,
    ToolCallComparator,
    ToolCallComparatorConfig,
    validate_tool_call_against_declared_schema,
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
    list_f1_match_details: list[ListF1MatchDetail] = Field(default_factory=list)


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
        tool_call_count = sum(1 for output_item in body.response.output if output_item.type == "function_call")
        extracted_content = extract_tool_call_or_text(body.response)
        if extracted_content is None:
            return SingleStepToolUseArgumentComparisonVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                category=StepRewardCategory.NO_ACTION_FOUND,
            )

        expected_action = body.expected_action
        list_f1_match_details: list[ListF1MatchDetail] = []
        match expected_action.type:
            case "function_call":
                if tool_call_count > 1:
                    reward = 0.0
                    category = StepRewardCategory.MULTIPLE_TOOL_CALLS_FOUND

                elif extracted_content.type == "function_call":
                    schema_category = validate_tool_call_against_declared_schema(
                        extracted_content,
                        body.responses_create_params.tools,
                    )
                    if schema_category is not None:
                        reward = 0.0
                        category = schema_category
                    else:
                        tool_call_comparator = ToolCallComparator(config=self.config.tool_call_comparator_config)
                        reward, category = tool_call_comparator.compare_tool_call(expected_action, extracted_content)
                        list_f1_match_details = tool_call_comparator.list_f1_match_details

                else:
                    reward = 0.0
                    category = StepRewardCategory.NO_EXPECTED_TOOL_CALL

            case "message":
                if extracted_content.type == "output_text":
                    # Currently, any chat message is assigned a reward of one.
                    reward = 1.0
                    category = StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND

                else:
                    reward = 0.0
                    category = StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE

            case _:
                raise NotImplementedError

        return SingleStepToolUseArgumentComparisonVerifyResponse(
            **body.model_dump(),
            reward=reward,
            category=category,
            list_f1_match_details=list_f1_match_details,
        )


if __name__ == "__main__":
    SingleStepToolUseArgumentComparisonResourcesServer.run_webserver()
