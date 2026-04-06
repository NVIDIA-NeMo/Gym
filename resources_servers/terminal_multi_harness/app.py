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
from resources_servers.terminal_multi_harness.common.response_utils import extract_action
from resources_servers.terminal_multi_harness.common.verification_utils import (
    ActionComparator,
    ExpectedAction,
    StepRewardCategory,
    ToolCallComparatorConfig,
)


class TerminalMultiHarnessResourcesServerConfig(BaseResourcesServerConfig):
    tool_call_comparator_config: ToolCallComparatorConfig


class TerminalMultiHarnessRunRequest(BaseRunRequest):
    harness: str = "generic"
    expected_action: ExpectedAction


class TerminalMultiHarnessVerifyRequest(TerminalMultiHarnessRunRequest, BaseVerifyRequest):
    pass


class TerminalMultiHarnessVerifyResponse(BaseVerifyResponse):
    harness: str
    expected_action: ExpectedAction
    category: StepRewardCategory


class TerminalMultiHarnessResourcesServer(SimpleResourcesServer):
    config: TerminalMultiHarnessResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: TerminalMultiHarnessVerifyRequest) -> TerminalMultiHarnessVerifyResponse:
        extracted_action = extract_action(body.response)
        if extracted_action is None:
            return TerminalMultiHarnessVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                category=StepRewardCategory.NO_ACTION_FOUND,
            )

        comparator = ActionComparator(config=self.config.tool_call_comparator_config)
        action_matches, category = comparator.compare_action(
            expected_action=body.expected_action,
            actual_action=extracted_action,
            harness=body.harness,
        )

        return TerminalMultiHarnessVerifyResponse(
            **body.model_dump(),
            reward=1.0 if action_matches else 0.0,
            category=category,
        )


if __name__ == "__main__":
    TerminalMultiHarnessResourcesServer.run_webserver()
