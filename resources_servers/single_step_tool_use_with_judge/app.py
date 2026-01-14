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
from resources_servers.single_step_tool_use_with_judge.common.response_utils import (
    extract_tool_call_or_output_message_from_response,
)
from resources_servers.single_step_tool_use_with_judge.common.verification_utils import (
    ModelStepVerificationResult,
    ModelStepVerifier,
    ModelStepVerifierConfig,
    StepRewardCategory,
)


class SingleStepToolUseJudgeResourcesServerConfig(BaseResourcesServerConfig):
    model_step_verifier_config: ModelStepVerifierConfig


class SingleStepToolUseJudgeRunRequest(BaseRunRequest):
    domain_policy: str


class SingleStepToolUseJudgeVerifyRequest(SingleStepToolUseJudgeRunRequest, BaseVerifyRequest):
    pass


class SingleStepToolUseJudgeVerifyResponse(ModelStepVerificationResult, BaseVerifyResponse):
    pass


class SingleStepToolUseJudgeResourcesServer(SimpleResourcesServer):
    config: SingleStepToolUseJudgeResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: SingleStepToolUseJudgeVerifyRequest) -> SingleStepToolUseJudgeVerifyResponse:
        extracted_content = extract_tool_call_or_output_message_from_response(body.response)
        if extracted_content is None:
            return SingleStepToolUseJudgeVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                category=StepRewardCategory.NO_ACTION_FOUND,
            )

        responses_create_params = body.responses_create_params
        model_verifier = ModelStepVerifier(
            server_client=self.server_client,
            config=self.config.model_step_verifier_config,
            domain_policy=body.domain_policy,
            agent_tools=responses_create_params.tools,
            responses_create_params=responses_create_params,
        )
        verification_result = await model_verifier.verify_step(extracted_content)
        return SingleStepToolUseJudgeVerifyResponse(
            **body.model_dump(),
            **verification_result.model_dump(),
        )


if __name__ == "__main__":
    SingleStepToolUseJudgeResourcesServer.run_webserver()
