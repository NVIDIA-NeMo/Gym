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
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


class DummyModelConfig(BaseResponsesAPIModelConfig):
    pass

    def model_post_init(self, context):
        return super().model_post_init(context)


class DummyModel(SimpleResponsesAPIModel):
    config: DummyModelConfig

    def model_post_init(self, context):
        return super().model_post_init(context)

    async def responses(
        self, request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise RuntimeError(
            "DummyModel.responses: should never actually be called!"
        )

    async def chat_completions(
        self, request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise RuntimeError(
            "DummyModel.chat_completions: should never actually be called!"
        )


if __name__ == "__main__":
    DummyModel.run_webserver()
