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

from openai import AzureOpenAI
from fastapi import Request
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_models.vllm_model.app import VLLMConverter


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    default_query: dict


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def model_post_init(self, context):
        self._client = AzureOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
            api_version=self.config.default_query.get("api-version"),
        )
        self._converter = VLLMConverter(return_token_id_information=False)
        print("post init client", self._client)
        return super().model_post_init(context)

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        print("responses body", body)
        # Response Create Params -> Chat Completion Create Params
        chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
        print("chat_completion_create_params", chat_completion_create_params)
        
        # Convert to dict and set model
        chat_completion_params_dict = chat_completion_create_params.model_dump(exclude_unset=True)
        chat_completion_params_dict.setdefault("model", self.config.openai_model)
        
        print("Final API call params:", chat_completion_params_dict)
        print("About to call AzureOpenAI with base_url:", self._client.base_url)
        
        try:
            # Call AzureOpenAI client directly
            chat_completion_response = await self._client.chat.completions.create(**chat_completion_params_dict)
            print("AzureOpenAI response received successfully")
        except Exception as e:
            print(f"AzureOpenAI API Error: {e}")
            print(f"Error type: {type(e)}")
            raise
            
        choice = chat_completion_response.choices[0]
        response_output = self._converter.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]
        return NeMoGymResponse.model_validate(response_output_dicts)

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        print("chat completions body", body)
        body_dict = body.model_dump(exclude_unset=True)
        body_dict.setdefault("model", self.config.openai_model)

        # Chat Completion Create Params -> Chat Completion
        openai_response_dict = await self._client.chat.completions.create(**body_dict)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
