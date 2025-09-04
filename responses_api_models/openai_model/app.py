from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
    Body,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymResponse,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )
        return super().model_post_init(context)

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict.setdefault("model", self.config.openai_model)
        openai_response = await self._client.responses.create(**body_dict)
        return NeMoGymResponse(**openai_response.model_dump())

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict.setdefault("model", self.config.openai_model)
        openai_response = await self._client.chat.completions.create(**body_dict)
        return NeMoGymChatCompletion(**openai_response.model_dump())


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
