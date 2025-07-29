from nemo_gym.base_responses_api_agent import (
    SimpleResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    Body,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
)


class SimpleChatAgentConfig(BaseResponsesAPIAgentConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model_name: str


class SimpleChatAgent(SimpleResponsesAPIAgent):
    config: SimpleChatAgentConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )
        return super().model_post_init(context)

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        body.setdefault("model", self.config.openai_model_name)
        return await self._client.responses.create(**body)


if __name__ == "__main__":
    """
    Test
    ```bash
    curl -X POST http://0.0.0.0:8080/v1/responses \
        -H "Content-Type: application/json" \
        -d '{"input": [{"role": "user", "content": "hello"}]}'
    ```
    """
    config = SimpleChatAgentConfig(
        host="0.0.0.0",
        port=8080,
        openai_base_url="https://api.openai.com/v1",
        openai_api_key=None,
        openai_model_name="gpt-4.1-2025-04-14",
    )
    agent = SimpleChatAgent(config=config)

    agent.run_webserver()
