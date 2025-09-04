from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModel,
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymChatCompletion,
    NeMoGymResponse,
)

from unittest.mock import MagicMock


class TestBaseResponsesAPIModel:
    def test_BaseResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(
            host="", port=0, openai_api_key="123", entrypoint=""
        )
        BaseResponsesAPIModel(config=config)

    def test_SimpleResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(
            host="", port=0, openai_api_key="123", entrypoint=""
        )

        class TestSimpleResponsesAPIModel(SimpleResponsesAPIModel):
            async def chat_completions(
                self, request: NeMoGymResponseCreateParamsNonStreaming
            ) -> NeMoGymChatCompletion:
                raise NotImplementedError

            async def responses(
                self, request: NeMoGymResponseCreateParamsNonStreaming
            ) -> NeMoGymResponse:
                raise NotImplementedError

        model = TestSimpleResponsesAPIModel(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        model.setup_webserver()
