from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)


class TestBaseResponsesAPIAgent:
    def test_BaseResponsesAPIAgent(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0)
        BaseResponsesAPIAgent(config=config)

    def test_SimpleResponsesAPIAgent(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0)

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

        agent = TestSimpleResponsesAPIAgent(config=config)
        agent.setup_webserver()
