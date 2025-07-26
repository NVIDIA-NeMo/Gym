from responses_api_agents.base import BaseResponsesAPIAgent


class TestBaseResponsesAPIAgent:
    def test_sanity(self) -> None:
        BaseResponsesAPIAgent()
