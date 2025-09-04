from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


class TestOpenAIUtils:
    def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")
