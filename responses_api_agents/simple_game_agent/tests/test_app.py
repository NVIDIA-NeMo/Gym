from unittest.mock import MagicMock

from app import TextGameAgent, TextGameAgentConfig

from nemo_gym.server_utils import ModelServerRef, ResourcesServerRef, ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = TextGameAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            resources_server=ResourcesServerRef(type="resources_servers", name="simple_sudoku"),
            model_server=ModelServerRef(type="responses_api_models", name="openai_model"),
            max_moves=50,
        )
        TextGameAgent(config=config, server_client=MagicMock(spec=ServerClient))
