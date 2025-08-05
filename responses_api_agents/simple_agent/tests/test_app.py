from nemo_gym.server_utils import ServerClient

from app import SimpleAgent, SimpleAgentConfig, ResourcesServerRef, ModelServerRef

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
