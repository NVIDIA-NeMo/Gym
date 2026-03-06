from nemo_gym.server_utils import ServerClient

from app import WordleResourcesServer, WordleResourcesServerConfig

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = WordleResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        WordleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
