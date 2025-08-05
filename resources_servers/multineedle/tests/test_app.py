from nemo_gym.server_utils import ServerClient

from app import MultiNeedleResourcesServer, MultiNeedleResourcesServerConfig

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = MultiNeedleResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        MultiNeedleResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
