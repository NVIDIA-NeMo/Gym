from nemo_gym.server_utils import ServerClient

from app import MinesweeperResourcesServer, MinesweeperResourcesServerConfig

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = MinesweeperResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        MinesweeperResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
