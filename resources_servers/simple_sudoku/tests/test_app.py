from nemo_gym.server_utils import ServerClient

from app import SudokuResourcesServer, SudokuResourcesServerConfig

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = SudokuResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        SudokuResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
