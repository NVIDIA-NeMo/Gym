from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.mini_swe_resource.app import (
    MiniSweResourceResourcesServer,
    MiniSweResourceResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = MiniSweResourceResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        MiniSweResourceResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
