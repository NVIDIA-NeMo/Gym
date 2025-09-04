from nemo_gym.server_utils import ServerClient

from resources_servers.simple_weather.app import (
    SimpleWeatherResourcesServer,
    SimpleWeatherResourcesServerConfig,
)

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleWeatherResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        SimpleWeatherResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
