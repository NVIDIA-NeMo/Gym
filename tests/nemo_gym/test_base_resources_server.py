from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
)
from nemo_gym.server_utils import ServerClient

from unittest.mock import MagicMock


class TestBaseResourcesServer:
    def test_sanity(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="")

        class TestSimpleResourcesServer(SimpleResourcesServer):
            async def verify(self, body):
                pass

        agent = TestSimpleResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        agent.setup_webserver()
