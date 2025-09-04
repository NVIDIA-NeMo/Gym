from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from app import PythonExecutorResourcesServer, PythonExecutorResourcesServerConfig


class TestApp:
    """Tests for the Python Executor server."""

    SERVER_NAME = "python_math_exec"

    def test_sanity(self) -> None:
        """Basic instantiation test - always runs."""
        config = PythonExecutorResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        PythonExecutorResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
