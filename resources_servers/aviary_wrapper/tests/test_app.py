import pytest
from unittest.mock import MagicMock, patch

from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        """Test that the server config can be instantiated"""
        # Mock the aviary imports to avoid dependency issues during testing
        with patch.dict('sys.modules', {
            'aviary.envs.hotpotqa.env': MagicMock(),
            'aviary.core': MagicMock(),
        }):
            from resources_servers.aviary_wrapper.app import (
                AviaryWrapperResourcesServerConfig
            )
            
            config = AviaryWrapperResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                max_steps=10,
                correct_reward=1.0,
                incorrect_reward=0.0,
            )
            assert config is not None
            assert config.max_steps == 10

    def test_parse_action_mock(self) -> None:
        """Test XML action parsing with mocked imports"""
        # Mock the aviary imports
        with patch.dict('sys.modules', {
            'aviary.envs.hotpotqa.env': MagicMock(),
            'aviary.core': MagicMock(),
        }):
            from resources_servers.aviary_wrapper.app import (
                AviaryWrapperResourcesServer,
                AviaryWrapperResourcesServerConfig
            )
            
            config = AviaryWrapperResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
            )
            server = AviaryWrapperResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            
            # Test search action
            result = server._parse_action("<search>Barack Obama</search>")
            assert result == ("search", "Barack Obama")
            
            # Test lookup action
            result = server._parse_action("<lookup>birth date</lookup>")
            assert result == ("lookup", "birth date")
            
            # Test submit answer
            result = server._parse_action("<submit_answer>Hawaii</submit_answer>")
            assert result == ("submit_answer", "Hawaii")
            
            # Test invalid format
            result = server._parse_action("just plain text")
            assert result is None
