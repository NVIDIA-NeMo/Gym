from nemo_gym.server_utils import ServerClient

from app import (
    GoogleSearchResourcesServer,
    GoogleSearchResourcesServerConfig,
    box_parser,
)

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = GoogleSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            google_api_key="dummy_key",
            google_cx="dummy_cx",
        )
        GoogleSearchResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    def test_box_parser_valid_content(self) -> None:
        """Test box_parser with valid boxed content"""
        # Test basic boxed content
        result = box_parser("The answer is \\boxed{42}")
        assert result == "42"

        # Test with complex content
        result = box_parser("After calculation: \\boxed{x + y = 10}")
        assert result == "x + y = 10"

        # Test with no boxed content
        result = box_parser("No boxed content here")
        assert result is None

        # Test with empty string
        result = box_parser("")
        assert result is None
