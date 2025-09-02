
from nemo_gym.server_utils import ServerClient

from resources_servers.offline_search.app import (
    OfflineSearchResourcesServer,
    OfflineSearchResourcesServerConfig,
    OfflineSearchVerifyRequest,
    BaseSearchQueryRequest,
    _extract_last_assistant_text,
)

from unittest.mock import MagicMock, patch
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText


class TestApp:
    def test_sanity(self) -> None:
        """Test basic server initialization"""
        config = OfflineSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            base_url="http://localhost:5000"
        )
        server = OfflineSearchResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        assert server.config.base_url == "http://localhost:5000"

    @patch('requests.post')
    async def test_search_endpoint(self, mock_post) -> None:
        """Test the search endpoint with mocked external API"""
        # Setup
        config = OfflineSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            base_url="http://localhost:5000"
        )
        server = OfflineSearchResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"title": "Test Result", "content": "Test content"}]}
        mock_post.return_value = mock_response
        
        # Test request
        request = BaseSearchQueryRequest(query="test query", topk=5)
        response = await server.search(request)
        
        # Verify the call was made correctly
        mock_post.assert_called_once_with(
            "http://localhost:5000/retrieve",
            json={
                "queries": ["test query"],
                "topk": 5,
                "return_scores": False
            }
        )
        
        # Verify response format
        assert "results" in response.search_results
        assert "Test Result" in response.search_results

    async def test_verify_function(self) -> None:
        """Test the verify function with correct and incorrect answers"""
        config = OfflineSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            base_url="http://localhost:5000"
        )
        server = OfflineSearchResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        
        # Test with correct answer
        correct_message = NeMoGymResponseOutputMessage(
            id="test_id_1",
            content=[
                NeMoGymResponseOutputText(
                    text="The answer is \\boxed{A}",
                    type="output_text",
                    annotations=[]
                )
            ],
            role="assistant",
            status="completed",
            type="message"
        )
        
        verify_request = OfflineSearchVerifyRequest(
            expected_answer="A",
            responses_create_params={"input": []},  # Required field
            response=NeMoGymResponse(
                id="test_response_1",
                created_at=1234567890.0,
                model="test_model",
                object="response",
                output=[correct_message],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[]
            )
        )
        
        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.parsed_option == "A"
        
        # Test with incorrect answer
        incorrect_message = NeMoGymResponseOutputMessage(
            id="test_id_2",
            content=[
                NeMoGymResponseOutputText(
                    text="The answer is \\boxed{B}",
                    type="output_text",
                    annotations=[]
                )
            ],
            role="assistant",
            status="completed",
            type="message"
        )
        
        verify_request_wrong = OfflineSearchVerifyRequest(
            expected_answer="A",
            responses_create_params={"input": []},  # Required field
            response=NeMoGymResponse(
                id="test_response_2",
                created_at=1234567890.0,
                model="test_model",
                object="response",
                output=[incorrect_message],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[]
            )
        )
        
        result_wrong = await server.verify(verify_request_wrong)
        assert result_wrong.reward == 0.0
        assert result_wrong.parsed_option == "B"