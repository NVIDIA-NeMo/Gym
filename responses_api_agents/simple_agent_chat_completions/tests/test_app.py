from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient

from responses_api_agents.simple_agent_chat_completions.app import (
    SimpleAgentChatCompletions,
    SimpleAgentChatCompletionsConfig,
)

from fastapi.testclient import TestClient

from unittest.mock import MagicMock
from pytest import MonkeyPatch


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleAgentChatCompletionsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        SimpleAgentChatCompletions(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        config = SimpleAgentChatCompletionsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
        )
        server = SimpleAgentChatCompletions(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        app = server.setup_webserver()
        client = TestClient(app)

        mock_chat_data = {
            "id": "chatcmpl-BzRdCFjIEIp59xXLBNYjdPPrcpDaa",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Hello! How can I help you today?",
                        "role": "assistant",
                    },
                }
            ],
            "created": 1753983922,
            "model": "dummy_model",
            "object": "chat.completion",
        }

        dotjson_mock = MagicMock()
        dotjson_mock.json.return_value = mock_chat_data
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post(
            "/v1/responses", json={"input": [{"role": "user", "content": "hello"}]}
        )
        assert res_no_model.status_code == 200
        server.server_client.post.assert_called_with(
            server_name="my server name",
            url_path="/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "chatcmpl-BzRdCFjIEIp59xXLBNYjdPPrcpDaa",
            "created_at": 0.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "",
            "object": "response",
            "output": [
                {
                    "id": "chatcmpl-BzRdCFjIEIp59xXLBNYjdPPrcpDaa",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
        }
        assert expected_responses_dict == actual_responses_dict
