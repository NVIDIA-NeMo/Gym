from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymChatCompletion
from nemo_gym.server_utils import ServerClient

from responses_api_models.openai_model.app import (
    SimpleModelServer,
    SimpleModelServerConfig,
)

from unittest.mock import AsyncMock, MagicMock
from pytest import MonkeyPatch


class TestApp:
    def _setup_server(self):
        config = SimpleModelServerConfig(
            host="0.0.0.0",
            port=8081,
            openai_base_url="https://api.openai.com/v1",
            openai_api_key="dummy_key",
            openai_model="dummy_model",
            entrypoint="",
        )
        return SimpleModelServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    async def test_sanity(self) -> None:
        self._setup_server()

    async def test_chat_completions(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server()
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

        called_args_chat = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_args_chat
            called_args_chat = kwargs
            return NeMoGymChatCompletion(**mock_chat_data)

        mock_chat = AsyncMock(side_effect=mock_create_chat)

        monkeypatch.setattr(server._client.chat.completions, "create", mock_chat)

        chat_no_model = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert chat_no_model.status_code == 200
        assert called_args_chat.get("model") == "dummy_model"

        chat_with_model = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "override_model",
            },
        )
        assert chat_with_model.status_code == 200
        assert called_args_chat.get("model") == "override_model"

        mock_chat.assert_any_await(
            messages=[{"role": "user", "content": "hi"}],
            model="override_model",
        )

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        called_args_response = {}

        async def mock_create_responses(**kwargs):
            nonlocal called_args_response
            called_args_response = kwargs
            return NeMoGymResponse(**mock_response_data)

        mock_response = AsyncMock(side_effect=mock_create_responses)

        monkeypatch.setattr(server._client.responses, "create", mock_response)

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": "hello"})
        assert res_no_model.status_code == 200
        assert called_args_response.get("model") == "dummy_model"

        # model provided should override config
        res_with_model = client.post(
            "/v1/responses", json={"input": "hello", "model": "override_model"}
        )
        assert res_with_model.status_code == 200
        assert called_args_response.get("model") == "override_model"

        mock_response.assert_any_await(input="hello", model="override_model")
