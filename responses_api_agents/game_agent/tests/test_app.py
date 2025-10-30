"""
Tests for responses_api_agents.game_agent.app
Mirrors the structure used by simple_agent tests.
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.game_agent.app import (
    GameAgent,
    GameAgentConfig,
)


class TestGameAgentApp:
    def test_sanity(self) -> None:
        config = GameAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="game_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        )
        GameAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses_tool_call_and_done(self, monkeypatch: MonkeyPatch) -> None:
        config = GameAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="game_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="model"),
            max_steps=10,
        )
        server = GameAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        # Mock model server to emit one function_call then an assistant message
        first_model_response = NeMoGymResponse.model_construct(
            id="resp1",
            object="response",
            created_at=0.0,
            status="completed",
            output=[
                {
                    "type": "function_call",
                    "name": "step",
                    "arguments": '{"actions":["Down"]}',
                    "call_id": "call_1",
                }
            ],
            model="gpt-4.1",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        second_model_response = NeMoGymResponse.model_construct(
            id="resp2",
            object="response",
            created_at=0.0,
            status="completed",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "<answer>Down</answer>",
                }
            ],
            model="gpt-4.1",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        async def fake_post(server_name: str, url_path: str, json=None, cookies=None):  # noqa: A002
            response = MagicMock()
            # /v1/responses goes to model server
            if url_path == "/v1/responses":
                payload = first_model_response if not getattr(fake_post, "_called", False) else second_model_response
                setattr(fake_post, "_called", True)
                response.json = AsyncMock(return_value=payload.model_dump(mode="json"))
                response.cookies = {}
                response.status = 200
                return response
            # resources server tool call
            if url_path == "/step":
                # Return done=true to trigger termination
                response.content.read = AsyncMock(return_value=b'{"observation":"____","reward":0.0,"done":true}')
                response.cookies = {}
                response.status = 200
                return response
            raise AssertionError(f"Unexpected call: {server_name} {url_path}")

        server.server_client.post = AsyncMock(side_effect=fake_post)

        body = NeMoGymResponseCreateParamsNonStreaming(input=[NeMoGymEasyInputMessage(role="user", content="play")])
        resp = client.post("/v1/responses", json=body.model_dump(mode="json"))
        assert resp.status_code == 200
        out = resp.json()
        # We expect both the function_call_output and the assistant message
        types = [item["type"] for item in out["output"]]
        assert "function_call_output" in types
        assert "message" in types

    async def test_run_flow(self, monkeypatch: MonkeyPatch) -> None:
        config = GameAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="game_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        )
        server = GameAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        async def fake_post(server_name: str, url_path: str, json=None, cookies=None):  # noqa: A002
            response = MagicMock()
            response.cookies = {}
            response.status = 200
            if url_path == "/seed_session":
                response.json = AsyncMock(return_value={"observation": "Init"})
                return response
            if url_path == "/v1/responses":
                response.json = AsyncMock(
                    return_value=NeMoGymResponse.model_construct(
                        id="r1",
                        object="response",
                        created_at=0.0,
                        status="completed",
                        output=[],
                        model="gpt-4.1",
                        parallel_tool_calls=True,
                        tool_choice="auto",
                        tools=[],
                    ).model_dump(mode="json")
                )
                return response
            if url_path == "/verify":
                response.json = AsyncMock(return_value={"success": True, "reward": 1.0})
                return response
            raise AssertionError(f"Unexpected call: {server_name} {url_path}")

        server.server_client.post = AsyncMock(side_effect=fake_post)

        run_body = {
            "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "play"}]
            ).model_dump(mode="json"),
        }
        resp = client.post("/run", json=run_body)
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["success"] is True
        assert payload["reward"] == 1.0
