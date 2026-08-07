# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import Cookies

from nemo_gym.server_utils import ServerClient
from resources_servers.browserbase_webvoyager.app import (
    BrowserbaseWebVoyagerResourcesServer,
    BrowserbaseWebVoyagerResourcesServerConfig,
)


class StatelessCookies(Cookies):
    def extract_cookies(self, response) -> None:  # pragma: no cover - behavior hook for TestClient
        pass


def _make_server() -> BrowserbaseWebVoyagerResourcesServer:
    config = BrowserbaseWebVoyagerResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        browserbase_api_key="browserbase-key",
        project_id="project-id",
        model_api_key="model-key",
    )
    return BrowserbaseWebVoyagerResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )


def _make_verify_payload(final_text: str) -> dict:
    return {
        "question": "Find the important link on the page.",
        "start_url": "https://example.com",
        "expected_answer": "More information",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "Go to the page and report the important link.",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
        "response": {
            "id": "resp_123",
            "object": "response",
            "created_at": 1,
            "model": "test-model",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": final_text, "annotations": []}],
                }
            ],
        },
    }


class TestBrowserbaseWebVoyagerApp:
    def test_init_requires_credentials(self) -> None:
        with pytest.raises(ValueError, match="Missing Browserbase API key"):
            BrowserbaseWebVoyagerResourcesServer(
                config=BrowserbaseWebVoyagerResourcesServerConfig(
                    host="0.0.0.0",
                    port=8080,
                    entrypoint="",
                    name="",
                ),
                server_client=MagicMock(spec=ServerClient),
            )

    def test_seed_session_and_tool_calls_are_session_scoped(self) -> None:
        server = _make_server()
        runtime_a = SimpleNamespace(stagehand_session_id="sid-a")
        runtime_b = SimpleNamespace(stagehand_session_id="sid-b")

        server.runtime_manager.create_runtime = AsyncMock(side_effect=[runtime_a, runtime_b])
        server.runtime_manager.navigate = AsyncMock(side_effect=["navigated-a", "navigated-b"])
        server.runtime_manager.cleanup_runtime = AsyncMock()

        client = TestClient(server.setup_webserver())
        client._cookies = StatelessCookies(client._cookies)

        seed_a = client.post("/seed_session", json={"start_url": "https://example.com"})
        cookies_a = seed_a.cookies
        assert seed_a.status_code == 200

        seed_b = client.post("/seed_session", json={"start_url": "https://example.org"})
        cookies_b = seed_b.cookies
        assert seed_b.status_code == 200

        response_a = client.post("/navigate", json={"url": "https://example.com/a"}, cookies=cookies_a)
        response_b = client.post("/navigate", json={"url": "https://example.org/b"}, cookies=cookies_b)

        assert response_a.text == "navigated-a"
        assert response_b.text == "navigated-b"
        assert server.runtime_manager.navigate.await_args_list[0].args[0] is runtime_a
        assert server.runtime_manager.navigate.await_args_list[1].args[0] is runtime_b

    def test_verify_scores_and_cleans_up_runtime(self) -> None:
        server = _make_server()
        runtime = SimpleNamespace(stagehand_session_id="sid-verify")

        server.runtime_manager.create_runtime = AsyncMock(return_value=runtime)
        server.runtime_manager.cleanup_runtime = AsyncMock()

        client = TestClient(server.setup_webserver())
        client._cookies = StatelessCookies(client._cookies)

        seed = client.post("/seed_session", json={"start_url": "https://example.com"})
        cookies = seed.cookies

        verify_response = client.post(
            "/verify",
            json=_make_verify_payload("The page contains the More information link."),
            cookies=cookies,
        )

        assert verify_response.status_code == 200
        verify_json = verify_response.json()
        assert verify_json["reward"] == 1.0
        assert verify_json["matched_expected_answer"] is True
        assert verify_json["stagehand_session_id"] == "sid-verify"

        post_verify_tool = client.post("/navigate", json={"url": "https://example.com/after"}, cookies=cookies)
        assert post_verify_tool.text == "No active session. Call /seed_session first."
        server.runtime_manager.cleanup_runtime.assert_awaited_once()

    def test_verify_returns_zero_when_expected_answer_not_found(self) -> None:
        server = _make_server()
        runtime = SimpleNamespace(stagehand_session_id="sid-miss")

        server.runtime_manager.create_runtime = AsyncMock(return_value=runtime)
        server.runtime_manager.cleanup_runtime = AsyncMock()

        client = TestClient(server.setup_webserver())
        client._cookies = StatelessCookies(client._cookies)

        seed = client.post("/seed_session", json={"start_url": "https://example.com"})
        cookies = seed.cookies

        verify_response = client.post(
            "/verify",
            json=_make_verify_payload("The page only says hello world."),
            cookies=cookies,
        )

        assert verify_response.status_code == 200
        verify_json = verify_response.json()
        assert verify_json["reward"] == 0.0
        assert verify_json["matched_expected_answer"] is False
