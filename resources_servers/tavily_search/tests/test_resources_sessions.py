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
"""tavily_search serves the typed resources sessions API through its real routes, next to unseeded tool callers."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.tavily_search.app import TavilySearchResourcesServer, TavilySearchResourcesServerConfig


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
TASK = TaskId(taskset="tavily_search", task_id="0")
TASK_DATA = {"question": "What is the capital of France?", "ground_truth": "Paris"}
SEARCH_RESULTS = {"results": [{"url": "https://example.com/paris", "title": "Paris", "content": "Capital of France."}]}


def make_server(**config: Any) -> TavilySearchResourcesServer:
    server = TavilySearchResourcesServer(
        config=TavilySearchResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="tavily_search",
            tavily_api_key="test_api_key",  # pragma: allowlist secret
            exclude_domains_file_path=str(Path(__file__).parent / "dummy_exclude_domains_file.json"),
            use_judge=False,
            **config,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    backend = MagicMock()
    backend.search = AsyncMock(return_value=SEARCH_RESULTS)
    server._async_tavily_clients = [backend]
    return server


def seed_json(resources_session_id: str = "resources-session", **update: Any) -> dict[str, Any]:
    seed = ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id, episode_id=EPISODE, task_id=TASK, task_data=TASK_DATA
    )
    return seed.model_copy(update=update).model_dump(mode="json")


def close_json(resources_session_id: str = "resources-session") -> dict[str, Any]:
    return {"resources_session_id": resources_session_id, "episode_id": EPISODE.model_dump(mode="json")}


def verify_json() -> dict[str, Any]:
    """The flat verify body the single_agent_turn Environment Server builds from the task's fields."""
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="model",
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Answer: Paris\nConfidence: 90%", "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return TASK_DATA | {
        "responses_create_params": {"input": TASK_DATA["question"]},
        "response": response.model_dump(mode="json"),
    }


def test_follows_the_session_contract() -> None:
    server = make_server()
    check_resources_session_contract(
        server.setup_webserver(),
        ResourcesSeedSessionRequest(
            resources_session_id="contract-session", episode_id=EPISODE, task_id=TASK, task_data=TASK_DATA
        ),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in server._session_id_to_metrics,
    )


def test_a_typed_session_keeps_its_metrics_until_close() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=seed_json())
    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["resources_session_id"] == "resources-session"
    assert client.post("/web_search", json={"query": "capital of France"}).status_code == 200

    # A retried seed returns the same session, with the search already recorded.
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    verified = client.post("/verify", json=verify_json()).json()
    assert verified["reward"] == 1.0
    assert [call["function"] for call in verified["metrics"]["async_tavily_calls"]] == ["search"]
    # The Environment Server closes after verifying, so verify leaves the session for that close.
    assert "resources-session" in server._session_id_to_metrics

    # The close names the session, so it releases it without the seed's cookie.
    client.cookies.clear()
    assert client.post("/close_session", json=close_json()).status_code == 200
    assert "resources-session" not in server._session_id_to_metrics


def test_a_late_tool_call_does_not_recreate_a_closed_session() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=seed_json())
    client.post("/close_session", json=close_json())

    called = client.post("/web_search", json={"query": "capital of France"})

    assert called.status_code == 404
    assert called.json()["detail"]["failure_kind"] == SESSION_LOST
    assert "resources-session" not in server._session_id_to_metrics
    # The score reads only the response, so a verify after the close still scores it, without metrics.
    verified = client.post("/verify", json=verify_json()).json()
    assert verified["reward"] == 1.0
    assert not verified.get("mask_sample")
    assert verified["metrics"] == {"async_tavily_calls": []}


def test_a_seed_whose_task_data_is_not_a_task_is_refused_for_good() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    response = client.post("/seed_session", json=seed_json(task_data={"question": TASK_DATA["question"]}))

    assert response.status_code == 422
    assert "task_data is not a tavily_search task" in response.json()["detail"]
    assert server._session_id_to_metrics == {}
    # The refusal records nothing, so a corrected seed for the same ID succeeds.
    assert client.post("/seed_session", json=seed_json()).status_code == 200


def test_an_unseeded_callers_metrics_are_released_by_verify() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    # An Agent's /run seeds with its run body, which records nothing; the tools work either way.
    assert client.post("/seed_session", json=TASK_DATA | {"responses_create_params": {"input": "hi"}}).json() == {}
    assert client.post("/web_search", json={"query": "capital of France"}).status_code == 200
    assert len(server._session_id_to_metrics) == 1

    verified = client.post("/verify", json=verify_json()).json()

    assert len(verified["metrics"]["async_tavily_calls"]) == 1
    assert server._session_id_to_metrics == {}


def test_the_exit_dump_keeps_released_metrics() -> None:
    server = make_server(dump_session_id_to_metrics_on_exit=True)
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=seed_json())
    client.post("/web_search", json={"query": "capital of France"})

    assert client.post("/close_session", json=close_json()).status_code == 200

    assert len(server._session_id_to_metrics["resources-session"].async_tavily_calls) == 1
