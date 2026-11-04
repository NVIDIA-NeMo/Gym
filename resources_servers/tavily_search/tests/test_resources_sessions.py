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

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from omegaconf import DictConfig

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import (
    CHECKPOINT_CLOSE_FOLLOWS_HEADER,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.tavily_search.app import (
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
TASK = TaskId(taskset="tavily_search", task_id="0")
TASK_DATA = {"question": "What is the capital of France?", "ground_truth": "Paris"}
SEARCH_RESULTS = {"results": [{"url": "https://example.com/paris", "title": "Paris", "content": "Capital of France."}]}


def make_server(*, checkpoint: bool = False, **config: Any) -> TavilySearchResourcesServer:
    server_client = MagicMock(spec=ServerClient)
    if checkpoint:
        server_client.global_config_dict = DictConfig(
            {"checkpoint": {"enabled": True, "control_auth_token": "test-token"}}
        )
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
        server_client=server_client,
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
        verify_body=verify_json(),
    )


def test_a_typed_verify_releases_live_metrics_and_preserves_its_retry() -> None:
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
    assert "resources-session" not in server._session_id_to_metrics
    assert "resources-session" not in server._typed_sessions
    resent = client.post("/verify", json=verify_json()).json()
    assert resent["metrics"] == verified["metrics"]
    assert (resent["reward"], resent["mask_sample"]) == (1.0, False)
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert "resources-session" not in server._session_id_to_metrics
    assert client.post("/web_search", json={"query": "capital of France"}).status_code == 404

    # The close names the session, so it releases it without the seed's cookie.
    client.cookies.clear()
    assert client.post("/close_session", json=close_json()).status_code == 200
    assert "resources-session" not in server._session_id_to_metrics
    assert server._verified_sessions.get("resources-session").metrics is None


def test_a_late_tool_call_does_not_recreate_a_closed_session() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=seed_json())
    client.post("/close_session", json=close_json())

    called = client.post("/web_search", json={"query": "capital of France"})

    assert called.status_code == 404
    assert called.json()["detail"]["failure_kind"] == SESSION_LOST
    assert "resources-session" not in server._session_id_to_metrics
    # Masked like every server under the contract, although the answer alone would score 1.
    verified = client.post("/verify", json=verify_json()).json()
    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
    assert verified["metrics"] == {"async_tavily_calls": []}


async def test_a_search_in_flight_across_close_does_not_recreate_the_session() -> None:
    server = make_server()
    searching, finish = asyncio.Event(), asyncio.Event()

    async def slow_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
        searching.set()
        await finish.wait()
        return SEARCH_RESULTS

    server._async_tavily_clients[0].search = slow_search
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "resources-session"}
    await server.seed_session(request, ResourcesSeedSessionRequest.model_validate(seed_json()))
    search = asyncio.create_task(server.web_search(request, TavilySearchRequest(query="capital of France")))
    await asyncio.wait_for(searching.wait(), timeout=5)

    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(close_json()))
    finish.set()
    await asyncio.wait_for(search, timeout=5)

    # The search records its call on the metrics it read before the close, not on a new record.
    assert "resources-session" not in server._session_id_to_metrics


def test_a_seed_for_the_same_episode_and_another_task_is_refused() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=seed_json()).status_code == 200

    other_task = client.post("/seed_session", json=seed_json(task_id=TaskId(taskset="tavily_search", task_id="1")))

    assert other_task.status_code == 409


def test_a_seed_whose_task_data_is_not_a_task_is_refused_and_records_nothing() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    response = client.post("/seed_session", json=seed_json(task_data={"question": TASK_DATA["question"]}))

    assert response.status_code == 422
    assert "task_data is not a tavily_search task" in response.json()["detail"]
    assert server._session_id_to_metrics == {}
    # The refusal records nothing, so a corrected seed for the same ID succeeds.
    assert client.post("/seed_session", json=seed_json()).status_code == 200


def test_an_unseeded_callers_metrics_are_released_by_verify() -> None:
    """A caller that never sends a typed seed, such as an Agent's /run or an MCP client, is scored as before."""
    server = make_server()
    client = TestClient(server.setup_webserver())

    # An Agent's /run seeds with its run body, which records nothing; the tools work either way.
    assert client.post("/seed_session", json=TASK_DATA | {"responses_create_params": {"input": "hi"}}).json() == {}
    assert client.post("/web_search", json={"query": "capital of France"}).status_code == 200
    assert len(server._session_id_to_metrics) == 1

    verified = client.post("/verify", json=verify_json()).json()

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert len(verified["metrics"]["async_tavily_calls"]) == 1
    assert server._session_id_to_metrics == {}
    resent = client.post("/verify", json=verify_json()).json()
    assert resent["metrics"] == verified["metrics"]


def test_the_exit_dump_keeps_released_metrics() -> None:
    server = make_server(dump_session_id_to_metrics_on_exit=True)
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=seed_json())
    client.post("/web_search", json={"query": "capital of France"})

    assert client.post("/close_session", json=close_json()).status_code == 200

    assert len(server._session_id_to_metrics["resources-session"].async_tavily_calls) == 1


@pytest.mark.parametrize("state", ["active", "verified", "closed"])
def test_a_run_seed_cannot_reuse_a_typed_session(state: str) -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    if state == "verified":
        assert client.post("/verify", json=verify_json()).status_code == 200
    elif state == "closed":
        assert client.post("/close_session", json=close_json()).status_code == 200
    assert (
        client.post("/seed_session", json=TASK_DATA | {"responses_create_params": {"input": "hi"}}).status_code == 409
    )


def test_a_typed_seed_cannot_claim_an_unseeded_callers_metrics() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    assert client.post("/web_search", json={"query": "capital of France"}).status_code == 200
    [session_id] = server._session_id_to_metrics
    original = server._session_id_to_metrics[session_id]
    assert client.post("/seed_session", json=seed_json(session_id)).status_code == 409
    assert server._session_id_to_metrics[session_id] is original


def test_failed_metrics_initialization_leaves_no_binding_and_can_be_retried() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    with patch("resources_servers.tavily_search.app.TavilySearchMetrics", side_effect=MemoryError):
        assert client.post("/seed_session", json=seed_json()).status_code == 500
    assert server._typed_sessions == {}
    assert server._session_id_to_metrics == {}
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert "resources-session" in server._session_id_to_metrics


def test_checkpoint_wait_verify_leaves_only_expiring_metrics() -> None:
    server = make_server(checkpoint=True)
    client = TestClient(server.setup_webserver())
    assert client.post("/ng-rollout/rollout/seed_session", json=seed_json()).status_code == 200
    assert client.post("/ng-rollout/rollout/web_search", json={"query": "capital of France"}).status_code == 200
    assert server._checkpoint.readiness().counts["sessions"] == 1
    verified = client.post(
        "/ng-rollout/rollout/verify", json=verify_json(), headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
    )
    assert verified.status_code == 200
    assert server._checkpoint.readiness().counts["sessions"] == 0
    assert server._session_id_to_metrics == {}
    assert server._typed_sessions == {}
    resent = client.post("/ng-rollout/rollout/verify", json=verify_json()).json()
    assert resent["metrics"] == verified.json()["metrics"]


@pytest.mark.parametrize("typed", [False, True])
def test_checkpoint_retirement_releases_unverified_metrics(typed: bool) -> None:
    server = make_server(checkpoint=True)
    client = TestClient(server.setup_webserver())
    assert client.post("/ng-rollout/rollout/seed_session", json=seed_json() if typed else TASK_DATA).status_code == 200
    assert client.post("/ng-rollout/rollout/web_search", json={"query": "capital of France"}).status_code == 200
    assert len(server._session_id_to_metrics) == 1
    retired = client.post(
        "/ng-control/v1/checkpoint/retire",
        json={"checkpoint_id": "retire", "deadline_ts": time.time() + 5, "episode_ids": [EPISODE.model_dump()]},
        headers={"authorization": "Bearer test-token"},
    )
    assert retired.status_code == 200, retired.text
    assert server._session_id_to_metrics == {}
    assert server._typed_sessions == {}


def test_verified_metrics_expire_and_close_still_checks_the_episode() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    with patch("nemo_gym.server_runtime.base_resources_server.monotonic", return_value=0):
        assert client.post("/seed_session", json=seed_json()).status_code == 200
        assert client.post("/web_search", json={"query": "capital of France"}).status_code == 200
        assert client.post("/verify", json=verify_json()).json()["reward"] == 1.0
        other = EpisodeId(rollout_id="other", attempt=0)
        assert client.post("/seed_session", json=seed_json(episode_id=other)).status_code == 409
        assert client.post("/close_session", json=close_json() | {"episode_id": other.model_dump()}).status_code == 409
    with patch("nemo_gym.server_runtime.base_resources_server.monotonic", return_value=301):
        assert client.post("/seed_session", json=seed_json(episode_id=other)).status_code == 200
        assert server._session_id_to_metrics["resources-session"].async_tavily_calls == []


@pytest.mark.parametrize("field", ["resources_session_id", "episode_id", "task_id", "task_data"])
def test_a_malformed_typed_seed_cannot_fall_back_to_a_run_seed(field: str) -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    body = seed_json()
    body.pop(field)
    # An extra top-level environment also must not let a malformed typed seed use the /run arm.
    body["environment"] = {}
    assert client.post("/seed_session", json=body).status_code == 422
    assert server._typed_sessions == {}


def test_a_run_seed_keeps_a_row_with_its_own_task_id() -> None:
    # A dataset row's scalar task_id is a row field, not a typed seed's TaskId.
    server = make_server()
    client = TestClient(server.setup_webserver())
    body = TASK_DATA | {"responses_create_params": {"input": "hi"}, "task_id": "row-0"}
    assert client.post("/seed_session", json=body).status_code == 200
    assert client.post("/web_search", json={"query": "capital of France"}).status_code == 200
    assert server._typed_sessions == {}


def test_repeated_runs_rotate_the_cookie_and_do_not_mix_metrics() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    body = TASK_DATA | {"responses_create_params": {"input": "hi"}}
    assert client.post("/seed_session", json=body).status_code == 200
    assert client.post("/web_search", json={"query": "capital of France"}).status_code == 200
    [first_id] = server._session_id_to_metrics
    first_cookies = dict(client.cookies)
    first = client.post("/verify", json=verify_json()).json()
    assert client.post("/seed_session", json=body).status_code == 200
    assert client.post("/web_search", json={"query": "capital of France"}).status_code == 200
    [second_id] = server._session_id_to_metrics
    assert second_id != first_id
    second = client.post("/verify", json=verify_json()).json()
    assert len(first["metrics"]["async_tavily_calls"]) == len(second["metrics"]["async_tavily_calls"]) == 1
    client.cookies.clear()
    client.cookies.update(first_cookies)
    assert client.post("/verify", json=verify_json()).json()["metrics"] == first["metrics"]
    assert server._session_id_to_metrics == {}


@pytest.mark.parametrize("typed", [False, True])
async def test_a_failed_judge_retry_preserves_metrics_without_caching_the_failure(typed: bool) -> None:
    from nemo_gym.verification import JudgeError
    from resources_servers.tavily_search.app import JudgeEvaluation, TavilySearchVerifyRequest

    server = make_server()
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "resources-session"}
    if typed:
        await server.seed_session(request, ResourcesSeedSessionRequest.model_validate(seed_json()))
    await server.web_search(request, TavilySearchRequest(query="capital of France"))
    body = TavilySearchVerifyRequest.model_validate(verify_json())
    server.config.use_judge = True
    failed_judge = AsyncMock(
        return_value=(JudgeEvaluation(reasoning="", extracted_final_answer="", reward=0), JudgeError("unavailable"))
    )
    with patch.object(type(server), "_verify_answer_with_judge", failed_judge):
        with pytest.raises(JudgeError, match="unavailable"):
            await server.verify(request, body)
    assert server._session_id_to_metrics == {}
    assert server._typed_sessions == {}
    server.config.use_judge = False
    retried = await server.verify(request, body)
    assert retried.reward == 1.0 and retried.mask_sample is False
    assert [call.function for call in retried.metrics.async_tavily_calls] == ["search"]
