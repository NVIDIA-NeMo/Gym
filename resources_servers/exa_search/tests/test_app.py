# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientConnectionError
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, check_resources_session_contract
from resources_servers.exa_search import app


_KEY = "exa-test-key"  # pragma: allowlist secret
_RESULT = {
    "title": "Exa",
    "url": "https://exa.ai",
    "publishedDate": "2026-01-01",
    "author": None,
    "text": "Exa is a search engine.",
    "id": "ignored",
}


class _Response:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status = status
        self._payload = payload
        self.released = False

    async def json(self) -> Any:
        return self._payload

    def release(self) -> None:
        self.released = True


def _server(**config: Any) -> app.ExaSearchResourcesServer:
    return app.ExaSearchResourcesServer(
        config=app.ExaSearchResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="exa_search", exa_api_key=_KEY, **config
        ),
        server_client=MagicMock(spec=ServerClient),
    )


class _Exa:
    """Records every call to Exa and answers each with the next queued reply (by default, one result)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies: list[Any] = []

    async def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        reply = self.replies.pop(0) if self.replies else _Response(200, {"results": [_RESULT]})
        if isinstance(reply, BaseException):
            raise reply
        return reply


@pytest.fixture
def exa(monkeypatch: pytest.MonkeyPatch) -> _Exa:
    fake = _Exa()

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(app, "request", fake.request)
    monkeypatch.setattr(app.asyncio, "sleep", no_sleep)
    return fake


async def test_search_returns_the_page_text_of_each_result(exa: _Exa) -> None:
    server = _server(exa_base_url="https://exa.example/", max_characters=1200)
    result = await server.exa_search(app.ExaSearchRequest(query="  what is exa  ", num_results=3))
    assert json.loads(result.results_string) == [
        {
            "title": "Exa",
            "url": "https://exa.ai",
            "published_date": "2026-01-01",
            "author": None,
            "text": "Exa is a search engine.",
        }
    ]
    (call,) = exa.calls
    assert call["method"] == "POST" and call["url"] == "https://exa.example/search"
    assert call["headers"]["x-api-key"] == _KEY
    assert call["json"] == {"query": "what is exa", "numResults": 3, "contents": {"text": {"maxCharacters": 1200}}}


@pytest.mark.parametrize(("asked", "sent"), [(None, 5), (0, 5), (50, app.MAX_NUM_RESULTS), (-2, 1)])
async def test_result_count_defaults_and_stays_in_range(exa: _Exa, asked: int | None, sent: int) -> None:
    await _server().exa_search(app.ExaSearchRequest(query="q", num_results=asked))
    assert exa.calls[0]["json"]["numResults"] == sent


@pytest.mark.parametrize("query", [None, "   ", "x" * (app.MAX_QUERY_CHARS + 1)])
async def test_unusable_query_is_answered_without_calling_exa(exa: _Exa, query: str | None) -> None:
    result = await _server().exa_search(app.ExaSearchRequest(query=query))
    assert result.results_string.startswith("Exa search")
    assert exa.calls == []


async def test_retryable_failures_are_retried(exa: _Exa) -> None:
    throttled = _Response(429)
    exa.replies.extend([throttled, ClientConnectionError("reset")])
    result = await _server().exa_search(app.ExaSearchRequest(query="q"))
    assert json.loads(result.results_string)[0]["url"] == "https://exa.ai"
    assert len(exa.calls) == 3
    assert throttled.released


@pytest.mark.parametrize(
    ("replies", "message", "attempts"),
    [
        ([_Response(401, {"error": f"bad key {_KEY}"})], "Exa search failed with HTTP 401.", 1),
        ([_Response(503)] * 3, "Exa search failed with HTTP 503.", 3),
        ([TimeoutError()] * 3, "Exa search failed to reach Exa (TimeoutError).", 3),
        ([_Response(200, ["not", "an", "object"])], "Exa search returned an unexpected reply.", 1),
    ],
)
async def test_failures_reach_the_agent_without_the_provider_body(
    exa: _Exa, replies: list[Any], message: str, attempts: int
) -> None:
    exa.replies.extend(replies)
    result = await _server().exa_search(app.ExaSearchRequest(query="q"))
    assert result.results_string == message
    assert _KEY not in result.results_string
    assert len(exa.calls) == attempts


async def test_no_results_is_said_plainly(exa: _Exa) -> None:
    exa.replies.append(_Response(200, {"results": []}))
    assert (
        await _server().exa_search(app.ExaSearchRequest(query="q"))
    ).results_string == "Exa search found no results."


def test_the_tool_is_served_over_http_and_verify_is_refused(exa: _Exa) -> None:
    server = _server()
    assert server.mcp_allowed_tools_for_session({}) == ["exa_search"]
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        searched = client.post("/exa_search", json={"query": "q"})
        assert searched.status_code == 200 and "exa.ai" in searched.json()["results_string"]
        response = {
            "id": "r",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        verified = client.post("/verify", json={"responses_create_params": {"input": []}, "response": response})
        assert verified.status_code == 400, verified.text


def test_api_key_stays_out_of_the_config_dump() -> None:
    assert _KEY not in repr(_server().config)


def test_session_conformance() -> None:
    seed = ResourcesSeedSessionRequest(
        resources_session_id="tool-session",
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="deepsearchqa", task_id="0"),
        task_data={"question": "anything the episode's task carries"},
    )
    check_resources_session_contract(_server().setup_webserver(), seed, keeps_state=False)
