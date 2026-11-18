# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the gym server client reads the gym's answers, against canned HTTP responses."""

import asyncio
import json
from typing import Any

import pytest
from aiohttp import ClientConnectionError

from nemo_gym.sandbox import SandboxEndpoint
from resources_servers.enterpriseops_gym import gym_client
from resources_servers.enterpriseops_gym.gym_client import (
    GymRequestError,
    GymServerClient,
    GymUnavailableError,
    ToolCallResult,
)


class FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._raw = body if isinstance(body, bytes) else json.dumps(body).encode()

    async def read(self) -> bytes:
        return self._raw


@pytest.fixture
def http(monkeypatch) -> dict[str, Any]:
    """Answers each request with the next queued response and records what was sent."""
    state: dict[str, Any] = {"responses": [], "requests": []}

    async def request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        state["requests"].append({"method": method, "url": url, **kwargs})
        response = state["responses"].pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(gym_client, "request", request)
    return state


def _client() -> GymServerClient:
    return GymServerClient(
        SandboxEndpoint(endpoint="https://proxy.test/sandboxes/1/proxy/8005/", headers={"OPEN-SANDBOX-API-KEY": "k"}),
        verify_tls=False,
        timeout_s=30,
    )


def test_a_tool_call_returns_the_text_the_gym_answered_and_never_resends(http):
    http["responses"].append(
        FakeResponse(200, {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "{}"}]}})
    )

    result = asyncio.run(_client().call_tool("list_labels", {"userId": "me"}, headers={"x-database-id": "db"}))

    assert result == ToolCallResult(text="{}", is_error=False)
    (sent,) = http["requests"]
    assert sent["url"] == "https://proxy.test/sandboxes/1/proxy/8005/mcp"
    assert sent["json"]["method"] == "tools/call"
    assert sent["json"]["params"] == {"name": "list_labels", "arguments": {"userId": "me"}}
    # Task context cannot replace the provider's routing headers.
    assert sent["headers"]["OPEN-SANDBOX-API-KEY"] == "k"
    assert sent["headers"]["x-database-id"] == "db"
    assert sent["_resend_after_disconnect"] is False
    assert sent["ssl"] is False


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            200,
            {"result": {"content": [{"type": "text", "text": "❌ bad"}], "isError": True}},
            ToolCallResult(text="❌ bad", is_error=True),
        ),
        (
            200,
            {"result": {"content": [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]}},
            ToolCallResult(text="a\nb", is_error=False),
        ),
        (
            200,
            {"error": {"code": -32601, "message": "Method not found"}},
            ToolCallResult(text="Tool call failed: Method not found", is_error=True),
        ),
        (
            500,
            b"Internal Server Error",
            ToolCallResult(text="Tool call failed with HTTP 500: Internal Server Error", is_error=True),
        ),
    ],
)
def test_a_tool_the_gym_could_not_run_is_an_error_the_model_sees(http, status, body, expected):
    http["responses"].append(FakeResponse(status, body))

    assert asyncio.run(_client().call_tool("t", {}, headers={"x-database-id": "db"})) == expected


@pytest.mark.parametrize("failure", [FakeResponse(502, b"proxy"), ClientConnectionError("refused")])
def test_an_unreachable_gym_is_an_infrastructure_error(http, failure):
    http["responses"].append(failure)

    with pytest.raises(GymUnavailableError):
        asyncio.run(_client().call_tool("t", {}, headers={"x-database-id": "db"}))


@pytest.mark.parametrize(
    "body",
    [
        {"success": True, "data": {"database_id": "db"}},
        {"message": "✅ Successfully seeded database with 29 SQL statements"},
    ],
)
def test_a_seed_any_gym_reports_without_a_failure_succeeds(http, body):
    http["responses"].append(FakeResponse(200, body))

    asyncio.run(_client().seed_database("db", "INSERT INTO t VALUES (1);"))

    (sent,) = http["requests"]
    assert sent["json"]["database_id"] == "db"
    assert sent["json"]["sql_content"] == "INSERT INTO t VALUES (1);"


@pytest.mark.parametrize(
    ("status", "body"),
    [(400, {"detail": "near TABL: syntax error"}), (200, {"success": False, "message": "could not seed"})],
)
def test_a_refused_seed_raises_with_the_gyms_reason(http, status, body):
    http["responses"].append(FakeResponse(status, body))

    with pytest.raises(GymRequestError) as error:
        asyncio.run(_client().seed_database("db", "CREATE TABL oops;"))

    assert (error.value.status, error.value.detail) == (status, body.get("detail") or body.get("message"))


def test_sql_answers_are_returned_and_rejected_queries_raise(http):
    answer = {"success": True, "data": [{"cnt": 1}]}
    http["responses"] += [FakeResponse(200, answer), FakeResponse(400, {"detail": "no such table: nope"})]
    client = _client()
    headers = {"x-database-id": "db", "x-user-email": "a@example.com"}

    assert asyncio.run(client.run_sql("SELECT COUNT(*) AS cnt FROM t", headers=headers)) == answer
    with pytest.raises(GymRequestError, match="no such table"):
        asyncio.run(client.run_sql("SELECT * FROM nope", headers=headers))
    assert http["requests"][0]["json"] == {"query": "SELECT COUNT(*) AS cnt FROM t", "database_id": "db"}
    assert http["requests"][0]["headers"]["x-user-email"] == "a@example.com"


def test_waiting_for_health_polls_until_the_gym_answers(http):
    http["responses"] += [ClientConnectionError("starting"), FakeResponse(502, b"proxy"), FakeResponse(200, {})]

    asyncio.run(_client().wait_until_healthy(30, poll_interval_s=0))

    assert [request["url"] for request in http["requests"]] == ["https://proxy.test/sandboxes/1/proxy/8005/health"] * 3


def test_a_gym_that_never_becomes_healthy_times_out(http):
    http["responses"] += [FakeResponse(502, b"proxy")] * 3

    with pytest.raises(GymUnavailableError, match="HTTP 502"):
        asyncio.run(_client().wait_until_healthy(0.01, poll_interval_s=0.02))
