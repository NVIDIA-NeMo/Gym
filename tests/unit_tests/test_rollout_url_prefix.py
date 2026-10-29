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
"""NVIDIA-NeMo/Gym#4724: a client with a fixed base URL names its rollout by rewriting each request URL."""

import asyncio
import logging
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import requests
import uvicorn
from fastapi import FastAPI, Request
from requests.adapters import HTTPAdapter

from nemo_gym.rollout_correlation import RolloutContextMiddleware, current_rollout_id, rollout_context
from nemo_gym.server_utils import (
    RequestContextIsolationMiddleware,
    async_rollout_prefix_hook,
    rollout_prefix_hook,
    with_rollout_prefix,
)


def _echo_app() -> RequestContextIsolationMiddleware:
    """A Gym-style server that reports the rollout and routed path it saw, isolated per request as in production."""
    app = FastAPI()

    @app.post("/v1/responses")
    async def _responses(request: Request) -> dict:
        return {"rollout_id": current_rollout_id(), "path": request.url.path}

    app.add_middleware(RolloutContextMiddleware)
    return RequestContextIsolationMiddleware(app)


@pytest.fixture
def server_url() -> Iterator[str]:
    server = uvicorn.Server(uvicorn.Config(_echo_app(), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the test server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.parametrize(
    ("url", "base_urls", "expected"),
    [
        (
            "http://gym:8000/v1/responses?x=1",
            ["http://gym:8000"],
            "http://gym:8000/ng-rollout/3-0-a1/v1/responses?x=1",
        ),
        ("http://gym:8000/v1/responses", ["http://gym:8000/"], "http://gym:8000/ng-rollout/3-0-a1/v1/responses"),
        (
            "http://gym:8000/api/v1/responses",
            ["http://gym:8000/api"],
            "http://gym:8000/api/ng-rollout/3-0-a1/v1/responses",
        ),
        # httpx and requests normalize host case and the default port before a hook sees the URL.
        (
            "http://model-server:8000/v1/responses",
            ["http://Model-Server:8000"],
            "http://model-server:8000/ng-rollout/3-0-a1/v1/responses",
        ),
        ("http://gym/v1/responses", ["http://gym:80"], "http://gym/ng-rollout/3-0-a1/v1/responses"),
        ("http://gym:80/v1/responses", ["http://gym"], "http://gym:80/ng-rollout/3-0-a1/v1/responses"),
        ("https://gym/v1/responses", ["https://gym:443"], "https://gym/ng-rollout/3-0-a1/v1/responses"),
        # Whole segments and the exact host and port only.
        ("http://gym:18000/v1/responses", ["http://gym:1800"], "http://gym:18000/v1/responses"),
        ("http://gym:8000/api10/v1/responses", ["http://gym:8000/api1"], "http://gym:8000/api10/v1/responses"),
        ("https://gym:8000/v1/responses", ["http://gym:8000"], "https://gym:8000/v1/responses"),
        ("http://other:8000/v1/responses", ["http://gym:8000"], "http://other:8000/v1/responses"),
        # Port 0 is a port, not the default one.
        ("http://gym:0/v1/responses", ["http://gym"], "http://gym:0/v1/responses"),
        # Repeated slashes are kept as sent; the prefix goes right after the base's segments.
        (
            "http://gym:8000//api/v1/responses",
            ["http://gym:8000/api"],
            "http://gym:8000//api/ng-rollout/3-0-a1/v1/responses",
        ),
        (
            "http://gym:8000/api//v1/responses",
            ["http://gym:8000//api/"],
            "http://gym:8000/api/ng-rollout/3-0-a1//v1/responses",
        ),
        # A prefix already on the request wins, so a client addressing another key is left alone.
        (
            "http://gym:8000/ng-rollout/7-1/v1/responses",
            ["http://gym:8000"],
            "http://gym:8000/ng-rollout/7-1/v1/responses",
        ),
        (
            "http://gym:8000/ng-rollout/7-1/training-token-capture/v1/responses",
            ["http://gym:8000"],
            "http://gym:8000/ng-rollout/7-1/training-token-capture/v1/responses",
        ),
    ],
)
def test_with_rollout_prefix_rewrites_only_urls_under_a_base(url: str, base_urls: list[str], expected: str) -> None:
    with rollout_context("3-0-a1"):
        assert with_rollout_prefix(url, base_urls) == expected


def test_without_a_current_rollout_the_url_is_unchanged_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="nemo_gym.server_utils"):
        assert (
            with_rollout_prefix("http://gym:8000/v1/responses", ["http://gym:8000"]) == "http://gym:8000/v1/responses"
        )
    assert "No current rollout" in caplog.text


@pytest.mark.parametrize("make", [lambda base: with_rollout_prefix("http://h:8000/v1", base), rollout_prefix_hook])
def test_a_single_base_url_string_is_refused(make) -> None:
    with pytest.raises(TypeError, match="not a single string"):
        make("http://h:8000")


def test_the_sync_httpx_hook_names_the_rollout_on_a_real_server(server_url: str) -> None:
    with httpx.Client(base_url=server_url, event_hooks={"request": [rollout_prefix_hook([server_url])]}) as client:
        with rollout_context("3-0-a1"):
            assert client.post("/v1/responses").json() == {"rollout_id": "3-0-a1", "path": "/v1/responses"}
        assert client.post("/v1/responses").json() == {"rollout_id": None, "path": "/v1/responses"}


def test_the_documented_requests_adapter_names_the_rollout_on_a_real_server(server_url: str) -> None:
    class RolloutPrefixAdapter(HTTPAdapter):
        def send(self, request, **kwargs):
            request.url = with_rollout_prefix(request.url, [server_url])
            return super().send(request, **kwargs)

    session = requests.Session()
    session.mount("http://", RolloutPrefixAdapter())
    session.mount("https://", RolloutPrefixAdapter())
    with rollout_context("4-2"):
        assert session.post(f"{server_url}/v1/responses").json() == {"rollout_id": "4-2", "path": "/v1/responses"}


async def test_a_synchronous_client_in_to_thread_sees_the_rollout(server_url: str) -> None:
    with httpx.Client(base_url=server_url, event_hooks={"request": [rollout_prefix_hook([server_url])]}) as client:
        with rollout_context("5-0"):
            response = await asyncio.to_thread(client.post, "/v1/responses")
    assert response.json()["rollout_id"] == "5-0"


async def test_a_shared_async_client_tags_each_concurrent_request_with_its_own_rollout() -> None:
    base_url = "http://gym-model"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_echo_app()),
        base_url=base_url,
        event_hooks={"request": [async_rollout_prefix_hook([base_url])]},
    ) as client:

        async def call(rollout_id: str) -> str:
            with rollout_context(rollout_id):
                await asyncio.sleep(0)
                return (await client.post("/v1/responses")).json()["rollout_id"]

        rollout_ids = [f"{task}-0" for task in range(20)]
        assert await asyncio.gather(*(call(rollout_id) for rollout_id in rollout_ids)) == rollout_ids


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync hook", "async hook"])
async def test_a_request_sent_again_is_rewritten_for_the_current_rollout(asynchronous: bool) -> None:
    """A retried or replayed request object must not keep the rollout of its first send."""
    base_url = "http://gym-model"
    hook = async_rollout_prefix_hook([base_url]) if asynchronous else rollout_prefix_hook([base_url])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_echo_app()), base_url=base_url) as client:
        request = client.build_request("POST", "/v1/responses")

        async def send(rollout_id: str | None) -> dict:
            if rollout_id is None:
                if asynchronous:
                    await hook(request)
                else:
                    hook(request)
            else:
                with rollout_context(rollout_id):
                    if asynchronous:
                        await hook(request)
                    else:
                        hook(request)
            return (await client.send(request)).json()

        seen = [(await send(rollout_id))["rollout_id"] for rollout_id in ("first", "second", None)]
    assert seen == ["first", "second", None]


@pytest.mark.parametrize(
    ("location", "expected_urls"),
    [
        pytest.param(
            "done",
            ["http://gym-model/ng-rollout/r/start", "http://gym-model/ng-rollout/r/done"],
            id="relative to the prefixed path, which already carries the prefix",
        ),
        pytest.param(
            "/done",
            ["http://gym-model/ng-rollout/r/start", "http://gym-model/ng-rollout/r/done"],
            id="absolute path under the base",
        ),
        pytest.param(
            "http://gym-model/done",
            ["http://gym-model/ng-rollout/r/start", "http://gym-model/ng-rollout/r/done"],
            id="same origin, absolute",
        ),
        pytest.param(
            "http://elsewhere/done",
            ["http://gym-model/ng-rollout/r/start", "http://elsewhere/done"],
            id="cross origin",
        ),
    ],
)
async def test_a_redirect_is_followed_and_rewritten_on_its_own_terms(location: str, expected_urls: list[str]) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("/start"):
            return httpx.Response(307, headers={"Location": location})
        return httpx.Response(200, json={"ok": True})

    base_url = "http://gym-model"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=base_url,
        follow_redirects=True,
        event_hooks={"request": [async_rollout_prefix_hook([base_url])]},
    ) as client:
        with rollout_context("r"):
            response = await client.post("/start")
    assert response.status_code == 200
    assert seen == expected_urls
