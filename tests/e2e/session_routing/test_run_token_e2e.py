# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run that requires its token: its servers call each other with it, and a caller without it is refused.

The deployment of ``test_session_routing_e2e`` (the ``/run`` relay over Simple Agent, the counter resources server with
4 workers, and a fake backend) with ``require_run_token: true`` and the run's token in every server's environment, as
the launcher gives it. Skipped unless NEMO_GYM_MULTIWORKER_E2E=1; under a minute.
"""

import asyncio
import os
import secrets
from collections import Counter
from typing import Any, Iterator

import aiohttp
import pytest

from nemo_gym.core.run_token import RUN_TOKEN_ENV, RUN_TOKEN_HEADER
from nemo_gym.server_runtime.base_resources_server import NEMO_GYM_MCP_SESSION_TOKEN_HEADER
from tests.e2e.session_routing.test_session_routing_e2e import EPISODES, Deployment, counter_row


pytestmark = pytest.mark.skipif(
    os.getenv("NEMO_GYM_MULTIWORKER_E2E") != "1", reason="set NEMO_GYM_MULTIWORKER_E2E=1 to run"
)

TOKEN = secrets.token_urlsafe(32)
RPC_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Deployment]:
    deployment = Deployment(tmp_path_factory.mktemp("run_token"))
    deployment.config["require_run_token"] = True
    with pytest.MonkeyPatch.context() as patch:
        # Every server inherits the run's token from its environment, as from the launcher.
        patch.setenv(RUN_TOKEN_ENV, TOKEN)
        try:
            deployment.start()
            yield deployment
        finally:
            deployment.stop()


async def _status(client: aiohttp.ClientSession, url: str, body: Any, headers: dict | None = None) -> int:
    async with client.post(url, json=body, headers=headers or {}) as response:
        await response.read()
        return response.status


def test_episodes_score_with_the_token_passed_from_server_to_server(deployment: Deployment) -> None:
    """The environment server calls the agent, which calls the model and the episode routes of the resources server's
    workers: each call carries the run's token, or the episode would not score."""

    async def run_all() -> list[float]:
        async with aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(), headers={RUN_TOKEN_HEADER: TOKEN}
        ) as client:

            async def one(index: int) -> float:
                async with client.post(f"{deployment.url('environment')}/run", json=counter_row(index)) as response:
                    assert response.status == 200, await response.text()
                    return (await response.json())["reward"]

            return await asyncio.gather(*(one(index) for index in range(EPISODES)))

    assert Counter(asyncio.run(run_all())) == Counter({1.0: EPISODES})


def test_a_caller_without_the_token_reaches_no_orchestration_route(deployment: Deployment) -> None:
    resources = deployment.url("resources")

    async def statuses() -> tuple[list[int], int, int]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:
            refused = [
                await _status(client, f"{deployment.url('environment')}/run", counter_row(0)),
                await _status(
                    client, f"{deployment.url('environment')}/run", counter_row(0), {RUN_TOKEN_HEADER: "another-run"}
                ),
                await _status(client, f"{deployment.url('agent')}/run", counter_row(0)),
                await _status(client, f"{resources}/seed_session", {"initial_count": 0}),
                await _status(client, f"{resources}/verify", {}),
                await _status(client, f"{resources}/close_session", {}),
            ]
            # What code in a sandbox calls with credentials of its own stays open: the MCP endpoint, with the session
            # token a seed hands out, and model inference.
            async with client.post(
                f"{resources}/seed_session", json={"initial_count": 0}, headers={RUN_TOKEN_HEADER: TOKEN}
            ) as seed:
                session_token = (await seed.json())["mcp"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
            listed = await _status(
                client,
                f"{resources}/mcp",
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                {**RPC_HEADERS, NEMO_GYM_MCP_SESSION_TOKEN_HEADER: session_token},
            )
            inference = await _status(
                client,
                f"{deployment.url('policy_model')}/v1/chat/completions",
                {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
            )
            return refused, listed, inference

    refused, listed, inference = asyncio.run(statuses())
    assert refused == [401] * 6
    assert listed == 200
    assert inference != 401
