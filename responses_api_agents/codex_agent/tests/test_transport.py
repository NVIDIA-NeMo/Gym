# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket

import pytest
from aiohttp import web

from nemo_gym.client import get_global_aiohttp_client, request
from responses_api_agents.codex_agent.transport import model_retry_proxy, retry_delay


@pytest.mark.parametrize(
    "statuses",
    [(429, 429, 200), (429, 429, 429), (408, 425, 200), (500, 503, 503), (400,), (401,), (404,), (409,), (422,)],
)
async def test_only_statuses_that_may_clear_retry_and_every_attempt_keeps_body_and_owner(
    statuses, monkeypatch
) -> None:
    monkeypatch.setattr("nemo_gym.client.http_client.get_global_config_dict", lambda **kwargs: {})
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
    seen = []

    async def upstream(incoming):
        seen.append((await incoming.read(), incoming.headers.getall("x-session-id")))
        return web.Response(status=statuses[len(seen) - 1], body=b"payload", headers={"Retry-After": "0"})

    app = web.Application()
    app.router.add_post("/responses", upstream)
    runner = web.AppRunner(app)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        await runner.setup()
        try:
            await web.SockSite(runner, sock).start()
            async with model_retry_proxy(f"http://127.0.0.1:{sock.getsockname()[1]}", "owner") as url:
                response = await request(
                    "POST", f"{url}/responses", data=b'{"input":[]}', headers={"X-Session-Id": "wrong"}
                )
                try:
                    assert response.status == statuses[-1]
                    assert await response.read() == b"payload"
                finally:
                    response.release()
            assert seen == [(b'{"input":[]}', ["owner"])] * len(statuses)
        finally:
            await runner.cleanup()
            await get_global_aiohttp_client().close()


@pytest.mark.parametrize(
    ("retry_after", "delays"),
    [
        (None, [1, 2]),
        ("0", [0, 0]),
        (" 2.5 ", [2.5, 2.5]),
        ("45", [10, 10]),
        ("-3", [0, 0]),
        ("nan", [1, 2]),
        ("inf", [1, 2]),
        ("soon", [1, 2]),
    ],
)
def test_retry_delay_is_bounded_and_invalid_values_back_off(retry_after, delays) -> None:
    assert [retry_delay(retry_after, attempt) for attempt in (0, 1)] == delays


async def test_refused_connection_is_retried_twice_then_answers_502(monkeypatch) -> None:
    monkeypatch.setattr("nemo_gym.client.http_client.get_global_config_dict", lambda **kwargs: {})
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
    waits = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("responses_api_agents.codex_agent.transport.asyncio.sleep", sleep)
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    try:
        async with model_retry_proxy(f"http://127.0.0.1:{port}", "owner") as url:
            response = await request("POST", f"{url}/responses", data=b"{}")
            try:
                assert response.status == 502
            finally:
                response.release()
        # The patch reaches every asyncio.sleep; the forwarder's own waits are the nonzero ones.
        assert [wait for wait in waits if wait] == [1, 2]
    finally:
        await get_global_aiohttp_client().close()
