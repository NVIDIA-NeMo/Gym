# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket

import pytest
from aiohttp import web

from nemo_gym.server_utils import get_global_aiohttp_client, request
from responses_api_agents.codex_agent.transport import rate_limit_proxy


@pytest.mark.parametrize("statuses", [(429, 429, 200), (429, 429, 429), (400,), (500,)])
async def test_only_rate_limits_retry_and_every_attempt_keeps_body_and_owner(statuses, monkeypatch) -> None:
    monkeypatch.setattr("nemo_gym.server_utils.get_global_config_dict", lambda **kwargs: {})
    monkeypatch.setattr("nemo_gym.server_utils._GLOBAL_AIOHTTP_CLIENT", None)
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
            async with rate_limit_proxy(f"http://127.0.0.1:{sock.getsockname()[1]}", "owner") as url:
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
