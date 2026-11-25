# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode's agent session: how a turn's failures reach the episode."""

import aiohttp

from environment_servers.bird_interact.tests.test_app import (
    _run,
    _server,
    _Tools,
    sandbox_workspace,  # noqa: F401 (an autouse fixture)
)


async def test_a_socket_timeout_in_a_turn_fails_the_episode_at_the_agent_stage_for_a_retry() -> None:
    # aiohttp's socket and connection timeouts are TimeoutErrors that come before the turn's deadline: the agent's
    # failure, not the turn's deadline and not an unhandled error.
    async def unreachable(_: _Tools) -> str:
        raise aiohttp.ServerTimeoutError("Timeout on reading data from socket")

    server, client = _server(turns=[unreachable])

    response = await _run(server)

    assert (response.failure.stage, response.failure.terminal) == ("agent", False), response.failure
    assert client.closes == 1
