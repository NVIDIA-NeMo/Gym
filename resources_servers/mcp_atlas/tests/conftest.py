# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import aiohttp
import pytest

from nemo_gym import server_utils


@pytest.fixture(autouse=True)
async def global_aiohttp_client(monkeypatch):
    """Gym's shared aiohttp client, bound to this test's event loop and built without a run config."""
    session = aiohttp.ClientSession()
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
    yield session
    await session.close()


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    from resources_servers.mcp_atlas import sandbox

    monkeypatch.setattr(sandbox, "POLL_INTERVAL_S", 0.01)
