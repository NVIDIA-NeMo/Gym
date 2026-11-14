# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import aiohttp
import pytest

import nemo_gym.client.http_client as http_client
from resources_servers.mcp_atlas.credentials import SERVICE_ENV_NAMES


@pytest.fixture(autouse=True)
async def global_aiohttp_client(monkeypatch):
    """Gym's shared aiohttp client, bound to this test's event loop and built without a run config."""
    session = aiohttp.ClientSession()
    monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT", session)
    yield session
    await session.close()


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    from resources_servers.mcp_atlas import sandbox

    monkeypatch.setattr(sandbox, "POLL_INTERVAL_S", 0.01)


@pytest.fixture(autouse=True)
def isolated_workload_environment(monkeypatch):
    """Tests supply synthetic service credentials and inherited aliases explicitly."""
    for name in list(os.environ):
        if name in SERVICE_ENV_NAMES or name.startswith(("APPTAINERENV_", "SINGULARITYENV_")):
            monkeypatch.delenv(name, raising=False)
