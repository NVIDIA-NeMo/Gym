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
import asyncio
import socket
import threading
import time
from os import environ
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest
import uvicorn

from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME


# The server's outbound MCP calls lazily build the global aiohttp client, which reads the
# global config. Outside ng_run there is no head server and Hydra would try to parse
# pytest's argv (and SystemExit) — pre-seed the documented child-process env var instead.
environ.setdefault(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "{}")

import nemo_gym.server_utils as server_utils  # noqa: E402
from nemo_gym.server_utils import ServerClient
from resources_servers.enterpriseops_gym.app import (
    EnterpriseOpsGymResourcesServer,
    EnterpriseOpsGymResourcesServerConfig,
)
from resources_servers.enterpriseops_gym.tests.stub_gym import StubGymState, create_stub_gym_app


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def reset_global_aiohttp_client():
    """The pooled aiohttp session binds to the first event loop that uses it; TestClient
    creates a fresh loop per test, so drop the session between tests (same close pattern
    as nemo_gym.server_utils.global_aiohttp_client_exit)."""
    yield
    client = server_utils._GLOBAL_AIOHTTP_CLIENT
    if client is not None:
        try:
            asyncio.run(client.close())
        except Exception:
            pass
        server_utils._GLOBAL_AIOHTTP_CLIENT = None


def start_stub_gym():
    """Start a stub gym server (uvicorn in a daemon thread). Returns (url, state, stop_fn)."""
    state = StubGymState()
    app = create_stub_gym_app(state)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("Stub gym server failed to start")
        time.sleep(0.05)

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=5)

    return f"http://127.0.0.1:{port}", state, stop


@pytest.fixture(scope="session")
def stub_gym():
    """A real HTTP stub gym server the resources server can call."""
    url, state, stop = start_stub_gym()
    yield url, state
    stop()


@pytest.fixture
def gym_env(stub_gym):
    """Per-test view of the stub gym with clean state."""
    url, state = stub_gym
    state.reset()
    return url, state


@pytest.fixture
def make_server() -> Callable[..., EnterpriseOpsGymResourcesServer]:
    def _make(**config_overrides) -> EnterpriseOpsGymResourcesServer:
        config = EnterpriseOpsGymResourcesServerConfig(
            **(
                dict(
                    host="0.0.0.0",
                    port=8080,
                    entrypoint="",
                    name="enterpriseops_gym",
                    seed_sql_root=str(FIXTURES_DIR),
                    janitor_interval_seconds=0,  # tests drive cleanup_expired_sessions directly
                )
                | config_overrides
            )
        )
        return EnterpriseOpsGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    return _make
