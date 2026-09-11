# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ServerDisconnectedError
from omegaconf import DictConfig

import nemo_gym.rollout_collection as collection
import nemo_gym.server_utils as transport
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.server_utils import ServerClient


@pytest.mark.parametrize("retry", [False, True])
async def test_run_dispatch_preserves_single_attempt_and_ordinary_retry_defaults(monkeypatch, retry):
    client = ServerClient(
        head_server_config=BaseServerConfig(host="head", port=12345),
        global_config_dict=DictConfig(
            {"agent": {"responses_api_agents": {"simple_agent": {"host": "agent", "port": 12346}}}}
        ),
    )
    monkeypatch.setattr(collection, "setup_server_client_utils", lambda _: client)
    effects = []
    response = MagicMock()
    response.ok = True
    response.read = AsyncMock(return_value=b'{"reward":1}')

    async def execute_then_lose_response(**kwargs):
        effects.append(kwargs)
        if len(effects) == 1:
            raise ServerDisconnectedError("agent executed; response lost")
        return response

    monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: MagicMock(request=execute_then_lose_response))
    monkeypatch.setattr(transport.asyncio, "sleep", AsyncMock())
    row = {"agent_ref": {"name": "agent"}, "_ng_rollout_id": "group_g0"}
    helper = RolloutCollectionHelper()
    if retry:
        results = [await task for task in helper.run_examples([row])]
        assert results == [(row, {"reward": 1})]
        assert len(effects) == 2
    else:
        with pytest.raises(ServerDisconnectedError, match="response lost"):
            for task in helper.run_examples([row], retry_requests=False):
                await task
        assert len(effects) == 1
    assert all(call["method"] == "POST" and call["url"].endswith("/run") for call in effects)
