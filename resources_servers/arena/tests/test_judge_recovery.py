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
"""A judge outage keeps the policy's answer for judging again."""

import json
from unittest.mock import AsyncMock

from aiohttp import ClientConnectionError
from pytest import mark

from nemo_gym.verification import judge_failsafe
from resources_servers.arena.app import ArenaResourcesServer
from resources_servers.arena.scripts.remove_failed_rollouts import is_failed_rollout
from resources_servers.arena.tests import test_app as _server_tests


class TestJudgeRecovery:
    # The server fixtures and request builder of the main server tests, without re-running those tests here.
    config = _server_tests.TestArenaResourcesServer.config
    server = _server_tests.TestArenaResourcesServer.server
    _make_verify_request = _server_tests.TestArenaResourcesServer._make_verify_request

    @mark.parametrize("error_type", [ClientConnectionError, TimeoutError])
    async def test_a_judge_outage_is_a_judge_failure_that_keeps_the_answer(
        self, server: ArenaResourcesServer, error_type
    ):
        request = self._make_verify_request("Paris is the capital.")
        request.category = "lmarena_v3"
        server.server_client.post = AsyncMock(side_effect=error_type("judge unreachable"))

        row = json.loads((await judge_failsafe(server.verify)(body=request)).body)

        assert row["_ng_failure_class"] == "judge_failed" and row["mask_sample"] is True
        assert "judge unreachable" in row["_ng_failure_judge_error"]
        assert row["response"] == request.response.model_dump(mode="json")
        assert server.server_client.post.await_count == 2
        assert all(
            call.kwargs["server_name"] == server.config.judge_model_server.name
            for call in server.server_client.post.await_args_list
        )

    def test_an_unjudged_row_is_still_removed(self) -> None:
        assert is_failed_rollout({"category": "lmarena_v3", "games": None}) is True
