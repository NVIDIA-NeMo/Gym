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
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from resources_servers.collab_intel.app import CollabIntelConfig, CollabIntelResourcesServer, CollabIntelVerifyRequest


RESPONSE = {
    "id": "collab_intel",
    "object": "response",
    "created_at": 0,
    "model": "policy_model",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


def _server() -> CollabIntelResourcesServer:
    config = CollabIntelConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return CollabIntelResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(result: dict) -> CollabIntelVerifyRequest:
    return CollabIntelVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "x"}]}, response=RESPONSE, collab_intel=result
    )


@pytest.mark.parametrize(
    "result, reward, alone_pass, test_pass, unfinished, mask_sample",
    [
        ({"alone_pass": True, "alone": {"pass": True}, "test": {"pass": True}}, 1.0, True, True, False, False),
        # a completed test failure is a measured zero
        ({"alone_pass": True, "alone": {"pass": True}, "test": {"pass": False}}, 0.0, True, False, False, False),
        # failed alone: a measured, unmasked row that collab_intel/score leaves out by alone_pass
        ({"alone_pass": False, "alone": {"pass": False}, "test": None}, 0.0, False, None, False, False),
        # a crashed harness or a missing verdict is unfinished, not a failure
        (
            {"alone_pass": True, "alone": {"pass": True}, "test": {"pass": False, "failure_kind": "agent_run_error"}},
            0.0,
            True,
            None,
            True,
            True,
        ),
        (
            {"alone_pass": False, "alone": {"pass": False, "failure_kind": "judge_failed"}, "test": None},
            0.0,
            False,
            None,
            True,
            True,
        ),
        # the policy's own timeout or context limit is a measured fail
        (
            {"alone_pass": True, "alone": {"pass": True}, "test": {"pass": False, "failure_kind": "agent_timeout"}},
            0.0,
            True,
            False,
            False,
            False,
        ),
        (
            {"alone_pass": False, "alone": {"pass": False, "failure_kind": "model_context_limit"}, "test": None},
            0.0,
            False,
            None,
            False,
            False,
        ),
    ],
)
async def test_verify(result, reward, alone_pass, test_pass, unfinished, mask_sample):
    response = await _server().verify(_request(result))
    assert (response.reward, response.alone_pass, response.test_pass, response.unfinished, response.mask_sample) == (
        reward,
        alone_pass,
        test_pass,
        unfinished,
        mask_sample,
    )
    last = result["test"] or result["alone"]
    assert response.failure_kind == last.get("failure_kind")
