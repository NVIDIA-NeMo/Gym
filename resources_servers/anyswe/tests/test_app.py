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

from unittest.mock import AsyncMock, MagicMock, patch

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.anyswe.app import AnySweConfig, AnySweResourcesServer, AnySweVerifyRequest


def _server() -> AnySweResourcesServer:
    config = AnySweConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="anyswe",
        sandbox_provider={"opensandbox": {}},
    )
    return AnySweResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(metadata: dict) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        metadata=metadata,
    )


def _request(response_metadata: dict) -> AnySweVerifyRequest:
    return AnySweVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], metadata={"docker_image": "img:tag"}
        ),
        response=_response(response_metadata),
        verifier_metadata={
            "instance_id": "django__django-1",
            "test_patch": "diff",
            "fail_to_pass": ["t1"],
            "pass_to_pass": ["t2"],
            "instance_dict": {"instance_id": "django__django-1"},
        },
    )


async def test_verify_resolved_patch_rewards_one():
    report = MagicMock(resolved=True)
    with patch("resources_servers.anyswe.app.verify_task", new=AsyncMock(return_value=report)) as vt:
        result = await _server().verify(_request({"model_patch": "diff --git a b"}))
    assert result.reward == 1.0
    task = vt.call_args.args[1]
    assert task.instance_id == "django__django-1"
    assert task.image == "img:tag"
    assert task.metadata["instance_dict"] == {"instance_id": "django__django-1"}


async def test_verify_unresolved_patch_rewards_zero():
    report = MagicMock(resolved=False)
    with patch("resources_servers.anyswe.app.verify_task", new=AsyncMock(return_value=report)):
        result = await _server().verify(_request({"model_patch": "diff --git a b"}))
    assert result.reward == 0.0


async def test_verify_without_patch_rewards_zero():
    with patch("resources_servers.anyswe.app.verify_task", new=AsyncMock()) as vt:
        result = await _server().verify(_request({}))
    assert result.reward == 0.0
    vt.assert_not_called()


async def test_verify_uses_benchmark_from_metadata():
    report = MagicMock(resolved=True)
    with patch("resources_servers.anyswe.app.verify_task", new=AsyncMock(return_value=report)) as vt:
        req = _request({"model_patch": "diff --git a b"})
        req.verifier_metadata["benchmark"] = "r2e-gym"
        result = await _server().verify(req)
    assert result.reward == 1.0
    assert vt.call_args.args[1].benchmark == "r2e-gym"


def test_r2e_grade_strict_passed_only():
    from resources_servers.anyswe.app import EvalArtifacts, R2EGymHarness, SweTask

    harness = R2EGymHarness()
    task = SweTask(
        instance_id="r2e-1",
        model_patch="diff",
        fail_to_pass=["t_a"],
        pass_to_pass=["t_b"],
        benchmark="r2e-gym",
    )
    log = "PASSED t_a\nXFAIL t_b\n"
    report = harness.grade(task, EvalArtifacts(test_output=log, return_code=0, patch_applied=True, raw={}))
    assert report.resolved is False
    log = "PASSED t_a\nPASSED - t_b\n"
    report = harness.grade(task, EvalArtifacts(test_output=log, return_code=0, patch_applied=True, raw={}))
    assert report.resolved is True
