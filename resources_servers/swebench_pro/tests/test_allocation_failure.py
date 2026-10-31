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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.rollout_collection import _episode_record
from resources_servers.swebench_pro import app as swe
from resources_servers.swebench_pro.tests.test_app import client_with_task_sandbox, make_server, request_body


@pytest.mark.parametrize("golden", [False, True])
@pytest.mark.parametrize("retries", [0, 2])
def test_failed_verifier_allocation_preserves_ungraded_patch(monkeypatch, golden, retries):
    server = make_server(golden=golden, inconclusive_verification_retries=retries)
    server.server_client.global_config_dict = {"observability_enabled": True}
    saved_patch = "saved policy patch"
    monkeypatch.setattr(server, "_extract_model_patch", AsyncMock(return_value=saved_patch))
    create = AsyncMock(side_effect=OSError("allocation unavailable"))
    monkeypatch.setattr(server, "_create_sandbox", create)
    monkeypatch.setattr(server, "_image_info", MagicMock(return_value={}))
    body = request_body()
    body["patch"] = saved_patch
    client = TestClient(server.setup_webserver()) if golden else client_with_task_sandbox(server)
    response = client.post("/ng-rollout/saved-patch/verify", json=body)
    assert response.status_code == 200
    row = response.json()
    assert create.await_count == retries + 1
    assert row["verification_attempts"] == retries + 1
    assert row["model_patch"] == saved_patch
    assert row["response"] == swe.SWEBenchProVerifyRequest.model_validate(body).response.model_dump(mode="json")
    assert row["evaluation_completed"] is False
    assert row["mask_sample"] is True
    assert row["failure_kind"] == "verifier_error"
    # The native episode path rejects collector-owned markers; the same actual HTTP result must remain persistable.
    projected = _episode_record({"task_id": {"taskset": "swe", "task_id": "saved-patch"}, "result": row})
    assert projected["model_patch"] == saved_patch
    assert projected["mask_sample"] is True
    assert projected["verification_attempts"] == retries + 1
    assert projected["failure_kind"] == "verifier_error"
    assert "_ng_failure_class" not in row
    assert "_ng_failure_judge_error" not in row
    assert row["failure_reason"] == "Verification failed: allocation unavailable"
    assert "_ng_failure_terminal" not in row
    (observation,) = row["ng_verifier_invocations"]
    assert observation["status"] == "incomplete"
    assert observation["failure_kind"] == "verifier_error"
    assert observation["invocation_id"].startswith("verify-")


@pytest.mark.parametrize(
    "tests,timed_out", [([], False), ([{"name": "new_test", "status": "FAILED"}], False), ([], True)]
)
def test_valid_wrong_zero_after_allocation_retry_is_still_graded(monkeypatch, tests, timed_out):
    server = make_server(golden=False)
    monkeypatch.setattr(server, "_extract_model_patch", AsyncMock(return_value="saved policy patch"))
    sandbox = SimpleNamespace(stop=AsyncMock())
    create = AsyncMock(side_effect=[OSError("allocation unavailable"), sandbox])
    monkeypatch.setattr(server, "_create_sandbox", create)
    monkeypatch.setattr(server, "_image_info", MagicMock(return_value={}))
    monkeypatch.setattr(
        swe,
        "run_verification",
        AsyncMock(
            return_value=swe.VerificationResult(
                completed=True,
                resolved=False,
                patch_applied=True,
                test_results={"tests": tests},
                timed_out=timed_out,
            )
        ),
    )
    body = request_body() | {
        "_ng_failure_class": "judge_failed",
        "_ng_failure_judge_error": "caller supplied",
        "verification_attempts": 999,
    }
    response = client_with_task_sandbox(server).post("/verify", json=body)
    row = response.json()
    assert create.await_count == 2
    assert row["reward"] == 0
    assert row["evaluation_completed"] is True
    assert row["mask_sample"] is False
    assert row["failure_kind"] is None
    assert "_ng_failure_class" not in row
    assert row["verification_attempts"] == 2
    assert "_ng_failure_judge_error" not in row
    sandbox.stop.assert_awaited_once()
