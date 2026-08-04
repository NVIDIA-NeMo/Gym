# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Addressing the model server per rollout so model-call capture can attribute calls.

Capture only records calls arriving under ``/ng-rollout/<rollout_id>``; on a bare URL the
middleware forwards them unattributed and the run has no request/response record at all.
"""

from responses_api_agents.pinchbench.tests.test_app import make_agent


ROLLOUT = "abc123"


def test_bare_url_by_default():
    env = make_agent(model_base_url="http://host:8000/v1")._task_env("task_x", ROLLOUT)

    assert env["MODEL_BASE_URL"] == "http://host:8000/v1"


def test_prefix_is_inserted_ahead_of_the_api_version():
    agent = make_agent(model_base_url="http://host:8000/v1", rollout_scoped_model_url=True)

    assert agent._task_env("task_x", ROLLOUT)["MODEL_BASE_URL"] == f"http://host:8000/ng-rollout/{ROLLOUT}/v1"


def test_without_a_rollout_id_the_url_is_unchanged():
    agent = make_agent(model_base_url="http://host:8000/v1", rollout_scoped_model_url=True)

    assert agent._task_env("task_x")["MODEL_BASE_URL"] == "http://host:8000/v1"


def test_a_url_with_no_api_version_still_gets_the_prefix():
    agent = make_agent(model_base_url="http://host:8000", rollout_scoped_model_url=True)

    assert agent._task_env("task_x", ROLLOUT)["MODEL_BASE_URL"] == f"http://host:8000/ng-rollout/{ROLLOUT}"


def test_the_sandbox_spec_carries_the_scoped_url():
    agent = make_agent(model_base_url="http://host:8000/v1", rollout_scoped_model_url=True)

    assert agent._build_spec("task_x", ROLLOUT).env["MODEL_BASE_URL"].endswith(f"/ng-rollout/{ROLLOUT}/v1")
