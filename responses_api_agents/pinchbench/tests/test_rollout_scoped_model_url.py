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
middleware forwards them unattributed and the run ends with an empty capture directory.
"""

from nemo_gym.global_config import (
    OBSERVABILITY_ENABLED_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from responses_api_agents.pinchbench.tests.test_app import make_agent


ROLLOUT = "abc123"


def _observable(agent, enabled=True):
    """Give the agent a global config; capture gating is read from there, not the agent config."""
    agent.server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: enabled}
    return agent


def test_prefix_is_inserted_ahead_of_the_api_version():
    agent = make_agent(model_base_url="http://host:8000/v1")

    assert agent._task_env("task_x", ROLLOUT)["MODEL_BASE_URL"] == f"http://host:8000/ng-rollout/{ROLLOUT}/v1"


def test_opting_out_leaves_the_url_bare():
    agent = make_agent(model_base_url="http://host:8000/v1", rollout_scoped_model_url=False)

    assert agent._task_env("task_x", ROLLOUT)["MODEL_BASE_URL"] == "http://host:8000/v1"


def test_without_a_rollout_id_the_url_is_unchanged():
    agent = make_agent(model_base_url="http://host:8000/v1")

    assert agent._task_env("task_x")["MODEL_BASE_URL"] == "http://host:8000/v1"


def test_a_url_with_no_api_version_still_gets_the_prefix():
    agent = make_agent(model_base_url="http://host:8000")

    assert agent._task_env("task_x", ROLLOUT)["MODEL_BASE_URL"] == f"http://host:8000/ng-rollout/{ROLLOUT}"


def test_a_trailing_slash_does_not_produce_a_double_slash():
    agent = make_agent(model_base_url="http://host:8000/v1/")

    assert agent._task_env("task_x", ROLLOUT)["MODEL_BASE_URL"] == f"http://host:8000/ng-rollout/{ROLLOUT}/v1"


def test_the_sandbox_spec_carries_the_scoped_url():
    agent = make_agent(model_base_url="http://host:8000/v1")

    assert agent._build_spec("task_x", ROLLOUT).env["MODEL_BASE_URL"].endswith(f"/ng-rollout/{ROLLOUT}/v1")


def test_the_id_is_gyms_rollout_id_not_a_fresh_uuid():
    """The prefix must key capture files to the collected rollout.

    ``run()`` also mints a uuid to name its work and transcript directories; prefixing with
    that instead writes ``<uuid>.capture.jsonl`` files that no rollout can be matched back to.
    """
    agent = _observable(make_agent(model_base_url="http://host:8000/v1"))
    body = {TASK_INDEX_KEY_NAME: 12, ROLLOUT_INDEX_KEY_NAME: 0}

    assert agent.rollout_id_from_run(body) == "12-0"
    assert agent._task_env("task_x", "12-0")["MODEL_BASE_URL"] == "http://host:8000/ng-rollout/12-0/v1"


def test_capture_disabled_yields_no_rollout_id():
    """Gating lives in ``rollout_id_from_run``, so the default-on prefix costs nothing when off."""
    agent = _observable(make_agent(model_base_url="http://host:8000/v1"), enabled=False)
    body = {TASK_INDEX_KEY_NAME: 12, ROLLOUT_INDEX_KEY_NAME: 0}

    assert agent.rollout_id_from_run(body) is None
    assert agent._task_env("task_x", None)["MODEL_BASE_URL"] == "http://host:8000/v1"
