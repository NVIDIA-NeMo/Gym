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
"""The OpenClaw agent id the harness derives from ``model_name``.

The harness builds ``bench-<slug(model_name)>`` and then looks the agent's
workspace up by that id. An id it cannot match sends every task to a fallback
workspace that is never read back, so the run scores near zero while looking
healthy.
"""

import pytest

from responses_api_agents.pinchbench.tests.test_app import make_agent


WORKABLE = "model-a"
UNRESOLVABLE = "model-" + "b" * 110


def test_a_name_whose_agent_id_cannot_be_resolved_is_refused():
    with pytest.raises(ValueError, match="agent id"):
        make_agent(model_name=UNRESOLVABLE)


def test_a_workable_name_is_accepted():
    assert make_agent(model_name=WORKABLE).config.model_name == WORKABLE


def test_the_limit_counts_the_harness_prefix_not_just_the_name():
    make_agent(model_name="a" * 58)
    with pytest.raises(ValueError, match="agent id"):
        make_agent(model_name="a" * 59)


def test_the_limit_is_configurable():
    assert make_agent(model_name=UNRESOLVABLE, max_agent_id_length=256).config.model_name
