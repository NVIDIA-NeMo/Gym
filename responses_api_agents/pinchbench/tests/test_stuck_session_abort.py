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
"""OpenClaw's stalled-run abort threshold (``diagnostics.stuckSessionAbortMs``).

Left unset, OpenClaw abort-drains a stalled run after at least 5 minutes and 3x
the warn threshold. A saturated model endpoint makes ordinary generations exceed
that, so sessions are cut mid-task and the partial work grades as a failure
rather than surfacing as a timeout.
"""

from responses_api_agents.pinchbench.tests.test_app import make_agent


def test_threshold_is_not_sent_when_unset():
    env = make_agent()._task_env("task_x")

    assert "PINCHBENCH_STUCK_SESSION_ABORT_SECONDS" not in env


def test_threshold_reaches_the_sandbox_in_seconds():
    env = make_agent(openclaw_stuck_session_abort_seconds=3600)._task_env("task_x")

    assert env["PINCHBENCH_STUCK_SESSION_ABORT_SECONDS"] == "3600"


def test_threshold_is_independent_of_the_provider_timeout():
    env = make_agent(openclaw_provider_timeout_seconds=14400)._task_env("task_x")

    assert env["PINCHBENCH_PROVIDER_TIMEOUT_SECONDS"] == "14400"
    assert "PINCHBENCH_STUCK_SESSION_ABORT_SECONDS" not in env


def test_the_sandbox_writes_it_to_openclaw_config_as_milliseconds(tmp_path):
    wrapper = make_agent()._write_direct_exec_wrapper(tmp_path).read_text()

    assert "PINCHBENCH_STUCK_SESSION_ABORT_SECONDS" in wrapper
    assert 'cfg.setdefault("diagnostics", {})["stuckSessionAbortMs"] = stuck_abort_s * 1000' in wrapper
