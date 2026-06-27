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

"""Unit tests for the r2e-gym flat (host-graded) harness.

r2e-gym now grades host-side via the shared flat-eval path (the apptainer-only nested
``run_local_evaluation`` grader was removed when PR #1694 took over the apptainer provider; the
nested re-wiring is tracked for a follow-up PR). These tests cover provisioning, the agent-phase
test-hiding command shape, ``reset_repo``, and the flat ``run_eval`` + ``grade`` path against a
scripted ``_FakeProvider``.
"""

from __future__ import annotations

import asyncio

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.harness import SweTask, reward_from_report
from responses_api_agents.swe_env.harnesses.r2egym import R2EGymHarness


_PASSING_LOG = ">>>>> Start Test Output\nPASSED t::a\nPASSED t::b\n>>>>> End Test Output\n"


class _FakeProvider:
    """Scripted provider: returns a canned eval log for the eval-script run; records uploads."""

    name = "fake-r2egym"

    def __init__(self, *, log_text="", exec_rc=0, **_):
        self._log_text = log_text
        self._exec_rc = exec_rc
        self.uploaded: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        rc = 0 if command.startswith("cat ") else self._exec_rc
        return SandboxExecResult(stdout=self._log_text, stderr="", return_code=rc)

    async def upload_file(self, handle, local_path, remote_path):
        try:
            with open(local_path, encoding="utf-8") as fh:
                self.uploaded[remote_path] = fh.read()
        except OSError:
            self.uploaded[remote_path] = ""
        return None

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        return None

    async def aclose(self):
        return None


register_provider("fake-r2egym", _FakeProvider, override=True)


def _task(**overrides) -> SweTask:
    """Build an r2e-gym ``SweTask`` with sensible defaults."""
    base = dict(
        instance_id="repo__inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="r2e-gym",
        split="test",
    )
    base.update(overrides)
    return SweTask(**base)


def test_harness_identity():
    harness = R2EGymHarness()
    assert harness.name == "r2e-gym"
    assert harness.grade_strategy == "flat-host-grade"


def test_build_spec_image_workdir_metadata():
    spec = R2EGymHarness().build_spec(_task())
    assert spec.image == "img:tag"
    assert spec.workdir == "/testbed"
    assert spec.metadata["harness"] == "r2e-gym"


def test_build_spec_truncates_long_instance_id():
    spec = R2EGymHarness().build_spec(_task(instance_id="x" * 100))
    assert len(spec.metadata["instance_id"]) == 63


def test_supports_provider_any_exec_capable():
    harness = R2EGymHarness()
    assert harness.supports_provider("docker") is True
    assert harness.supports_provider("apptainer") is True


def test_hide_eval_tests_commands_shape():
    commands = R2EGymHarness().hide_eval_tests_commands()
    assert len(commands) == 3
    assert all("r2e_tests" in c for c in commands)


def test_materialize_writes_patch_diff():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = R2EGymHarness()
        task = _task()
        env = await AsyncSweEnvironment.start({"fake-r2egym": {}}, harness.build_spec(task))
        await harness.materialize(env, task)
        return env.sandbox._provider

    provider = asyncio.run(run())
    assert provider.uploaded.get("/root/patch.diff") == "diff --git a/x b/x\n"


def test_run_eval_then_grade_flat_resolved():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = R2EGymHarness()
        task = _task(metadata={"eval_script": "echo run"})
        env = await AsyncSweEnvironment.start({"fake-r2egym": {"log_text": _PASSING_LOG}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_run_eval_missing_eval_script_is_unmasked_unresolved():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    # A missing/unbuildable eval script grades UNMASKED unresolved (reward 0), not eval_error:
    # only genuine sandbox/timeout infra failures are masked.
    async def run():
        harness = R2EGymHarness()
        task = _task()
        env = await AsyncSweEnvironment.start({"fake-r2egym": {}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.error_kind is None
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_reset_repo_is_noop():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = R2EGymHarness()
        task = _task()
        env = await AsyncSweEnvironment.start({"fake-r2egym": {}}, harness.build_spec(task))
        await harness.reset_repo(env, task)  # must not raise

    asyncio.run(run())
