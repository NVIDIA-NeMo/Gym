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

"""Unit tests for the swe-bench / swe-bench-multilingual flat (host-graded) harness.

The harness runs the instance's eval script in the sandbox and grades the produced log
host-side (swebench's per-repo parser, falling back to the generic flat parser), so it runs on
any exec-capable provider. These tests validate provisioning (``build_spec`` / ``materialize``),
the flat ``run_eval`` + ``grade`` path, and family validation, against a scripted ``FakeSandbox``.
"""

from __future__ import annotations

import asyncio

import pytest

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask, reward_from_report
from responses_api_agents.swe_env.harnesses.swebench import SweBenchHarness


# Canned eval-script log with the SWE-bench sentinels + pytest-style passing lines.
_PASSING_LOG = ">>>>> Start Test Output\nPASSED t::a\nPASSED t::b\n>>>>> End Test Output\n"


class _FakeProvider:
    """Scripted provider: returns a canned eval log for the eval-script run; records uploads.

    Args:
        log_text: Text returned by the eval-script (``bash``) and ``cat`` commands.
        exec_rc: Return code for the eval-script command.
    """

    name = "fake-swebench"

    def __init__(self, *, log_text="", exec_rc=0, **_):
        self._log_text = log_text
        self._exec_rc = exec_rc
        self.uploaded: dict[str, str] = {}
        self.commands: list[str] = []

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
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


register_provider("fake-swebench", _FakeProvider, override=True)


def _task(**overrides) -> SweTask:
    """Build a swe-bench ``SweTask`` with sensible defaults."""
    base = dict(
        instance_id="repo__inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="swe-bench",
        split="test",
    )
    base.update(overrides)
    return SweTask(**base)


def test_grade_strategy_is_flat():
    assert SweBenchHarness("swe-bench").grade_strategy == "flat-host-grade"
    assert SweBenchHarness("swe-bench-multilingual").grade_strategy == "flat-host-grade"


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        SweBenchHarness("not-a-family")


def test_build_spec_image_workdir_metadata():
    spec = SweBenchHarness("swe-bench").build_spec(_task())
    assert spec.image == "img:tag"
    assert spec.workdir == "/testbed"
    assert spec.metadata["instance_id"] == "repo__inst-1"
    assert spec.metadata["harness"] == "swe-bench"


def test_build_spec_preserves_task_provider_options():
    spec = SweBenchHarness("swe-bench").build_spec(_task(metadata={"provider_options": {"network": "host"}}))
    assert spec.provider_options.get("network") == "host"


def test_supports_provider_any_exec_capable():
    harness = SweBenchHarness("swe-bench")
    assert harness.supports_provider("docker") is True
    assert harness.supports_provider("apptainer") is True
    assert harness.supports_provider("opensandbox") is True


def test_with_flat_eval_is_self():
    harness = SweBenchHarness("swe-bench")
    assert harness.with_flat_eval() is harness


def test_materialize_writes_patch_diff():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task()
        env = await AsyncSweEnvironment.start({"fake-swebench": {}}, harness.build_spec(task))
        await harness.materialize(env, task)
        return env.sandbox._provider

    provider = asyncio.run(run())
    assert provider.uploaded.get("/root/patch.diff") == "diff --git a/x b/x\n"


def test_materialize_empty_patch_writes_nothing():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task(model_patch="")
        env = await AsyncSweEnvironment.start({"fake-swebench": {}}, harness.build_spec(task))
        await harness.materialize(env, task)
        return env.sandbox._provider

    provider = asyncio.run(run())
    assert "/root/patch.diff" not in provider.uploaded


def test_run_eval_then_grade_flat_resolved():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    # eval_script preset so flat_run_eval executes it; no instance_dict -> grade falls back to
    # the generic flat parser over the canned passing log.
    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task(metadata={"eval_script": "echo run"})
        env = await AsyncSweEnvironment.start({"fake-swebench": {"log_text": _PASSING_LOG}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_run_eval_missing_eval_script_is_unmasked_unresolved():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    # No instance_dict + no preset eval_script -> _flat_eval_script returns "" -> the run tags an
    # eval_error, but grading no longer masks it: per main an unbuildable/empty spec grades as a
    # legitimate unmasked unresolved (reward 0), not an eval_error mask.
    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task()
        env = await AsyncSweEnvironment.start({"fake-swebench": {}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.error_kind is None
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_infra_error():
    report = SweBenchHarness("swe-bench").grade(_task(), EvalArtifacts(raw={"error_type": "timeout"}))
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


def test_flat_eval_script_empty_without_instance_dict():
    assert SweBenchHarness("swe-bench")._flat_eval_script(_task()) == ""


def test_grade_fails_loud_when_swebench_unavailable(monkeypatch):
    """A SWE-bench instance whose ``swebench`` install is missing fails loud, not silent-degrade.

    Degrading to the generic pytest-only parser would mis-score non-pytest repos (e.g. django) as
    unresolved, silently skewing the resolve rate. Instead grading raises ``GraderDependencyError``
    so the misconfiguration surfaces.
    """
    import sys

    from responses_api_agents.swe_env.harness import GraderDependencyError

    # Simulate a missing / broken swebench install for the import inside _swebench_flat_grade.
    monkeypatch.setitem(sys.modules, "swebench.harness.constants", None)
    harness = SweBenchHarness("swe-bench")
    task = _task(metadata={"instance_dict": {"instance_id": "repo__inst-1", "repo": "x/y"}})
    artifacts = EvalArtifacts(test_output=_PASSING_LOG, return_code=0, raw={})
    with pytest.raises(GraderDependencyError):
        harness.grade(task, artifacts)
