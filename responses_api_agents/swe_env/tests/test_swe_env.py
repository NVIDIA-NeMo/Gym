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

"""Unit tests for the swe_env library, driven by a FakeSandbox provider."""

from __future__ import annotations

import asyncio

import responses_api_agents.swe_env.harnesses  # noqa: F401  (registers harnesses)
from nemo_gym.sandbox import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from resources_servers.swe_env.verify_task import ProviderCapabilityError, verify_task
from responses_api_agents.swe_env import (
    compute_resolved,
    get_harness,
    list_harnesses,
    reward_from_report,
)
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask
from responses_api_agents.swe_env.harnesses.swe_bench_ext import SweBenchExtHarness


# Trailing-status pytest text (``<node_id> PASSED``) is the format the test
# parser recognizes; node ids carry a ``.py`` path so they normalize to the
# F2P/P2P ids below.
_PASS_OUTPUT = "tests/test_x.py::a PASSED\ntests/test_x.py::b PASSED\n"
_F2P_FAIL_OUTPUT = "tests/test_x.py::a FAILED\ntests/test_x.py::b PASSED\n"


class _FakeProvider:
    """Scripted provider: pytest commands return a canned transcript."""

    name = "fake-swe"

    def __init__(self, *, test_output="", test_rc=0, apply_rc=0, create_error=False, **_):
        """Configure the scripted provider's responses.

        Args:
            test_output: Stdout returned for pytest commands.
            test_rc: Return code returned for pytest commands.
            apply_rc: Return code returned for ``git apply`` commands.
            create_error: When True, ``create`` raises a SandboxCreateError.
            **_: Ignored extra keyword arguments.
        """
        self._test_output = test_output
        self._test_rc = test_rc
        self._apply_rc = apply_rc
        self._create_error = create_error

    async def create(self, spec):
        if self._create_error:
            raise SandboxCreateError("simulated create failure")
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "pytest" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=self._test_rc)
        if "git apply" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=self._apply_rc)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, *a, **k):
        return None

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        return None

    async def aclose(self):
        return None


register_provider("fake-swe", _FakeProvider, override=True)


def _task(**overrides) -> SweTask:
    """Build a SweTask with sensible defaults, overridable per keyword.

    Args:
        **overrides: Field overrides merged onto the default task fields.

    Returns:
        A SweTask configured for the swe-bench-ext benchmark.
    """
    base = dict(
        instance_id="inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        test_command="python -m pytest -rA -q",
        model_patch="diff --git a/x b/x\n",
        test_framework="pytest",
        fail_to_pass=["tests/test_x.py::a"],
        pass_to_pass=["tests/test_x.py::b"],
        benchmark="swe-bench-ext",
    )
    base.update(overrides)
    return SweTask(**base)


# ---- pure helpers -----------------------------------------------------------


def test_compute_resolved():
    """``compute_resolved`` is True only when all required tests are in the passed set."""
    assert compute_resolved(fail_to_pass=["a"], pass_to_pass=["b"], passed=["a", "b"]) is True
    assert compute_resolved(fail_to_pass=["a"], pass_to_pass=["b"], passed=["a"]) is False
    assert compute_resolved(fail_to_pass=[], pass_to_pass=[], passed=["a"]) is False


def test_reward_from_report():
    """``reward_from_report`` is 1.0 for a resolved report and 0.0 otherwise or when masked."""
    assert reward_from_report(SweEvalReport(instance_id="i", resolved=True)) == 1.0
    assert reward_from_report(SweEvalReport(instance_id="i", resolved=False)) == 0.0
    assert reward_from_report(SweEvalReport(instance_id="i", resolved=True, error_kind="sandbox")) == 0.0


def test_registry_and_build_spec():
    """The swe-bench-ext harness is registered and builds the expected sandbox spec."""
    assert "swe-bench-ext" in list_harnesses()
    harness = get_harness("swe-bench-ext")
    assert isinstance(harness, SweBenchExtHarness)
    spec = harness.build_spec(_task())
    assert spec.image == "img:tag"
    assert spec.workdir == "/testbed"
    assert spec.metadata["instance_id"] == "inst-1"


def test_grade_masks_on_infra_error():
    """Grading masks an infra error to reward 0.0 and records its error kind."""
    harness = get_harness("swe-bench-ext")
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout"}))
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


# ---- verify_task orchestrator (fresh-sandbox, FakeProvider) -----------------


def test_verify_task_resolved():
    """``verify_task`` resolves a task whose required tests all pass."""
    provider = {"fake-swe": {"test_output": _PASS_OUTPUT, "test_rc": 0}}
    report = asyncio.run(verify_task(provider, _task()))
    assert report.resolved is True
    assert report.patch_applied is True
    assert reward_from_report(report) == 1.0


def test_verify_task_unresolved():
    """``verify_task`` leaves a task unresolved when a required test fails."""
    provider = {"fake-swe": {"test_output": _F2P_FAIL_OUTPUT, "test_rc": 1}}
    report = asyncio.run(verify_task(provider, _task()))
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_verify_task_empty_patch_fast_path():
    """An empty model patch short-circuits to an unresolved report."""
    report = asyncio.run(verify_task({"fake-swe": {}}, _task(model_patch="")))
    assert report.patch_exists is False
    assert report.resolved is False


def test_verify_task_infra_error_masked():
    """A sandbox creation failure is masked to reward 0.0 with a sandbox error kind."""
    report = asyncio.run(verify_task({"fake-swe": {"create_error": True}}, _task()))
    assert report.error_kind == "sandbox"
    assert reward_from_report(report) == 0.0


def test_verify_task_golden():
    """Running with ``run_golden`` applies the golden patch and resolves the task."""
    provider = {"fake-swe": {"test_output": _PASS_OUTPUT}}
    task = _task(model_patch="", metadata={"golden_patch": "diff --git a/x b/x\n"})
    report = asyncio.run(verify_task(provider, task, run_golden=True))
    assert report.resolved is True


def test_verify_task_patch_apply_failure_does_not_gate_resolved():
    """A failed patch apply is recorded but does not gate ``resolved``.

    The patch is applied best-effort and grading is based on the tests only, so a
    failed apply (patch_applied=False) does not flip a tests-passing run to
    unresolved.
    """
    provider = {"fake-swe": {"test_output": _PASS_OUTPUT, "apply_rc": 1}}
    report = asyncio.run(verify_task(provider, _task()))
    assert report.patch_applied is False
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_unsupported_provider_raises():
    """``verify_task`` raises when the harness does not support the given provider."""

    class _NestedOnly(SweBenchExtHarness):
        name = "nested-only-test"

        def supports_provider(self, provider_name: str) -> bool:
            """Report support for every provider except ``fake-swe``.

            Args:
                provider_name: The provider name being checked.

            Returns:
                True for any provider other than ``fake-swe``.
            """
            return provider_name != "fake-swe"

    from responses_api_agents.swe_env.registry import register_harness

    register_harness(_NestedOnly(), override=True)
    task = _task(benchmark="nested-only-test")
    try:
        asyncio.run(verify_task({"fake-swe": {}}, task))
    except ProviderCapabilityError:
        return
    raise AssertionError("expected ProviderCapabilityError")
