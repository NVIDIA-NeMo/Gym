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

import ast
import asyncio
from pathlib import Path

import responses_api_agents.swe_env.harnesses  # noqa: F401  (registers harnesses)
from nemo_gym.sandbox import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env import (
    compute_resolved,
    get_harness,
    list_harnesses,
    reward_from_report,
)
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask
from responses_api_agents.swe_env.harnesses.swe_bench_ext import SweBenchExtHarness
from responses_api_agents.swe_env.verify_task import ProviderCapabilityError, verify_task


# Trailing-status pytest text (``<node_id> PASSED``) is the format the test
# parser recognizes; node ids carry a ``.py`` path so they normalize to the
# F2P/P2P ids below.
_PASS_OUTPUT = "tests/test_x.py::a PASSED\ntests/test_x.py::b PASSED\n"
_F2P_FAIL_OUTPUT = "tests/test_x.py::a FAILED\ntests/test_x.py::b PASSED\n"


class _FakeProvider:
    """Scripted provider: pytest commands return a canned transcript."""

    name = "fake-swe"

    def __init__(self, *, test_output="", test_rc=0, apply_rc=0, create_error=False, sink=None, **_):
        """Configure the scripted provider's responses.

        Args:
            test_output: Stdout returned for pytest commands.
            test_rc: Return code returned for pytest commands.
            apply_rc: Return code returned for ``git apply`` commands.
            create_error: When True, ``create`` raises a SandboxCreateError.
            sink: Optional list each created spec is appended to, for asserting on what
                ``verify_task`` passed the provider (e.g. the stamped ``ttl_s``).
            **_: Ignored extra keyword arguments.
        """
        self._test_output = test_output
        self._test_rc = test_rc
        self._apply_rc = apply_rc
        self._create_error = create_error
        self._sink = sink

    async def create(self, spec):
        if self._sink is not None:
            self._sink.append(spec)
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


def test_compute_resolved_fail_only():
    """The ``fail_only`` eval type mirrors swebench's ``check_fail_only``.

    A required test is success UNLESS it is present in the status map AND ==FAILED, so an
    absent test (silent success) still resolves; a present-and-FAILED test does not.
    """
    # Required test absent from the status map -> success (silent) -> resolved.
    assert (
        compute_resolved(fail_to_pass=["a"], pass_to_pass=["b"], passed=[], eval_type="fail_only", status_map={})
        is True
    )
    # A present-and-FAILED required test -> failure -> unresolved.
    assert (
        compute_resolved(
            fail_to_pass=["a"],
            pass_to_pass=["b"],
            passed=["b"],
            eval_type="fail_only",
            status_map={"a": "FAILED", "b": "PASSED"},
        )
        is False
    )
    # Present but not FAILED (e.g. SKIPPED/ERROR) -> success under fail_only -> resolved.
    assert (
        compute_resolved(
            fail_to_pass=["a"],
            pass_to_pass=["b"],
            passed=[],
            eval_type="fail_only",
            status_map={"a": "SKIPPED", "b": "ERROR"},
        )
        is True
    )
    # Empty required set is still unresolved under fail_only (the validated edge).
    assert compute_resolved(fail_to_pass=[], pass_to_pass=[], passed=[], eval_type="fail_only") is False


def test_compute_resolved_pass_and_fail_status_map():
    """The default ``pass_and_fail`` rule with a populated status_map mirrors swebench.

    This is the path that runs for SWE-bench Verified: a required test is a failure only when it
    is absent or its status is FAILED/ERROR; PASSED/XFAIL pass and any other status (SKIPPED/XPASS)
    is neutral (excluded, not a failure). Locking it in guards the swebench-equivalence this PR
    depends on.
    """
    f2p, p2p = ["a"], ["b"]
    # All required tests PASSED -> resolved.
    assert compute_resolved(fail_to_pass=f2p, pass_to_pass=p2p, passed=[], status_map={"a": "PASSED", "b": "PASSED"})
    # A required test FAILED -> unresolved.
    assert not compute_resolved(
        fail_to_pass=f2p, pass_to_pass=p2p, passed=[], status_map={"a": "FAILED", "b": "PASSED"}
    )
    # A required test ERROR -> unresolved.
    assert not compute_resolved(
        fail_to_pass=f2p, pass_to_pass=p2p, passed=[], status_map={"a": "ERROR", "b": "PASSED"}
    )
    # A required test absent from the status_map -> unresolved.
    assert not compute_resolved(fail_to_pass=f2p, pass_to_pass=p2p, passed=[], status_map={"a": "PASSED"})
    # XFAIL passes; SKIPPED/XPASS are neutral (not failures) -> resolved.
    assert compute_resolved(fail_to_pass=f2p, pass_to_pass=p2p, passed=[], status_map={"a": "XFAIL", "b": "SKIPPED"})


def test_agent_adapters_do_not_call_grading_methods():
    """Agent-facing swe_env modules never call the grader-only harness methods.

    ``harness.py`` documents a trust boundary: ``reset_repo`` / ``run_eval`` / ``grade`` are used
    ONLY by the grader (``verify_task``). This AST guard enforces it — the agent adapters
    (``self_drive``, ``sandbox``) must reach grading through ``verify_task``, never by calling
    those methods directly — so the boundary the docstring promises cannot silently regress.
    """
    grading_only = {"reset_repo", "run_eval", "grade"}
    adapter_dir = Path(__file__).resolve().parent.parent
    for module in ("self_drive.py", "sandbox.py"):
        tree = ast.parse((adapter_dir / module).read_text())
        referenced = sorted(
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in grading_only
        )
        assert not referenced, f"{module} calls grader-only methods {referenced}; route grading via verify_task"


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


def test_verify_task_sandbox_create_error_masked():
    """A ``SandboxCreateError`` (e.g. a docker image-pull failure) is masked as error_kind='sandbox'.

    An infra failure provisioning the eval sandbox must not depress the resolve rate or leak into
    the training signal, so it is masked (reward 0.0) rather than reported as an unmasked reward-0.
    """
    report = asyncio.run(verify_task({"fake-swe": {"create_error": True}}, _task()))
    assert report.error_kind == "sandbox"
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_verify_task_generic_eval_failure_unmasked():
    """A non-timeout, non-infra eval-stage exception stays unmasked (resolved=False, reward 0.0).

    Mirrors main's app.py, which catches a generic eval exception, scores reward 0, and leaves the
    sample in the gradient (error_kind=None) — only timeouts and sandbox-create failures are masked.
    """
    from responses_api_agents.swe_env.harness import register_harness

    class _BoomGrade(SweBenchExtHarness):
        name = "boom-grade-test"

        def grade(self, task, artifacts):
            """Raise a generic error during grading to exercise the unmasked branch.

            Args:
                task: The task being graded.
                artifacts: The eval artifacts (unused).

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("boom")

    register_harness(_BoomGrade(), override=True)
    report = asyncio.run(verify_task({"fake-swe": {"test_output": _PASS_OUTPUT}}, _task(benchmark="boom-grade-test")))
    assert report.error_kind is None
    assert report.resolved is False
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

    from responses_api_agents.swe_env.harness import register_harness

    register_harness(_NestedOnly(), override=True)
    task = _task(benchmark="nested-only-test")
    try:
        asyncio.run(verify_task({"fake-swe": {}}, task))
    except ProviderCapabilityError:
        return
    raise AssertionError("expected ProviderCapabilityError")


def test_verify_task_propagates_grader_dependency_error():
    """``verify_task`` propagates ``GraderDependencyError`` instead of swallowing it to reward-0.

    A missing grading dependency (e.g. swebench for a SWE-bench instance) must fail loud rather
    than silently degrade the resolve rate, so it is re-raised, not caught by the unmasked
    eval-stage handler.
    """
    from responses_api_agents.swe_env.harness import GraderDependencyError, register_harness

    class _MissingGrader(SweBenchExtHarness):
        name = "missing-grader-test"

        def grade(self, task, artifacts):
            """Simulate a harness whose required grading dependency is unavailable.

            Args:
                task: The task being graded.
                artifacts: The eval artifacts (unused).

            Raises:
                GraderDependencyError: Always, to exercise the propagation path.
            """
            raise GraderDependencyError("grading dependency missing")

    register_harness(_MissingGrader(), override=True)
    try:
        asyncio.run(verify_task({"fake-swe": {"test_output": _PASS_OUTPUT}}, _task(benchmark="missing-grader-test")))
    except GraderDependencyError:
        return
    raise AssertionError("expected GraderDependencyError to propagate")


def test_verify_task_flat_eval_metadata():
    """``metadata['flat_eval']`` routes grading through the harness's flat variant."""
    provider = {"fake-swe": {"test_output": _PASS_OUTPUT, "test_rc": 0}}
    report = asyncio.run(verify_task(provider, _task(metadata={"flat_eval": True})))
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_verify_task_stamps_ttl_when_unset():
    """``verify_task`` stamps ``ttl_s = eval_timeout_s + slack`` when the harness leaves it unset.

    The stamp lets TTL-honoring backends (opensandbox) self-expire an eval sandbox orphaned by a
    hard crash; harnesses that already set ``ttl_s`` (e.g. swe-bench-ext) keep their own value.
    """
    import dataclasses

    from responses_api_agents.swe_env.harness import register_harness
    from responses_api_agents.swe_env.verify_task import _TTL_SLACK_S

    class _NoTtl(SweBenchExtHarness):
        name = "no-ttl-test"

        def build_spec(self, task):
            """Build the swe-bench-ext spec but clear ``ttl_s`` so verify_task must stamp it.

            Args:
                task: The task to build a spec for.

            Returns:
                The base spec with ``ttl_s`` reset to None.
            """
            return dataclasses.replace(super().build_spec(task), ttl_s=None)

    register_harness(_NoTtl(), override=True)
    captured: list = []
    provider = {"fake-swe": {"test_output": _PASS_OUTPUT, "sink": captured}}
    asyncio.run(verify_task(provider, _task(benchmark="no-ttl-test"), eval_timeout_s=120))
    assert captured, "expected create() to be called with a stamped spec"
    assert captured[-1].ttl_s == 120 + _TTL_SLACK_S


def test_report_to_reward_wrapper():
    """``report_to_reward`` is a thin wrapper that scores a report like ``reward_from_report``."""
    from responses_api_agents.swe_env.verify_task import report_to_reward

    assert report_to_reward(SweEvalReport(instance_id="i", resolved=True)) == 1.0
    assert report_to_reward(SweEvalReport(instance_id="i", resolved=False)) == 0.0
