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

"""Unit tests for the opt-in flat (host-graded) eval mode of the nested families.

The suite has three layers:

* Parser unit tests on recorded fixture logs cover the SWE-bench eval-script log
  parser (``parse_eval_log``) on a success log, a failure log, the bad-code logs
  (patch-apply-failed / timeout), a no-markers log, and the
  output-outside-markers fallback. The fixtures use the
  ``>>>>> Start/End Test Output`` shape the SWE-bench eval script emits.

* Flat run_eval and grade via FakeSandbox drive the flat path of both nested
  harnesses (``swe-bench``, ``r2e-gym``) end-to-end with a scripted provider that
  returns a fixture log, asserting ``resolved`` is computed from ``FAIL_TO_PASS``
  / ``PASS_TO_PASS``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask, reward_from_report
from responses_api_agents.swe_env.harnesses import flat_eval
from responses_api_agents.swe_env.harnesses.r2egym import R2EGymHarness
from responses_api_agents.swe_env.harnesses.swebench import SweBenchHarness


_FIXTURES = Path(__file__).parent / "fixtures" / "flat_eval"


def _fixture(name: str) -> str:
    """Read a recorded fixture log by name.

    Fixtures are stored with a ``.txt`` suffix, so a caller may pass either the
    ``.log`` stem name or the real ``.txt`` name.

    Args:
        name: The fixture file name, with either a ``.log`` or ``.txt`` suffix.

    Returns:
        The fixture file contents as text.
    """
    path = _FIXTURES / name
    if not path.exists() and path.suffix == ".log":
        path = path.with_suffix(".txt")
    return path.read_text()


# ---- parser: recorded fixture logs (CI) -------------------------------------


def test_parse_success_log_all_pass():
    """A success log parses to a status map with the expected passed and skipped tests."""
    status_map, applied = flat_eval.parse_eval_log(_fixture("resolved_success.log"))
    assert applied is True
    assert status_map == {
        "tests/test_ext_autodoc.py::test_format_signature": "PASSED",
        "tests/test_ext_autodoc.py::test_autodoc_inherited": "PASSED",
        "tests/test_ext_autodoc.py::test_autodoc_exclude_members": "PASSED",
        "tests/test_ext_autodoc.py::test_optional_feature": "SKIPPED",
    }
    assert sorted(flat_eval.passed_tests(status_map)) == [
        "tests/test_ext_autodoc.py::test_autodoc_exclude_members",
        "tests/test_ext_autodoc.py::test_autodoc_inherited",
        "tests/test_ext_autodoc.py::test_format_signature",
    ]


def test_parse_failure_log_strips_failed_reason():
    """A failure log parses with the failure reason stripped down to the node id."""
    status_map, applied = flat_eval.parse_eval_log(_fixture("unresolved_failure.log"))
    assert applied is True
    # The "FAILED <id> - <reason>" line keeps only the node id.
    assert status_map["tests/test_ext_autodoc.py::test_format_signature"] == "FAILED"
    assert "tests/test_ext_autodoc.py::test_autodoc_inherited" in flat_eval.passed_tests(status_map)


def test_parse_apply_patch_failed_is_untrusted():
    """A patch-apply-failed log yields an empty status map and patch_applied False."""
    status_map, applied = flat_eval.parse_eval_log(_fixture("apply_patch_failed.log"))
    assert status_map == {}
    assert applied is False


def test_parse_timeout_is_untrusted():
    """A timeout log yields an empty status map and patch_applied False."""
    status_map, applied = flat_eval.parse_eval_log(_fixture("tests_timeout.log"))
    assert status_map == {}
    assert applied is False


def test_parse_no_markers_is_untrusted():
    """A log with no test-output markers yields an empty status map and patch_applied False."""
    status_map, applied = flat_eval.parse_eval_log(_fixture("no_markers.log"))
    assert status_map == {}
    assert applied is False


def test_parse_fallback_outside_markers():
    """Per-test lines appearing after the End marker are recovered by the whole-log fallback."""
    status_map, applied = flat_eval.parse_eval_log(_fixture("fallback_outside_markers.log"))
    assert applied is True
    assert len(flat_eval.passed_tests(status_map)) == 3


def test_parse_duplicate_node_last_status_wins():
    """For a duplicated node id the last reported status wins.

    A node first reported FAILED then re-reported PASSED (e.g. via a rerun plugin)
    ends up PASSED, and vice versa.
    """
    log = "\n".join(
        [
            flat_eval.APPLY_PATCH_PASS,
            flat_eval.START_TEST_OUTPUT,
            "FAILED tests/test_x.py::test_flaky",
            "PASSED tests/test_x.py::test_flaky",
            "PASSED tests/test_x.py::test_regressed",
            "FAILED tests/test_x.py::test_regressed",
            flat_eval.END_TEST_OUTPUT,
        ]
    )
    status_map, applied = flat_eval.parse_eval_log(log)
    assert applied is True
    # Last line wins for each node, not the first.
    assert status_map["tests/test_x.py::test_flaky"] == "PASSED"
    assert status_map["tests/test_x.py::test_regressed"] == "FAILED"
    assert flat_eval.passed_tests(status_map) == ["tests/test_x.py::test_flaky"]


def test_parse_xfail_counts_as_pass():
    """An XFAIL node counts as a passed test."""
    log = "\n".join(
        [
            flat_eval.APPLY_PATCH_PASS,
            flat_eval.START_TEST_OUTPUT,
            "XFAIL tests/test_x.py::test_known_bug",
            "PASSED tests/test_x.py::test_ok",
            flat_eval.END_TEST_OUTPUT,
        ]
    )
    status_map, applied = flat_eval.parse_eval_log(log)
    assert applied is True
    assert set(flat_eval.passed_tests(status_map)) == {
        "tests/test_x.py::test_known_bug",
        "tests/test_x.py::test_ok",
    }


# ---- flat_grade over parsed fixtures (CI) -----------------------------------


def _task(benchmark: str = "swe-bench", **overrides) -> SweTask:
    """Build a SweTask with sensible defaults, overridable per keyword.

    Args:
        benchmark: The benchmark name for the task.
        **overrides: Field overrides merged onto the default task fields.

    Returns:
        A SweTask configured for the given benchmark.
    """
    base = dict(
        instance_id="repo__inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["tests/test_ext_autodoc.py::test_format_signature"],
        pass_to_pass=["tests/test_ext_autodoc.py::test_autodoc_inherited"],
        benchmark=benchmark,
    )
    base.update(overrides)
    return SweTask(**base)


def _flat_artifacts(log: str) -> EvalArtifacts:
    """Wrap an eval log in flat-eval EvalArtifacts.

    Args:
        log: The eval-script log text.

    Returns:
        EvalArtifacts carrying the log with a clean (non-error) flat raw payload.
    """
    return EvalArtifacts(test_output=log, return_code=0, patch_applied=True, raw={"error_type": None, "flat": True})


def test_flat_grade_resolved_on_success():
    """Flat grading resolves a success log with reward 1.0."""
    report = flat_eval.flat_grade(_task(), _flat_artifacts(_fixture("resolved_success.log")))
    assert report.resolved is True
    assert report.patch_applied is True
    assert report.patch_exists is True
    assert reward_from_report(report) == 1.0


def test_flat_grade_unresolved_on_failure():
    """Flat grading leaves a failure log unresolved with reward 0.0."""
    report = flat_eval.flat_grade(_task(), _flat_artifacts(_fixture("unresolved_failure.log")))
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_flat_grade_unresolved_on_apply_failed():
    """A failed patch apply grades as a legitimate unresolved, not an infra mask."""
    report = flat_eval.flat_grade(_task(), _flat_artifacts(_fixture("apply_patch_failed.log")))
    assert report.resolved is False
    assert report.patch_applied is False
    assert report.error_kind is None
    assert reward_from_report(report) == 0.0


# ---- consistency of flat grading --------------------------------------------
#
# Flat grading takes ``resolved`` straight from the parser's verdict (all F2P +
# all P2P passed) and never re-gates it on ``patch_applied``. The parser's
# ``log_patch_applied`` flag never changes ``resolved`` relative to a pure
# ``compute_resolved`` verdict: whenever ``parse_eval_log`` reports
# ``patch_applied=False`` it also returns an empty status map, so
# ``compute_resolved`` already yields False. These tests lock in that invariant
# so a future edit cannot reintroduce a divergent gate.


@pytest.mark.parametrize(
    "fixture_name",
    [
        "resolved_success.log",
        "unresolved_failure.log",
        "apply_patch_failed.log",
        "tests_timeout.log",
        "no_markers.log",
        "fallback_outside_markers.log",
    ],
)
def test_flat_grade_resolved_matches_ungated_compute_resolved(fixture_name):
    """``flat_grade``'s resolved verdict agrees with a bare ``compute_resolved`` over the parsed passed-set.

    The patch-applied gate is redundant and never flips the verdict True<->False.

    Args:
        fixture_name: The recorded fixture log to parse and grade.
    """
    from responses_api_agents.swe_env.harness import compute_resolved

    task = _task()
    log = _fixture(fixture_name)
    status_map, _applied = flat_eval.parse_eval_log(log)
    ungated = compute_resolved(
        fail_to_pass=task.fail_to_pass,
        pass_to_pass=task.pass_to_pass,
        passed=flat_eval.passed_tests(status_map),
    )
    report = flat_eval.flat_grade(task, _flat_artifacts(log))
    assert report.resolved is ungated


@pytest.mark.parametrize(
    "bad_code_attr",
    ["APPLY_PATCH_FAIL", "RESET_FAILED", "TESTS_ERROR", "TESTS_TIMEOUT"],
)
def test_parse_eval_log_bad_code_empties_status_map_even_with_status_lines(bad_code_attr):
    """A bad code forces an empty status map and patch_applied False even with per-test status lines.

    This is what makes the flat_grade patch-applied gate redundant: no path yields
    patch_applied=False together with a non-empty status map.

    Args:
        bad_code_attr: Name of the bad-code marker attribute on ``flat_eval``.
    """
    bad_code = getattr(flat_eval, bad_code_attr)
    log = "\n".join(
        [
            bad_code,
            flat_eval.START_TEST_OUTPUT,
            "PASSED tests/test_ext_autodoc.py::test_format_signature",
            "PASSED tests/test_ext_autodoc.py::test_autodoc_inherited",
            flat_eval.END_TEST_OUTPUT,
        ]
    )
    status_map, applied = flat_eval.parse_eval_log(log)
    assert applied is False
    assert status_map == {}
    # And it grades as a legitimate unresolved (not an infra mask): error_kind
    # stays None, resolved False -> reward 0.0, matching the flat families.
    report = flat_eval.flat_grade(_task(), _flat_artifacts(log))
    assert report.resolved is False
    assert report.error_kind is None
    assert reward_from_report(report) == 0.0


def test_flat_grade_resolved_does_not_gate_on_artifact_patch_applied():
    """Flat ``resolved`` is the parser's verdict only and ignores the artifact's patch_applied flag.

    Even if the EvalArtifacts carries patch_applied False (e.g. the model patch
    did not cleanly apply), a passing eval log still resolves, since grading is
    based on the tests rather than the apply status.
    """
    artifacts = EvalArtifacts(
        test_output=_fixture("resolved_success.log"),
        return_code=0,
        patch_applied=False,
        raw={"error_type": None, "flat": True},
    )
    report = flat_eval.flat_grade(_task(), artifacts)
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_flat_grade_neutral_skipped_required_test_is_not_a_failure():
    """A required test reported SKIPPED is neutral (excluded), not a failure.

    This mirrors swebench's ``get_eval_tests_report`` + ``get_resolution_status``: a
    required test counts as a failure only when absent or FAILED/ERROR. A neutral
    status (SKIPPED/XPASS) is excluded from both the success and failure tallies, so a
    run whose only "non-pass" required test is SKIPPED still resolves. A bare
    ``passed``-set membership check (the prior behavior) would have treated the
    SKIPPED test as a failure and wrongly graded it unresolved.
    """
    log = "\n".join(
        [
            flat_eval.APPLY_PATCH_PASS,
            flat_eval.START_TEST_OUTPUT,
            "PASSED tests/test_ext_autodoc.py::test_format_signature",
            "SKIPPED tests/test_ext_autodoc.py::test_autodoc_inherited",
            flat_eval.END_TEST_OUTPUT,
        ]
    )
    report = flat_eval.flat_grade(_task(), _flat_artifacts(log))
    # F2P passed; the SKIPPED P2P is neutral (excluded) -> zero failures -> resolved.
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_flat_grade_absent_required_test_is_a_failure():
    """A required test absent from the status map is a failure (not neutral).

    Per swebench's ``test_failed`` (``case not in sm``), an absent required test counts
    as a failure, so the run must grade unresolved.
    """
    log = "\n".join(
        [
            flat_eval.APPLY_PATCH_PASS,
            flat_eval.START_TEST_OUTPUT,
            "PASSED tests/test_ext_autodoc.py::test_format_signature",
            flat_eval.END_TEST_OUTPUT,
        ]
    )
    # P2P (test_autodoc_inherited) is absent from the log -> failure -> unresolved.
    report = flat_eval.flat_grade(_task(), _flat_artifacts(log))
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_flat_grade_masks_infra_error():
    """Flat grading masks an infra timeout to reward 0.0 with a timeout error kind."""
    artifacts = EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout", "flat": True})
    report = flat_eval.flat_grade(_task(), artifacts)
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


def test_flat_grade_unbuildable_eval_script_is_unmasked_unresolved():
    """An unbuildable / missing eval script grades UNMASKED unresolved (reward 0), not eval_error.

    Per main, only genuine sandbox/timeout infra failures are masked; an empty/unbuildable eval
    spec produces no test markers and so grades as a legitimate unresolved (``error_kind`` None).
    """
    artifacts = EvalArtifacts(test_output="", return_code=1, raw={"error_type": "eval_error", "flat": True})
    report = flat_eval.flat_grade(_task(), artifacts)
    assert report.error_kind is None
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


# ---- gating (CI) ------------------------------------------------------------


def test_flat_eval_enabled_harness_flag():
    """The harness-level flat-eval flag enables flat eval."""
    assert flat_eval.flat_eval_enabled(True, _task()) is True


def test_flat_eval_enabled_task_metadata():
    """Per-task ``flat_eval`` metadata enables flat eval."""
    assert flat_eval.flat_eval_enabled(False, _task(metadata={"flat_eval": True})) is True


def test_flat_eval_disabled_by_default():
    """Flat eval is disabled when neither the harness flag nor task metadata enables it."""
    assert flat_eval.flat_eval_enabled(False, _task()) is False


def test_swebench_supports_provider_gating():
    """The swe-bench harness is host-graded (flat), so it runs on any exec-capable provider."""
    harness = SweBenchHarness("swe-bench")
    assert harness.supports_provider("docker") is True
    assert harness.supports_provider("apptainer") is True
    assert harness.supports_provider("opensandbox") is True
    assert harness.grade_strategy == "flat-host-grade"


def test_r2egym_supports_provider_gating():
    """The r2e-gym harness is host-graded (flat), so it runs on any exec-capable provider."""
    harness = R2EGymHarness()
    assert harness.supports_provider("docker") is True
    assert harness.supports_provider("apptainer") is True
    assert harness.supports_provider("opensandbox") is True
    assert harness.grade_strategy == "flat-host-grade"


# ---- flat run_eval end-to-end via FakeSandbox (CI) --------------------------


class _FakeFlatProvider:
    """Scripted provider: ``bash eval.sh ...`` streams a fixture log; ``cat`` echoes it."""

    name = "fake-flat-eval"

    def __init__(self, *, log_text="", run_rc=0, error_type=None, stream_empty=False, **_):
        """Configure the scripted flat-eval provider's responses.

        Args:
            log_text: The eval-script log text returned by the run and ``cat``.
            run_rc: Return code returned for the eval-script run.
            error_type: Optional error type attached to the run result.
            stream_empty: When True, the eval-script run streams empty stdout so
                the harness falls back to reading the tee'd log file.
            **_: Ignored extra keyword arguments.
        """
        self._log_text = log_text
        self._run_rc = run_rc
        self._error_type = error_type
        self._stream_empty = stream_empty
        self.commands: list[str] = []
        self.uploaded: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
        if command.startswith("cat "):
            return SandboxExecResult(stdout=self._log_text, stderr="", return_code=0)
        # The eval script run.
        stdout = "" if self._stream_empty else self._log_text
        return SandboxExecResult(stdout=stdout, stderr="", return_code=self._run_rc, error_type=self._error_type)

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


register_provider("fake-flat-eval", _FakeFlatProvider, override=True)


def _drive_flat(harness, task, *, log_text, run_rc=0, error_type=None, stream_empty=False):
    """Drive materialize -> run_eval -> grade for a flat harness via the scripted provider.

    Args:
        harness: The flat-capable harness under test.
        task: The SweTask to evaluate.
        log_text: The eval-script log text the provider returns.
        run_rc: Return code returned for the eval-script run.
        error_type: Optional error type attached to the run result.
        stream_empty: When True, the run streams empty stdout so the harness falls
            back to reading the tee'd log file.

    Returns:
        A tuple of the graded report, the EvalArtifacts, and the provider instance.
    """
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def _go():
        provider = {
            "fake-flat-eval": {
                "log_text": log_text,
                "run_rc": run_rc,
                "error_type": error_type,
                "stream_empty": stream_empty,
            }
        }
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        try:
            await harness.materialize(env, task)
            artifacts = await harness.run_eval(env, task)
            return harness.grade(task, artifacts), artifacts, env.sandbox._provider
        finally:
            await env.cleanup()

    return asyncio.run(_go())


def test_swebench_flat_run_eval_resolved():
    """The swe-bench flat path resolves a success run and uploads the eval script."""
    harness = SweBenchHarness("swe-bench")
    task = _task(metadata={"eval_script": "echo running", "flat_eval": True})
    report, artifacts, provider = _drive_flat(harness, task, log_text=_fixture("resolved_success.log"))
    assert artifacts.raw["flat"] is True
    assert report.resolved is True
    assert reward_from_report(report) == 1.0
    # The eval script was uploaded into the sandbox.
    assert provider.uploaded.get(flat_eval.EVAL_SCRIPT_PATH, "").startswith("echo running")


def test_swebench_flat_run_eval_unresolved():
    """The swe-bench flat path leaves a failure run unresolved."""
    harness = SweBenchHarness("swe-bench")
    task = _task(metadata={"eval_script": "echo running"})
    report, _artifacts, _ = _drive_flat(harness, task, log_text=_fixture("unresolved_failure.log"))
    assert report.resolved is False


def test_swebench_flat_run_eval_stream_empty_uses_log_file():
    """When streamed output is empty, run_eval reads back the tee'd log file."""
    harness = SweBenchHarness("swe-bench")
    task = _task(metadata={"eval_script": "echo running"})
    report, _artifacts, provider = _drive_flat(
        harness, task, log_text=_fixture("resolved_success.log"), stream_empty=True
    )
    assert any(cmd.startswith("cat ") for cmd in provider.commands)
    assert report.resolved is True


def test_swebench_flat_run_eval_masks_sandbox_error():
    """The swe-bench flat path masks a sandbox error reported by the run."""
    harness = SweBenchHarness("swe-bench")
    task = _task(metadata={"eval_script": "echo running"})
    report, artifacts, _ = _drive_flat(harness, task, log_text="", run_rc=1, error_type="sandbox")
    assert artifacts.raw["error_type"] == "sandbox"
    assert report.error_kind == "sandbox"


def test_swebench_flat_run_eval_missing_script_is_unmasked_unresolved():
    """A missing/unbuildable eval script grades UNMASKED unresolved (reward 0), not eval_error.

    ``flat_run_eval`` still tags the artifact ``error_type == "eval_error"`` (so callers can log
    it), but grading no longer masks on it: per main only genuine sandbox/timeout infra failures
    are masked, and an empty spec simply produces no test markers and grades unresolved.
    """
    harness = SweBenchHarness("swe-bench")
    task = _task(metadata={})  # no eval_script
    report, artifacts, _ = _drive_flat(harness, task, log_text="")
    assert artifacts.raw["error_type"] == "eval_error"
    assert report.error_kind is None
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_r2egym_flat_run_eval_resolved_via_task_metadata():
    """Per-task ``flat_eval`` metadata drives the r2e-gym flat path to a resolved run."""
    harness = R2EGymHarness()
    task = _task(benchmark="r2e-gym", instance_id="r2e__pkg-1", metadata={"eval_script": "echo run"})
    report, artifacts, _ = _drive_flat(harness, task, log_text=_fixture("resolved_success.log"))
    assert artifacts.raw["flat"] is True
    assert report.resolved is True
