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

"""Tests for the swe-bench-ext harness grading.

These cover two grading behaviors:

* ``grade`` delegates to the vendored lighthouse parser
  (``parse_and_check_tests``) — so junit-xml parsing, ``normalize_test_id`` plus
  4-stage fuzzy matching, the 20+ framework dispatch, and the
  ``::build``/``::compile`` synthetic-PASS injection all drive ``resolved``.
  Recorded fixture logs (one per parser path) anchor the expectation.
* ``resolved`` is the parser's verdict only; a failed ``git apply`` is recorded
  in ``patch_applied`` but never gates ``resolved``.

The harness is flat / host-graded (no nested container), so ``run_eval`` runs
against a scripted ``FakeSandbox`` rather than a real image.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask
from responses_api_agents.swe_env.harnesses.swe_bench_ext import SweBenchExtHarness


_FIXTURES = Path(__file__).parent / "fixtures" / "swe_bench_ext"


def _fixture(name: str) -> str:
    """Read a recorded fixture log by file name.

    Args:
        name: The fixture file name under the ``swe_bench_ext`` fixtures dir.

    Returns:
        str: The fixture file contents.
    """
    return (_FIXTURES / name).read_text()


def _task(**overrides) -> SweTask:
    """Build a swe-bench-ext ``SweTask`` with sensible defaults.

    Args:
        **overrides: Field values overriding the defaults.

    Returns:
        SweTask: A task populated from the defaults merged with overrides.
    """
    base = dict(
        instance_id="repo__inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        test_command="python -m pytest -rA -q",
        test_framework="pytest",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["tests/test_core.py::test_fix_applied"],
        pass_to_pass=["tests/test_core.py::test_regression_guard"],
        benchmark="swe-bench-ext",
    )
    base.update(overrides)
    return SweTask(**base)


def _artifacts(test_output: str, *, patch_applied: bool = True, error_type=None) -> EvalArtifacts:
    """Build ``EvalArtifacts`` for a graded run.

    Args:
        test_output: The captured test transcript handed to the parser.
        patch_applied: Whether the model patch applied cleanly.
        error_type: Infrastructure error kind, or None for a clean run.

    Returns:
        EvalArtifacts: The artifacts passed to ``grade``.
    """
    return EvalArtifacts(
        test_output=test_output,
        return_code=0,
        patch_applied=patch_applied,
        raw={"error_type": error_type},
    )


# --- vendored parser drives resolved ----------------------------------------


def test_grade_junit_xml_resolved():
    """junit-xml parsing + fuzzy id matching resolves a clean F2P/P2P pass."""
    harness = SweBenchExtHarness()
    report = harness.grade(_task(), _artifacts(_fixture("pytest_junit.xml")))
    assert report.resolved is True
    assert reward_from_report(report) == 1.0
    # The parser report is surfaced for inspection.
    assert report.tests_status["framework"] == "pytest"
    assert report.tests_status["f2p_passed"] == 1
    assert report.tests_status["p2p_passed"] == 1


def test_grade_junit_xml_unresolved_when_p2p_fails():
    harness = SweBenchExtHarness()
    task = _task(pass_to_pass=["tests/test_core.py::test_unrelated_broken"])
    report = harness.grade(task, _artifacts(_fixture("pytest_junit.xml")))
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_grade_pytest_text_fuzzy_id_match():
    """Normalized/fuzzy id matching: ``src/pkg/...py::test`` log id resolves a
    differently-delimited expected id via normalize_test_id."""
    harness = SweBenchExtHarness()
    task = _task(
        fail_to_pass=["src/pkg/tests/test_widget.py::test_alpha"],
        pass_to_pass=["src/pkg/tests/test_widget.py::test_beta"],
    )
    report = harness.grade(task, _artifacts(_fixture("pytest_text_fuzzy.txt")))
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_grade_pytest_text_unresolved_when_f2p_fails():
    harness = SweBenchExtHarness()
    task = _task(
        fail_to_pass=["src/pkg/tests/test_widget.py::test_gamma"],
        pass_to_pass=["src/pkg/tests/test_widget.py::test_beta"],
    )
    report = harness.grade(task, _artifacts(_fixture("pytest_text_fuzzy.txt")))
    assert report.resolved is False


def test_grade_build_synthetic_pass_injection():
    """An F2P entry ending ``::build`` not present in the parsed output is
    injected as PASSED (synthetic build/compile handling)."""
    harness = SweBenchExtHarness()
    task = _task(
        fail_to_pass=["src/pkg/tests/test_widget.py::test_alpha", "mypkg::build"],
        pass_to_pass=["src/pkg/tests/test_widget.py::test_beta"],
    )
    report = harness.grade(task, _artifacts(_fixture("pytest_text_fuzzy.txt")))
    assert report.resolved is True
    assert report.tests_status["fail_to_pass_results"]["mypkg::build"] == "PASSED"


def test_grade_non_pytest_framework_go_json():
    """A non-pytest framework (``go``) dispatches to the go-json parser."""
    harness = SweBenchExtHarness()
    task = _task(
        test_framework="go",
        fail_to_pass=["github.com/acme/widget::TestAlpha"],
        pass_to_pass=["github.com/acme/widget::TestBeta"],
    )
    report = harness.grade(task, _artifacts(_fixture("go_json.txt")))
    assert report.resolved is True
    assert report.tests_status["framework"] == "go"


def test_grade_non_pytest_framework_go_json_unresolved():
    harness = SweBenchExtHarness()
    task = _task(
        test_framework="go",
        fail_to_pass=["github.com/acme/widget::TestGamma"],
        pass_to_pass=["github.com/acme/widget::TestBeta"],
    )
    report = harness.grade(task, _artifacts(_fixture("go_json.txt")))
    assert report.resolved is False


# --- empty framework is passed VERBATIM (NOT coerced to pytest) --------------


def test_grade_empty_framework_passed_verbatim_not_coerced_to_pytest():
    """``test_framework`` is passed through UNCHANGED — an empty framework reaches
    ``parse_and_check_tests`` as ``""`` and hits the parser's auto-detect path, NOT
    the pytest junit-xml parser.

    Coercing ``""`` -> ``"pytest"`` would let junit-xml parse and report
    ``resolved`` for an instance that should auto-detect. We assert the framework
    reaches the parser verbatim (recorded in ``report.framework``) and that
    junit-xml is therefore NOT parsed under an empty framework.
    """
    harness = SweBenchExtHarness()
    task = _task(test_framework="")
    report = harness.grade(task, _artifacts(_fixture("pytest_junit.xml")))
    # Framework recorded verbatim — not silently rewritten to "pytest".
    assert report.tests_status["framework"] == ""
    # Auto-detect path does not understand junit-xml -> nothing parsed -> unresolved.
    assert report.tests_status["parsed_count"] == 0
    assert report.resolved is False


def test_grade_empty_framework_uses_autodetect_path():
    """An empty framework grades via parse_test_output's auto-detect path (TAP /
    Mocha-Hardhat console) when the instance ships no framework. Here a TAP
    transcript resolves without any framework hint."""
    harness = SweBenchExtHarness()
    tap_output = (
        "<<<SWE_BENCH_EXT_TEST_OUTPUT_START>>>\n"
        "TAP version 13\n"
        "1..2\n"
        "ok 1 - test_fix_applied\n"
        "ok 2 - test_regression_guard\n"
        "<<<SWE_BENCH_EXT_TEST_OUTPUT_END>>>\n"
    )
    task = _task(
        test_framework="",
        fail_to_pass=["test_fix_applied"],
        pass_to_pass=["test_regression_guard"],
    )
    report = harness.grade(task, _artifacts(tap_output))
    assert report.tests_status["framework"] == ""
    assert report.tests_status["parsed_count"] >= 2
    assert report.resolved is True


def test_run_eval_and_grade_share_framework_value():
    """run_eval (flag/result-file selection) and grade (parsing) use the SAME
    framework. With an empty framework, run_eval must NOT inject pytest's
    ``--junitxml`` flag and must wrap the bare command, and grade must parse under
    ``""`` — proving the two share ``_resolve_framework`` rather than diverging on a
    pytest default."""
    task = _task(test_framework="", test_command="run-my-tests")
    _, _, provider = _run_eval(task, test_output="", run_cmd="run-my-tests")
    eval_cmds = [c for c in provider.commands if "run-my-tests" in c]
    assert eval_cmds, "expected the bare framework command to be wrapped"
    wrapped = eval_cmds[-1]
    # Empty framework => default framework config => no output flag, no result file.
    assert "--junitxml" not in wrapped
    assert "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>" not in wrapped
    # The mkdir parent-dir creation is present regardless.
    assert "mkdir -p /workspace/test-results" in wrapped


# --- patch_applied does not gate resolved -----------------------------------


def test_grade_resolved_even_when_patch_apply_failed():
    """Grading is on tests ONLY; a failed apply is recorded but never flips a
    tests-passing run to unresolved."""
    harness = SweBenchExtHarness()
    report = harness.grade(_task(), _artifacts(_fixture("pytest_junit.xml"), patch_applied=False))
    assert report.patch_applied is False
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


# --- infra masking (unchanged behavior) -------------------------------------


def test_grade_masks_on_infra_error():
    harness = SweBenchExtHarness()
    report = harness.grade(_task(), _artifacts("", error_type="timeout"))
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


# --- run_eval against a scripted FakeSandbox --------------------------------


class _FakeExtProvider:
    """Scripted provider that records git-apply attempts and returns a transcript.

    Args:
        test_output: The transcript returned for the wrapped eval command.
        apply_rc: Return code for ``git apply`` commands.
        run_cmd: Substring identifying the wrapped eval command.
    """

    name = "fake-ext"

    def __init__(self, *, test_output="", apply_rc=0, run_cmd="pytest", **_):
        self._test_output = test_output
        self._apply_rc = apply_rc
        # Marker that identifies the wrapped eval command (defaults to the pytest
        # command); tests with a custom command pass run_cmd.
        self._run_cmd = run_cmd
        self.commands: list[str] = []
        self.uploaded: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
        if "git apply" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=self._apply_rc)
        if self._run_cmd in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

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


register_provider("fake-ext", _FakeExtProvider, override=True)


def _run_eval(task: SweTask, *, test_output: str, apply_rc: int = 0, run_cmd: str = "pytest"):
    """Run the harness through a scripted provider and return the run outputs.

    Args:
        task: The task to evaluate.
        test_output: The transcript the provider returns for the eval command.
        apply_rc: Return code for ``git apply`` commands.
        run_cmd: Substring identifying the wrapped eval command.

    Returns:
        tuple: The harness, the produced ``EvalArtifacts``, and the provider
        instance (for command inspection).
    """
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    async def run():
        harness = SweBenchExtHarness()
        env = await AsyncSweEnvironment.start(
            {"fake-ext": {"test_output": test_output, "apply_rc": apply_rc, "run_cmd": run_cmd}},
            harness.build_spec(task),
        )
        await harness.materialize(env, task)
        artifacts = await harness.run_eval(env, task)
        return harness, artifacts, env.sandbox._provider

    return asyncio.run(run())


def test_run_eval_uses_legacy_apply_flags_and_grades_resolved():
    task = _task()
    harness, artifacts, provider = _run_eval(task, test_output=_fixture("pytest_junit.xml"))
    apply_cmds = [c for c in provider.commands if "git apply" in c]
    assert apply_cmds, "expected a git-apply attempt"
    # The git-apply flag set, with no --3way fallback.
    assert all("--reject --recount --ignore-space-change --ignore-whitespace" in c for c in apply_cmds)
    assert all("--3way" not in c for c in apply_cmds)
    assert artifacts.patch_applied is True
    report = harness.grade(task, artifacts)
    assert report.resolved is True


def test_run_eval_apply_failure_still_resolves_on_tests():
    # End-to-end through run_eval -> grade: a failed apply records
    # patch_applied=False but a tests-passing run still resolves.
    task = _task()
    harness, artifacts, _ = _run_eval(task, test_output=_fixture("pytest_junit.xml"), apply_rc=1)
    assert artifacts.patch_applied is False
    report = harness.grade(task, artifacts)
    assert report.patch_applied is False
    assert report.resolved is True


def test_run_eval_wraps_command_with_structured_output_and_markers():
    # run_eval wraps the command — add the structured-output flag (--junitxml) via
    # get_test_command_with_output and run between the SWE_BENCH_EXT markers (plus
    # result-file dump), so parse_and_check_tests receives junit-xml / marked
    # output rather than raw "-rA" text it cannot parse.
    task = _task()
    _, _, provider = _run_eval(task, test_output=_fixture("pytest_junit.xml"))
    eval_cmds = [c for c in provider.commands if "pytest" in c and "git apply" not in c]
    assert eval_cmds, "expected a wrapped pytest eval command"
    wrapped = eval_cmds[-1]
    assert "<<<SWE_BENCH_EXT_TEST_OUTPUT_START>>>" in wrapped
    assert "<<<SWE_BENCH_EXT_TEST_OUTPUT_END>>>" in wrapped
    assert "--junitxml=" in wrapped  # structured-output flag from get_test_command_with_output
    assert "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>" in wrapped  # junit result-file dumped for the parser
    # The result-file parent dir is created first.
    assert "mkdir -p /workspace/test-results" in wrapped
