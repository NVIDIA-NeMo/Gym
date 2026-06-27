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

"""Unit tests for the nv-internal-1 harness, driven by a FakeSandbox provider.

nv-internal-1 is flat + host-graded, so it runs on any exec-capable provider.
The scripted provider returns the parsing_script ``output.json`` report on the
``cat /root/output.json`` hop; grading is a pure host-side parse.
"""

from __future__ import annotations

import asyncio
import json

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask, reward_from_report
from responses_api_agents.swe_env.harnesses.nv_internal import (
    NV_DEFAULT_WORKDIR,
    NVInternalHarness,
    _coerce_test_list,
    _format_test_files,
    _nv_workdir,
    _parse_dockerfile_env,
    _resolve_required_tests,
    parse_passed_tests,
)
from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment


class _FakeProvider:
    """Scripted provider: ``cat /root/output.json`` returns a canned report."""

    name = "fake-nv"

    def __init__(self, *, report="", apply_rc=0, **_):
        """Configure the scripted provider's responses.

        Args:
            report: JSON report stdout returned for ``cat /root/output.json``.
            apply_rc: Return code returned for ``git apply`` commands.
            **_: Ignored extra keyword arguments.
        """
        self._report = report
        self._apply_rc = apply_rc

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "cat /root/output.json" in command:
            return SandboxExecResult(stdout=self._report, stderr="", return_code=0)
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


register_provider("fake-nv", _FakeProvider, override=True)


class _RecordingProvider:
    """Provider that records exec ``cwd`` per command and captures uploads.

    Uploads are captured as ``{target_path: content}`` by reading the temp file
    that ``write_text`` hands to ``upload_file``; execs are captured as a list of
    ``(command, cwd)`` so tests can assert which directory each hop ran in.
    """

    name = "fake-nv-rec"

    def __init__(self, *, report="", **_):
        """Configure the recording provider's canned report.

        Args:
            report: JSON report stdout returned for ``cat /root/output.json``.
            **_: Ignored extra keyword arguments.
        """
        self._report = report
        self.execs: list[tuple[str, str | None]] = []
        self.uploads: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append((command, cwd))
        if "cat /root/output.json" in command:
            return SandboxExecResult(stdout=self._report, stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, handle, source_path, target_path):
        with open(source_path, encoding="utf-8") as fh:
            self.uploads[target_path] = fh.read()

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        return None

    async def aclose(self):
        return None


register_provider("fake-nv-rec", _RecordingProvider, override=True)


def _task(**overrides) -> SweTask:
    """Build an nv-internal-1 SweTask with sensible defaults, overridable per keyword.

    Args:
        **overrides: Field overrides merged onto the default task fields.

    Returns:
        A SweTask configured for the nv-internal-1 benchmark.
    """
    base = dict(
        instance_id="nv-inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/app",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["pkg/test_x.py::a"],
        pass_to_pass=["pkg/test_x.py::b"],
        benchmark="nv-internal-1",
        metadata={
            "run_script": "echo run\n",
            "parsing_script": "import sys\n",
            "selected_test_files_to_run": ["pkg/test_x.py"],
        },
    )
    base.update(overrides)
    return SweTask(**base)


def _report(*passed, failed=()):
    """Build a JSON test report with the given passed and failed test names.

    Args:
        *passed: Names of tests reported as PASSED.
        failed: Names of tests reported as FAILED.

    Returns:
        The report serialized as a JSON string under a ``tests`` key.
    """
    tests = [{"name": name, "status": "PASSED"} for name in passed]
    tests += [{"name": name, "status": "FAILED"} for name in failed]
    return json.dumps({"tests": tests})


async def _run(provider_cfg, task) -> SweEvalReport:
    """Drive reset -> materialize -> run_eval -> grade against a scripted provider.

    Args:
        provider_cfg: Provider configuration mapping for the ``fake-nv`` provider.
        task: The SweTask to evaluate.

    Returns:
        The graded SweEvalReport for the run.
    """
    harness = NVInternalHarness()
    env = await AsyncSweEnvironment.start({"fake-nv": provider_cfg}, harness.build_spec(task))
    try:
        await harness.reset_repo(env, task)
        await harness.materialize(env, task)
        artifacts = await harness.run_eval(env, task)
    finally:
        await env.cleanup()
    return harness.grade(task, artifacts)


# ---- pure helpers -----------------------------------------------------------


def test_parse_passed_tests():
    """``parse_passed_tests`` returns only PASSED names and ignores malformed entries."""
    report = {"tests": [{"name": "a", "status": "PASSED"}, {"name": "b", "status": "FAILED"}]}
    assert parse_passed_tests(report) == ["a"]
    assert parse_passed_tests({}) == []
    # Malformed entries are ignored, not crashed on.
    assert parse_passed_tests({"tests": ["junk", {"status": "PASSED"}]}) == []


def test_format_test_files():
    """``_format_test_files`` joins list/JSON/CSV inputs into a comma-separated string."""
    assert _format_test_files(["a", "b"]) == "a,b"
    assert _format_test_files('["a", "b"]') == "a,b"
    assert _format_test_files("a,b") == "a,b"
    assert _format_test_files(None) == ""


def test_format_test_files_single_quoted_list():
    """``_format_test_files`` parses repr-style single-quoted lists.

    Single-quoted lists are not valid JSON, so they are parsed with
    ``ast.literal_eval``; unparseable bracketed text falls back to the raw string.
    """
    assert _format_test_files("['pkg/test_x.py', 'pkg/test_y.py']") == "pkg/test_x.py,pkg/test_y.py"
    # A single-element single-quoted list.
    assert _format_test_files("['only.py']") == "only.py"
    # Unparseable bracketed text falls back to the raw string, not a crash.
    assert _format_test_files("[not a list") == "[not a list"


def test_build_spec():
    """The nv-internal-1 harness builds a sandbox spec from a task."""
    harness = NVInternalHarness()
    assert harness.name == "nv-internal-1"
    assert harness.grade_strategy == "flat-host-grade"
    spec = harness.build_spec(_task())
    assert spec.image == "img:tag"
    assert spec.workdir == "/app"
    assert spec.metadata["instance_id"] == "nv-inst-1"


def test_supports_any_provider():
    """The nv-internal-1 harness supports any exec-capable provider."""
    assert NVInternalHarness().supports_provider("docker") is True
    assert NVInternalHarness().supports_provider("apptainer") is True


def test_grade_masks_on_infra_error():
    """Grading masks an infra timeout to reward 0.0 and records its error kind."""
    harness = NVInternalHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout"}))
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_sandbox_error():
    """Grading masks a sandbox error to reward 0.0 and records its error kind."""
    harness = NVInternalHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "sandbox"}))
    assert report.error_kind == "sandbox"
    assert reward_from_report(report) == 0.0


def test_grade_empty_report_is_unresolved():
    """An empty report grades as unresolved."""
    harness = NVInternalHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=0, patch_applied=True))
    assert report.resolved is False


def test_grade_malformed_report_is_unresolved():
    """A malformed (non-JSON) report grades as unresolved."""
    harness = NVInternalHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="not json", return_code=0, patch_applied=True))
    assert report.resolved is False


# ---- full reset -> materialize -> run_eval -> grade -------------------------


def test_resolved():
    """A run with all required tests passing resolves with reward 1.0."""
    report = _report("pkg/test_x.py::a", "pkg/test_x.py::b")
    result = asyncio.run(_run({"report": report}, _task()))
    assert result.patch_applied is True
    assert result.resolved is True
    assert reward_from_report(result) == 1.0


def test_unresolved_failing_required_test():
    """A failing fail-to-pass test leaves the run unresolved with reward 0.0."""
    report = _report("pkg/test_x.py::b", failed=["pkg/test_x.py::a"])
    result = asyncio.run(_run({"report": report}, _task()))
    assert result.resolved is False
    assert reward_from_report(result) == 0.0


def test_unresolved_missing_required_test():
    """A required test missing from the report leaves the run unresolved."""
    report = _report("pkg/test_x.py::a")
    result = asyncio.run(_run({"report": report}, _task()))
    assert result.resolved is False


def test_patch_apply_rc_does_not_gate_resolved():
    """A non-zero patch-apply return code does not gate ``resolved``.

    Grading derives ``resolved`` from the tests alone, so a rejected patch
    (apply_rc != 0) with all required tests passing is still resolved.
    """
    report = _report("pkg/test_x.py::a", "pkg/test_x.py::b")
    result = asyncio.run(_run({"report": report, "apply_rc": 1}, _task()))
    assert result.patch_applied is False
    assert result.resolved is True
    assert reward_from_report(result) == 1.0


# ---- *_select precedence ----------------------------------------------------


def test_resolve_required_tests_prefers_select_keys():
    """``fail_to_pass_select`` / ``pass_to_pass_select`` take precedence over the plain keys."""
    task = _task(
        fail_to_pass=["plain::f2p"],
        pass_to_pass=["plain::p2p"],
        metadata={
            "fail_to_pass_select": ["sel::f2p"],
            "pass_to_pass_select": ["sel::p2p"],
        },
    )
    f2p, p2p = _resolve_required_tests(task)
    assert f2p == ["sel::f2p"]
    assert p2p == ["sel::p2p"]


def test_resolve_required_tests_falls_back_to_plain_keys():
    """Without ``*_select`` keys, the plain fail_to_pass / pass_to_pass keys are used."""
    task = _task(fail_to_pass=["plain::f2p"], pass_to_pass=["plain::p2p"], metadata={})
    f2p, p2p = _resolve_required_tests(task)
    assert f2p == ["plain::f2p"]
    assert p2p == ["plain::p2p"]


def test_resolve_required_tests_parses_stringified_select():
    """A ``*_select`` value given as a repr-style stringified list is parsed."""
    task = _task(
        metadata={
            "fail_to_pass_select": "['sel::f2p']",
            "pass_to_pass_select": "['sel::p2p']",
        },
    )
    f2p, p2p = _resolve_required_tests(task)
    assert f2p == ["sel::f2p"]
    assert p2p == ["sel::p2p"]


def test_coerce_test_list():
    """``_coerce_test_list`` accepts lists and stringified lists, returning [] on bad input."""
    assert _coerce_test_list(["a", "b"]) == ["a", "b"]
    assert _coerce_test_list("['a', 'b']") == ["a", "b"]
    assert _coerce_test_list('["a", "b"]') == ["a", "b"]
    assert _coerce_test_list("not a list") == []
    assert _coerce_test_list("[broken") == []


def test_resolved_uses_select_tests_end_to_end():
    """End to end, ``*_select`` precedence resolves a run whose report has only the select tests."""
    # The report only contains the *_select tests; the plain keys would be unmet.
    report = _report("sel::f2p", "sel::p2p")
    task = _task(
        fail_to_pass=["plain::f2p"],
        pass_to_pass=["plain::p2p"],
        metadata={
            "run_script": "echo run\n",
            "parsing_script": "import sys\n",
            "selected_test_files_to_run": ["pkg/test_x.py"],
            "fail_to_pass_select": ["sel::f2p"],
            "pass_to_pass_select": ["sel::p2p"],
        },
    )
    result = asyncio.run(_run({"report": report}, task))
    assert result.resolved is True


# ---- dockerfile ENV replay --------------------------------------------------


def test_parse_dockerfile_env_equals_and_space_forms():
    """``_parse_dockerfile_env`` parses both ``ENV K=V`` and ``ENV K V`` forms, skipping non-ENV lines."""
    task = _task(
        metadata={
            "base_dockerfile": "FROM ubuntu\nENV FOO=bar\nENV SPACED  spaced_value\n",
            "instance_dockerfile": "ENV BAZ = qux\nRUN echo hi\n",
        },
    )
    env = _parse_dockerfile_env(task)
    assert env["FOO"] == "bar"
    assert env["SPACED"] == "spaced_value"
    assert env["BAZ"] == "qux"
    assert "RUN" not in env


def test_parse_dockerfile_env_absent_is_noop():
    """``_parse_dockerfile_env`` returns an empty mapping when no dockerfile is present."""
    assert _parse_dockerfile_env(_task(metadata={})) == {}


def test_build_spec_injects_dockerfile_env():
    """``build_spec`` injects dockerfile ENV entries while preserving the existing git env."""
    task = _task(metadata={"base_dockerfile": "ENV PATH=/custom/bin:$PATH\n"})
    spec = NVInternalHarness().build_spec(task)
    # Existing git env preserved; dockerfile ENV injected.
    assert spec.env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert spec.env["PATH"] == "/custom/bin:$PATH"


# ---- dotted script keys are uploaded ----------------------------------------


async def _run_recording(task) -> _RecordingProvider:
    """Drive reset -> materialize -> run_eval with a recording provider.

    Args:
        task: The SweTask to evaluate.

    Returns:
        The recording provider, so tests can inspect captured execs and uploads.
    """
    provider = _RecordingProvider(report=_report("pkg/test_x.py::a", "pkg/test_x.py::b"))
    harness = NVInternalHarness()
    env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
    try:
        await harness.reset_repo(env, task)
        await harness.materialize(env, task)
        await harness.run_eval(env, task)
    finally:
        await env.cleanup()
    return provider


def test_materialize_reads_dotted_script_keys():
    """``materialize`` uploads scripts stored under the dotted keys ``run_script.sh`` / ``parsing_script.py``."""
    task = _task(
        repo_workdir="/app",
        metadata={
            "run_script.sh": "echo DOTTED_RUN\n",
            "parsing_script.py": "print('DOTTED_PARSE')\n",
            "selected_test_files_to_run": ["pkg/test_x.py"],
        },
    )
    provider = asyncio.run(_run_recording(task))
    assert provider.uploads["/root/run_script.sh"] == "echo DOTTED_RUN\n"
    assert provider.uploads["/root/parsing_script.py"] == "print('DOTTED_PARSE')\n"


def test_materialize_dotted_keys_take_precedence_over_extensionless():
    """When both dotted and extensionless script keys are present, the dotted keys win."""
    task = _task(
        repo_workdir="/app",
        metadata={
            "run_script.sh": "echo DOTTED\n",
            "run_script": "echo EXTLESS\n",
            "parsing_script.py": "print('DOTTED')\n",
            "parsing_script": "print('EXTLESS')\n",
            "selected_test_files_to_run": ["pkg/test_x.py"],
        },
    )
    provider = asyncio.run(_run_recording(task))
    assert provider.uploads["/root/run_script.sh"] == "echo DOTTED\n"
    assert provider.uploads["/root/parsing_script.py"] == "print('DOTTED')\n"


def test_materialize_falls_back_to_extensionless_keys():
    """When only the extensionless script keys are present, they are used."""
    task = _task(
        repo_workdir="/app",
        metadata={
            "run_script": "echo EXTLESS_RUN\n",
            "parsing_script": "print('EXTLESS_PARSE')\n",
            "selected_test_files_to_run": ["pkg/test_x.py"],
        },
    )
    provider = asyncio.run(_run_recording(task))
    assert provider.uploads["/root/run_script.sh"] == "echo EXTLESS_RUN\n"
    assert provider.uploads["/root/parsing_script.py"] == "print('EXTLESS_PARSE')\n"


# ---- hops run in /app -------------------------------------------------------


def test_nv_workdir_defaults_to_app():
    """``_nv_workdir`` maps the generic /testbed default (or empty) to /app, honoring pinned paths."""
    assert _nv_workdir(_task(repo_workdir="/testbed")) == NV_DEFAULT_WORKDIR
    assert _nv_workdir(_task(repo_workdir="")) == NV_DEFAULT_WORKDIR
    # A row that pins a non-default workdir is honored.
    assert _nv_workdir(_task(repo_workdir="/srv/repo")) == "/srv/repo"
    assert _nv_workdir(_task(repo_workdir="/app")) == "/app"


def test_build_spec_workdir_defaults_to_app_for_generic_default():
    """``build_spec`` rewrites the generic /testbed default workdir to /app."""
    spec = NVInternalHarness().build_spec(_task(repo_workdir="/testbed"))
    assert spec.workdir == NV_DEFAULT_WORKDIR


def test_all_hops_run_in_app_for_generic_default():
    """With the generic /testbed default, every reset/apply/run/parse/cat hop runs in /app."""
    task = _task(
        repo_workdir="/testbed",
        metadata={
            "run_script.sh": "echo run\n",
            "parsing_script.py": "import sys\n",
            "selected_test_files_to_run": ["pkg/test_x.py"],
        },
    )
    provider = asyncio.run(_run_recording(task))
    cwds = {cwd for _, cwd in provider.execs}
    assert cwds == {NV_DEFAULT_WORKDIR}
    # Spot-check that the key hops were exercised in /app.
    by_cwd = {cmd: cwd for cmd, cwd in provider.execs}
    assert any("git reset --hard" in cmd and cwd == "/app" for cmd, cwd in provider.execs)
    assert any("git apply" in cmd and cwd == "/app" for cmd, cwd in provider.execs)
    assert any("run_script.sh" in cmd and cwd == "/app" for cmd, cwd in provider.execs)
    assert any("parsing_script.py" in cmd and cwd == "/app" for cmd, cwd in provider.execs)
    assert by_cwd["cat /root/output.json"] == "/app"


def test_all_hops_honor_explicit_non_default_workdir():
    """A row that pins ``repo_workdir`` to a non-default path runs every hop there."""
    task = _task(
        repo_workdir="/srv/repo",
        metadata={
            "run_script.sh": "echo run\n",
            "parsing_script.py": "import sys\n",
            "selected_test_files_to_run": ["pkg/test_x.py"],
        },
    )
    provider = asyncio.run(_run_recording(task))
    assert {cwd for _, cwd in provider.execs} == {"/srv/repo"}
