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

"""Unit tests for the swe-rebench harness (FakeSandbox provider).

A tiny fake ``agent/log_parsers.py`` is written to a tmp dir so the real
``_load_rebench_log_parsers`` import and ``NAME_TO_PARSER`` resolution path is
exercised end to end, then the resolved / unresolved / masked grade paths are
driven.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask
from responses_api_agents.swe_env.harnesses.swe_rebench import (
    SweRebenchHarness,
    _normalize_test_name,
)


class _FakeProvider:
    """Scripted provider: test command returns a canned transcript."""

    name = "fake-rebench"

    def __init__(self, *, test_output="", test_rc=0, apply_rc=0, **_):
        """Initialize the scripted provider.

        Args:
            test_output: Transcript returned for the test command.
            test_rc: Return code for the test command.
            apply_rc: Return code for ``git apply`` commands.
        """
        self._test_output = test_output
        self._test_rc = test_rc
        self._apply_rc = apply_rc

    async def create(self, spec):
        raw = {"workdir": spec.workdir, "env": spec.env}
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw=raw)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "git apply" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=self._apply_rc)
        if "pytest" in command or "test" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=self._test_rc)
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


register_provider("fake-rebench", _FakeProvider, override=True)


class _RecordingProvider:
    """Scripted provider that records every exec command, in order."""

    name = "recording-rebench"
    commands: list[str] = []
    # (command, timeout_s) for every exec, so tests can assert the eval timeout
    # is threaded into the test exec.
    exec_calls: list[tuple[str, object]] = []

    def __init__(self, *, test_output="", test_rc=0, apply_rc=0, **_):
        """Initialize the recording provider.

        Args:
            test_output: Transcript returned for the test command.
            test_rc: Return code for the test command.
            apply_rc: Return code for ``git apply`` commands.
        """
        self._test_output = test_output
        self._test_rc = test_rc
        self._apply_rc = apply_rc

    async def create(self, spec):
        return SandboxHandle(sandbox_id="rec", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        type(self).commands.append(command)
        type(self).exec_calls.append((command, timeout_s))
        if "git apply" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=self._apply_rc)
        if "pytest" in command or "test" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=self._test_rc)
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


register_provider("recording-rebench", _RecordingProvider, override=True)


# A standalone log_parsers module the harness imports dynamically. The parser
# splits "<node> <STATUS>" lines into {node: STATUS} and exposes a
# NAME_TO_PARSER registry of callables, matching the shape the harness expects.
_FAKE_LOG_PARSERS = textwrap.dedent(
    """
    def parse_simple(log):
        results = {}
        for line in log.splitlines():
            line = line.strip()
            if not line:
                continue
            node, _, status = line.rpartition(" ")
            if node and status:
                results[node] = status
        return results

    NAME_TO_PARSER = {"simple": parse_simple}
    """
)


def _write_fake_parsers(tmp_path: Path) -> Path:
    """Write the fake ``agent/log_parsers.py`` module under a tmp repo dir.

    Args:
        tmp_path: The pytest tmp dir to create the repo under.

    Returns:
        Path: The created ``SWE-rebench-V2`` repo directory.
    """
    repo_dir = tmp_path / "SWE-rebench-V2"
    (repo_dir / "agent").mkdir(parents=True)
    (repo_dir / "agent" / "log_parsers.py").write_text(_FAKE_LOG_PARSERS)
    return repo_dir


def _task(**overrides) -> SweTask:
    """Build a swe-rebench ``SweTask`` with sensible defaults.

    Args:
        **overrides: Field values overriding the defaults.

    Returns:
        SweTask: A task populated from the defaults merged with overrides.
    """
    base = dict(
        instance_id="rebench-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        test_command="python -m pytest -rA -q",
        model_patch="diff --git a/x b/x\n",
        test_patch="diff --git a/t b/t\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="swe-rebench",
    )
    base.update(overrides)
    return SweTask(**base)


# ---- pure helpers -----------------------------------------------------------


def test_normalize_test_name_strips_timing():
    assert _normalize_test_name("t::a [ 12 ms ]") == "t::a"
    assert _normalize_test_name("t::a [0.3s]") == "t::a"
    assert _normalize_test_name("t::a in 1.2 sec") == "t::a"
    assert _normalize_test_name("t::a (5 ms)") == "t::a"
    assert _normalize_test_name("  t::a  ") == "t::a"
    # No timing suffix -> unchanged.
    assert _normalize_test_name("pkg::mod::test_x") == "pkg::mod::test_x"


def test_build_spec_sets_java_env():
    harness = SweRebenchHarness()
    spec = harness.build_spec(_task())
    assert spec.env["_JAVA_OPTIONS"] == "-Djava.net.preferIPv6Addresses=false"
    assert spec.metadata["harness"] == "swe-rebench"
    assert spec.image == "img:tag"


# ---- grade paths (real dynamic-import of the fake parser) --------------------


def test_grade_resolved(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}},
    )
    # Both required tests pass; timing suffix on one exercises normalization.
    artifacts = EvalArtifacts(test_output="t::a [ 12 ms ] PASSED\nt::b PASSED\n", patch_applied=True)
    report = harness.grade(task, artifacts)
    assert report.resolved is True
    assert report.error_kind is None
    assert set(report.tests_status["passed"]) == {"t::a", "t::b"}


def test_grade_unresolved_missing_pass_to_pass(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}},
    )
    artifacts = EvalArtifacts(test_output="t::a PASSED\nt::b FAILED\n", patch_applied=True)
    report = harness.grade(task, artifacts)
    assert report.resolved is False
    assert report.error_kind is None


def test_grade_no_patch_applied_gate(tmp_path):
    """``resolved`` is the test verdict ONLY and does not gate on patch_applied.
    So even when the model patch failed to apply (``patch_applied=False``), a run
    where every F2P/P2P test passes scores resolved=True."""
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}},
    )
    artifacts = EvalArtifacts(test_output="t::a PASSED\nt::b PASSED\n", patch_applied=False)
    report = harness.grade(task, artifacts)
    assert report.resolved is True
    assert report.error_kind is None


def test_grade_masks_missing_clone():
    harness = SweRebenchHarness()
    # No rebench_repo_dir in metadata -> the clone is not provisioned.
    report = harness.grade(_task(), EvalArtifacts(test_output="t::a PASSED\n", patch_applied=True))
    assert report.error_kind == "eval_error"
    assert report.resolved is False


def test_grade_masks_unknown_parser(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "does_not_exist"}},
    )
    report = harness.grade(task, EvalArtifacts(test_output="t::a PASSED\n", patch_applied=True))
    assert report.error_kind == "eval_error"


def test_grade_masks_on_infra_error():
    harness = SweRebenchHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout"}))
    assert report.error_kind == "timeout"


# ---- run_eval (FakeSandbox) -------------------------------------------------


def test_run_eval_then_grade_resolved(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={
            "rebench_repo_dir": str(repo_dir),
            "install_config": {"log_parser": "simple", "test_cmd": "python -m pytest -rA -q"},
        },
    )
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    provider = {"fake-rebench": {"test_output": "t::a PASSED\nt::b PASSED\n", "test_rc": 0}}

    async def _run():
        spec = harness.build_spec(task)
        env = await AsyncSweEnvironment.start(provider, spec)
        try:
            await harness.reset_repo(env, task)
            await harness.materialize(env, task)
            artifacts = await harness.run_eval(env, task)
        finally:
            await env.cleanup()
        return artifacts

    artifacts = asyncio.run(_run())
    assert artifacts.patch_applied is True
    report = harness.grade(task, artifacts)
    assert report.resolved is True


def test_run_eval_patch_not_applied_still_grades_on_tests(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}})
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    # apply_rc=1 -> model patch fails to apply -> patch_applied False, but grading
    # is on the tests only (no patch_applied gate), so a run where every F2P/P2P
    # test passes is still resolved=True.
    provider = {"fake-rebench": {"test_output": "t::a PASSED\nt::b PASSED\n", "apply_rc": 1}}

    async def _run():
        spec = harness.build_spec(task)
        env = await AsyncSweEnvironment.start(provider, spec)
        try:
            await harness.run_eval(env, task)
            return await harness.run_eval(env, task)
        finally:
            await env.cleanup()

    artifacts = asyncio.run(_run())
    assert artifacts.patch_applied is False
    assert harness.grade(task, artifacts).resolved is True


# ---- apply order ------------------------------------------------------------


def test_run_eval_applies_model_patch_before_test_patch(tmp_path):
    """The model patch (/root/patch.diff) is applied BEFORE the test patch
    (/root/test_patch.diff)."""
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}})
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    _RecordingProvider.commands = []
    _RecordingProvider.exec_calls = []
    provider = {"recording-rebench": {"test_output": "t::a PASSED\nt::b PASSED\n"}}

    async def _run():
        spec = harness.build_spec(task)
        env = await AsyncSweEnvironment.start(provider, spec)
        try:
            await harness.run_eval(env, task)
        finally:
            await env.cleanup()

    asyncio.run(_run())
    applies = [c for c in _RecordingProvider.commands if "git apply" in c]
    assert len(applies) == 2
    assert "/root/patch.diff" in applies[0], applies
    assert "/root/test_patch.diff" in applies[1], applies


# ---- eval timeout threaded into the test exec -------------------------------


def _rebench_test_exec_timeout(commands_and_timeouts):
    """Return the timeout_s passed to the test exec (the one running the tests).

    The test block is the only exec that is neither a ``git apply`` nor an
    install command; in these tests the test command always contains ``pytest``.

    Args:
        commands_and_timeouts: An iterable of ``(command, timeout_s)`` pairs.

    Returns:
        The ``timeout_s`` value recorded for the test exec.

    Raises:
        AssertionError: If no test exec is found in the recorded calls.
    """
    for command, timeout_s in commands_and_timeouts:
        if "git apply" not in command and ("pytest" in command or "test" in command):
            return timeout_s
    raise AssertionError(f"no test exec found in {commands_and_timeouts!r}")


def test_run_eval_threads_tests_timeout_into_test_exec(tmp_path):
    """The test exec receives timeout_s = task.metadata['tests_timeout'] when
    present so a stuck run is bounded instead of hanging the verifier. Uses a
    non-default value (600) so this distinguishes an explicit override from the
    1800 default."""
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={
            "rebench_repo_dir": str(repo_dir),
            "install_config": {"log_parser": "simple", "test_cmd": "python -m pytest -rA -q"},
            "tests_timeout": 600,
        },
    )
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    _RecordingProvider.commands = []
    _RecordingProvider.exec_calls = []
    provider = {"recording-rebench": {"test_output": "t::a PASSED\nt::b PASSED\n"}}

    async def _run():
        spec = harness.build_spec(task)
        env = await AsyncSweEnvironment.start(provider, spec)
        try:
            await harness.run_eval(env, task)
        finally:
            await env.cleanup()

    asyncio.run(_run())
    assert _rebench_test_exec_timeout(_RecordingProvider.exec_calls) == 600


def test_run_eval_tests_timeout_absent_defaults_to_1800(tmp_path):
    """The timeout (default 30*60) is applied to every swe-rebench run. Rows that
    carry no tests_timeout (including SWE-bench-Verified) still get the 1800s
    bound rather than an unbounded (None) run."""
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={
            "rebench_repo_dir": str(repo_dir),
            "install_config": {"log_parser": "simple", "test_cmd": "python -m pytest -rA -q"},
        },
    )
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    _RecordingProvider.commands = []
    _RecordingProvider.exec_calls = []
    provider = {"recording-rebench": {"test_output": "t::a PASSED\nt::b PASSED\n"}}

    async def _run():
        spec = harness.build_spec(task)
        env = await AsyncSweEnvironment.start(provider, spec)
        try:
            await harness.run_eval(env, task)
        finally:
            await env.cleanup()

    asyncio.run(_run())
    assert _rebench_test_exec_timeout(_RecordingProvider.exec_calls) == 1800


# ---- grading parity / empty-required ----------------------------------------


def test_grade_empty_required_does_not_resolve(tmp_path):
    """A degenerate row with no FAIL_TO_PASS and no PASS_TO_PASS does NOT resolve.

    The empty-required guard (matching ``compute_resolved``) keeps swe-rebench consistent with the
    other families: an empty required set must not vacuously resolve to reward 1.0 just because the
    empty set is a subset of any passed set.
    """
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        fail_to_pass=[],
        pass_to_pass=[],
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}},
    )
    artifacts = EvalArtifacts(test_output="something PASSED\n", patch_applied=True)
    report = harness.grade(task, artifacts)
    assert report.resolved is False
    assert report.error_kind is None
