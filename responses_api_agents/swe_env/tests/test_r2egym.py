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

"""Unit tests for the r2e-gym nested harness, driven by a FakeSandbox provider.

r2e-gym is a nested-harness family. These tests cover spec construction, the
apptainer-only provider gate, the agent-phase test-hiding command shape, and
report parsing fed a scripted ``report.json``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.environment import AsyncSweEnvironment
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask
from responses_api_agents.swe_env.harnesses.r2egym import R2EGymHarness


class _FakeProvider:
    """Scripted provider: the eval command returns a canned rc; ``cat`` returns the report.

    Records the commands it executes and the files uploaded so tests can assert
    on the in-sandbox side effects (e.g. that ``materialize`` writes the
    predictions JSONL and that ``reset_repo`` issues no ``git reset``).
    """

    name = "fake-r2egym"

    def __init__(self, *, report_text="", eval_rc=0, **_):
        """Initialize the scripted provider.

        Args:
            report_text: Text returned by any ``cat`` command (the report).
            eval_rc: Return code for the ``run_local_evaluation.py`` command.
        """
        self._report_text = report_text
        self._eval_rc = eval_rc
        self.commands: list[str] = []
        self.uploads: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
        if command.startswith("cat "):
            return SandboxExecResult(stdout=self._report_text, stderr="", return_code=0)
        if "run_local_evaluation.py" in command:
            return SandboxExecResult(stdout="eval done", stderr="", return_code=self._eval_rc)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, handle, source_path, target_path):
        self.uploads[target_path] = Path(source_path).read_text()
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
    """Build an r2e-gym ``SweTask`` with sensible defaults.

    Args:
        **overrides: Field values overriding the defaults.

    Returns:
        SweTask: A task populated from the defaults merged with overrides.
    """
    base = dict(
        instance_id="r2e__pkg-42",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="r2e-gym",
    )
    base.update(overrides)
    return SweTask(**base)


def _report(instance_id: str, resolved: bool) -> str:
    """Build a serialized nested-harness ``report.json`` for one instance.

    Args:
        instance_id: The instance id keying the report entry.
        resolved: Whether the instance is marked resolved.

    Returns:
        str: The JSON-encoded report.
    """
    return json.dumps(
        {
            instance_id: {
                "resolved": resolved,
                "tests_status": {"FAIL_TO_PASS": {"success": ["t::a"], "failure": []}},
            }
        }
    )


# ---- spec + provider gate ---------------------------------------------------


def test_harness_identity():
    harness = R2EGymHarness()
    assert harness.name == "r2e-gym"
    assert harness.grade_strategy == "nested-harness"


def test_build_spec_mounts_setup_dir():
    harness = R2EGymHarness()
    spec = harness.build_spec(_task(metadata={"r2egym_setup_dir": "/abs/setup"}))
    assert spec.image == "img:tag"
    assert spec.workdir == "/testbed"
    assert spec.metadata["instance_id"] == "r2e__pkg-42"
    assert spec.metadata["harness"] == "r2e-gym"
    mounts = spec.provider_options["mounts"]
    # Bind-mounted at both /r2egym_setup and its original absolute path.
    assert {"src": "/abs/setup", "dst": "/r2egym_setup"} in mounts
    assert {"src": "/abs/setup", "dst": "/abs/setup"} in mounts


def test_build_spec_truncates_long_instance_id():
    harness = R2EGymHarness()
    spec = harness.build_spec(_task(instance_id="x" * 100))
    assert len(spec.metadata["instance_id"]) == 63


def test_supports_provider_apptainer_only():
    harness = R2EGymHarness()
    assert harness.supports_provider("apptainer") is True
    assert harness.supports_provider("docker") is False
    assert harness.supports_provider("fake-r2egym") is False


def test_hide_eval_tests_commands_shape():
    harness = R2EGymHarness()
    commands = harness.hide_eval_tests_commands()
    # One command per checkout root (root, /root, /testbed).
    assert len(commands) == 3
    joined = " ".join(commands)
    assert "rm -rf /r2e_tests" in joined
    assert "rm -rf /root/r2e_tests" in joined
    assert "rm -rf /testbed/r2e_tests" in joined
    # Substring guard before deleting run_tests.sh.
    assert "grep -qs r2e_tests" in commands[0]


# ---- grade() over the nested report.json ------------------------------------


def test_grade_resolved_from_report():
    harness = R2EGymHarness()
    report = _report("r2e__pkg-42", resolved=True)
    out = harness.grade(_task(), EvalArtifacts(test_output=report, return_code=0, raw={"report_json": report}))
    assert out.resolved is True
    assert out.patch_exists is True
    assert reward_from_report(out) == 1.0


def test_grade_unresolved_from_report():
    harness = R2EGymHarness()
    report = _report("r2e__pkg-42", resolved=False)
    out = harness.grade(_task(), EvalArtifacts(test_output=report, return_code=0, raw={"report_json": report}))
    assert out.resolved is False
    assert reward_from_report(out) == 0.0


def test_grade_single_entry_fallback_on_key_mismatch():
    harness = R2EGymHarness()
    # Report keyed by a different id than the task; sole entry is used.
    report = _report("some-other-id", resolved=True)
    out = harness.grade(_task(), EvalArtifacts(test_output=report, return_code=0, raw={"report_json": report}))
    assert out.resolved is True


def test_grade_masks_on_infra_error():
    harness = R2EGymHarness()
    out = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout"}))
    assert out.error_kind == "timeout"
    assert reward_from_report(out) == 0.0


def test_grade_unparseable_report_is_eval_error():
    harness = R2EGymHarness()
    out = harness.grade(_task(), EvalArtifacts(test_output="not json", return_code=0, raw={"report_json": "not json"}))
    assert out.error_kind == "eval_error"
    assert reward_from_report(out) == 0.0


# ---- run_eval over the FakeSandbox ------------------------------------------


def _run_eval(report_text: str, eval_rc: int = 0) -> EvalArtifacts:
    """Run the harness eval over the FakeSandbox and return its artifacts.

    Args:
        report_text: The report contents the provider returns for ``cat``.
        eval_rc: Return code for the nested eval command.

    Returns:
        EvalArtifacts: The artifacts produced by ``run_eval``.
    """

    async def _go() -> EvalArtifacts:
        harness = R2EGymHarness()
        task = _task()
        provider = {"fake-r2egym": {"report_text": report_text, "eval_rc": eval_rc}}
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        try:
            return await harness.run_eval(env, task)
        finally:
            await env.cleanup()

    return asyncio.run(_go())


def test_run_eval_then_grade_resolved():
    report = _report("r2e__pkg-42", resolved=True)
    artifacts = _run_eval(report)
    assert artifacts.return_code == 0
    assert artifacts.patch_applied is True
    out = R2EGymHarness().grade(_task(), artifacts)
    assert out.resolved is True


def test_run_eval_eval_failure_marks_not_applied():
    artifacts = _run_eval("", eval_rc=1)
    assert artifacts.return_code == 1
    assert artifacts.patch_applied is False


# ---- materialize writes the predictions JSONL ------------------------------


def test_materialize_writes_predictions_jsonl():
    # The model patch is delivered to the nested grader via a SWE-bench
    # predictions JSONL keyed by instance_id (--predictions_path), not a bare
    # /root/patch.diff.
    async def _go() -> dict[str, str]:
        harness = R2EGymHarness()
        task = _task(model_patch="diff --git a/x b/x\n+new line\n")
        env = await AsyncSweEnvironment.start({"fake-r2egym": {}}, harness.build_spec(task))
        try:
            await harness.materialize(env, task)
            return dict(env.sandbox._provider.uploads)
        finally:
            await env.cleanup()

    uploads = asyncio.run(_go())
    assert "/root/predictions.jsonl" in uploads
    record = json.loads(uploads["/root/predictions.jsonl"])
    assert record["instance_id"] == "r2e__pkg-42"
    assert record["model_patch"] == "diff --git a/x b/x\n+new line\n"
    assert record["model_name_or_path"] == "nemo-gym"
    # The bare patch.diff path is NOT written for r2e-gym.
    assert "/root/patch.diff" not in uploads


def _r2e_prediction(model_patch: str) -> dict:
    """Materialize a task with the given patch and return the prediction record.

    Args:
        model_patch: The model patch placed on the task.

    Returns:
        dict: The prediction record decoded from the uploaded JSONL.
    """

    async def _go() -> str:
        harness = R2EGymHarness()
        task = _task(model_patch=model_patch)
        env = await AsyncSweEnvironment.start({"fake-r2egym": {}}, harness.build_spec(task))
        try:
            await harness.materialize(env, task)
            return env.sandbox._provider.uploads["/root/predictions.jsonl"]
        finally:
            await env.cleanup()

    return json.loads(asyncio.run(_go()))


def test_materialize_normalizes_patch_trailing_newline():
    # A non-empty patch missing its trailing newline gets one appended before
    # being handed to the nested grader, so the upstream ``git apply`` does not
    # fail.
    assert _r2e_prediction("diff --git a/x b/x")["model_patch"] == "diff --git a/x b/x\n"


def test_materialize_empty_patch_stays_empty():
    # An empty patch stays "" (only a truthy patch is normalized) — it must not
    # become a bare "\n".
    assert _r2e_prediction("")["model_patch"] == ""


def test_materialize_predictions_path_feeds_run_eval():
    # The path materialize writes must match the --predictions_path run_eval
    # passes to run_local_evaluation.py, or the patch is never read.
    harness = R2EGymHarness()

    async def _go() -> list[str]:
        task = _task()
        provider = {"fake-r2egym": {"report_text": _report("r2e__pkg-42", True)}}
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        try:
            await harness.materialize(env, task)
            await harness.run_eval(env, task)
            return list(env.sandbox._provider.commands)
        finally:
            await env.cleanup()

    commands = asyncio.run(_go())
    eval_cmd = next(c for c in commands if "run_local_evaluation.py" in c)
    assert "--predictions_path /root/predictions.jsonl" in eval_cmd


# ---- reset_repo is a no-op for r2e-gym --------------------------------------


def test_reset_repo_is_noop():
    # No host-orchestrated reset happens: the nested run_local_evaluation resets
    # inside its own container. The base `git reset --hard <base_commit>` must NOT
    # fire for r2e-gym.
    async def _go() -> list[str]:
        harness = R2EGymHarness()
        task = _task(base_commit="deadbeef")
        env = await AsyncSweEnvironment.start({"fake-r2egym": {}}, harness.build_spec(task))
        try:
            await harness.reset_repo(env, task)
            return list(env.sandbox._provider.commands)
        finally:
            await env.cleanup()

    commands = asyncio.run(_go())
    assert all("git reset" not in c for c in commands)


# ---- setup-dir mount is on the channel the apptainer provider consumes ------


def test_build_spec_mount_consumed_by_apptainer_provider(tmp_path):
    # The provider reads provider_options["mounts"]; assert the venv setup dir is
    # bound in via the SAME channel _mount_binds reads, so {setup}/R2E-Gym/venv
    # is actually available in the container (no dead no-op). Use a real host dir
    # so the canonical self-bind survives the dataset self-bind guard (which only
    # drops self-binds whose source does not exist on the host).
    from responses_api_agents.swe_env.providers.apptainer_provider import ApptainerSandboxProvider

    setup = tmp_path / "setup"
    setup.mkdir()
    harness = R2EGymHarness()
    spec = harness.build_spec(_task(metadata={"r2egym_setup_dir": str(setup)}))
    binds = ApptainerSandboxProvider._mount_binds(spec)
    assert f"{setup}:/r2egym_setup" in binds
    assert f"{setup}:{setup}" in binds
