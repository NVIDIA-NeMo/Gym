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

"""Unit tests for the nested swe-bench / swe-bench-multilingual harness.

The nested families run the ``run_local_evaluation`` harness inside an apptainer
sandbox. These tests validate provisioning (``build_spec`` / ``supports_provider``
/ ``materialize``) and host-side ``grade`` parsing of a sample ``report.json``
against a scripted ``FakeSandbox``.
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
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask
from responses_api_agents.swe_env.harnesses.swebench import (
    _DATASET_PATH,
    _PREDICTIONS_PATH,
    _REPORT_PATH,
    SweBenchHarness,
)


class _FakeProvider:
    """Scripted sandbox provider for the nested swe-bench harness.

    ``run_local_evaluation`` is a no-op, ``cat`` returns a canned report, and
    uploaded text is recorded so ``materialize`` can be asserted.

    Args:
        report_text: Text returned by any ``cat`` command (the report contents).
        report_rc: Return code for the ``cat`` command.
        eval_rc: Return code for the eval/collect command.
    """

    name = "fake-swebench"

    def __init__(self, *, report_text="", report_rc=0, eval_rc=0, **_):
        self._report_text = report_text
        self._report_rc = report_rc
        self._eval_rc = eval_rc
        self.uploaded: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command.startswith("cat "):
            return SandboxExecResult(stdout=self._report_text, stderr="", return_code=self._report_rc)
        # The eval and collect step.
        return SandboxExecResult(stdout="ran nested harness", stderr="", return_code=self._eval_rc)

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
    """Build a swe-bench ``SweTask`` with sensible defaults.

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
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="swe-bench",
        split="test",
    )
    base.update(overrides)
    return SweTask(**base)


def _sample_report(instance_id: str, resolved: bool) -> str:
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
                "patch_is_None": False,
                "patch_successfully_applied": True,
                "tests_status": {"FAIL_TO_PASS": {"success": ["t::a"], "failure": []}},
            }
        }
    )


# ---- provisioning -----------------------------------------------------------


def test_grade_strategy_is_nested():
    assert SweBenchHarness("swe-bench").grade_strategy == "nested-harness"
    assert SweBenchHarness("swe-bench-multilingual").grade_strategy == "nested-harness"


def test_unknown_family_rejected():
    try:
        SweBenchHarness("not-a-family")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown family")


def test_build_spec_image_and_mounts():
    harness = SweBenchHarness("swe-bench")
    task = _task(metadata={"host_setup_dir": "/host/swe_swebench_setup"})
    spec = harness.build_spec(task)
    assert spec.image == "img:tag"
    assert spec.workdir == "/testbed"
    assert spec.metadata["instance_id"] == "repo__inst-1"
    assert spec.metadata["harness"] == "swe-bench"
    # Mounts live on provider_options (typed dict[str, Any]) — the channel the
    # apptainer provider consumes. metadata is dict[str, str] and never carries
    # the mount list.
    assert "mounts" not in spec.metadata
    mounts = spec.provider_options["mounts"]
    dsts = {m["dst"] for m in mounts}
    assert "/root/dataset/data.jsonl" in dsts
    # Host setup dir bind-mounted at both the alias and its canonical path.
    assert "/swebench_setup" in dsts
    assert "/host/swe_swebench_setup" in dsts
    # Both setup-dir binds point at the host setup dir.
    setup_binds = {m["src"] for m in mounts if m["dst"] in {"/swebench_setup", "/host/swe_swebench_setup"}}
    assert setup_binds == {"/host/swe_swebench_setup"}


def test_build_spec_multilingual_mount_alias():
    harness = SweBenchHarness("swe-bench-multilingual")
    task = _task(benchmark="swe-bench-multilingual", metadata={"host_setup_dir": "/host/ml"})
    spec = harness.build_spec(task)
    mounts = spec.provider_options["mounts"]
    dsts = {m["dst"] for m in mounts}
    # Multilingual alias plus the canonical host path.
    assert "/swebench_multilingual_setup" in dsts
    assert "/host/ml" in dsts


def test_build_spec_preserves_task_provider_options():
    # A task-supplied provider_options (e.g. instance_args) must survive; only
    # the default mounts are filled in when absent.
    harness = SweBenchHarness("swe-bench")
    task = _task(
        metadata={
            "host_setup_dir": "/host/s",
            "provider_options": {"instance_args": ["--nv"]},
        }
    )
    spec = harness.build_spec(task)
    assert spec.provider_options["instance_args"] == ["--nv"]
    assert "mounts" in spec.provider_options


def test_supports_provider_fail_fast_on_docker():
    harness = SweBenchHarness("swe-bench")
    assert harness.supports_provider("apptainer") is True
    assert harness.supports_provider("docker") is False
    assert harness.supports_provider("fake-swebench") is False


def test_materialize_writes_predictions_jsonl():
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    provider = {"fake-swebench": {}}

    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task()
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        # Reach into the underlying provider instance to inspect uploads.
        await harness.materialize(env, task)
        return env.sandbox._provider

    sandbox_provider = asyncio.run(run())
    assert _PREDICTIONS_PATH in sandbox_provider.uploaded
    prediction = json.loads(sandbox_provider.uploaded[_PREDICTIONS_PATH])
    assert prediction["instance_id"] == "repo__inst-1"
    assert prediction["model_patch"] == "diff --git a/x b/x\n"


# ---- grade (sample report.json) ---------------------------------------------


def test_grade_resolved_from_report():
    harness = SweBenchHarness("swe-bench")
    task = _task()
    artifacts = EvalArtifacts(
        test_output="ran",
        return_code=0,
        patch_applied=True,
        raw={"error_type": None, "report_json": _sample_report(task.instance_id, True)},
    )
    report = harness.grade(task, artifacts)
    assert report.resolved is True
    assert report.patch_applied is True
    assert report.patch_exists is True
    assert reward_from_report(report) == 1.0


def test_grade_unresolved_from_report():
    harness = SweBenchHarness("swe-bench")
    task = _task()
    artifacts = EvalArtifacts(raw={"error_type": None, "report_json": _sample_report(task.instance_id, False)})
    report = harness.grade(task, artifacts)
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_infra_error():
    harness = SweBenchHarness("swe-bench")
    report = harness.grade(_task(), EvalArtifacts(raw={"error_type": "timeout"}))
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_missing_report():
    harness = SweBenchHarness("swe-bench")
    report = harness.grade(_task(), EvalArtifacts(raw={"error_type": None, "report_json": ""}))
    assert report.error_kind == "eval_error"
    assert reward_from_report(report) == 0.0


# ---- run_eval (FakeSandbox: nested command issued, report read back) --------


def test_run_eval_reads_report_and_grades():
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    task = _task()
    report_text = _sample_report(task.instance_id, True)
    provider = {"fake-swebench": {"report_text": report_text}}

    async def run():
        harness = SweBenchHarness("swe-bench")
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        await harness.materialize(env, task)
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts), artifacts

    report, artifacts = asyncio.run(run())
    assert artifacts.raw["report_json"] == report_text
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_run_eval_report_path_constant_is_stable():
    # The collect step copies the nested harness report to this fixed path.
    assert _REPORT_PATH == "/root/report.json"


class _RecordingProvider(_FakeProvider):
    """Like ``_FakeProvider`` but records every command issued to ``exec``."""

    name = "rec-swebench"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.commands: list[str] = []

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
        return await super().exec(handle, command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


register_provider("rec-swebench", _RecordingProvider, override=True)


def _eval_command(*, family="swe-bench", **task_overrides) -> str:
    """Run run_eval through a recording provider and return the eval command string.

    Args:
        family: The swe-bench family to instantiate the harness for.
        **task_overrides: Field values overriding the task defaults.

    Returns:
        str: The single eval command containing ``run_local_evaluation``.
    """
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    async def run():
        harness = SweBenchHarness(family)
        task = _task(benchmark=family, **task_overrides)
        env = await AsyncSweEnvironment.start({"rec-swebench": {}}, harness.build_spec(task))
        await harness.materialize(env, task)
        await harness.run_eval(env, task)
        # The eval+collect step is the only non-``cat`` command issued.
        cmds = [c for c in env.sandbox._provider.commands if "run_local_evaluation" in c]
        return cmds[0]

    return asyncio.run(run())


def test_run_eval_command_has_uv_exports_swebench():
    # The command exports UV_INSTALL_DIR / UV_PYTHON_INSTALL_DIR / PATH pointing
    # at the mounted portable uv+python so the prebuilt venv resolves its
    # hardcoded toolchain.
    cmd = _eval_command(family="swe-bench", metadata={"setup_dir": "/swebench_setup"})
    assert 'export UV_INSTALL_DIR="/swebench_setup/uv"' in cmd
    assert 'export UV_PYTHON_INSTALL_DIR="/swebench_setup/python"' in cmd
    assert 'export PATH="/swebench_setup/uv/bin:$PATH"' in cmd
    # Dataset and prebuilt venv paths.
    assert "--dataset_name /root/dataset/data.jsonl" in cmd
    assert "/swebench_setup/SWE-bench/venv/bin/python" in cmd
    assert "env -u VIRTUAL_ENV" in cmd
    assert "cd /swebench_setup/SWE-bench " in cmd


def test_run_eval_command_has_uv_exports_multilingual():
    cmd = _eval_command(family="swe-bench-multilingual", metadata={"setup_dir": "/swebench_multilingual_setup"})
    assert 'export UV_INSTALL_DIR="/swebench_multilingual_setup/uv"' in cmd
    assert 'export UV_PYTHON_INSTALL_DIR="/swebench_multilingual_setup/python"' in cmd
    assert 'export PATH="/swebench_multilingual_setup/uv/bin:$PATH"' in cmd
    assert "/swebench_multilingual_setup/SWE-bench_Multilingual/venv/bin/python" in cmd


def test_run_eval_command_uses_custom_setup_dir():
    # When the verifier provisions a real host setup dir, the exports + venv path
    # track it (uv venvs hardcode this absolute path).
    cmd = _eval_command(family="swe-bench", metadata={"setup_dir": "/host/swe_swebench_setup"})
    assert 'export UV_INSTALL_DIR="/host/swe_swebench_setup/uv"' in cmd
    assert "/host/swe_swebench_setup/SWE-bench/venv/bin/python" in cmd


# ---- mount source and eval cd/UV/venv path read ONE unified key -------------


def _setup_srcs(spec) -> set[str]:
    """Return the host source(s) of the setup-dir binds (every non-dataset mount).

    Args:
        spec: The sandbox spec whose ``provider_options["mounts"]`` is inspected.

    Returns:
        set[str]: The set of host source paths for non-dataset mounts.
    """
    return {m["src"] for m in spec.provider_options["mounts"] if m["dst"] != _DATASET_PATH}


def test_setup_key_unified_via_host_setup_dir():
    # build_spec mounts the host dir from the SAME key that run_eval reads for
    # cd/UV/venv, so the prebuilt venv is bound and invoked from one path.
    harness = SweBenchHarness("swe-bench")
    meta = {"host_setup_dir": "/host/swe_swebench_setup"}
    spec = harness.build_spec(_task(metadata=dict(meta)))
    cmd = _eval_command(family="swe-bench", metadata=dict(meta))
    # The bind source is exactly the path the eval command cd's into and runs
    # the venv from.
    assert _setup_srcs(spec) == {"/host/swe_swebench_setup"}
    assert "cd /host/swe_swebench_setup/SWE-bench " in cmd
    assert "/host/swe_swebench_setup/SWE-bench/venv/bin/python" in cmd
    assert 'export UV_INSTALL_DIR="/host/swe_swebench_setup/uv"' in cmd


def test_setup_key_unified_via_setup_dir():
    # The same unified key resolves through the ``setup_dir`` alias too, so either
    # config key keeps mount-source and eval-path in lockstep.
    harness = SweBenchHarness("swe-bench")
    meta = {"setup_dir": "/host/alt_setup"}
    spec = harness.build_spec(_task(metadata=dict(meta)))
    cmd = _eval_command(family="swe-bench", metadata=dict(meta))
    assert _setup_srcs(spec) == {"/host/alt_setup"}
    assert "cd /host/alt_setup/SWE-bench " in cmd
    assert "/host/alt_setup/SWE-bench/venv/bin/python" in cmd


def test_setup_key_default_alias_is_self_consistent():
    # With NO host dir provisioned (SWE-bench-Verified default), both halves fall
    # back to the family in-container alias, so the bind target == cd path and the
    # venv resolves against the alias-mounted setup.
    harness = SweBenchHarness("swe-bench")
    spec = harness.build_spec(_task())
    cmd = _eval_command(family="swe-bench")
    assert _setup_srcs(spec) == {"/swebench_setup"}
    assert "cd /swebench_setup/SWE-bench " in cmd
    assert "/swebench_setup/SWE-bench/venv/bin/python" in cmd


# ---- model_patch trailing-newline normalization in predictions -------------


def _materialized_prediction(**task_overrides) -> dict:
    """Materialize a task and return the parsed predictions record.

    Args:
        **task_overrides: Field values overriding the task defaults.

    Returns:
        dict: The single prediction record decoded from the uploaded JSONL.
    """
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task(**task_overrides)
        env = await AsyncSweEnvironment.start({"fake-swebench": {}}, harness.build_spec(task))
        await harness.materialize(env, task)
        return env.sandbox._provider.uploaded[_PREDICTIONS_PATH]

    return json.loads(asyncio.run(run()))


def test_materialize_normalizes_patch_trailing_newline():
    # A non-empty patch missing its trailing newline gets one appended so the
    # upstream ``git apply`` does not fail.
    prediction = _materialized_prediction(model_patch="diff --git a/x b/x")
    assert prediction["model_patch"] == "diff --git a/x b/x\n"


def test_materialize_preserves_existing_trailing_newline():
    # An already-terminated patch is left byte-for-byte unchanged (no double \n).
    prediction = _materialized_prediction(model_patch="diff --git a/x b/x\n")
    assert prediction["model_patch"] == "diff --git a/x b/x\n"


def test_materialize_empty_patch_stays_empty():
    # Only a truthy patch is normalized; an empty patch stays "" (it must not
    # become a bare "\n", which would be a non-empty no-op patch to the grader).
    prediction = _materialized_prediction(model_patch="")
    assert prediction["model_patch"] == ""


# ---- eval timeout threaded into env.execute ---------------------------------


class _TimeoutRecordingProvider(_FakeProvider):
    """Records the ``timeout_s`` passed to the eval exec call."""

    name = "timeout-swebench"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.eval_timeout_s = None

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "run_local_evaluation" in command:
            self.eval_timeout_s = timeout_s
        return await super().exec(handle, command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


register_provider("timeout-swebench", _TimeoutRecordingProvider, override=True)


def test_run_eval_threads_eval_timeout():
    # run_eval must pass timeout_s = tests_timeout + 120 so a stuck nested harness
    # is killed and masked rather than hanging the verifier.
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task(metadata={"tests_timeout": 600})
        env = await AsyncSweEnvironment.start({"timeout-swebench": {}}, harness.build_spec(task))
        await harness.materialize(env, task)
        await harness.run_eval(env, task)
        return env.sandbox._provider.eval_timeout_s

    assert asyncio.run(run()) == 600 + 120


def test_run_eval_threads_default_eval_timeout():
    # With no explicit tests_timeout the default (1800) + 120 headroom is used.
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task()
        env = await AsyncSweEnvironment.start({"timeout-swebench": {}}, harness.build_spec(task))
        await harness.materialize(env, task)
        await harness.run_eval(env, task)
        return env.sandbox._provider.eval_timeout_s

    assert asyncio.run(run()) == 1800 + 120
