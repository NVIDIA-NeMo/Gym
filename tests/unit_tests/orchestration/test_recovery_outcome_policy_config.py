# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`driver.benchmarks.<name>.resumable.recovery_outcome_policy`: a benchmark's partial override of the recovery
outcome policy, read from the run's saved config, with the Python API's validation and an explicit precedence."""

import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.agents.agent_outcome import OutcomeAction
from nemo_gym.orchestration.api import ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord, load_submission_record
from nemo_gym.orchestration.recovery import RecoveryAction, RecoveryOutcomePolicyConfig
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_eval_resubmit_decision import INFRA


def _submit_config(tmp_path: Path, policy: dict | None) -> SubmitConfig:
    resumable = {"max_retries": 3, "max_walltime": "16:00:00"}
    if policy is not None:
        resumable["recovery_outcome_policy"] = policy
    return SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "a", "hostname": None}},
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {"resumable": resumable}}},
            "job": {"output_path": str(tmp_path)},
            "otel": {"enabled": False},
        }
    )


def _failed_run(tmp_path: Path, policy: dict | None, failure: dict) -> Path:
    """A submitted resumable run: one scored row and one saved failure, spent budgets left over."""
    config = _submit_config(tmp_path, policy)
    run_dir = tmp_path / f"gym-job-{tmp_path.name}"
    job_dir = run_dir / "gsm8k"
    (job_dir / "artifacts").mkdir(parents=True)
    record = SubmissionRecord(
        gym_job_id=run_dir.name,
        gym_version="0",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir=str(run_dir),
        cluster="cluster",
        executor="slurm",
        submitted_by="someone",
        benchmarks=[BenchmarkJob(benchmark="gsm8k", job_dir=str(job_dir), job_id="100")],
    )
    SlurmExecutor().persist(record, config, LocalConnection().write_text)
    compute = next(iter(config.compute.values()))
    (job_dir / "job.sh").write_text(
        build_sbatch_script(config, "gsm8k", config.driver.benchmarks["gsm8k"], compute, job_dir)
    )
    rows = [{"_ng_task_index": t, "_ng_rollout_index": 0} for t in range(2)]
    artifacts = job_dir / "artifacts"
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (artifacts / "rollouts.jsonl").write_text(json.dumps(rows[0] | {"reward": 1.0}) + "\n")
    (artifacts / "rollouts_failures.jsonl").write_text(json.dumps(rows[1] | failure) + "\n")
    (job_dir / ".gym_job_chain").write_text("100\n101\n")
    (job_dir / ".gym_infra_retries").write_text("1\n")
    (job_dir / ".gym_accumulated_walltime").write_text("7200\n")
    return job_dir


# A nonterminal agent timeout and a nonterminal model rejection with HTTP 429: both retry by default.
AGENT_TIMEOUT = INFRA | {"failure_kind": "agent_timeout"}
RATE_LIMITED = INFRA | {"_ng_failure_http_status": 429}


# --- Partial merge and validation -------------------------------------------------------------------------------


def test_the_config_section_is_a_partial_override_of_the_default_policy() -> None:
    from nemo_gym.orchestration.recovery import configured_outcome_policy

    resume = ResumeConfig(
        recovery_outcome_policy={"by_failure_kind": {"agent_timeout": "fail"}, "by_status_code": {429: "fail"}}
    )
    policy = configured_outcome_policy(resume.recovery_outcome_policy)
    default = RecoveryOutcomePolicyConfig()
    # Named entries change; every entry it does not name keeps its default, as with the Python API.
    assert policy.by_failure_kind == default.by_failure_kind | {"agent_timeout": OutcomeAction.FAIL}
    assert policy.by_status_code == default.by_status_code | {429: OutcomeAction.FAIL}
    assert policy.default_action == default.default_action
    assert policy == RecoveryOutcomePolicyConfig(
        by_failure_kind={"agent_timeout": "fail"}, by_status_code={429: "fail"}
    )
    # YAML and JSON spell a status code as a string key: it is the same override.
    assert configured_outcome_policy({"by_status_code": {"429": "fail"}}).by_status_code[429] == OutcomeAction.FAIL


def test_an_unset_section_keeps_saved_configs_unchanged() -> None:
    assert ResumeConfig().model_dump() == {"max_retries": 3, "max_walltime": None}
    assert "recovery_outcome_policy" not in ResumeConfig().model_dump_json()


@pytest.mark.parametrize(
    "policy,message",
    [
        ({"by_failure_kind": {"agent_timout": "fail"}}, "Unregistered failure kinds: ['agent_timout']"),
        ({"by_status_code": {399: "fail"}}, "greater than or equal to 400"),
        ({"by_status_code": {600: "fail"}}, "less than or equal to 599"),
        ({"by_status_code": {"rate_limited": "fail"}}, "valid integer"),
        ({"default_action": "count_as_zero"}, "does not support count_as_zero"),
        ({"by_failure_kind": {"agent_timeout": "count_as_zero"}}, "does not support count_as_zero"),
        ({"by_status_code": {429: "count_as_zero"}}, "does not support count_as_zero"),
        ({"by_failure_kind": {"agent_timeout": "skip"}}, "'verify', 'retry', 'fail' or 'count_as_zero'"),
        ({"by_kind": {"agent_timeout": "fail"}}, "Extra inputs are not permitted"),
    ],
)
def test_a_bad_section_fails_config_load_with_the_python_api_validation(tmp_path, policy, message) -> None:
    with pytest.raises(ValidationError) as config_error:
        _submit_config(tmp_path, policy)
    assert "recovery_outcome_policy" in str(config_error.value) and message in str(config_error.value)
    # The Python API refuses the same section for the same reason.
    with pytest.raises(ValidationError, match=re.escape(message)):
        RecoveryOutcomePolicyConfig.model_validate(policy)


def test_a_bad_section_in_the_saved_config_fails_recovery_loudly(tmp_path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import inspect_recovery

    job_dir = _failed_run(tmp_path, {"by_failure_kind": {"agent_timeout": "fail"}}, AGENT_TIMEOUT)
    resolved = job_dir.parent / "resolved-config.yaml"
    saved = yaml.safe_load(resolved.read_text())
    saved["driver"]["benchmarks"]["gsm8k"]["resumable"]["recovery_outcome_policy"] = {
        "by_failure_kind": {"agent_timout": "fail"}
    }
    resolved.write_text(yaml.safe_dump(saved, sort_keys=False))
    with pytest.raises(ValidationError, match=r"recovery_outcome_policy.*Unregistered failure kinds"):
        inspect_recovery(*_handle(job_dir))


# --- Precedence -------------------------------------------------------------------------------------------------


def test_python_wins_over_config_which_wins_over_the_defaults() -> None:
    from nemo_gym.orchestration.recovery import effective_outcome_policy

    configured = {"by_failure_kind": {"agent_timeout": "fail"}}
    explicit = RecoveryOutcomePolicyConfig(by_status_code={429: "fail"})
    assert effective_outcome_policy(None, None) == (RecoveryOutcomePolicyConfig(), "default")
    assert effective_outcome_policy(None, configured) == (
        RecoveryOutcomePolicyConfig(by_failure_kind={"agent_timeout": "fail"}),
        "config",
    )
    # An explicit argument replaces the config section; it is not merged with it.
    assert effective_outcome_policy(explicit, configured) == (explicit, "python")


# --- Snapshot read and the end-to-end decision ------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy,failure",
    [
        ({"by_failure_kind": {"agent_timeout": "fail"}}, AGENT_TIMEOUT),
        ({"by_status_code": {429: "fail"}}, RATE_LIMITED),
        # A default of fail with one status code retried: the config can also widen what recovery retries.
        ({"default_action": "fail", "by_status_code": {429: "retry"}}, INFRA),
    ],
)
def test_the_saved_config_changes_the_recovery_decision(tmp_path, slurm_bin, policy, failure) -> None:
    from nemo_gym.orchestration.submit import inspect_recovery

    default_dir = _failed_run(tmp_path / "default", None, failure)
    configured_dir = _failed_run(tmp_path / "configured", policy, failure)
    default = inspect_recovery(*_handle(default_dir)).decision
    configured = inspect_recovery(*_handle(configured_dir)).decision
    assert default.action == RecoveryAction.RESUBMIT and default.identities["retryable"] == [(1, 0)]
    assert configured.action == RecoveryAction.STOP and configured.identities["terminal"] == [(1, 0)]
    assert default.outcome_policy == {"source": "default", **RecoveryOutcomePolicyConfig().model_dump(mode="json")}
    assert configured.outcome_policy == {
        "source": "config",
        **RecoveryOutcomePolicyConfig.model_validate(policy).model_dump(mode="json"),
    }


def test_a_status_code_retried_by_config_resubmits_a_run_the_default_fail_would_stop(tmp_path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import inspect_recovery

    base = {"default_action": "fail"}
    stopped = inspect_recovery(*_handle(_failed_run(tmp_path / "a", base, RATE_LIMITED))).decision
    retried = _failed_run(tmp_path / "b", base | {"by_status_code": {429: "retry"}}, RATE_LIMITED)
    resumed = inspect_recovery(*_handle(retried)).decision
    assert stopped.action == RecoveryAction.STOP and stopped.identities["terminal"] == [(1, 0)]
    assert resumed.action == RecoveryAction.RESUBMIT and resumed.identities["retryable"] == [(1, 0)]


def test_an_explicit_python_policy_wins_over_the_saved_config(tmp_path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job_dir = _failed_run(tmp_path, {"by_failure_kind": {"agent_timeout": "fail"}}, AGENT_TIMEOUT)
    explicit = RecoveryOutcomePolicyConfig(by_failure_kind={"agent_timeout": "retry"})
    inspected = inspect_recovery(*_handle(job_dir), outcome_policy=explicit).decision
    assert inspected.action == RecoveryAction.RESUBMIT and inspected.outcome_policy["source"] == "python"
    # The CLI budget flags are orthogonal: an extension does not change which policy applies.
    extended = inspect_recovery(*_handle(job_dir), extend_retries=2).decision
    assert extended.action == RecoveryAction.STOP and extended.outcome_policy["source"] == "config"
    out = recover(*_handle(job_dir), outcome_policy=explicit)
    assert out.dispatched == "resubmit", out.decision.reason


def test_recovery_reads_the_policy_from_the_saved_snapshot_and_the_cli_reports_it(tmp_path, slurm_bin, capsys):
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import run_cli

    job_dir = _failed_run(tmp_path, {"by_failure_kind": {"agent_timeout": "fail"}}, AGENT_TIMEOUT)
    # The local record names the run directory: recovery reads that directory's saved config, nothing else.
    record = load_submission_record(job_dir.parent.name)
    assert Path(record.run_dir) == job_dir.parent
    _, out = run_cli(["eval", "resubmit", job_dir.parent.name, "gsm8k", "--recover", "--inspect"], capsys)
    assert out["decision"]["outcome_policy"]["source"] == "config"
    assert out["decision"]["outcome_policy"]["by_failure_kind"]["agent_timeout"] == "fail"
    assert out["decision"]["action"] == "stop"


# --- A resubmit with a changed policy ---------------------------------------------------------------------------


def test_a_policy_changed_after_the_evidence_was_collected_refuses_the_resubmit(tmp_path, slurm_bin) -> None:
    job_dir = _failed_run(tmp_path, {"by_status_code": {429: "fail"}}, AGENT_TIMEOUT)
    executor = SlurmExecutor()
    record = load_submission_record(job_dir.parent.name)
    decision, collected = executor.recovery_evidence(record, "gsm8k")
    assert decision.action == RecoveryAction.RESUBMIT
    script = (job_dir / "job.sh").read_text()
    resolved = job_dir.parent / "resolved-config.yaml"
    saved = yaml.safe_load(resolved.read_text())
    saved["driver"]["benchmarks"]["gsm8k"]["resumable"]["recovery_outcome_policy"] = {
        "by_failure_kind": {"agent_timeout": "fail"}
    }
    resolved.write_text(yaml.safe_dump(saved, sort_keys=False))
    # The policy decides which rows run again: a change since the decision is refused, never resumed across.
    with pytest.raises(RuntimeError, match="changed since the recovery evidence was collected"):
        executor.resubmit(
            record,
            "gsm8k",
            keep_budget=True,
            accounted_walltime_s=collected.accounted_walltime_s,
            accounting_snapshot=collected.accounting_snapshot,
        )
    assert (job_dir / "job.sh").read_text() == script and not (slurm_bin / "env").exists()


def test_the_in_job_continuation_applies_the_saved_policy(tmp_path) -> None:
    config = _submit_config(tmp_path, {"by_failure_kind": {"agent_timeout": "fail"}})
    plain = _submit_config(tmp_path, None)
    compute = next(iter(config.compute.values()))
    job_dir = tmp_path / "job"
    script = build_sbatch_script(config, "gsm8k", config.driver.benchmarks["gsm8k"], compute, job_dir)
    unset = build_sbatch_script(plain, "gsm8k", plain.driver.benchmarks["gsm8k"], compute, job_dir)
    # The admission step that decides an automatic continuation is rendered with the run's policy section...
    assert '{"by_failure_kind": {"agent_timeout": "fail"}}' in script and "sys.argv[7]" in script
    # ...and a run without one renders exactly as before.
    assert "sys.argv[7]" not in unset and "recovery_outcome_policy" not in unset


def test_the_continuation_admission_decides_with_the_saved_policy(tmp_path, monkeypatch) -> None:
    from nemo_gym.orchestration import recovery
    from tests.unit_tests.orchestration.test_allocation_cut_recovery import _cut_run

    job = tmp_path / "job"
    _cut_run(job, scored=2, in_flight=0, not_started=2)
    seen = []
    decide = recovery.decide_from_collected
    monkeypatch.setattr(
        recovery,
        "decide_from_collected",
        lambda collected, **options: seen.append(options) or decide(collected, **options),
    )
    policy = json.dumps({"by_failure_kind": {"agent_timeout": "fail"}})
    recovery.continuation_admission(str(job), 2, None, "0:15", "100", "TIMEOUT", policy)
    [options] = seen
    assert options["outcome_policy"] == RecoveryOutcomePolicyConfig(by_failure_kind={"agent_timeout": "fail"})
    assert options["outcome_policy_source"] == "config"
