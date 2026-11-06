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
"""`gym eval resubmit --recover --repin`: a recorded move of a run to another Gym commit, rows and budgets untouched."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import nemo_gym.orchestration.executors.slurm as slurm_executor
from nemo_gym.cli.main import build_parser
from nemo_gym.orchestration.api import ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.slurm import REPIN_JOURNAL_NAME, SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin


OLD, NEW = "a" * 40, "d" * 40


def _repinnable_run(tmp_path: Path, config_paths: list[str] | None = None) -> Path:
    """One benchmark staged on Gym OLD with its MLflow pin tags, one saved row, one missing row and spent budgets."""
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "a", "hostname": None}},
            "driver": {
                "container": "python:3.12",
                "gym_install": {"repo": "https://example.invalid/gym", "ref": OLD},
                "benchmarks": {
                    "gsm8k": {
                        "run": {
                            **({"config_paths": config_paths} if config_paths else {}),
                            "mlflow_run_tags": {"gym_sha": f"'{OLD}'", "gym_commit": f"'{OLD}'", "model": "'m'"},
                        },
                        "resumable": ResumeConfig(max_retries=3, max_walltime="16:00:00").model_dump(),
                        **({"prepare": {"config_paths": config_paths[:1]}} if config_paths else {}),
                    }
                },
            },
            "job": {"output_path": str(tmp_path)},
            "otel": {"enabled": False},
        }
    )
    run_dir = tmp_path / "gym-job-x"
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
    (job_dir / "artifacts" / "rollouts_materialized_inputs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (job_dir / "artifacts" / "rollouts.jsonl").write_text(json.dumps(rows[0] | {"reward": 1.0}) + "\n")
    (job_dir / ".gym_job_chain").write_text("100\n101\n")
    (job_dir / ".gym_infra_retries").write_text("2\n")
    (job_dir / ".gym_accumulated_walltime").write_text("7200\n")
    return job_dir


def _digests(job_dir: Path) -> dict[str, str]:
    names = ["artifacts/rollouts.jsonl", "artifacts/rollouts_materialized_inputs.jsonl", ".gym_infra_retries"]
    return {name: hashlib.sha256((job_dir / name).read_bytes()).hexdigest() for name in names}


@pytest.fixture
def launcher_at(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_SUBMIT_NO_REF_CHECK", "true")

    def set_commit(commit):
        monkeypatch.setattr(slurm_executor, "installed_gym_commit", lambda: commit)

    return set_commit


def test_a_repin_renders_the_run_again_on_the_new_commit_and_keeps_rows_and_budgets(tmp_path, slurm_bin, launcher_at):
    from nemo_gym.orchestration.submit import recover

    launcher_at(NEW)
    job_dir = _repinnable_run(tmp_path)
    run_dir = job_dir.parent
    before = _digests(job_dir)
    old_script = (job_dir / "job.sh").read_text()
    out = recover(*_handle(job_dir), repin=NEW, repin_reason="continue on the fixed revision")
    assert out.dispatched == "resubmit", out.decision.reason

    script = (job_dir / "job.sh").read_text()
    assert NEW in script and OLD not in script
    [kept_script] = job_dir.glob("job.sh.pre-repin-*")
    assert kept_script.read_text() == old_script
    saved = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
    assert saved["driver"]["gym_install"]["ref"] == NEW
    assert saved["driver"]["benchmarks"]["gsm8k"]["run"]["mlflow_run_tags"] == {
        "gym_sha": f"'{NEW}'",
        "gym_commit": f"'{NEW}'",
        "model": "'m'",
    }
    [kept_config] = run_dir.glob("resolved-config.yaml.pre-repin-*")
    assert yaml.safe_load(kept_config.read_text())["driver"]["gym_install"]["ref"] == OLD
    [entry] = [json.loads(line) for line in (run_dir / REPIN_JOURNAL_NAME).read_text().splitlines()]
    assert (entry["benchmark"], entry["from"], entry["to"], entry["reason"]) == (
        "gsm8k",
        OLD,
        NEW,
        "continue on the fixed revision",
    )
    [job] = json.loads((run_dir / "gym-job.json").read_text())["benchmarks"]
    assert (job["job_id"], job["gym_commit"]) == ("4242", NEW)
    # Saved rows and the retry counter are untouched; the walltime keeps its terminal accounting (7200 + 60).
    assert _digests(job_dir) == before
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "7260\n"


@pytest.mark.parametrize(
    "repin,launcher,error",
    [
        ("main", "main", "full 40-character commit hash"),
        (NEW, "e" * 40, "not the repin target"),
    ],
)
def test_a_repin_is_refused_before_anything_changes(tmp_path, slurm_bin, launcher_at, repin, launcher, error):
    from nemo_gym.orchestration.submit import recover

    launcher_at(launcher)
    job_dir = _repinnable_run(tmp_path)
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", job_dir.parent / "resolved-config.yaml"]}
    with pytest.raises((RuntimeError, ValueError), match=error):
        recover(*_handle(job_dir), repin=repin, repin_reason="reason")
    assert {path: path.read_bytes() for path in files} == files
    assert not (slurm_bin / "env").exists()
    assert not (job_dir.parent / REPIN_JOURNAL_NAME).exists()


def test_a_repin_whose_job_is_not_queued_restores_both_files(tmp_path, slurm_bin, launcher_at):
    from nemo_gym.orchestration.submit import recover

    launcher_at(NEW)
    job_dir = _repinnable_run(tmp_path)
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", job_dir.parent / "resolved-config.yaml"]}
    (slurm_bin / "sbatch").write_text("#!/bin/bash\necho 'sbatch: error: Batch job submission failed'; exit 1\n")
    out = recover(*_handle(job_dir), repin=NEW, repin_reason="reason")
    assert not out.ok and out.dispatched is None
    assert out.scheduler_refusal["scheduler_text"] == "sbatch: error: Batch job submission failed"
    assert out.scheduler_refusal["job_dir_restored"] is True
    assert {path: path.read_bytes() for path in files} == files
    assert not (job_dir.parent / REPIN_JOURNAL_NAME).exists()


def test_the_resubmit_help_lists_repin_and_a_repin_needs_recover(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "resubmit", "--help"])
    assert "--repin" in capsys.readouterr().out
    from nemo_gym.cli.main import _eval_resubmit

    args = parser.parse_args(["eval", "resubmit", "job", "bench", "--repin", NEW, "--repin-reason", "x"])
    with pytest.raises(SystemExit):
        _eval_resubmit(args, [])
    assert "--repin and --repin-reason need --recover" in capsys.readouterr().err


def test_an_inspect_at_a_repin_reads_the_plan_there_and_changes_nothing(tmp_path, slurm_bin, launcher_at, monkeypatch):
    # The dry run of a repinned recovery: the Store is read at the target commit, nothing is rendered or queued.
    from nemo_gym.orchestration.submit import inspect_recovery

    launcher_at(NEW)
    job_dir = _repinnable_run(tmp_path)
    before = _digests(job_dir) | {"job.sh": hashlib.sha256((job_dir / "job.sh").read_bytes()).hexdigest()}
    seen = []
    real = SlurmExecutor.recovery_evidence

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("reader_ref"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(SlurmExecutor, "recovery_evidence", spy)
    out = inspect_recovery(*_handle(job_dir), repin=NEW, repin_reason="preview the fixed revision")
    assert seen == [NEW]
    assert out.dispatched is None
    assert any(w.startswith(f"INSPECTED AT REPIN {NEW}") for w in out.gate_task["warnings"]), out.gate_task
    after = _digests(job_dir) | {"job.sh": hashlib.sha256((job_dir / "job.sh").read_bytes()).hexdigest()}
    assert after == before
    assert not list(job_dir.glob("job.sh.pre-repin-*")) and not (job_dir / REPIN_JOURNAL_NAME).exists()


# --- A run pinned at another commit is planned on the installed Gym; its pin moves only when the plan needs it. ---


def _pinned_plans_differently(monkeypatch) -> None:
    """The pinned revision's Store reader plans fewer rows than the installed one (a row only newer rules plan)."""
    real = SlurmExecutor.recovery_evidence

    def evidence(self, *args, **kwargs):
        decision, collected = real(self, *args, **kwargs)
        if kwargs.get("reader_ref") is None:
            decision = decision.model_copy(update={"counts": {**decision.counts, "unowned": 1}})
        return decision, collected

    monkeypatch.setattr(SlurmExecutor, "recovery_evidence", evidence)


def test_an_old_pin_run_the_pinned_revision_plans_alike_keeps_its_pin(tmp_path, slurm_bin, launcher_at):
    from nemo_gym.orchestration.submit import recover

    launcher_at(NEW)
    job_dir = _repinnable_run(tmp_path)
    out = recover(*_handle(job_dir))
    assert out.dispatched == "resubmit", out.decision.reason
    assert out.pin_plan == {
        "installed": NEW,
        "pinned": OLD,
        "repin": False,
        "reason": f"planned on {NEW} for a run pinned at {OLD}; the pin stays",
    }
    assert out.pin_plan["reason"] in out.gate_task["warnings"]
    assert OLD in (job_dir / "job.sh").read_text() and not (job_dir.parent / REPIN_JOURNAL_NAME).exists()


def test_an_old_pin_run_whose_plan_needs_the_installed_revision_is_repinned_automatically(
    tmp_path, slurm_bin, launcher_at, monkeypatch
):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    launcher_at(NEW)
    job_dir = _repinnable_run(tmp_path)
    _pinned_plans_differently(monkeypatch)
    before = (job_dir / "job.sh").read_text()
    # --inspect reports the same plan and pin decision, and changes nothing.
    inspected = inspect_recovery(*_handle(job_dir))
    assert inspected.dispatched is None and inspected.pin_plan["repin"] is True
    assert (job_dir / "job.sh").read_text() == before and not (job_dir.parent / REPIN_JOURNAL_NAME).exists()

    out = recover(*_handle(job_dir))
    assert out.dispatched == "resubmit", out.decision.reason
    assert out.pin_plan == inspected.pin_plan
    reason = out.pin_plan["reason"]
    assert reason.startswith(f"automatic repin: planned on {NEW} for a run pinned at {OLD}: the pinned revision plans")
    assert NEW in (job_dir / "job.sh").read_text()
    [entry] = [json.loads(line) for line in (job_dir.parent / REPIN_JOURNAL_NAME).read_text().splitlines()]
    assert (entry["from"], entry["to"], entry["reason"]) == (OLD, NEW, reason)
    [job] = json.loads((job_dir.parent / "gym-job.json").read_text())["benchmarks"]
    assert job["gym_commit"] == NEW


def test_no_auto_repin_refuses_a_plan_that_needs_the_installed_revision_and_names_repin(
    tmp_path, slurm_bin, launcher_at, monkeypatch
):
    from nemo_gym.orchestration.submit import recover

    launcher_at(NEW)
    job_dir = _repinnable_run(tmp_path)
    _pinned_plans_differently(monkeypatch)
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", job_dir.parent / "resolved-config.yaml"]}
    out = recover(*_handle(job_dir), auto_repin=False)
    assert not out.ok and out.dispatched is None and out.decision.action.value == "stop"
    assert f"--repin {NEW} --repin-reason TEXT" in out.decision.reason and out.pin_plan["refused"] is True
    assert {path: path.read_bytes() for path in files} == files
    assert not (slurm_bin / "env").exists() and not (job_dir.parent / REPIN_JOURNAL_NAME).exists()


def test_no_auto_repin_needs_recover_and_excludes_repin(capsys):
    from nemo_gym.cli.main import _eval_resubmit

    parser = build_parser()
    for argv in (
        ["eval", "resubmit", "job", "bench", "--no-auto-repin"],
        ["eval", "resubmit", "job", "bench", "--recover", "--no-auto-repin", "--repin", NEW, "--repin-reason", "x"],
    ):
        with pytest.raises(SystemExit):
            _eval_resubmit(parser.parse_args(argv), [])
        assert "--no-auto-repin needs --recover and cannot be used with --repin" in capsys.readouterr().err


# --- An automatic repin never moves a run to an installed commit that lacks the config files the run launched with. ---

BENCHMARK_CONFIG = "benchmarks/<benchmark>/config.yaml"


@pytest.fixture
def installed_tree(tmp_path, monkeypatch):
    """A stand-in install root holding the files named; the recovery reads the installed Gym's files there."""
    import nemo_gym.orchestration.submit as submit_module

    root = tmp_path / "installed"
    root.mkdir()
    monkeypatch.setattr(submit_module, "PARENT_DIR", root)

    def holds(*names):
        for name in names:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("{}\n")

    return holds


@pytest.mark.parametrize("auto_repin", [True, False])
def test_a_repin_is_refused_when_the_installed_commit_lacks_a_config_path_of_the_run(
    tmp_path, slurm_bin, launcher_at, monkeypatch, installed_tree, auto_repin
):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    launcher_at(NEW)
    installed_tree("benchmarks/other/config.yaml")
    job_dir = _repinnable_run(tmp_path, [BENCHMARK_CONFIG, "benchmarks/other/config.yaml"])
    _pinned_plans_differently(monkeypatch)
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", job_dir.parent / "resolved-config.yaml"]}

    out = recover(*_handle(job_dir), auto_repin=auto_repin)
    assert not out.ok and out.dispatched is None and out.decision.action.value == "stop"
    assert out.pin_plan["refused"] is True and out.pin_plan["repin"] is False
    assert out.pin_plan["missing_config_paths"] == [BENCHMARK_CONFIG]
    assert f"lacks config_paths '{BENCHMARK_CONFIG}'" in out.decision.reason
    assert "benchmarks/other/config.yaml" not in out.decision.reason
    assert out.decision.reason in out.gate_task["errors"]
    assert {path: path.read_bytes() for path in files} == files
    assert not (slurm_bin / "env").exists() and not (job_dir.parent / REPIN_JOURNAL_NAME).exists()
    # --inspect reports the same refusal and changes nothing.
    assert inspect_recovery(*_handle(job_dir), auto_repin=auto_repin).pin_plan == out.pin_plan


def test_a_repin_proceeds_when_the_installed_commit_holds_every_config_path_of_the_run(
    tmp_path, slurm_bin, launcher_at, monkeypatch, installed_tree
):
    from nemo_gym.orchestration.submit import recover

    launcher_at(NEW)
    installed_tree(BENCHMARK_CONFIG)
    # An absolute entry resolves unchanged, so it is not looked up in the installed tree.
    job_dir = _repinnable_run(tmp_path, [BENCHMARK_CONFIG, "/elsewhere/extra.yaml"])
    _pinned_plans_differently(monkeypatch)
    out = recover(*_handle(job_dir))
    assert out.dispatched == "resubmit", out.decision.reason
    assert out.pin_plan["repin"] is True and "missing_config_paths" not in out.pin_plan
    assert NEW in (job_dir / "job.sh").read_text()


def test_a_run_whose_pin_stays_is_not_checked_for_config_paths(tmp_path, slurm_bin, launcher_at, installed_tree):
    from nemo_gym.orchestration.submit import recover

    launcher_at(NEW)
    job_dir = _repinnable_run(tmp_path, [BENCHMARK_CONFIG])
    out = recover(*_handle(job_dir))
    assert out.dispatched == "resubmit", out.decision.reason
    assert out.pin_plan["repin"] is False and "refused" not in out.pin_plan
    assert OLD in (job_dir / "job.sh").read_text()
