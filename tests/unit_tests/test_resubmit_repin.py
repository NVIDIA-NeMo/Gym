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
from tests.unit_tests.test_eval_resubmit import _handle
from tests.unit_tests.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.test_eval_resubmit import slurm_bin as slurm_bin


OLD, NEW = "a" * 40, "d" * 40


def _repinnable_run(tmp_path: Path) -> Path:
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
                        "run": {"mlflow_run_tags": {"gym_sha": f"'{OLD}'", "gym_commit": f"'{OLD}'", "model": "'m'"}},
                        "resumable": ResumeConfig(max_retries=3, max_walltime="16:00:00").model_dump(),
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
