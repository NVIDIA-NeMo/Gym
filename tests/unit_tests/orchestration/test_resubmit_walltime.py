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
"""`gym eval resubmit --recover --max-walltime T --walltime-reason R`: an explicit, journaled walltime raise."""

import json
from pathlib import Path

import pytest
import yaml

from nemo_gym.cli.main import build_parser
from nemo_gym.orchestration.api import ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.slurm import REPIN_JOURNAL_NAME, WALLTIME_JOURNAL_NAME, SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resubmit_repin import NEW, OLD, _digests
from tests.unit_tests.orchestration.test_resubmit_repin import launcher_at as launcher_at


@pytest.fixture(autouse=True)
def _launcher_at_the_pinned_revision(launcher_at):
    """The launcher is Gym OLD, the revision ``_timed_run`` pins, so a raise may render its job.sh again."""
    launcher_at(OLD)


def _timed_run(tmp_path: Path) -> Path:
    """One benchmark with a 4 h Slurm walltime and a 16 h chain cap, one saved row, one missing, spent budgets."""
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "a", "hostname": None, "walltime": "04:00:00"}},
            "driver": {
                "container": "python:3.12",
                "gym_install": {"repo": "https://example.invalid/gym", "ref": OLD},
                "benchmarks": {
                    "gsm8k": {"resumable": ResumeConfig(max_retries=3, max_walltime="16:00:00").model_dump()}
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


@pytest.mark.parametrize("raise_to,cap", [("12:00:00", "16:00:00"), ("20:00:00", "20:00:00")])
def test_a_walltime_raise_renders_the_run_again_and_keeps_rows_and_budgets(tmp_path, slurm_bin, raise_to, cap):
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    run_dir = job_dir.parent
    before = _digests(job_dir)
    old_script = (job_dir / "job.sh").read_text()
    out = recover(*_handle(job_dir), max_walltime=raise_to, walltime_reason="slow but healthy policy")
    assert out.dispatched == "resubmit", out.decision.reason

    assert f"#SBATCH --time={raise_to}" in (job_dir / "job.sh").read_text()
    [kept_script] = job_dir.glob("job.sh.pre-walltime-*")
    assert kept_script.read_text() == old_script and "#SBATCH --time=04:00:00" in old_script
    saved = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
    assert saved["compute"]["cluster"]["walltime"] == raise_to
    # The chain cap is at least the new segment walltime; every other field is as saved.
    assert saved["driver"]["benchmarks"]["gsm8k"]["resumable"] == {"max_retries": 3, "max_walltime": cap}
    [kept_config] = run_dir.glob("resolved-config.yaml.pre-walltime-*")
    assert yaml.safe_load(kept_config.read_text())["compute"]["cluster"]["walltime"] == "04:00:00"
    [entry] = [json.loads(line) for line in (run_dir / WALLTIME_JOURNAL_NAME).read_text().splitlines()]
    assert {k: entry[k] for k in ("benchmark", "from", "to", "max_walltime_from", "max_walltime_to", "reason")} == {
        "benchmark": "gsm8k",
        "from": "04:00:00",
        "to": raise_to,
        "max_walltime_from": "16:00:00",
        "max_walltime_to": cap,
        "reason": "slow but healthy policy",
    }
    assert entry["actor"] and entry["at"]
    [job] = json.loads((run_dir / "gym-job.json").read_text())["benchmarks"]
    assert (job["job_id"], job["walltime"], job["gym_commit"]) == ("4242", raise_to, None)
    # Saved rows and the retry counter are untouched; the walltime keeps its terminal accounting (7200 + 60).
    assert _digests(job_dir) == before
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "7260\n"
    assert not (run_dir / REPIN_JOURNAL_NAME).exists()


def test_a_walltime_raise_adds_no_field_the_saved_record_did_not_hold(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    run_dir = job_dir.parent
    # A record saved by a Gym that predates a defaulted field (here `driver.policy_host`).
    saved_path = run_dir / "resolved-config.yaml"
    saved = yaml.safe_load(saved_path.read_text())
    del saved["driver"]["policy_host"]
    saved_path.write_text(yaml.safe_dump(saved, sort_keys=False))

    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="slow but healthy policy")
    assert out.dispatched == "resubmit", out.decision.reason

    after = yaml.safe_load(saved_path.read_text())
    assert "policy_host" not in after["driver"]
    # Only the walltime and the chain cap differ from the record as launched.
    after["compute"]["cluster"]["walltime"] = saved["compute"]["cluster"]["walltime"]
    after["driver"]["benchmarks"]["gsm8k"]["resumable"]["max_walltime"] = "16:00:00"
    assert after == saved


def test_a_walltime_raise_lets_a_chain_that_spent_its_old_cap_plan_again(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    # 16 h of the 16 h chain cap are spent: 57600 s stored before job 101, which ran 60 s more.
    (job_dir / ".gym_accumulated_walltime").write_text("57600\n")
    (slurm_bin / "accounting").write_text("100|TIMEOUT|57600|0\n101|FAILED|60|0\n")
    script = (job_dir / "job.sh").read_text()
    kept = recover(*_handle(job_dir))
    assert (kept.dispatched, kept.decision.reason) == (None, "the stored walltime budget is used (57660 of 57600 s)")
    assert (job_dir / "job.sh").read_text() == script and not (slurm_bin / "env").exists()

    out = recover(*_handle(job_dir), max_walltime="20:00:00", walltime_reason="slow but healthy policy")
    assert out.dispatched == "resubmit", out.decision.reason
    assert out.decision.reason.endswith("; walltime cap raised from 16:00:00 to 20:00:00 for this plan")
    assert "#SBATCH --time=20:00:00" in (job_dir / "job.sh").read_text()
    # The used walltime is accounted as before and checked against the raised cap: it is not reset.
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "57660\n"


@pytest.mark.parametrize(
    "value,reason,error",
    [
        ("02:00:00", "x", "below the recorded walltime 04:00:00"),
        ("twelve hours", "x", "needs a Slurm duration"),
        ("12:00:00", " ", "needs a nonempty --walltime-reason"),
    ],
)
def test_a_walltime_raise_is_refused_before_anything_changes(tmp_path, slurm_bin, value, reason, error):
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", job_dir.parent / "resolved-config.yaml"]}
    with pytest.raises((RuntimeError, ValueError), match=error):
        recover(*_handle(job_dir), max_walltime=value, walltime_reason=reason)
    assert {path: path.read_bytes() for path in files} == files
    assert not (slurm_bin / "env").exists()
    assert not (job_dir.parent / WALLTIME_JOURNAL_NAME).exists()


def test_a_walltime_raise_whose_job_is_not_queued_restores_both_files(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", job_dir.parent / "resolved-config.yaml"]}
    (slurm_bin / "sbatch").write_text("#!/bin/bash\necho 'sbatch: error: Batch job submission failed'; exit 1\n")
    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="reason")
    assert not out.ok and out.dispatched is None
    assert out.scheduler_refusal["scheduler_text"] == "sbatch: error: Batch job submission failed"
    assert out.scheduler_refusal["job_dir_restored"] is True
    assert {path: path.read_bytes() for path in files} == files
    assert not (job_dir.parent / WALLTIME_JOURNAL_NAME).exists()


def test_a_repin_and_a_walltime_raise_render_once_and_journal_both(tmp_path, slurm_bin, launcher_at):
    from nemo_gym.orchestration.submit import recover

    launcher_at(NEW)
    job_dir = _timed_run(tmp_path)
    run_dir = job_dir.parent
    out = recover(
        *_handle(job_dir), repin=NEW, repin_reason="fixed revision", max_walltime="12:00:00", walltime_reason="slow"
    )
    assert out.dispatched == "resubmit", out.decision.reason
    script = (job_dir / "job.sh").read_text()
    assert NEW in script and OLD not in script and "#SBATCH --time=12:00:00" in script
    saved = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
    assert (saved["driver"]["gym_install"]["ref"], saved["compute"]["cluster"]["walltime"]) == (NEW, "12:00:00")
    assert len(list(job_dir.glob("job.sh.pre-repin-*"))) == len(list(job_dir.glob("job.sh.pre-walltime-*"))) == 1
    assert json.loads((run_dir / REPIN_JOURNAL_NAME).read_text())["to"] == NEW
    assert json.loads((run_dir / WALLTIME_JOURNAL_NAME).read_text())["to"] == "12:00:00"
    [job] = json.loads((run_dir / "gym-job.json").read_text())["benchmarks"]
    assert (job["gym_commit"], job["walltime"]) == (NEW, "12:00:00")


@pytest.mark.parametrize(
    "argv,error",
    [
        (["--max-walltime", "12:00:00", "--walltime-reason", "x"], "need --recover"),
        (
            ["--recover", "--inspect", "--max-walltime", "12:00:00", "--walltime-reason", "x"],
            "cannot be used with --inspect",
        ),
        (["--recover", "--max-walltime", "12:00:00"], "go together"),
    ],
)
def test_the_resubmit_help_lists_the_walltime_flags_and_refuses_them_out_of_place(capsys, argv, error):
    from nemo_gym.cli.main import _eval_resubmit

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "resubmit", "--help"])
    assert "--max-walltime" in capsys.readouterr().out
    args = parser.parse_args(["eval", "resubmit", "job", "bench", *argv])
    with pytest.raises(SystemExit):
        _eval_resubmit(args, [])
    assert error in capsys.readouterr().err


@pytest.mark.parametrize(
    "change",
    [
        {"max_walltime": "12:00:00", "walltime_reason": "slow but healthy policy"},
        {"server_spinup_timeout_seconds": 1800, "spinup_timeout_reason": "cold server installs"},
        {"unsafe_resume_reason": "identity check reviewed"},
    ],
    ids=["walltime", "spinup-timeout", "unsafe-resume"],
)
def test_a_limit_raise_from_a_launcher_of_another_revision_is_refused_before_anything_changes(
    tmp_path, slurm_bin, launcher_at, change
):
    # The run is pinned to Gym OLD. This launcher is NEW: its job.sh templates may need modules OLD does not have.
    from nemo_gym.orchestration.executors.slurm import IncompatibleRerenderError
    from nemo_gym.orchestration.submit import recover

    launcher_at(NEW)
    job_dir = _timed_run(tmp_path)
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", job_dir.parent / "resolved-config.yaml"]}
    with pytest.raises(IncompatibleRerenderError, match=f"pinned to Gym {OLD}.*this launcher is Gym {NEW}") as raised:
        recover(*_handle(job_dir), **change)
    assert (raised.value.pinned, raised.value.launcher) == (OLD, NEW)
    assert {path: path.read_bytes() for path in files} == files
    assert not (slurm_bin / "env").exists()
    assert not list(job_dir.parent.glob("*-journal.jsonl")) and not list(job_dir.glob("*-journal.jsonl"))


def test_a_limit_raise_from_the_pinned_revision_renders_the_run_again(tmp_path, slurm_bin, launcher_at):
    from nemo_gym.orchestration.submit import recover

    launcher_at(OLD)
    job_dir = _timed_run(tmp_path)
    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="slow but healthy policy")

    assert out.dispatched == "resubmit", out.decision.reason
    assert "#SBATCH --time=12:00:00" in (job_dir / "job.sh").read_text()
