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

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from nemo_gym.cli.job_render import render_info, render_list, render_status
from nemo_gym.cli.main import _reject_overrides, main
from nemo_gym.orchestration.executors.resume_script import JOB_CHAIN_NAME
from nemo_gym.orchestration.jobs import (
    BenchmarkInfo,
    BenchmarkJob,
    Failure,
    FailureCategory,
    JobInfo,
    JobState,
    JobStatus,
    LogSource,
    SubmissionRecord,
    SubmissionStatus,
)


_SOURCES = [
    LogSource(name="driver", kind="driver", location="logs/driver-{job_id}.log"),
    LogSource(name="vllm", kind="service", location="logs/vllm-{job_id}.log"),
]


def _record(
    gym_job_id: str,
    benchmarks: dict[str, str | None],
    *,
    submitted_at: str = "2026-10-29T09:30:12Z",
    executor: str = "slurm",
    run_dir: str | None = None,
) -> SubmissionRecord:
    run_dir = run_dir or f"/lustre/runs/{gym_job_id}"
    return SubmissionRecord(
        gym_job_id=gym_job_id,
        gym_version="0.4.0",
        gym_commit="c155b77c8420283982ec30347a46b92ae6ee1af5",
        submitted_at=submitted_at,
        run_dir=run_dir,
        cluster="cluster-a",
        executor=executor,
        submitted_by="tgrzegorzek",
        benchmarks=[
            BenchmarkJob(benchmark=name, job_dir=f"{run_dir}/{name}", job_id=job_id, log_sources=_SOURCES)
            for name, job_id in benchmarks.items()
        ],
    )


# --- What the commands print ---------------------------------------------------------------------------------


def test_the_list_shows_what_each_submission_is_in_columns():
    records = [
        _record("gym-job-20261029T093012Z-4f1a9c", {"gpqa": "1", "ruler": "2", "ifbench": "3"}),
        _record("gym-job-20261028T154501Z-77be02", {"ruler": "4"}, submitted_at="2026-10-28T15:45:01Z"),
    ]

    assert render_list(records) == dedent(
        """\
        ID                               SUBMITTED             CLUSTER    EXECUTOR  BENCHMARKS
        gym-job-20261029T093012Z-4f1a9c  2026-10-29 09:30:12Z  cluster-a  slurm     gpqa, ruler, ifbench
        gym-job-20261028T154501Z-77be02  2026-10-28 15:45:01Z  cluster-a  slurm     ruler"""
    )


def test_an_empty_index_says_so():
    assert render_list([]) == "No submissions in this machine's job index."


def _submission_status(record: SubmissionRecord, jobs: list[JobStatus]) -> SubmissionStatus:
    from nemo_gym.orchestration.operations import _roll_up

    states = [job.state for job in jobs]
    counts = {state: states.count(state) for state in JobState if state in states}
    return SubmissionStatus(record=record, jobs=jobs, state=_roll_up(states), counts=counts)


def test_status_shows_each_benchmark_under_its_submission_with_the_failure_reason_and_logs_pointer():
    running = _submission_status(
        _record("gym-job-20261029T093012Z-4f1a9c", {"gpqa": "8123401", "ruler": "8123402", "ifbench": "8123403"}),
        [
            JobStatus(benchmark="gpqa", state=JobState.SUCCEEDED, job_id="8123401", detail="COMPLETED"),
            JobStatus(benchmark="ruler", state=JobState.RUNNING, job_id="8123455", detail="RUNNING"),
            JobStatus(benchmark="ifbench", state=JobState.SUCCEEDED, job_id="8123403", detail="COMPLETED"),
        ],
    )
    failed = _submission_status(
        _record("gym-job-20261028T154501Z-77be02", {"ruler": "8119870"}),
        [
            JobStatus(
                benchmark="ruler",
                state=JobState.FAILED,
                job_id="8119870",
                detail="FAILED",
                failure=Failure(
                    category=FailureCategory.INCOMPLETE,
                    message="Chain stopped at the walltime cap with rollouts missing; resubmit to continue.",
                    source="job directory",
                    logs="gym job logs gym-job-20261028T154501Z-77be02 ruler --tail 100",
                ),
            )
        ],
    )

    assert render_status([running, failed], note="2 of 4 submissions shown (--all for the rest).") == dedent(
        """\
        gym-job-20261029T093012Z-4f1a9c  cluster-a  RUNNING (1 RUNNING, 2 SUCCEEDED)
          BENCHMARK  STATE      JOB ID                     DETAIL
          gpqa       SUCCEEDED  8123401                    COMPLETED
          ruler      RUNNING    8123455 (record: 8123402)  RUNNING
          ifbench    SUCCEEDED  8123403                    COMPLETED

        gym-job-20261028T154501Z-77be02  cluster-a  FAILED (1 FAILED)
          BENCHMARK  STATE   JOB ID   DETAIL
          ruler      FAILED  8119870  FAILED
                     Chain stopped at the walltime cap with rollouts missing; resubmit to continue.
                     logs: gym job logs gym-job-20261028T154501Z-77be02 ruler --tail 100

        2 of 4 submissions shown (--all for the rest)."""
    )


def test_a_status_of_a_job_the_scheduler_does_not_know_shows_dashes():
    unknown = _submission_status(
        _record("gym-job-x", {"gpqa": None}), [JobStatus(benchmark="gpqa", state=JobState.UNKNOWN)]
    )

    assert "  gpqa       UNKNOWN  -       -" in render_status([unknown])


def test_info_describes_the_submission_and_fills_the_newest_job_id_into_the_log_locations():
    record = _record("gym-job-20261029T093012Z-4f1a9c", {"ruler": "8123402"})
    described = JobInfo(
        record=record,
        benchmarks=[
            BenchmarkInfo(
                benchmark="ruler",
                job_dir=record.benchmarks[0].job_dir,
                job_ids=["8123402", "8123455"],
                log_sources=_SOURCES,
            )
        ],
        config={"driver": {"container": "gym:latest"}},
    )

    assert render_info(described, show_config=False) == dedent(
        """\
        Submission  gym-job-20261029T093012Z-4f1a9c
        Executor    slurm, cluster-a
        Submitted   2026-10-29 09:30:12Z by tgrzegorzek
        Gym         0.4.0 (commit c155b77c8)
        Run dir     /lustre/runs/gym-job-20261029T093012Z-4f1a9c

        ruler
          Job dir   /lustre/runs/gym-job-20261029T093012Z-4f1a9c/ruler
          Jobs      8123402, 8123455
          Logs      SOURCE  INSTANCE  LOCATION
                    driver  -         logs/driver-8123455.log
                    vllm    -         logs/vllm-8123455.log

        Config      not shown; use --config to print the resolved config"""
    )
    assert render_info(described, show_config=True).endswith("Config\ndriver:\n  container: gym:latest")


def test_info_says_when_a_submission_predates_log_sources_and_why_the_config_is_missing():
    record = _record("gym-job-x", {"ruler": "1"})
    described = JobInfo(
        record=record,
        benchmarks=[BenchmarkInfo(benchmark="ruler", job_dir="/d/ruler", job_ids=[])],
        config_unavailable="the run directory is not readable by this executor",
    )

    text = render_info(described, show_config=True)

    assert "Jobs      none recorded" in text
    assert "no log sources recorded (submitted before log sources were tracked)" in text
    assert "Config      not available: the run directory is not readable by this executor" in text


# --- gym job, through the real CLI ---------------------------------------------------------------------------


@pytest.fixture
def cluster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A job index in a temporary cache and a Slurm whose `squeue` and `sacct` answer from files."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("squeue", "sacct"):
        script = bin_dir / tool
        script.write_text(f'#!/bin/bash\ncat "$(dirname "$0")/{tool}_rows" 2>/dev/null\nexit 0\n')
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def _submitted(
    tmp_path: Path, gym_job_id: str, benchmarks: dict[str, str], *, submitted_at: str, executor: str = "slurm"
) -> SubmissionRecord:
    """A run as `gym eval submit` leaves it: the job dirs, the resolved config, and the record in the index."""
    run_dir = tmp_path / "runs" / gym_job_id
    for name in benchmarks:
        (run_dir / name).mkdir(parents=True)
    (run_dir / "resolved-config.yaml").write_text(yaml.safe_dump({"driver": {"container": "gym:latest"}}))
    record = _record(gym_job_id, benchmarks, submitted_at=submitted_at, executor=executor, run_dir=str(run_dir))
    record.write_local_index()
    return record


def _gym(monkeypatch: pytest.MonkeyPatch, capsys, *args: str, ok: bool = True) -> tuple[str, str]:
    monkeypatch.setattr(sys, "argv", ["gym", "job", *args])
    if ok:
        main()
    else:
        with pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code != 0
    captured = capsys.readouterr()
    return captured.out, captured.err


def test_list_shows_the_newest_first_and_json_is_the_records(tmp_path, cluster, monkeypatch, capsys):
    _submitted(tmp_path, "gym-job-old", {"gpqa": "1"}, submitted_at="2026-10-01T00:00:00Z")
    _submitted(tmp_path, "gym-job-new", {"ruler": "2"}, submitted_at="2026-10-29T00:00:00Z")

    out, _ = _gym(monkeypatch, capsys, "list")
    assert [line.split()[0] for line in out.splitlines()] == ["ID", "gym-job-new", "gym-job-old"]

    out, _ = _gym(monkeypatch, capsys, "list", "--limit", "1", "--json")
    assert [record["gym_job_id"] for record in json.loads(out)] == ["gym-job-new"]


def test_status_reports_the_scheduler_and_the_files_the_job_wrote(tmp_path, cluster, monkeypatch, capsys):
    record = _submitted(tmp_path, "gym-job-a", {"gpqa": "100", "ruler": "200"}, submitted_at="2026-10-29T00:00:00Z")
    run_dir = Path(record.run_dir)
    (run_dir / "ruler" / JOB_CHAIN_NAME).write_text("200\n201\n")
    (cluster / "squeue_rows").write_text(f"201|RUNNING|{run_dir / 'ruler'}\n")
    (cluster / "sacct_rows").write_text("100|COMPLETED|0:0\n200|TIMEOUT|0:0\n201|RUNNING|0:0\n")

    out, err = _gym(monkeypatch, capsys, "status", "gym-job-a", "--json")

    (result,) = json.loads(out)
    by_name = {job["benchmark"]: job for job in result["jobs"]}
    assert result["state"] == "RUNNING"
    assert (by_name["gpqa"]["state"], by_name["ruler"]["state"], by_name["ruler"]["job_id"]) == (
        "SUCCEEDED",
        "RUNNING",
        "201",
    )
    assert "experimental" in err and "experimental" not in out


def test_status_with_no_ids_shows_the_newest_submissions_and_says_how_many_it_left_out(
    tmp_path, cluster, monkeypatch, capsys
):
    _submitted(tmp_path, "gym-job-old", {"gpqa": "1"}, submitted_at="2026-10-01T00:00:00Z")
    _submitted(tmp_path, "gym-job-new", {"gpqa": "2"}, submitted_at="2026-10-29T00:00:00Z")
    (cluster / "sacct_rows").write_text("1|COMPLETED|0:0\n2|COMPLETED|0:0\n")

    out, _ = _gym(monkeypatch, capsys, "status", "--limit", "1")
    assert "gym-job-new" in out and "gym-job-old" not in out
    assert "1 of 2 submissions shown (--all for the rest)." in out

    out, _ = _gym(monkeypatch, capsys, "status", "--all")
    assert "gym-job-new" in out and "gym-job-old" in out and "submissions shown" not in out


def test_status_with_no_ids_only_shows_submissions_that_have_the_named_benchmark(
    tmp_path, cluster, monkeypatch, capsys
):
    _submitted(tmp_path, "gym-job-gpqa", {"gpqa": "1"}, submitted_at="2026-10-01T00:00:00Z")
    _submitted(tmp_path, "gym-job-ruler", {"ruler": "2"}, submitted_at="2026-10-29T00:00:00Z")

    out, _ = _gym(monkeypatch, capsys, "status", "--benchmark", "gpqa", "--json")

    assert [result["record"]["gym_job_id"] for result in json.loads(out)] == ["gym-job-gpqa"]


def test_status_with_ids_is_strict_about_a_benchmark_a_submission_lacks(tmp_path, cluster, monkeypatch, capsys):
    _submitted(tmp_path, "gym-job-a", {"ruler": "1"}, submitted_at="2026-10-29T00:00:00Z")

    out, err = _gym(monkeypatch, capsys, "status", "gym-job-a", "--benchmark", "gpqa", ok=False)

    assert out == ""
    assert "gym-job-a has no benchmark 'gpqa' (it has: ruler). Nothing was queried." in err


def test_status_names_every_id_it_does_not_know(tmp_path, cluster, monkeypatch, capsys):
    _submitted(tmp_path, "gym-job-a", {"ruler": "1"}, submitted_at="2026-10-29T00:00:00Z")

    out, err = _gym(monkeypatch, capsys, "status", "gym-job-a", "nope-1", "nope-2", ok=False)

    assert out == ""
    assert "'nope-1'" in err and "'nope-2'" in err


def test_status_collapses_a_repeated_id(tmp_path, cluster, monkeypatch, capsys):
    _submitted(tmp_path, "gym-job-a", {"ruler": "1"}, submitted_at="2026-10-29T00:00:00Z")

    out, _ = _gym(monkeypatch, capsys, "status", "gym-job-a", "gym-job-a", "--json")

    assert len(json.loads(out)) == 1


def test_an_executor_without_the_capability_says_so(tmp_path, cluster, monkeypatch, capsys):
    _submitted(tmp_path, "gym-job-k8s", {"gpqa": "name"}, submitted_at="2026-10-29T00:00:00Z", executor="kubernetes")

    _, err = _gym(monkeypatch, capsys, "status", "gym-job-k8s", ok=False)

    assert "kubernetes executor does not support status" in err


def test_info_hides_the_config_unless_asked_and_reads_it_from_the_run_directory(
    tmp_path, cluster, monkeypatch, capsys
):
    _submitted(tmp_path, "gym-job-a", {"ruler": "1"}, submitted_at="2026-10-29T00:00:00Z")

    out, _ = _gym(monkeypatch, capsys, "info", "gym-job-a")
    assert "not shown; use --config" in out and "container: gym:latest" not in out

    out, _ = _gym(monkeypatch, capsys, "info", "gym-job-a", "--config")
    assert "container: gym:latest" in out

    out, _ = _gym(monkeypatch, capsys, "info", "gym-job-a", "--json")
    assert json.loads(out)["config"] is None

    out, _ = _gym(monkeypatch, capsys, "info", "gym-job-a", "--json", "--config")
    assert json.loads(out)["config"] == {"driver": {"container": "gym:latest"}}


def test_info_of_a_benchmark_the_submission_lacks_exits_with_the_known_ones(tmp_path, cluster, monkeypatch, capsys):
    _submitted(tmp_path, "gym-job-a", {"ruler": "1"}, submitted_at="2026-10-29T00:00:00Z")

    _, err = _gym(monkeypatch, capsys, "info", "gym-job-a", "nope", ok=False)

    assert "has no benchmark 'nope'; it has: ruler" in err


def test_the_commands_reject_overrides_and_both_a_limit_and_all(tmp_path, cluster, monkeypatch, capsys):
    _, err = _gym(monkeypatch, capsys, "list", "foo=bar", ok=False)
    assert "does not accept Hydra overrides" in err

    _, err = _gym(monkeypatch, capsys, "status", "--limit", "2", "--all", ok=False)
    assert "not allowed with" in err


def test_only_the_verbose_override_is_let_through_and_a_stray_one_is_rejected():
    class _Parser:
        def error(self, message):
            raise SystemExit(message)

    args = type("Args", (), {"_parser": _Parser()})()

    _reject_overrides(args, ["+verbose=true"], "status")
    with pytest.raises(SystemExit, match="status does not accept Hydra overrides"):
        _reject_overrides(args, ["+verbose=true", "foo=bar"], "status")


def test_info_of_an_executor_without_the_capability_says_so(tmp_path, cluster, monkeypatch, capsys):
    _submitted(tmp_path, "gym-job-k8s", {"gpqa": "name"}, submitted_at="2026-10-29T00:00:00Z", executor="kubernetes")

    _, err = _gym(monkeypatch, capsys, "info", "gym-job-k8s", ok=False)

    assert "kubernetes executor does not support info" in err


def test_the_renderers_import_nothing_that_talks_to_a_scheduler():
    """Importing `job_render` must not load the dispatch layer or any executor, so what it prints can be
    checked without a cluster. Checked in a SUBPROCESS: this module imports the commands itself."""
    probe = (
        "import sys;"
        "import nemo_gym.cli.job_render;"
        "bad = sorted(m for m in sys.modules if m == 'nemo_gym.orchestration.submit'"
        " or m.startswith('nemo_gym.orchestration.executors.slurm')"
        " or m.startswith('nemo_gym.orchestration.executors.kubernetes'));"
        "print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "", f"job_render pulled in: {result.stdout.strip()}"
