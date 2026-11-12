# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`gym eval resubmit --recover --server-spinup-timeout-seconds N --spinup-timeout-reason R`: a journaled raise."""

import json

import pytest
import yaml

from nemo_gym.cli.main import build_parser
from nemo_gym.orchestration.executors.resume_script import SPINUP_TIMEOUT_REASON
from nemo_gym.orchestration.executors.slurm import SPINUP_TIMEOUT_JOURNAL_NAME, UNSAFE_RESUME_JOURNAL_NAME
from nemo_gym.rollouts.rollout_recovery import _is_semantic
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resubmit_repin import _digests
from tests.unit_tests.orchestration.test_resubmit_repin import launcher_at as launcher_at
from tests.unit_tests.orchestration.test_resubmit_walltime import (
    _launcher_at_the_pinned_revision as _launcher_at_the_pinned_revision,
)
from tests.unit_tests.orchestration.test_resubmit_walltime import _timed_run


def test_the_raise_renders_the_job_with_it_journals_it_and_the_decision_names_it(tmp_path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    run_dir = job_dir.parent
    before = _digests(job_dir)

    out = recover(*_handle(job_dir), server_spinup_timeout_seconds=1800, spinup_timeout_reason="cold server installs")

    assert out.dispatched == "resubmit", out.decision.reason
    assert out.decision.reason.endswith("; server spin-up timeout raised from 600 s to 1800 s: cold server installs")
    assert "+server_spinup_timeout_seconds=1800" in (job_dir / "job.sh").read_text()
    saved = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
    assert saved["driver"]["benchmarks"]["gsm8k"]["run"]["server_spinup_timeout_seconds"] == 1800
    assert list(job_dir.glob("job.sh.pre-spinup-timeout-*"))
    [entry] = [json.loads(line) for line in (job_dir / SPINUP_TIMEOUT_JOURNAL_NAME).read_text().splitlines()]
    assert {k: entry[k] for k in ("from_seconds", "to_seconds", "reason", "gym_job_id", "benchmark")} == {
        "from_seconds": 600.0,
        "to_seconds": 1800,
        "reason": "cold server installs",
        "gym_job_id": run_dir.name,
        "benchmark": "gsm8k",
    }
    assert entry["ts"] and entry["user"]
    assert not (run_dir / SPINUP_TIMEOUT_JOURNAL_NAME).exists()
    assert not (run_dir / UNSAFE_RESUME_JOURNAL_NAME).exists()
    assert _digests(job_dir) == before  # saved rows are untouched


def test_a_second_raise_starts_from_the_limit_the_first_recorded(tmp_path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    recover(*_handle(job_dir), server_spinup_timeout_seconds=1800, spinup_timeout_reason="first")
    (job_dir / ".gym_job_chain").unlink(missing_ok=True)

    with pytest.raises(ValueError, match="does not raise the recorded limit of 1800 s"):
        recover(*_handle(job_dir), server_spinup_timeout_seconds=1200, spinup_timeout_reason="lower")


@pytest.mark.parametrize(
    "seconds,reason,error",
    [
        (600, "same", "does not raise the recorded limit of 600 s"),
        (300, "lower", "does not raise the recorded limit of 600 s"),
        (-1, "negative", "0 for no limit"),
        (1800, "  ", "nonempty --spinup-timeout-reason"),
    ],
)
def test_a_lower_limit_or_a_blank_reason_is_refused_before_anything_changes(
    tmp_path, slurm_bin, seconds, reason, error
) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    old_script = (job_dir / "job.sh").read_text()
    with pytest.raises(ValueError, match=error):
        recover(*_handle(job_dir), server_spinup_timeout_seconds=seconds, spinup_timeout_reason=reason)
    assert (job_dir / "job.sh").read_text() == old_script
    assert not (job_dir / SPINUP_TIMEOUT_JOURNAL_NAME).exists()


def test_zero_is_a_raise_to_no_limit(tmp_path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    out = recover(*_handle(job_dir), server_spinup_timeout_seconds=0, spinup_timeout_reason="installs vary too much")

    assert out.dispatched == "resubmit", out.decision.reason
    assert out.decision.reason.endswith("raised from 600 s to 0 s (no limit): installs vary too much")
    assert "+server_spinup_timeout_seconds=0" in (job_dir / "job.sh").read_text()


def test_a_changed_limit_is_operational_drift_and_the_typed_stop_names_the_flag() -> None:
    assert not _is_semantic("server_spinup_timeout_seconds")
    assert "--server-spinup-timeout-seconds N --spinup-timeout-reason TEXT" in SPINUP_TIMEOUT_REASON


@pytest.mark.parametrize(
    "argv,error",
    [
        (["--recover", "--server-spinup-timeout-seconds", "1800"], "needs a nonempty --spinup-timeout-reason"),
        (["--recover", "--server-spinup-timeout-seconds", "1800", "--spinup-timeout-reason", " "], "nonempty"),
        (["--recover", "--spinup-timeout-reason", "x"], "goes with --server-spinup-timeout-seconds"),
        (["--server-spinup-timeout-seconds", "1800", "--spinup-timeout-reason", "x"], "needs --recover"),
        (
            ["--recover", "--inspect", "--server-spinup-timeout-seconds", "1800", "--spinup-timeout-reason", "x"],
            "--inspect",
        ),
        (["--recover", "--server-spinup-timeout-seconds", "-5", "--spinup-timeout-reason", "x"], "0 for no limit"),
    ],
)
def test_the_flag_without_a_reason_or_out_of_place_is_a_usage_error(capsys, argv, error) -> None:
    from nemo_gym.cli.main import _eval_resubmit

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "resubmit", "--help"])
    assert "--server-spinup-timeout-seconds" in capsys.readouterr().out
    args = parser.parse_args(["eval", "resubmit", "job", "bench", *argv])
    with pytest.raises(SystemExit):
        _eval_resubmit(args, [])
    assert error in capsys.readouterr().err
