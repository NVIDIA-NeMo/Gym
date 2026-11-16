# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`gym eval resubmit --recover --allow-unsafe-resume --unsafe-resume-reason R`: an explicit, journaled override."""

import json

import pytest
import yaml

from nemo_gym.cli.main import build_parser
from nemo_gym.orchestration.executors.slurm import UNSAFE_RESUME_JOURNAL_NAME, WALLTIME_JOURNAL_NAME
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resubmit_repin import _digests
from tests.unit_tests.orchestration.test_resubmit_repin import launcher_at as launcher_at
from tests.unit_tests.orchestration.test_resubmit_walltime import (
    _launcher_at_the_pinned_revision as _launcher_at_the_pinned_revision,
)
from tests.unit_tests.orchestration.test_resubmit_walltime import _timed_run


def test_the_override_renders_the_job_with_it_and_journals_the_reason_and_user(tmp_path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    run_dir = job_dir.parent
    before = _digests(job_dir)

    out = recover(*_handle(job_dir), unsafe_resume_reason="judge prompt fix, scores re-checked")

    assert out.dispatched == "resubmit", out.decision.reason
    assert "+allow_unsafe_resume=True" in (job_dir / "job.sh").read_text()
    saved = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
    assert saved["driver"]["benchmarks"]["gsm8k"]["run"]["allow_unsafe_resume"] is True
    assert list(job_dir.glob("job.sh.pre-unsafe-resume-*"))
    [entry] = [json.loads(line) for line in (run_dir / UNSAFE_RESUME_JOURNAL_NAME).read_text().splitlines()]
    assert (entry["benchmark"], entry["reason"]) == ("gsm8k", "judge prompt fix, scores re-checked")
    assert entry["actor"] and entry["at"]
    assert _digests(job_dir) == before  # saved rows are untouched
    assert not (run_dir / WALLTIME_JOURNAL_NAME).exists()


def test_a_blank_reason_is_refused_before_anything_changes(tmp_path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir = _timed_run(tmp_path)
    old_script = (job_dir / "job.sh").read_text()
    with pytest.raises(ValueError, match="nonempty --unsafe-resume-reason"):
        recover(*_handle(job_dir), unsafe_resume_reason="  ")
    assert (job_dir / "job.sh").read_text() == old_script
    assert not (job_dir.parent / UNSAFE_RESUME_JOURNAL_NAME).exists()


@pytest.mark.parametrize(
    "argv,error",
    [
        (["--recover", "--allow-unsafe-resume"], "needs a nonempty --unsafe-resume-reason"),
        (["--recover", "--allow-unsafe-resume", "--unsafe-resume-reason", " "], "needs a nonempty"),
        (["--recover", "--unsafe-resume-reason", "x"], "goes with --allow-unsafe-resume"),
        (["--allow-unsafe-resume", "--unsafe-resume-reason", "x"], "needs --recover"),
        (["--recover", "--inspect", "--allow-unsafe-resume", "--unsafe-resume-reason", "x"], "--inspect"),
    ],
)
def test_the_flag_without_a_reason_or_out_of_place_is_a_usage_error(capsys, argv, error) -> None:
    from nemo_gym.cli.main import _eval_resubmit

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "resubmit", "--help"])
    assert "--allow-unsafe-resume" in capsys.readouterr().out
    args = parser.parse_args(["eval", "resubmit", "job", "bench", *argv])
    with pytest.raises(SystemExit):
        _eval_resubmit(args, [])
    assert error in capsys.readouterr().err
