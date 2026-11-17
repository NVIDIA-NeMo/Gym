# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for CPU export handoff in a resumable Slurm run."""

from pathlib import Path

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script


def _render(*, separate: bool, resumable: bool) -> str:
    cluster = {"type": "slurm", "account": "test"}
    if separate:
        cluster["export_partition"] = "cpu"
    config = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"cluster": cluster},
            "driver": {
                "container": "installed-gym:test",
                "benchmarks": {"task": {"resumable": resumable}},
            },
            "job": {
                "output_path": "/jobs",
                "export": {"evalrdl": {"label": "nightly"}},
            },
        }
    )
    return build_sbatch_script(
        config, "task", config.driver.benchmarks["task"], config.compute["cluster"], Path("/jobs/run/task")
    )


def test_resumable_cpu_export_handoff_follows_chain_epilogue():
    script = _render(separate=True, resumable=True)
    assert "GYM_SEPARATE_EXPORT=1" in script
    assert script.index('_gym_end_own_chain "$_gym_driver_rc"') < script.index("_gym_eval_final_rc=$?")
    assert '"$_gym_eval_final_rc" -eq 0' in script
    assert '."$SLURM_JOB_ID"' in script
    assert ".gym_chain_incomplete" in script
    assert script.index("_gym_eval_final_rc=$?") < script.index("sbatch --parsable")
    assert 'exit "$_gym_eval_final_rc"' in script or (
        "DRIVER_RC=${_gym_eval_final_rc}" in script and "exit $DRIVER_RC" in script
    )


def test_nonresumable_cpu_export_does_not_require_chain_marker():
    script = _render(separate=True, resumable=False)
    assert "GYM_SEPARATE_EXPORT=1" not in script
    assert "_gym_eval_final_rc=$?" in script
    assert ".gym_chain_incomplete" not in script
    assert "sbatch --parsable" in script


def test_legacy_inline_export_does_not_submit_cpu_job():
    script = _render(separate=False, resumable=True)
    assert "GYM_SEPARATE_EXPORT=1" not in script
    assert "sbatch --parsable" not in script
    assert "_gym_eval_final_rc=$?" not in script
