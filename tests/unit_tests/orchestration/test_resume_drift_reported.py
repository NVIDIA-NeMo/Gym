# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `gym eval resubmit` command flow that the recovery decisions of tests/unit_tests/eval/recovery/test_resume_drift_reported.py feed."""

from pathlib import Path

from tests.unit_tests.eval.recovery.test_resume_drift_reported import drift_journal
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin


def test_inspect_reports_the_drift_of_a_submitted_run(tmp_path: Path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import inspect_recovery
    from tests.unit_tests.orchestration.test_eval_resubmit import _handle
    from tests.unit_tests.orchestration.test_resubmit_walltime import _timed_run

    job_dir = _timed_run(tmp_path)
    drift_journal(job_dir, ["servers.policy_model.responses_api_models.vllm_model.request_timeout_s"])

    out = inspect_recovery(*_handle(job_dir))

    assert out.decision.reason.endswith(
        "resumed across operational drift: servers.policy_model.responses_api_models.vllm_model.request_timeout_s"
    )
