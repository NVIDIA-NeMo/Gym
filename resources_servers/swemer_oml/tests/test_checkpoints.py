# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""swemer_oml's episode sessions in partial-rollout checkpoints (``RepoTaskSessionResourcesServer``)."""

from pathlib import Path
from runpy import run_path

import pytest


# gym env test runs from the server directory, whose local tests package shadows the root tests package.
_helpers = run_path(str(Path(__file__).resolve().parents[3] / "tests/helpers/swe_repo_task_checkpoints.py"))


@pytest.mark.parametrize("scenario", [*_helpers["SCENARIOS"], "restore_applies_the_restoring_job_s_egress_rule"])
async def test_checkpoints(monkeypatch, tmp_path, scenario):
    # The seed reads HEAD: the patch is taken against the commit the image's checkout is at.
    await _helpers["check_repo_task_checkpoints"](
        monkeypatch, tmp_path, "swemer_oml", "SwemerOml", scenario, records_base_commit=True
    )
