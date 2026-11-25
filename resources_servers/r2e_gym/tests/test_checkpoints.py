# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""r2e_gym's episode sessions in partial-rollout checkpoints (``RepoTaskSessionResourcesServer``)."""

from pathlib import Path
from runpy import run_path

import pytest


# gym env test runs from the server directory, whose local tests package shadows the root tests package.
_helpers = run_path(str(Path(__file__).resolve().parents[3] / "tests/helpers/swe_repo_task_checkpoints.py"))


@pytest.mark.parametrize("scenario", _helpers["SCENARIOS"])
async def test_checkpoints(monkeypatch, tmp_path, scenario):
    # The patch is taken against the checkout's HEAD, which the seed does not record.
    await _helpers["check_repo_task_checkpoints"](
        monkeypatch, tmp_path, "r2e_gym", "R2EGym", scenario, records_base_commit=False, base_commit="HEAD"
    )
