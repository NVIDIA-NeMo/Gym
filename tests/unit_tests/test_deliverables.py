# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dependency-neutral deliverable classification."""

from pathlib import Path

from nemo_gym.deliverables import IGNORE_FILES, is_deliverable


def test_is_deliverable_distinguishes_output_from_run_state(tmp_path: Path) -> None:
    output = tmp_path / "answer.txt"
    output.write_text("answer")
    run_state = tmp_path / "finish_params.json"
    run_state.write_text("{}")
    reference_files = tmp_path / "reference_files"
    reference_files.mkdir()

    assert is_deliverable(output)
    assert not is_deliverable(run_state)
    assert not is_deliverable(reference_files)
    assert {run_state.name, reference_files.name} <= IGNORE_FILES
