# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared preprocessed-dataset download tests."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from resources_servers.genrm_eval.preferences import validate_metadata
from resources_servers.genrm_eval.prepare import download_prepared_dataset, make_task


@pytest.mark.parametrize("valid", [True, False])
def test_download_validates_before_replacing_output(tmp_path: Path, valid: bool) -> None:
    output = tmp_path / "prepared.jsonl"
    output.write_text("existing file")
    row = make_task("test", [{"role": "user", "content": "Q"}], "A", "B", {"ground_truth_label": 1})
    data = json.dumps(row if valid else {"error": "download failed"}) + "\n"

    def fetch(config) -> None:
        assert config.dataset_name == "test_dataset"
        assert config.version == "0.1.0"
        assert config.artifact_fpath == "tasks.jsonl"
        Path(config.output_fpath).write_text(data)

    with patch("nemo_gym.gitlab_utils.download_jsonl_dataset", side_effect=fetch):
        if valid:
            assert download_prepared_dataset("test_dataset", "0.1.0", "tasks.jsonl", str(output)) == output
            assert output.read_text() == data
        else:
            with pytest.raises(ValueError, match="task ID"):
                download_prepared_dataset("test_dataset", "0.1.0", "tasks.jsonl", output)
            assert output.read_text() == "existing file"


@pytest.mark.parametrize(
    "defect",
    [
        "null_fields",
        "bad_request",
        "empty_input",
        "missing_candidate",
        "empty_id",
        "duplicate_id",
        "null_label",
        "bad_label",
        "bool_label",
        "tie_label",
        "bad_data_type",
    ],
)
def test_invalid_download_preserves_output(tmp_path: Path, defect: str) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text("existing dataset")
    row = make_task("task", [{"role": "user", "content": "Q"}], "A", "B", {"ground_truth_label": 2})
    if defect == "null_fields":
        row = dict.fromkeys(row)
    elif defect == "bad_request":
        row["responses_create_params"] = None
    elif defect == "empty_input":
        row["responses_create_params"]["input"] = []
    elif defect == "missing_candidate":
        row["responses_create_params"]["metadata"]["response_2"] = None
    elif defect == "empty_id":
        row["id"] = " "
    elif defect == "bad_data_type":
        row["verifier_metadata"]["data_type"] = ["chat"]
    elif defect.endswith("label"):
        row["verifier_metadata"]["ground_truth_label"] = {
            "null_label": None,
            "bad_label": 100,
            "bool_label": True,
            "tie_label": 3.5,
        }[defect]
    data = json.dumps(row) + "\n"
    if defect == "duplicate_id":
        data *= 2

    def fetch(config):
        Path(config.output_fpath).write_text(data)

    with patch("nemo_gym.gitlab_utils.download_jsonl_dataset", side_effect=fetch), pytest.raises(ValueError):
        download_prepared_dataset("test", "0.1.0", "tasks.jsonl", output, validate_metadata=validate_metadata)
    assert output.read_text() == "existing dataset"
    assert list(tmp_path.iterdir()) == [output]


def test_blank_lines_are_removed(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    row = make_task("task", [{"role": "user", "content": "Q"}], "A", "B", {"ground_truth_label": 2})

    def fetch(config):
        Path(config.output_fpath).write_text("\n" + json.dumps(row) + "\n  \n")

    with patch("nemo_gym.gitlab_utils.download_jsonl_dataset", side_effect=fetch):
        download_prepared_dataset("test", "0.1.0", "tasks.jsonl", output, validate_metadata=validate_metadata)
    assert [json.loads(line) for line in output.read_text().splitlines()] == [row]
