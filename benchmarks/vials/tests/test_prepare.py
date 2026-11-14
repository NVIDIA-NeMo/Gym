# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from benchmarks.vials import prepare as preparation
from benchmarks.vials.harbor.adapter import HF_CONFIG, HF_REPO, HF_REVISION, HF_SPLIT
from benchmarks.vials.tests.test_harbor_adapter import source_rows


@pytest.mark.parametrize("image", [None, "example/vials:tested"])
def test_prepare_exports_harbor_and_refreshes_example_index(tmp_path: Path, monkeypatch, image: str | None) -> None:
    output = tmp_path / "harbor"
    monkeypatch.setattr(preparation, "DEFAULT_OUTPUT_DIR", output)
    if image is None:
        monkeypatch.delenv("VIALS_HARBOR_IMAGE", raising=False)
    else:
        monkeypatch.setenv("VIALS_HARBOR_IMAGE", image)
    example = tmp_path / "harbor_example.jsonl"
    example.write_text("stale example\n")
    with patch("datasets.load_dataset", return_value=Dataset.from_list(source_rows())) as load:
        index = preparation.prepare()
        load.assert_called_once_with(HF_REPO, HF_CONFIG, split=HF_SPLIT, revision=HF_REVISION)
        # Preparation is idempotent and replaces generated tasks and the example index.
        (output / "vials-single/old-file").touch()
        assert preparation.prepare() == index

    assert index == output / "gym.jsonl"
    assert example.read_bytes() == (output / "example.jsonl").read_bytes()
    entries = [json.loads(line) for line in example.read_text().splitlines()]
    assert {entry["task_name"] for entry in entries} == {"vials-single", "vials-multi"}
    assert all(entry["responses_create_params"] == {"input": []} for entry in entries)
    assert "REFERENCE_ONLY_SENTINEL" not in example.read_text()
    assert not (output / "vials-single/old-file").exists()
    dockerfile = (output / "vials-single/environment/Dockerfile").read_text()
    assert dockerfile.startswith(f"FROM {image or 'vials-inputs:local'}\n")
