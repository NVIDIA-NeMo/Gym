# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from benchmarks.vials import prepare as preparation
from benchmarks.vials.harbor.adapter import HF_CONFIG, HF_REPO, HF_REVISION, HF_SPLIT
from benchmarks.vials.tests.test_harbor_adapter import source_rows


@pytest.mark.parametrize("image", [None, "example/vials:tested"])
def test_prepare_exports_native_task_folders(tmp_path: Path, monkeypatch, image: str | None) -> None:
    output = tmp_path / "vials"
    monkeypatch.setattr(preparation, "DEFAULT_OUTPUT_DIR", output)
    if image is None:
        monkeypatch.delenv("VIALS_HARBOR_IMAGE", raising=False)
    else:
        monkeypatch.setenv("VIALS_HARBOR_IMAGE", image)
    with patch("datasets.load_dataset", return_value=Dataset.from_list(source_rows())) as load:
        folder = preparation.prepare()
        load.assert_called_once_with(HF_REPO, HF_CONFIG, split=HF_SPLIT, revision=HF_REVISION)
        (output / "vials-single/old-file").touch()
        assert preparation.prepare() == folder
    assert folder == output
    assert (folder / "dataset.toml").is_file()
    assert not (folder / "gym.jsonl").exists()
    assert not (folder / "vials-single/old-file").exists()
    dockerfile = (folder / "vials-single/environment/Dockerfile").read_text()
    assert dockerfile.startswith(f"FROM {image or 'vials-inputs:local'}\n")
