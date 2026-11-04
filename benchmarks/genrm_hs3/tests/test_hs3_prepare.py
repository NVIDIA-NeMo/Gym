# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preprocessed HS3 download test."""

from pathlib import Path
from unittest.mock import patch

from benchmarks.genrm_hs3.prepare import prepare
from resources_servers.genrm_eval.preferences import validate_metadata


def test_download_configuration(tmp_path: Path) -> None:
    output = tmp_path / "prepared.jsonl"
    with patch("benchmarks.genrm_hs3.prepare.download_prepared_dataset", return_value=output) as download:
        assert prepare(str(output)) == output
    download.assert_called_once_with(
        "genrm_hs3", "0.1.0", "hs3_validation.jsonl", str(output), validate_metadata=validate_metadata
    )
