# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.opencode_multimodal_benchmarks.prepare import prepare as prepare_tsv


def test_tsv_checksum_checked_before_materialization(tmp_path):
    source = tmp_path / "wrong.tsv"
    source.write_text("wrong dataset")
    with pytest.raises(ValueError, match="source dataset checksum mismatch"):
        prepare_tsv("vstar", source, sha256="0" * 64)
