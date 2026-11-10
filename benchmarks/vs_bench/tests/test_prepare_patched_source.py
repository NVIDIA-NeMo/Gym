# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the reproducible source export against a real pinned checkout."""

import json
import os
import shutil
from pathlib import Path

import pytest

from benchmarks.vs_bench.prepare_patched_source import digest, prepare
from benchmarks.vs_bench.reference_identity import KAZ_ARRAY_FIX_SHA256


def test_patched_export_preserves_input_and_refuses_overwrite(tmp_path):
    source = Path(os.environ.get("VS_BENCH_TEST_CLEAN_UPSTREAM", "/nonexistent"))
    if not source.is_dir():
        pytest.skip("Clean pinned upstream source export required")
    original = digest(source / "envs/knights_archers_zombies.py")
    target = tmp_path / "patched"
    prepare(source, target)
    assert digest(source / "envs/knights_archers_zombies.py") == original
    assert digest(target / "envs/knights_archers_zombies.py") == KAZ_ARRAY_FIX_SHA256
    provenance = json.loads((target / "vsbench-source-provenance.json").read_text())
    assert provenance["patched_file_sha256"] == KAZ_ARRAY_FIX_SHA256
    with pytest.raises(FileExistsError):
        prepare(source, target)


def test_tampered_source_is_rejected_before_creating_output(tmp_path):
    source = Path(os.environ.get("VS_BENCH_TEST_CLEAN_UPSTREAM", "/nonexistent"))
    if not source.is_dir():
        pytest.skip("Clean pinned upstream source export required")
    altered = tmp_path / "altered"
    shutil.copytree(source, altered)
    path = altered / "envs/knights_archers_zombies.py"
    path.write_bytes(path.read_bytes() + b"\n# unexpected change\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="differs from pinned upstream"):
        prepare(altered, output)
    assert not output.exists()
