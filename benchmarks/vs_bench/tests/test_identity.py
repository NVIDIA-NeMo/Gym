# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate exported author code without requiring Git on execution nodes."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from benchmarks.vs_bench.reference_identity import validate_author_checkout


def test_exported_author_files_are_verified_and_tampering_is_rejected(tmp_path):
    source = Path(os.environ.get("VS_BENCH_TEST_UPSTREAM", "/nonexistent"))
    if not source.is_dir():
        pytest.skip("Pinned author checkout required")
    manifest = json.loads(Path("benchmarks/vs_bench/reference_source_manifest.json").read_text())
    for relative in manifest["sha256"]:
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, dest)
    validate_author_checkout(str(tmp_path), sys.executable)
    changed = tmp_path / "envs/base_env.py"
    original = changed.read_bytes()
    changed.write_bytes(original + b"\n# unexpected change\n")
    with pytest.raises(ValueError, match="differs"):
        validate_author_checkout(str(tmp_path), sys.executable)
    changed.write_bytes(original)
    (tmp_path / "envs/unexpected.py").write_text("pass\n")
    with pytest.raises(ValueError, match="unexpected Python"):
        validate_author_checkout(str(tmp_path), sys.executable)
