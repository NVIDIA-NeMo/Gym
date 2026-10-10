# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/scicode_verified/prepare.py``."""

import hashlib
import json

import pytest

from benchmarks.scicode_verified import prepare as scicode_verified_prepare


def _digest(contents: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, contents).hexdigest()


def _mock_release(monkeypatch, tmp_path):
    step = {
        "step_number": "1.2",
        "step_description_prompt": "Implement the second step.",
        "step_background": "Background.",
        "function_header": "def f():",
        "return_line": "return 1",
        "ground_truth_code": "def f():\n    return 1",
        "test_cases": ["assert f() == target"],
    }
    source_row = {
        "problem_id": "1",
        "problem_name": "fixture",
        "required_dependencies": "import numpy as np",
        "sub_steps": [step],
        "source_only_field": {"preserve": True},
    }
    problems_contents = (json.dumps(source_row) + "\n").encode()
    h5_contents = b"fixture-h5"
    manifest = {
        "version": "v2",
        "n_problems": 1,
        "problem_order": ["1"],
        "problems_test_jsonl_md5": _digest(problems_contents, "md5"),
        "h5_md5": _digest(h5_contents, "md5"),
    }
    manifest_contents = json.dumps(manifest).encode()

    problems_path = tmp_path / "problems_test.jsonl"
    manifest_path = tmp_path / "release-manifest.json"
    h5_path = tmp_path / "release-targets.h5"
    problems_path.write_bytes(problems_contents)
    manifest_path.write_bytes(manifest_contents)
    h5_path.write_bytes(h5_contents)

    monkeypatch.setattr(scicode_verified_prepare, "DATA_DIR", tmp_path / "prepared")
    monkeypatch.setattr(scicode_verified_prepare, "OUTPUT_FPATH", tmp_path / "prepared" / "benchmark.jsonl")
    monkeypatch.setattr(scicode_verified_prepare, "MANIFEST_FPATH", tmp_path / "prepared" / "manifest.json")
    monkeypatch.setattr(scicode_verified_prepare, "TEST_DATA_FPATH", tmp_path / "prepared" / "targets.h5")
    monkeypatch.setattr(scicode_verified_prepare, "EXPECTED_PROBLEMS", 1)
    monkeypatch.setattr(scicode_verified_prepare, "EXPECTED_TOTAL_SUBPROBLEMS", 1)
    monkeypatch.setattr(scicode_verified_prepare, "EXPECTED_SCORED_SUBPROBLEMS", 0)
    monkeypatch.setattr(scicode_verified_prepare, "EXPECTED_PROBLEMS_JSONL_MD5", _digest(problems_contents, "md5"))
    monkeypatch.setattr(
        scicode_verified_prepare,
        "EXPECTED_PROBLEMS_JSONL_SHA256",
        _digest(problems_contents, "sha256"),
    )
    monkeypatch.setattr(
        scicode_verified_prepare,
        "EXPECTED_MANIFEST_SHA256",
        _digest(manifest_contents, "sha256"),
    )
    monkeypatch.setattr(scicode_verified_prepare, "EXPECTED_H5_MD5", _digest(h5_contents, "md5"))
    monkeypatch.setattr(scicode_verified_prepare, "PREFILLED_STEP_SHA256", {"1.2": "unused-by-mock"})
    monkeypatch.setattr(scicode_verified_prepare, "PREFILLED_STEP_LOCATIONS", {("1", 0): "1.2"})
    monkeypatch.setattr(scicode_verified_prepare, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(scicode_verified_prepare, "_download_prefilled_steps", lambda: {"1.2": "official"})

    paths = {
        "data/problems_test.jsonl": problems_path,
        "manifest.json": manifest_path,
        "test_data_cleaned.h5": h5_path,
    }
    monkeypatch.setattr(scicode_verified_prepare, "_download_release_file", lambda filename, token: paths[filename])
    return source_row


def test_release_constants_are_pinned_to_v2() -> None:
    assert (
        scicode_verified_prepare.HF_REVISION == "eea11a866be6860725258702b39ef8651ed26abd"  # pragma: allowlist secret
    )
    assert (
        scicode_verified_prepare.UPSTREAM_GIT_REVISION
        == "ddab4a92f8d80a7113ab946628e994b52354d838"  # pragma: allowlist secret
    )
    assert scicode_verified_prepare.EXPECTED_PROBLEMS == 64
    assert scicode_verified_prepare.EXPECTED_TOTAL_SUBPROBLEMS == 290
    assert scicode_verified_prepare.EXPECTED_SCORED_SUBPROBLEMS == 287
    assert set(scicode_verified_prepare.PREFILLED_STEP_SHA256) == {"13.6", "62.1", "76.3"}
    assert scicode_verified_prepare.PREFILLED_STEP_LOCATIONS == {
        ("13", 5): "13.6",
        ("62", 0): "62.1",
        ("76", 2): "76.3",
    }


def test_prepare_preserves_source_and_adds_gym_transport(monkeypatch, tmp_path) -> None:
    source_row = _mock_release(monkeypatch, tmp_path)

    output_path = scicode_verified_prepare.prepare()
    row = json.loads(output_path.read_text())

    assert row["responses_create_params"] == {"input": []}
    assert row["uuid"] == "1"
    assert row["prefilled_steps_code"] == {"1.2": "official"}
    for key, value in source_row.items():
        assert row[key] == value
    assert scicode_verified_prepare.TEST_DATA_FPATH.read_bytes() == b"fixture-h5"
    assert json.loads(scicode_verified_prepare.MANIFEST_FPATH.read_text())["version"] == "v2"


def test_validate_release_rejects_wrong_h5(monkeypatch, tmp_path) -> None:
    _mock_release(monkeypatch, tmp_path)
    monkeypatch.setattr(scicode_verified_prepare, "EXPECTED_H5_MD5", "0" * 32)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        scicode_verified_prepare.prepare()
