# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise four current harness producers through the JSONL evidence checker."""

import json
from pathlib import Path

import pytest

from nemo_gym.harness_capabilities.checker import inspect_record
from nemo_gym.harness_capabilities.cli import inspect_matrix
from nemo_gym.harness_capabilities.reader import hydrate_record
from tests.unit_tests.harness_capabilities.producer_fixtures import HARNESSES, build_harness_record


@pytest.mark.parametrize("harness", HARNESSES)
def test_current_producer_evidence(harness, tmp_path):
    record = build_harness_record(harness, tmp_path / harness)
    result = inspect_record(hydrate_record(record))
    passed = {key for key, value in result["evidence"].items() if value["verdict"] == "fulfilled"}
    expected = {"TE-1", "TE-2", "TE-4", "TE-6", "TE-7"}
    if harness != "codex":
        expected |= {"TE-5", "TE-8"}
    assert passed == expected, result["findings"]
    assert result["verdict"] == "not_fulfilled"
    # Committed regression artifacts are reproducible outputs of these producers.
    fixture = Path(__file__).parent / "fixtures" / f"{harness}.jsonl"
    assert json.loads(fixture.read_text()) == record


def test_harness_matrix_uses_individual_report_verdicts(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    path, result = inspect_matrix({name: fixtures / f"{name}.jsonl" for name in HARNESSES}, output=tmp_path)
    table = (path / "harness_evidence.md").read_text()
    assert all(name in table for name in HARNESSES)
    assert all(row["verdict"] == "not_fulfilled" for row in result["harnesses"].values())
    assert "| codex | PASS | PASS | FAIL | PASS | FAIL | PASS | PASS | FAIL | FAIL | FAIL |" in table
    assert inspect_matrix({name: fixtures / f"{name}.jsonl" for name in HARNESSES}, output=tmp_path)[0] == path
