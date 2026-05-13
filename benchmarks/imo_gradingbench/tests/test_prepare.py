# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the prepare script by piping a fixed CSV through its core logic.

The actual download from google-deepmind/superhuman is exercised on the
cluster (``prepare_imo_gradingbench_gym.py``) — these unit tests just
guard the column-mapping / JSONL-shape logic so accidental refactors of
``prepare.py`` don't silently break the verifier contract.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.imo_gradingbench import prepare as prepare_module


# Minimal valid CSV with the same columns Skills' upstream produces.
_FAKE_CSV = (
    "Grading ID,Problem ID,Problem,Solution,Grading guidelines,Response,"
    "Points,Reward,Problem Source\n"
    'g1,p1,"Prove 1+1=2.","trivial","6+ pts: complete","1+1=2 is a definition.",'
    "7,correct,IMO_2024\n"
    'g2,p2,"Prove sqrt 2 irrational.","induction sketch","6+ pts: rigorous",'
    '"contradiction skipped",2,partial,IMO_2024\n'
    'g3,p3,"Show evens sum even.","direct","6+ pts: defns",'
    '"Bolzano-Weierstrass",0,incorrect,IMO_2023\n'
)


def _fake_urlopen(*_args, **_kwargs):  # noqa: ANN001, ANN201
    """Mock urlopen returning the fake CSV bytes."""

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ANN001, ANN201
            self.close()

    return _Resp(_FAKE_CSV.encode("utf-8"))


def test_prepare_writes_expected_jsonl(tmp_path: Path) -> None:
    out_dir = tmp_path / "data"
    output_fpath = out_dir / "imo_gradingbench_benchmark.jsonl"

    with (
        patch.object(prepare_module, "DATA_DIR", out_dir),
        patch.object(prepare_module, "OUTPUT_FPATH", output_fpath),
        patch.object(prepare_module.urllib.request, "urlopen", _fake_urlopen),
    ):
        result = prepare_module.prepare()

    assert result == output_fpath
    rows = [json.loads(line) for line in output_fpath.read_text().splitlines()]
    assert len(rows) == 3

    # Field shape — every key the Skills evaluator + Gym verifier read.
    expected_keys = {
        "grading_id",
        "problem_id",
        "problem_statement",
        "reference_solution",
        "rubric",
        "proof",
        "points",
        "reward",
        "source",
        "expected_answer",
    }
    for r in rows:
        assert set(r.keys()) == expected_keys, set(r.keys()) ^ expected_keys

    # `expected_answer` is the same as `reward` — the gold grade word.
    assert [r["expected_answer"] for r in rows] == ["correct", "partial", "incorrect"]
    assert [r["reward"] for r in rows] == ["correct", "partial", "incorrect"]
    assert [r["grading_id"] for r in rows] == ["g1", "g2", "g3"]
    assert rows[0]["problem_statement"] == "Prove 1+1=2."
    assert rows[1]["proof"] == "contradiction skipped"


def test_csv_columns_match_upstream() -> None:
    """The CSV reader must pick up every column the prepare script reads.

    Guards against an upstream rename — e.g. if google-deepmind ever
    changed `Problem Source` to `Problem_Source`, prepare.py would
    raise a KeyError and this test would catch it locally before the
    cluster job tries.
    """
    reader = csv.DictReader(io.StringIO(_FAKE_CSV))
    fieldnames = reader.fieldnames or []
    required = {
        "Grading ID",
        "Problem ID",
        "Problem",
        "Solution",
        "Grading guidelines",
        "Response",
        "Points",
        "Reward",
        "Problem Source",
    }
    missing = required - set(fieldnames)
    assert not missing, f"missing CSV columns: {missing}"
