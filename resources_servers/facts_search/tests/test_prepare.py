# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from benchmarks.facts_search.prepare import (
    ADVERTISED_V2_PUBLIC_ROWS,
    BRAVE_SEARCH_TOOL,
    CSV_SHA256,
    EXPECTED_PUBLISHED_ROWS,
    SYSTEM_PROMPT,
    prepare,
)


FIXTURE_CSV = b"example_id,problem,gold answer\nrow-1,Question one?,Answer one\nrow-2,Question two?,Answer two\n"


def test_public_contract_constants_match_the_pinned_source():
    assert EXPECTED_PUBLISHED_ROWS == 890
    assert ADVERTISED_V2_PUBLIC_ROWS == 921
    assert len(CSV_SHA256) == 64
    assert "one short sentence" in SYSTEM_PROMPT
    assert BRAVE_SEARCH_TOOL["name"] == "brave_search"


def test_prepare_is_deterministic_for_a_pinned_source(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_bytes(FIXTURE_CSV)
    monkeypatch.setattr("benchmarks.facts_search.prepare.CSV_SHA256", hashlib.sha256(FIXTURE_CSV).hexdigest())
    monkeypatch.setattr("benchmarks.facts_search.prepare.EXPECTED_PUBLISHED_ROWS", 2)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    prepare(str(source), str(first))
    prepare(str(source), str(second))
    assert first.read_bytes() == second.read_bytes()
    rows = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["responses_create_params"]["input"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rows[0]["problem"]},
    ]
    assert rows[0]["responses_create_params"]["tools"] == [BRAVE_SEARCH_TOOL]
    assert rows[0]["responses_create_params"]["parallel_tool_calls"] is True


def test_prepare_rejects_any_unpinned_csv(tmp_path):
    changed = tmp_path / "changed.csv"
    changed.write_bytes(FIXTURE_CSV)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare(str(changed), str(tmp_path / "out.jsonl"))
