# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from benchmarks.agent_arena import prepare as prep


EXAMPLE = Path(prep.__file__).resolve().parents[2] / ("resources_servers/agentic_turn_simulation/data/example.jsonl")


@pytest.fixture(autouse=True)
def _output_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "OUTPUT_FPATH", tmp_path / "data" / "agent_arena_benchmark.jsonl")
    monkeypatch.delenv(prep.SOURCE_ENV, raising=False)


def test_without_a_source_it_fails_closed():
    with pytest.raises(ValueError, match="not in git"):
        prep.prepare()
    assert not prep.OUTPUT_FPATH.exists()


def test_missing_source_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        prep.prepare(source_jsonl=str(tmp_path / "nope.jsonl"))


def test_copies_the_staged_file_byte_for_byte():
    out = prep.prepare(source_jsonl=str(EXAMPLE))
    assert out.read_bytes() == EXAMPLE.read_bytes()


def test_env_var_names_the_source_and_limit_cuts_rows(monkeypatch):
    monkeypatch.setenv(prep.SOURCE_ENV, str(EXAMPLE))
    out = prep.prepare(limit=2)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    assert all("responses_create_params" in row for row in rows)


def test_rows_without_responses_create_params_are_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"verifier_metadata": {}}) + "\n")
    with pytest.raises(ValueError, match="responses_create_params"):
        prep.prepare(source_jsonl=str(bad))
