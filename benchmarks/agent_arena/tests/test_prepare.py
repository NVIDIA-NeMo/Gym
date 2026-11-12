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


@pytest.mark.parametrize(
    "row",
    [
        {"responses_create_params": None},
        {"responses_create_params": {}},
        "not an object",
    ],
    ids=["null request", "empty request", "not an object"],
)
def test_a_row_the_environment_could_not_run_is_rejected(tmp_path, row):
    """The key alone proves nothing: collation writes these rows and the run rejects them."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        prep.prepare(source_jsonl=str(bad))


def test_a_minimal_request_is_kept_as_it_is(tmp_path):
    source = tmp_path / "minimal.jsonl"
    source.write_text(json.dumps({"responses_create_params": {"input": "hello"}}) + "\n")
    out = prep.prepare(source_jsonl=str(source))
    assert out.read_bytes() == source.read_bytes()


def test_a_blank_line_is_rejected_rather_than_skipped(tmp_path):
    """The copy keeps it, and collation reads every line, so the run would stop on it instead."""
    source = tmp_path / "blank.jsonl"
    source.write_text(EXAMPLE.read_text().splitlines()[0] + "\n\n")
    with pytest.raises(ValueError):
        prep.prepare(source_jsonl=str(source))
    assert not prep.OUTPUT_FPATH.exists()


def test_a_failed_preparation_leaves_the_previous_output_untouched(tmp_path):
    prepared = prep.prepare(source_jsonl=str(EXAMPLE)).read_bytes()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError):
        prep.prepare(source_jsonl=str(bad))
    assert prep.OUTPUT_FPATH.read_bytes() == prepared


def test_a_failed_preparation_writes_no_output_at_all(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError):
        prep.prepare(source_jsonl=str(bad))
    assert not prep.OUTPUT_FPATH.exists()


def test_preparing_a_subset_of_the_output_itself_keeps_its_rows():
    """`source_jsonl` naming the output must not truncate the file it reads."""
    out = prep.prepare(source_jsonl=str(EXAMPLE))
    again = prep.prepare(source_jsonl=str(out), limit=2)
    assert len([line for line in again.read_text().splitlines() if line]) == 2
