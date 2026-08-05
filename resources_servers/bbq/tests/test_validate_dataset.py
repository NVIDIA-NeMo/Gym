from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "validate_dataset", ROOT / "scripts/validate_dataset.py"
)
validate_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_dataset)


def run_main(tmp_path: Path, rows: list[dict], monkeypatch, extra_args: list[str] | None = None):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["validate_dataset.py", str(dataset), *(extra_args or [])],
    )
    return validate_dataset.main()


def test_fixture_rows_pass_row_contract(example_rows):
    failures = validate_dataset.Failures()
    parsed = validate_dataset.validate_rows(example_rows, failures)
    assert failures.messages == []
    assert len(parsed) == len(example_rows)


def test_fixture_file_passes_with_documented_relaxation(tmp_path, example_rows, monkeypatch):
    # The five curated fixtures predate DATA_SPEC and share the "teen athlete"
    # actor label across three Age rows; the section 5 cap is 2.
    assert run_main(tmp_path, example_rows, monkeypatch, ["--max-actor-label-rows", "3"]) == 0


def test_fixture_file_default_gate_flags_only_known_repetition(example_rows):
    failures = validate_dataset.Failures()
    parsed = validate_dataset.validate_rows(example_rows, failures)
    validate_dataset.validate_dataset_gates(
        example_rows,
        parsed,
        failures,
        max_actor_label_rows=2,
        max_question_opening_rows=3,
        max_five_gram_rows=2,
        question_opening_words=4,
    )
    assert len(failures.messages) == 1
    assert "teen athlete" in failures.messages[0]


def test_rejects_mcq_options(tmp_path, example_rows, monkeypatch):
    rows = copy.deepcopy(example_rows[:1])
    rows[0]["input"] += "\n\nA) the neighbor\nB) the athlete"
    assert run_main(tmp_path, rows, monkeypatch) == 1


def test_rejects_ambig_row_with_decisive_detail(tmp_path, example_rows, monkeypatch):
    rows = copy.deepcopy(
        [r for r in example_rows if r["expected_values"]["context_condition"] == "ambig"][:1]
    )
    rows[0]["expected_values"]["decisive_detail"] = "the neighbor mistyped every query"
    assert run_main(tmp_path, rows, monkeypatch) == 1


def test_rejects_missing_decoy_contract(tmp_path, example_rows, monkeypatch):
    rows = copy.deepcopy(
        [
            r
            for r in example_rows
            if r["expected_values"]["context_condition"] == "disambig"
            and r["expected_values"]["actor_count"] >= 3
        ][:1]
    )
    rows[0]["expected_values"]["wrong_neutral_actor_labels"] = []
    assert run_main(tmp_path, rows, monkeypatch) == 1


def test_rejects_duplicate_provenance_id(tmp_path, example_rows, monkeypatch):
    rows = copy.deepcopy(example_rows[:2])
    rows[1]["expected_values"]["provenance_id"] = rows[0]["expected_values"]["provenance_id"]
    assert run_main(tmp_path, rows, monkeypatch) == 1


def test_rejects_repeated_shingles(tmp_path, example_rows, monkeypatch):
    first = copy.deepcopy(example_rows[0])
    second = copy.deepcopy(example_rows[0])
    second["expected_values"]["provenance_id"] = "age-ambig-duplicate000001"
    assert run_main(tmp_path, [first, second], monkeypatch) == 1


def test_empty_dataset_fails(tmp_path, monkeypatch):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_dataset.py", str(dataset)])
    assert validate_dataset.main() == 1
