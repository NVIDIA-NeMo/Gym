# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path

import pytest

from resources_servers.ether0.scripts.migrate_completion_prefix import INSTRUCTION, migrate


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    original = tmp_path / "original.jsonl"
    current = tmp_path / "current.jsonl"
    output = tmp_path / "migrated.jsonl"
    row = {
        "id": "completion-1",
        "problem_type": "molecule-completion",
        "problem": "Complete O[C@H](C",
        "ideal": "O[C@H](C)C=O",
        "unformatted": "must not be used as the reference",
        "solution": "valid_mol_eval!:!O[C@H](C!:!molecule-completion",
    }
    original.write_text(json.dumps(row) + "\n")
    row["solution"] = "formula_eval!:!O[C@H](C)C=O!:!molecule-completion"
    current.write_bytes(
        b'{ "id": "other", "problem_type": "reaction-name", "problem": "name" }\r\n'
        + (json.dumps(row) + "\n").encode()
    )
    return original, current, output


def test_migration_preserves_rows_and_joins_prefix_by_id(tmp_path: Path) -> None:
    original, current, output = _write_inputs(tmp_path)
    originals = original.read_bytes(), current.read_bytes()
    result = migrate(original_path=original, current_path=current, output_path=output)
    before = current.read_bytes().splitlines(keepends=True)
    after = output.read_bytes().splitlines(keepends=True)
    assert result == {"rows": 2, "completion_rows_changed": 1, "rows_copied_byte_for_byte": 1}
    assert before[0] == after[0]
    old, new = json.loads(before[1]), json.loads(after[1])
    assert old.keys() == new.keys()
    assert {key for key in old if old[key] != new[key]} == {"problem", "solution"}
    name, target, kind = new["solution"].split("!:!")
    assert name == "completion_formula_eval"
    assert ast.literal_eval(target) == ("O[C@H](C", old["ideal"])
    assert kind == "molecule-completion"
    assert new["problem"] == old["problem"] + " " + INSTRUCTION
    assert (original.read_bytes(), current.read_bytes()) == originals


@pytest.mark.parametrize("destination", ["original", "current", "existing"])
def test_migration_refuses_overwrite(tmp_path: Path, destination: str) -> None:
    original, current, output = _write_inputs(tmp_path)
    output.write_bytes(b"existing output\n")
    target = {"original": original, "current": current, "existing": output}[destination]
    before = target.read_bytes()
    with pytest.raises(ValueError, match="Choose a new output file"):
        migrate(original_path=original, current_path=current, output_path=target)
    assert target.read_bytes() == before


@pytest.mark.parametrize("invalid", ["early_pipeline", "prefix_mismatch", "missing_id", "duplicate_original"])
def test_migration_failure_does_not_publish_partial_output(tmp_path: Path, invalid: str) -> None:
    original, current, output = _write_inputs(tmp_path)
    if invalid == "duplicate_original":
        original.write_bytes(original.read_bytes() * 2)
    else:
        rows = current.read_bytes().splitlines(keepends=True)
        row = json.loads(rows[1])
        if invalid == "early_pipeline":
            row["solution"] = "valid_mol_eval!:!O[C@H](C!:!molecule-completion"
        elif invalid == "prefix_mismatch":
            row["solution"] = "formula_eval!:!CCO!:!molecule-completion"
            row["ideal"] = "CCO"
        else:
            row["id"] = "unknown"
        current.write_bytes(rows[0] + (json.dumps(row) + "\n").encode())
    before = original.read_bytes(), current.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        migrate(original_path=original, current_path=current, output_path=output)
    assert not output.exists()
    assert set(tmp_path.iterdir()) == {original, current}
    assert (original.read_bytes(), current.read_bytes()) == before
