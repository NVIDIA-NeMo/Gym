# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add prefix-aware completion reward metadata after all existing data cleanups."""

import argparse
import json
import os
import tempfile
from pathlib import Path


INSTRUCTION = (
    "Copy the given partial SMILES unchanged at the beginning of your answer, "
    "append the missing characters, and return the full SMILES string."
)


def migrate(*, original_path: str | Path, current_path: str | Path, output_path: str | Path) -> dict[str, int]:
    """Migrate cleaned completion rows, preserving all other rows byte-for-byte."""
    original_path, current_path, output_path = map(Path, (original_path, current_path, output_path))
    if output_path.resolve() in {original_path.resolve(), current_path.resolve()} or output_path.exists():
        raise ValueError("Choose a new output file; do not overwrite either input.")
    prefixes = {}
    with original_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["problem_type"] == "molecule-completion":
                fn, prefix, _ = row["solution"].split("!:!", 2)
                if fn != "valid_mol_eval" or not prefix or row["id"] in prefixes:
                    raise ValueError(f"Invalid original completion metadata: {row['id']}")
                prefixes[row["id"]] = prefix
    total = changed = 0
    temporary = tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False)
    try:
        with current_path.open("rb") as stream, temporary as destination:
            for raw in stream:
                total += 1
                row = json.loads(raw)
                if row["problem_type"] != "molecule-completion":
                    destination.write(raw)
                    continue
                fn, reference, problem_type = row["solution"].split("!:!", 2)
                prefix = prefixes[row["id"]]
                if (
                    fn != "formula_eval"
                    or problem_type != "molecule-completion"
                    or reference != row["ideal"]
                    or not reference.startswith(prefix)
                    or prefix not in row["problem"]
                ):
                    raise ValueError(f"Unexpected current completion row: {row['id']}")
                row["solution"] = f"completion_formula_eval!:!{(prefix, reference)!r}!:!{problem_type}"
                row["problem"] = row["problem"].rstrip() + " " + INSTRUCTION
                encoded = json.dumps(row, ensure_ascii=True, separators=(",", ":")).replace("/", "\\/")
                destination.write((encoded + "\n").encode("utf-8"))
                changed += 1
            destination.flush()
            os.fsync(destination.fileno())
        if changed == 0:
            raise ValueError("No completion rows migrated.")
        # Exclusive creation protects an output created while this script was running.
        os.link(temporary.name, output_path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)
    return {"rows": total, "completion_rows_changed": changed, "rows_copied_byte_for_byte": total - changed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(migrate(original_path=args.original, current_path=args.current, output_path=args.output))
