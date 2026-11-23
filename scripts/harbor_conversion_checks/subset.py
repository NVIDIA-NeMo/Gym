# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`subset ROWS --id-field FIELD --count N --seed S --out IDS.txt`: a reproducible list of native task ids.

Ids are ordered by `sha256(f"{seed}:{id}")`, filtered by `--include` and `--exclude`, and the first N are written
with a header naming the rows file's sha256, so the list can be checked against the rows it came from. The first 5
ids are the smoke subset.
"""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.harbor_conversion_checks.messages import InputError, field_value, id_text, read_id_list


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("rows", type=Path, help="the native JSONL rows")
    parser.add_argument("--id-field", required=True, help="dotted field naming a row, or _index (0-based line)")
    parser.add_argument("--count", type=int, required=True, help="how many ids to keep")
    parser.add_argument("--seed", required=True, help="seed of the order")
    parser.add_argument("--include", type=Path, help="keep only these ids (one per line, # comments)")
    parser.add_argument("--exclude", type=Path, help="drop these ids, such as references the grader cannot award")
    parser.add_argument("--out", type=Path, required=True, help="the ids file to write")


def _row_ids(path: Path, field: str) -> tuple[list[str], str, int]:
    """The ids of all rows, the file's sha256 and its line count; streamed."""
    digest, ids, seen, lines = hashlib.sha256(), [], set(), 0
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    with handle:
        for index, raw in enumerate(handle):
            digest.update(raw)
            lines += 1
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                row_id = id_text(field_value(row, field, index))
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, AttributeError):
                raise InputError(f"{path} line {index + 1}: not a JSON object with field {field}") from None
            if row_id in seen:
                raise InputError(f"{path} line {index + 1}: duplicate id {row_id}")
            seen.add(row_id)
            ids.append(row_id)
    return ids, digest.hexdigest(), lines


def run(args: argparse.Namespace) -> int:
    if args.count < 1:
        raise InputError("--count must be at least 1")
    ids, sha256, lines = _row_ids(args.rows, args.id_field)
    include = set(read_id_list(args.include)) if args.include else None
    exclude = set(read_id_list(args.exclude)) if args.exclude else set()
    known = set(ids)
    # An include list is usually the server's list for the whole dataset, so ids it names beyond the rows are
    # expected; ids an exclude list names beyond the rows are listed (a mistyped id shows up there).
    if include is not None:
        print(f"subset: {len(include & known)} of {len(include)} --include id(s) are rows of {args.rows.name}")
    missing = sorted(exclude - known)
    if missing:
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        print(f"subset: {len(missing)} --exclude id(s) are not in {args.rows.name}: {shown}")
    eligible = [i for i in ids if (include is None or i in include) and i not in exclude]
    eligible.sort(key=lambda i: hashlib.sha256(f"{args.seed}:{i}".encode()).hexdigest())
    if len(eligible) < args.count:
        print(f"subset: only {len(eligible)} of {len(ids)} ids are eligible, fewer than --count {args.count}")
        return 1
    header = [
        f"# rows {args.rows.name} sha256 {sha256} lines {lines}",
        f"# seed {args.seed} count {args.count} include {args.include.name if args.include else 'none'} "
        f"exclude {args.exclude.name if args.exclude else 'none'}",
    ]
    args.out.write_text("\n".join(header + eligible[: args.count]) + "\n")
    print(f"subset: wrote {args.count} ids to {args.out}")
    return 0
