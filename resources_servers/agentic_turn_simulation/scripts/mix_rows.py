#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mix JSONL part files into one training file: every line of every part, in the order given, then
`random.Random(seed).shuffle`. Lines keep their exact bytes (a missing final newline is added). Fails on a
repeated row id (`verifier_metadata.row_id`), since a repeated row is a silent upsample. Lines are
shuffled as byte offsets, so memory does not grow with the file size. Standard library only.

    python3 mix_rows.py --seed 0 train_mix.jsonl source_a.jsonl source_b.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROW_ID_KEY = "row_id"
ROW_ID_FALLBACK_KEY = "annotating_node_id"  # older files name the row id differently


def row_id(metadata: Mapping[str, Any]) -> str:
    """A row's id: `verifier_metadata.row_id`, or the fallback key when the row has no `row_id`."""
    return metadata[ROW_ID_KEY] if ROW_ID_KEY in metadata else metadata[ROW_ID_FALLBACK_KEY]


def line_spans(path: Path) -> list[tuple[int, int]]:
    """(offset, length) of every line of a file."""
    spans, offset = [], 0
    with open(path, "rb") as f:
        for line in f:
            spans.append((offset, len(line)))
            offset += len(line)
    return spans


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("parts", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.resolve() in {p.resolve() for p in args.parts}:
        raise SystemExit("output must differ from every part")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists (pass --overwrite to replace it)")

    order = [(i, span) for i, part in enumerate(args.parts) for span in line_spans(part)]
    random.Random(args.seed).shuffle(order)

    files = [open(p, "rb") for p in args.parts]
    ids, digest = Counter(), hashlib.sha256()
    tmp = args.output.with_name(args.output.name + ".tmp")
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as dst:
            for i, (offset, length) in order:
                files[i].seek(offset)
                line = files[i].read(length)
                if not line.endswith(b"\n"):
                    line += b"\n"
                ids[row_id(json.loads(line)["verifier_metadata"])] += 1
                dst.write(line)
                digest.update(line)
    finally:
        for f in files:
            f.close()
    repeated = [k for k, v in ids.items() if v > 1]
    if repeated:
        tmp.unlink()
        raise SystemExit(f"{len(repeated)} row ids repeat across the parts, e.g. {repeated[0]}")
    os.replace(tmp, args.output)
    print(
        json.dumps(
            {
                "output": args.output.name,
                "rows": len(order),
                "parts": {p.name: sum(1 for i, _ in order if args.parts[i] == p) for p in args.parts},
                "seed": args.seed,
                "output_sha256": digest.hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
