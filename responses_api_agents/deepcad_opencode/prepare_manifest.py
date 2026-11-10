# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route an existing prepared/lazy CAD manifest to OpenCode without rendering it."""

import argparse
import json
from pathlib import Path


def prepare(source: Path, output: Path) -> int:
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output manifest")
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as src, output.open("x") as dst:
        for line in src:
            row = json.loads(line)
            if "verifier_metadata" not in row or "responses_create_params" not in row:
                raise ValueError("Expected a prepared or lazy Gym manifest, not raw CAD records")
            row["task_source"] = "deepcad_opencode_agent"
            dst.write(json.dumps(row) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Prepared {prepare(args.source, args.output)} rows; no geometry loaded or rendered")
