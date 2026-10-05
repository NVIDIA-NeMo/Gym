# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate cached continuation after a successful Gym evaluation process.

Exit 75 requests another allocation; 76 rejects broad failures, 77 rejects
no progress, and 78 reports invalid coverage. This checks completion, not score.
"""

import argparse
import json
import sys
from pathlib import Path


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        count = 0
        for count, line in enumerate(stream, start=1):
            if not isinstance(json.loads(line), dict):
                raise ValueError(f"Expected a JSON object at {path}:{count}")
    return count


def check_continuation(rollouts_path: Path, *, max_missing_percent: int) -> int:
    """Request a bounded retry only when valid coverage is small and improving."""
    if not 0 <= max_missing_percent <= 100:
        raise ValueError("max_missing_percent must be between 0 and 100")
    if rollouts_path.suffix != ".jsonl":
        raise ValueError("rollouts_path must end in .jsonl")
    base = rollouts_path.with_suffix("")
    expected = _count_rows(base.with_name(base.name + "_materialized_inputs.jsonl"))
    successful = _count_rows(rollouts_path) if rollouts_path.exists() else 0
    if expected <= 0 or successful > expected:
        raise ValueError(f"Invalid coverage: {successful}/{expected}")
    print(f"Gym coverage: {successful}/{expected}", file=sys.stderr)
    missing = expected - successful
    if missing == 0:
        return 0
    if missing * 100 > expected * max_missing_percent:
        print(f"Automatic continuation refused: missing rows exceed {max_missing_percent}%", file=sys.stderr)
        return 76
    state_path = base.with_name(base.name + "_continuation_success_count")
    if state_path.exists():
        previous = int(state_path.read_text(encoding="utf-8"))
        if not 0 <= previous <= expected:
            raise ValueError(f"Invalid continuation count in {state_path}")
        if successful <= previous:
            print("Automatic continuation stopped: no progress", file=sys.stderr)
            return 77
    temporary_path = state_path.with_name(state_path.name + ".tmp")
    temporary_path.write_text(f"{successful}\n", encoding="utf-8")
    temporary_path.replace(state_path)
    print("Requesting cached continuation for missing rows", file=sys.stderr)
    return 75


def main() -> int:
    """Expose the coverage decision as the batch launcher's exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts_path", type=Path)
    parser.add_argument("--max-missing-percent", type=int, required=True)
    args = parser.parse_args()
    try:
        return check_continuation(args.rollouts_path, max_missing_percent=args.max_missing_percent)
    except (OSError, ValueError) as error:
        print(f"Gym continuation coverage is invalid: {error}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    sys.exit(main())
