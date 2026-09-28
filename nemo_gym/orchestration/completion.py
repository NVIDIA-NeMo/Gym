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

"""Mandatory post-run coverage check for submitted evaluations.

Custom collection drivers must write the standard materialized-inputs JSONL,
including every expected rollout before retry filtering. Multi-stage drivers
must include stage_index in both inputs and results. This module deliberately
uses only the artifact contract and stdlib, so validation needs no live servers.
"""

import argparse
import json
import sys
from pathlib import Path

from nemo_gym.path_utils import failures_path_for, materialized_inputs_path_for


def _read_identities(path: Path, *, results: bool) -> set[tuple[int | None, int, int]]:
    identities = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSON at {location}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {location}")
            # Keep this reader independent of the collector's heavy runtime imports.
            fields = ("_ng_task_index", "_ng_rollout_index")
            if "stage_index" in row:
                fields += ("stage_index",)
            for field in fields:
                if type(row.get(field)) is not int or row[field] < 0:
                    raise ValueError(f"Missing or invalid {field} at {location}; expected a nonnegative integer")
            identity = (row.get("stage_index"), row["_ng_task_index"], row["_ng_rollout_index"])
            if results and (row.get("_ng_failure_class") is not None or row.get("_ng_no_persist")):
                continue
            if not results and identity in identities:
                raise ValueError(f"Duplicate expected sample identity at {location}")
            identities.add(identity)
    return identities


def validate_completion(output_fpath: Path) -> int:
    """Require a persisted result for every expected sample; return the complete count."""
    expected = _read_identities(materialized_inputs_path_for(output_fpath), results=False)
    completed = _read_identities(output_fpath, results=True)
    missing = expected - completed
    if missing:
        raise ValueError(f"{len(expected) - len(missing)}/{len(expected)} samples completed; {len(missing)} missing")
    return len(expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_jsonl_fpath", type=Path)
    output_fpath = parser.parse_args(argv).output_jsonl_fpath
    try:
        completed = validate_completion(output_fpath)
    except (OSError, ValueError) as error:
        print(
            f"EVAL FAILED: {error}\n"
            "Partial artifacts retained for diagnosis (including any aggregate metrics).\n"
            f"Expected inputs: {materialized_inputs_path_for(output_fpath)}\n"
            f"Rollouts: {output_fpath}\n"
            f"Failures: {failures_path_for(output_fpath)}",
            file=sys.stderr,
        )
        return 1
    print(f"EVAL SUCCEEDED: {completed}/{completed} samples completed. Rollouts: {output_fpath}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
