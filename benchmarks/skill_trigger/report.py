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
"""Summarize an explicit attempt ledger without hiding failed or pending rows."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Attempt(BaseModel):
    """One planned attempt; adapters must preserve unscored and pending records."""

    model_config = ConfigDict(extra="forbid", strict=True)
    attempt_id: str
    task_id: str
    model: str
    runtime: str
    profile: str
    collection: str
    release_sha256: str
    case_kind: Literal["positive", "negative"]
    status: Literal["scored", "unscored", "pending"]
    reward: Literal[0, 1] | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "Attempt":
        if (self.status == "scored") != (self.reward is not None):
            raise ValueError("Only scored attempts may have a reward")
        if self.status == "unscored" and not self.reason:
            raise ValueError("Unscored attempts require a reason")
        return self


def summarize(rows: list[Attempt]) -> list[dict]:
    """Keep releases and control kinds separate; zero denominators have no rate."""
    groups = defaultdict(list)
    seen = set()
    fields = ("model", "runtime", "profile", "collection", "release_sha256", "case_kind")
    for row in rows:
        if row.attempt_id in seen:
            raise ValueError(f"Duplicate attempt: {row.attempt_id}")
        seen.add(row.attempt_id)
        groups[tuple(getattr(row, field) for field in fields)].append(row)
    result = []
    for key, attempts in sorted(groups.items()):
        scored = [r for r in attempts if r.status == "scored"]
        reward_sum = sum(r.reward for r in scored)
        negative = key[-1] == "negative"
        numerator = len(scored) - reward_sum if negative else reward_sum
        result.append(
            {
                **dict(zip(fields, key)),
                "task_ids": sorted({r.task_id for r in attempts}),
                "planned": len(attempts),
                "scored": len(scored),
                "unscored": sum(r.status == "unscored" for r in attempts),
                "pending": sum(r.status == "pending" for r in attempts),
                "metric": "unnecessary_opening_rate" if negative else "target_opening_rate",
                "numerator": numerator,
                "denominator": len(scored),
                "rate": numerator / len(scored) if scored else None,
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    args = parser.parse_args()
    rows = [Attempt.model_validate_json(line) for line in args.attempts.read_text().splitlines() if line.strip()]
    print(json.dumps(summarize(rows), indent=2))
