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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    all_skill_openings: int | None = Field(default=None, ge=0)
    session_start_openings: int | None = Field(default=None, ge=0)
    unnecessary_skill_openings: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> "Attempt":
        if (self.status == "scored") != (self.reward is not None):
            raise ValueError("Only scored attempts may have a reward")
        if self.status == "unscored" and not self.reason:
            raise ValueError("Unscored attempts require a reason")
        counts = (self.all_skill_openings, self.session_start_openings, self.unnecessary_skill_openings)
        if self.case_kind == "negative" and self.status == "scored":
            if any(value is None for value in counts):
                raise ValueError("Scored negative controls require all opening counts")
            if counts[0] != counts[1] + counts[2] or self.reward != int(counts[2] == 0):
                raise ValueError("Opening counts contradict the negative reward")
        elif any(value is not None for value in counts):
            raise ValueError("Opening counts belong to scored negative controls")
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
        if negative:
            result[-1]["opening_metrics"] = {
                name: {
                    "numerator": sum(getattr(r, name) > 0 for r in scored),
                    "denominator": len(scored),
                    "rate": sum(getattr(r, name) > 0 for r in scored) / len(scored) if scored else None,
                    "distinct_skill_read_total": sum(getattr(r, name) for r in scored),
                    "unscored": result[-1]["unscored"],
                    "pending": result[-1]["pending"],
                }
                for name in ("all_skill_openings", "session_start_openings", "unnecessary_skill_openings")
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    args = parser.parse_args()
    rows = [Attempt.model_validate_json(line) for line in args.attempts.read_text().splitlines() if line.strip()]
    print(json.dumps(summarize(rows), indent=2))
