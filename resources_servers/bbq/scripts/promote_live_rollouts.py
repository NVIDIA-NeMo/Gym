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
"""Promote live collected rollouts to the canonical reference file.

Reads a live `ng_collect_rollouts` output (full verify responses), reduces each
row to the lean canonical reference schema, recomputes and cross-checks the
reward mapping, and writes `data/example_rollouts.jsonl` (rollout_kind
`live_reference`) plus a regenerated `data/example_metrics.json`. The previous
canonical file, if present, is preserved at
`data/example_rollouts_previous.jsonl`.

Usage:
  python scripts/promote_live_rollouts.py \
      --live data/example_rollouts_live.jsonl \
      --status-note "collected 2026-07-31 via <judge>/<policy>"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util import (  # noqa: E402
    AnswerJudgment,
    ExpectedValues,
    ExplanationJudgment,
    answer_reward,
    explanation_reward,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", default="data/example_rollouts_live.jsonl")
    parser.add_argument("--status-note", required=True)
    args = parser.parse_args()

    examples = [json.loads(line) for line in (ROOT / "data/example.jsonl").read_text().splitlines() if line]
    expected_by_provenance = {
        row["expected_values"]["provenance_id"]: ExpectedValues.model_validate(row["expected_values"])
        for row in examples
    }

    live_path = ROOT / args.live
    live_rows = [json.loads(line) for line in live_path.read_text().splitlines() if line]
    if len(live_rows) != len(examples):
        raise SystemExit(f"expected {len(examples)} live rollouts, found {len(live_rows)} in {live_path}")

    canonical_rows = []
    for row in live_rows:
        expected_values = row["expected_values"]
        provenance_id = expected_values["provenance_id"]
        expected = expected_by_provenance[provenance_id]
        answer = AnswerJudgment.model_validate(row["answer_judgment"])
        explanation = ExplanationJudgment.model_validate(row["explanation_judgment"])
        reward_1 = answer_reward(answer, expected)
        reward_2 = explanation_reward(explanation)
        for name, recomputed, recorded in (
            ("reward_answer", reward_1, row["reward_answer"]),
            ("reward_explanation_quality", reward_2, row["reward_explanation_quality"]),
            ("reward", reward_1 * reward_2, row["reward"]),
        ):
            if recomputed != recorded:
                raise SystemExit(f"{provenance_id}: {name} mismatch (recorded {recorded}, recomputed {recomputed})")
        canonical_rows.append(
            {
                "provenance_id": provenance_id,
                "source_id": expected_values["source_id"],
                "rollout_kind": "live_reference",
                "policy_response": row["model_response_text"],
                "answer_judgment": answer.model_dump(),
                "explanation_judgment": explanation.model_dump(),
                "reward_answer": reward_1,
                "reward_explanation_quality": reward_2,
                "reward": reward_1 * reward_2,
            }
        )

    canonical_path = ROOT / "data/example_rollouts.jsonl"
    if canonical_path.exists():
        backup = ROOT / "data/example_rollouts_previous.jsonl"
        n = 2
        while backup.exists():
            backup = ROOT / f"data/example_rollouts_previous_{n}.jsonl"
            n += 1
        shutil.copy2(canonical_path, backup)
    canonical_path.write_text("".join(json.dumps(row) + "\n" for row in canonical_rows))

    answer_classes = Counter(row["answer_judgment"]["answer_class"] for row in canonical_rows)
    quality_classes = Counter(row["explanation_judgment"]["quality_class"] for row in canonical_rows)
    metrics = {
        "Number of examples": len(canonical_rows),
        "Reference mean reward": sum(row["reward"] for row in canonical_rows) / len(canonical_rows),
        "Reference answer classes": {
            "correct_answer": answer_classes.get("correct_answer", 0),
            "incorrect_answer": answer_classes.get("incorrect_answer", 0),
        },
        "Reference explanation classes": {
            "high": quality_classes.get("high", 0),
            "low": quality_classes.get("low", 0),
            "unacceptable": quality_classes.get("unacceptable", 0),
        },
        "Status": f"Live endpoint rollouts; {args.status_note}",
    }
    (ROOT / "data/example_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(
        f"Promoted {len(canonical_rows)} live rollouts to {canonical_path.name}; "
        f"mean reward {metrics['Reference mean reward']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
