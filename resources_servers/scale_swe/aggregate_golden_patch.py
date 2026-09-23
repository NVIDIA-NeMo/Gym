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
"""Join repeated golden-patch passes into the set of rows that are actually usable.

A row is SUPPORTED when its golden patch resolved in every pass. That is the point of
repeating: a row that resolves only sometimes has a nondeterministic test, and its reward is
noise whether it is used for evaluation or for training.

The one distinction that matters, and the reason this is not a single boolean: a pass can fail
to produce a verdict at all (image pull, timeout, provider fault). That is evidence about the
infrastructure, not about the row. Collapsing the two would discard good tasks for an unrelated
hiccup, so incomplete passes are counted separately and those rows are reported as needing more
evidence rather than as broken.

    python resources_servers/scale_swe/aggregate_golden_patch.py \
        +runs=results/scale_swe_golden_patch_full \
        +output_jsonl=results/scale_swe_supported.jsonl
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nemo_gym.global_config import get_global_config_dict


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def classify(observations: list[dict[str, Any]], expected_passes: int) -> str:
    """Bucket one row from its per-pass observations."""
    completed = [o for o in observations if o.get("evaluation_completed")]
    resolved = sum(1 for o in completed if o.get("resolved"))

    if len(completed) < expected_passes:
        # Not enough verdicts to judge the row. Distinguished from a genuine failure so an
        # image-pull fault does not quietly delete a usable task.
        return "inconclusive"
    if resolved == len(completed):
        return "supported"
    if resolved == 0:
        return "broken"
    return "flaky"


def main() -> None:
    config = get_global_config_dict()
    runs_dir = Path(config["runs"])
    output_fpath = Path(config.get("output_jsonl") or "results/scale_swe_supported.jsonl")

    pass_files = sorted(p for p in runs_dir.glob("*.jsonl") if p.is_file())
    if not pass_files:
        raise SystemExit(f"no pass jsonl files under {runs_dir}")

    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    language: dict[str, str] = {}
    for pass_file in pass_files:
        for row in _rows(pass_file):
            instance_id = row.get("instance_id")
            if not instance_id or instance_id == "?":
                continue
            by_instance[instance_id].append(row)
            if row.get("language"):
                language[instance_id] = row["language"]

    expected = len(pass_files)
    buckets: Counter[str] = Counter()
    per_language: dict[str, Counter[str]] = defaultdict(Counter)

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as out:
        for instance_id, observations in sorted(by_instance.items()):
            verdict = classify(observations, expected)
            buckets[verdict] += 1
            per_language[language.get(instance_id, "unknown")][verdict] += 1
            out.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "language": language.get(instance_id, ""),
                        "verdict": verdict,
                        "passes_observed": len(observations),
                        "passes_completed": sum(1 for o in observations if o.get("evaluation_completed")),
                        "passes_resolved": sum(1 for o in observations if o.get("resolved")),
                        "expected_passes": expected,
                    }
                )
                + "\n"
            )

    total = sum(buckets.values())
    print(f"golden-patch aggregation over {total} row(s) from {expected} pass(es)\n")
    for verdict, description in (
        ("supported", "resolved in every pass - usable"),
        ("flaky", "resolved in some passes only - nondeterministic test"),
        ("broken", "never resolved - golden patch does not work"),
        ("inconclusive", "too few verdicts - re-run before judging"),
    ):
        count = buckets.get(verdict, 0)
        print(f"  {verdict:<13} {count:>6} ({100 * count / max(total, 1):5.1f}%)  {description}")

    print("\n  by language (supported / total):")
    for lang, counts in sorted(per_language.items(), key=lambda kv: -sum(kv[1].values())):
        rows_total = sum(counts.values())
        supported = counts.get("supported", 0)
        print(
            f"    {lang:<10} {supported:>5} / {rows_total:<6} ({100 * supported / rows_total:5.1f}%)"
            f"   flaky={counts.get('flaky', 0)} broken={counts.get('broken', 0)}"
            f" inconclusive={counts.get('inconclusive', 0)}"
        )
    print(f"\nper-row verdicts: {output_fpath}")


if __name__ == "__main__":
    main()
