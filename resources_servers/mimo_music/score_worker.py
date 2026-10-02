# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One isolated invocation of the original music scoring pipeline."""

import json
import sys

from resources_servers.mimo_music.scorer.pipeline import do


def score_abc(abc: str) -> dict:
    """Preserve native rewards; distinguish infrastructure exceptions from rejects."""
    details = do({"key": "rollout", "id": 0, "rep": 0, "abc": abc})
    skip = str(details.get("skip", ""))
    scorer_skip = str(details.get("scorer_skip", ""))
    if skip.startswith("abc2midi:") or scorer_skip.startswith("analyze:"):
        raise RuntimeError(skip or scorer_skip)
    reward = 0.0 if details.get("skip") or details.get("reject") else float(details.get("total", 0)) / 100
    return {"reward": max(0.0, min(1.0, reward)), "scorer_details": details}


if __name__ == "__main__":
    print(json.dumps(score_abc(json.load(sys.stdin)["abc"])))
