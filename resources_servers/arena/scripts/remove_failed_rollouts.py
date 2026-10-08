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
"""Remove failed LMArena rollouts so resume can retry them."""

import argparse
import json
from pathlib import Path


def is_failed_rollout(row: dict) -> bool:
    # Self-comparisons intentionally have no judgments.
    if row.get("self_comparison"):
        return False
    # v3 deliberately scores responses stopped at the output limit as zero.
    incomplete_reason = ((row.get("response") or {}).get("incomplete_details") or {}).get("reason")
    if row.get("category") == "lmarena_v3" and incomplete_reason == "max_output_tokens":
        return False

    games = row.get("games") or []
    # Successful judging produces one game for each answer ordering.
    if len(games) != 2:
        return True
    for game in games:
        # Missing or unparseable judge output has no verdict.
        if not game or game.get("verdict") is None:
            return True
        # A recorded judge response must contain generated output.
        if "response" in game and not (game.get("response") or {}).get("output"):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    args = parser.parse_args()

    backup = Path(f"{args.rollouts}.back")
    tmp = Path(f"{args.rollouts}.tmp")
    if tmp.exists():
        raise FileExistsError(f"Refusing to overwrite {tmp}")

    total = removed = 0
    with args.rollouts.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            if is_failed_rollout(json.loads(line)):
                removed += 1
            else:
                dst.write(line)

    # Preserve the first backup when retrying the cleanup more than once.
    if not backup.exists():
        args.rollouts.rename(backup)
    tmp.replace(args.rollouts)
    print(f"removed {removed}/{total}; backup: {backup}")


if __name__ == "__main__":
    main()
