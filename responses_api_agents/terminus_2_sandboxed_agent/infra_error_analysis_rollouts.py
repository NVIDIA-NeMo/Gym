# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
from collections import Counter

import orjson


parser = ArgumentParser()
parser.add_argument("--fpath", type=str, required=True)
args = parser.parse_args()


counts = Counter()
with open(args.fpath, "rb") as f:
    for i, line in enumerate(f):
        row = orjson.loads(line)

        if row["reward"] != 0:
            continue

        count = 0
        stuck_count = 0
        for output_item in row["response"]["output"]:
            if output_item.get("role") == "user":
                content = output_item["content"]
                count += "No valid JSON found in response" in content
            elif output_item.get("type") == "reasoning":
                content = output_item["summary"][0]["text"]
                stuck_count += "stuck" in content or "unresponsive" in content

        is_gt_10 = count > 10
        was_stuck = stuck_count > 10
        model_call_long = row["model_calls_gt_10min"] >= 6
        is_covered = is_gt_10 or was_stuck or model_call_long
        is_errored = bool(row["error"])
        is_timeout = is_errored and "raise TimeoutError from exc_val" in row["error"]
        if not is_covered:
            print(f"{i + 1}: {count > 10=} {was_stuck=} {model_call_long=}")

        counts["Long model calls"] += model_call_long
        counts["Model claims to be stuck"] += was_stuck
        counts["No valid JSON found in response"] += is_gt_10
        counts["Samples covered by the above errors"] += is_covered
        counts["Errored (excluding timeout)"] += is_errored - is_timeout
        counts["Timed out"] += is_timeout
        counts["Total samples with reward=0"] += 1

print_str = ""
for k, v in counts.items():
    print_str += f"{k}: {v} ({int(100 * v / counts['Total samples with reward=0'])}%)\n"
print(print_str)
