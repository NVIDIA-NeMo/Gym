# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

import orjson


parser = ArgumentParser()
parser.add_argument("--fpath", type=str, required=True)
args = parser.parse_args()


gt_10 = 0
num_stuck = 0
num_is_covered = 0
num_model_call_long = 0
total = 0
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
        if not is_covered:
            print(f"{i + 1}: {count > 10=} {was_stuck=} {model_call_long=}")

        num_model_call_long += model_call_long
        num_stuck += was_stuck
        gt_10 += is_gt_10
        num_is_covered += is_covered
        total += 1

print(gt_10, num_stuck, num_model_call_long, num_is_covered, total)
