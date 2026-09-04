# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
from collections import Counter
from contextlib import nullcontext
from os import stat

import orjson
from tqdm.auto import tqdm


parser = ArgumentParser()
parser.add_argument("--fpath", type=str, required=True)
parser.add_argument("--group-to-write-out", type=str, default=None)
args = parser.parse_args()


f_out_path = "temp.jsonl"
num_wrote_out = 0
counts = Counter()
with open(args.fpath, "rb") as f, open(f_out_path, "wb") if args.group_to_write_out else nullcontext() as f_out:
    for i, line in tqdm(enumerate(f)):
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

        is_errored = bool(row["error"])
        is_timeout = is_errored and "raise TimeoutError from exc_val" in row["error"]

        count_dict = {
            "Long model calls": row["model_calls_gt_10min"] >= 6,
            "Model claims to be stuck": stuck_count > 10,
            "No valid JSON found in response": count > 10,
            "Errored (excluding timeout)": is_errored - is_timeout,
            "Timed out": is_timeout,
            "Total samples with reward=0": 1,
        }
        count_dict["Samples covered by the above errors"] = (
            count_dict["Long model calls"] or count_dict["Model claims to be stuck"] or count_dict["Long model calls"]
        )
        counts.update(count_dict)

        if args.group_to_write_out and count_dict[args.group_to_write_out]:
            num_wrote_out += 1
            row = row | count_dict
            f_out.write(orjson.dumps(row) + b"\n")


print_str = ""
for k, v in counts.items():
    print_str += f"{k}: {v} ({int(100 * v / counts['Total samples with reward=0'])}%)\n"
print(print_str)

if args.group_to_write_out:
    print(
        f"Wrote out {num_wrote_out} rows ({stat(f_out_path).st_size / 1024**2} MB) for `{args.group_to_write_out}` to {f_out_path}"
    )
