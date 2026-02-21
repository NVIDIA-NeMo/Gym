# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Run:
```bash
python resources_servers/math_with_judge/filter_for_mixed_rewards.py \
    --input_fpath <> \
    --output_fpath <> \
    --source_fpath <>
```
"""

import json
from argparse import ArgumentParser
from collections import Counter

from tqdm.auto import tqdm


parser = ArgumentParser()
parser.add_argument("--input_fpath", type=str, required=True)
parser.add_argument("--output_fpath", type=str, required=True)
parser.add_argument("--source_fpath", type=str, required=True)
args = parser.parse_args()

# These are inclusive, for total of 16 rollouts per prompt
minimum_pass_at_k = 0
maximum_pass_at_k = 14

key_to_example = dict()
with open(args.source_fpath) as f:
    for line in tqdm(f, "Loading source dataset"):
        row = json.loads(line)
        # TODO this key is not generic. Eventually this filtering for mixed rewards will probably be integrated somehow into rollout collection
        key = json.dumps(row["responses_create_params"]["input"][0]["content"])
        key_to_example[key] = line

counter = Counter()
with open(args.input_fpath) as f:
    for line in tqdm(f, desc="Loading responses"):
        row = json.loads(line)
        key = json.dumps(row["responses_create_params"]["input"][0]["content"])
        counter[key] += row["reward"]

bucketed_counts = Counter(counter.values())
total_rollouts = sum(bucketed_counts.values())
total_prompts = len(counter)
print("Pass@k distribution")
for k, v in sorted(bucketed_counts.items()):
    pct = 100 * v / total_rollouts
    print(f"{k:>3}: {v:<8} ({pct:.2f}%)")


filtered_out = 0
with open(args.output_fpath, "w") as f:
    for key, count in tqdm(counter.items(), total=total_prompts):
        if not (minimum_pass_at_k <= count <= maximum_pass_at_k):
            filtered_out += 1
            continue

        f.write(key_to_example[key])

filtered_out_pct = 100 * filtered_out / total_prompts
remaining = total_prompts - filtered_out
remaining_pct = 100 * remaining / total_prompts
print(f"""Filtered out {filtered_out} examples ({filtered_out_pct:.2f}%)
Remaining: {remaining} examples ({remaining_pct:.2f}%)""")
