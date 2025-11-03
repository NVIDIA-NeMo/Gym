# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Run:
```bash
python resources_servers/library_judge_math/filter_for_mixed_rewards.py \
    --input_fpath <> \
    --output_fpath <>
```
"""

import json
from argparse import ArgumentParser
from collections import Counter

from tqdm.auto import tqdm


parser = ArgumentParser()
parser.add_argument("--input_fpath", type=str, required=True)
parser.add_argument("--output_fpath", type=str, required=True)
args = parser.parse_args()

# These are inclusive, for total of 16 rollouts per prompt
minimum_pass_at_k = 0
maximum_pass_at_k = 14

counter = Counter()
key_to_example = dict()
with open(args.input_fpath) as f:
    for line in tqdm(f):
        row = json.loads(line)
        key = json.dumps(row["responses_create_params"])
        counter[key] += row["reward"]

        # TODO this part is not generic. We should try to figure out how to make this generic.
        # Maybe some parameter?
        key_to_example[key] = {
            "responses_create_params": row["responses_create_params"],
            "question": row["question"],
            "expected_answer": row["expected_answer"],
        }


bucketed_counts = Counter(counter)
total_rollouts = sum(bucketed_counts.values())
total_prompts = len(counter)
print("Pass@k distribution")
for k, v in sorted(bucketed_counts.items()):
    pct = 100 * v / total_rollouts
    print(f"{k: 3}: {v:<8} ({pct:.2f}%)")


filtered_out = 0
with open(args.output_fpath, "w") as f:
    for key, count in tqdm(counter.items(), total=total_prompts):
        if not (minimum_pass_at_k <= count <= maximum_pass_at_k):
            filtered_out += 1
            continue

        example = key_to_example[key]
        f.write(json.dumps(example, separators=(",", ":")) + "\n")

filtered_out_pct = 100 * filtered_out / total_prompts
remaining = total_prompts - filtered_out
remaining_pct = 100 * remaining / total_prompts
print(f"""Filtered out {filtered_out} examples ({filtered_out_pct:.2f}%)
Remaining: {remaining} examples ({remaining_pct:.2f}%)""")
