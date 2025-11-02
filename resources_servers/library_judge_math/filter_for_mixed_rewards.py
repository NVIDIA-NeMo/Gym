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
import json
from collections import Counter

from tqdm.auto import tqdm


input_fpath = ""
output_fpath = ""

# These are inclusive, for total of 16 rollouts per prompt
minimum_pass_at_k = 0
maximum_pass_at_k = 14

counter = Counter()
question_to_example = dict()
with open(input_fpath) as f:
    for line in tqdm(f):
        row = json.loads(line)
        question = row["responses_create_params"]["input"][0]["content"]
        counter[question] += row["reward"]

        question_to_example[question] = {
            "responses_create_params": row["responses_create_params"],
            "question": row["question"],
            "expected_answer": row["expected_answer"],
        }


filtered_out = 0
with open(output_fpath, "w") as f:
    for question, count in tqdm(counter.items(), total=len(counter)):
        if not (minimum_pass_at_k <= count <= maximum_pass_at_k):
            filtered_out += 1
            continue

        example = question_to_example[question]
        f.write(json.dumps(example, separators=(",", ":")) + "\n")

filtered_out_pct = 100 * filtered_out / len(counter)
remaining = len(counter) - filtered_out
remaining_pct = 100 * remaining / len(counter)
print(f"""Filtered out {filtered_out} examples ({filtered_out_pct:.2f}%)
Remaining: {remaining} examples ({remaining_pct:.2f}%)""")
