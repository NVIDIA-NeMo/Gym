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
Run
```bash
python responses_api_agents/ext-taubench-env-agent/profile_steps.py
```

Qwen 3 4B Instruct agent Qwen 3 235BA22B Instruct 2507 User
```
Count distribution (columns are the number of steps and count of samples)
    0: 3     (avg reward 0.00)
   10: 20    (avg reward 0.05)
   20: 146   (avg reward 0.16)
   30: 289   (avg reward 0.26)
   40: 239   (avg reward 0.15)
   50: 178   (avg reward 0.17)
   60: 86    (avg reward 0.12)
   70: 38    (avg reward 0.13)
   80: 12    (avg reward 0.00)
   90: 4     (avg reward 0.00)
  100: 2     (avg reward 0.50)
  130: 2     (avg reward 0.00)
  160: 1     (avg reward 0.00)
  180: 1     (avg reward 0.00)
  200: 3     (avg reward 0.00)
```

GPT 4.1 agent and user
```
Count distribution (columns are the number of steps and count of samples)
    0: 1     (avg reward 0.00)
   10: 53    (avg reward 0.36)
   20: 358   (avg reward 0.46)
   30: 352   (avg reward 0.49)
   40: 144   (avg reward 0.44)
   50: 23    (avg reward 0.39)
   60: 3     (avg reward 0.00)
   90: 1     (avg reward 1.00)
  200: 1     (avg reward 0.00)
```
"""

import json
from collections import Counter


rail_to_nearest = 10
counter = Counter()
score = Counter()

# For W&B logged table:
# with open("responses_api_agents/ext-taubench-env-agent/data/full_result.table.json") as f:
#     data = json.load(f)["data"]
#     for line in data:
#         data = json.loads(line[0])

# For rollout collection output:
with open("responses_api_agents/ext-taubench-env-agent/data/ext-taubench-env-gpt4p1_train_rollouts.jsonl") as f:
    for line in f:
        data = json.loads(line)

        num_steps = len(data["raw_rollout"]["trajectory"])

        railed_num_steps = (num_steps // rail_to_nearest) * rail_to_nearest
        counter[railed_num_steps] += 1
        score[railed_num_steps] += data["reward"]

for k in score:
    score[k] /= counter[k]


print("Count distribution (columns are the number of steps and count of samples)")
for k, v in sorted(counter.items()):
    print(f"{k: 5}: {v:<5} (avg reward {score[k]:.2f})")
