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
Run as:
```bash
HF_HOME=.cache \
HF_TOKEN={your HF token} \
python resources_servers/code_gen/scripts/preprocess_train_dataset.py
```

Upload:
```bash
ng_upload_dataset_to_gitlab \
    +dataset_name=opencodereasoning_filtered \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/code_gen/data/opencodereasoning_filtered_25k_train.jsonl
```

Rollout collection. We match the LCB setting for reward profiling. For gpt-4o-2024-05-13 this should be around 33%.
```bash
ng_collect_rollouts +agent_name=code_gen_simple_agent \
    +input_jsonl_fpath=resources_servers/code_gen/data/opencodereasoning_filtered_25k_train.jsonl \
    +output_jsonl_fpath=resources_servers/code_gen/data/opencodereasoning_filtered_25k_train_1k_gpt-4o-2024-05-13_rollouts.jsonl \
    +responses_create_params.temperature=0.2 \
    +responses_create_params.max_output_tokens=2000 \
    +responses_create_params.top_p=0.95 \
    +num_samples_in_parallel=16 \
    +limit=1000
```
"""

import json
from typing import List

from datasets import load_dataset
from pydantic import BaseModel


ds = load_dataset("Nexusflow/comp_prog_filtered_no_function", split="train")

# Largely taken from https://github.com/NVIDIA/NeMo-Skills/blob/0af0b169ba62be9097f6362c4fb29202849ae036/nemo_skills/prompt/config/eval/livecodebench/python_codegen_reasoning.yaml
prompt_template = """You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.

Please use python programming language only.

You must use ```python for just the final solution code block with the following format:
```python
# Your code here
```

{question}"""


class UnitTests(BaseModel):
    inputs: List[str]
    outputs: List[str]


num_failed_examples = 0
with open("resources_servers/code_gen/data/opencodereasoning_filtered_25k_train.jsonl", "w") as f:
    for d in ds:
        try:
            UnitTests.model_validate_json(d["unit_tests"])
        except:
            from collections import Counter

            unit_tests = json.loads(d["unit_tests"])
            inputs = unit_tests["inputs"]
            outputs = unit_tests["outputs"]
            num_failed_examples += 1
            print(Counter([type(v) for v in inputs]), Counter([type(v) for v in outputs]))

            continue

        row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": prompt_template.format(question=d["question"]),
                    },
                ],
            },
            "verifier_metadata": {"unit_tests": UnitTests.model_validate_json(d["unit_tests"]).model_dump()},
            # Carry over original columns, even though they are unused for Gym
            "hash_id": d["hash_id"],
            "dataset": d["dataset"],
            "source": d["source"],
        }
        f.write(json.dumps(row) + "\n")

print(f"Skipped examples with improperly structured unit test test cases: {num_failed_examples}")
