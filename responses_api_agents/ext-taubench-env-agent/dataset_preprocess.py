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
python responses_api_agents/ext-taubench-env-agent/dataset_preprocess.py
```
"""

import json
from pathlib import Path
from random import seed, shuffle


input_json_folder = Path("responses_api_agents/ext-taubench-env-agent/data/final_delivery_dual_control")
output_jsonl_train_file = Path("responses_api_agents/ext-taubench-env-agent/data/train.jsonl")
output_jsonl_validation_file = Path("responses_api_agents/ext-taubench-env-agent/data/validation.jsonl")
validation_size = 64

all_examples = []
for file in input_json_folder.iterdir():
    if not file.suffix == ".json":
        continue

    data = json.loads(file.read_text())
    new_record = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "<placeholder, unused in downstream agent calls>"}],
        },
        **data,
    }
    all_examples.append(new_record)

seed(42)
shuffle(all_examples)

train_examples = all_examples[:-validation_size]
validation_examples = all_examples[-validation_size:]
assert len(train_examples) == (len(all_examples) - validation_size)
assert len(validation_examples) == validation_size


def write_to_file(examples: list[dict], fpath: Path):
    with fpath.open("w") as f:
        for example in examples:
            f.write(json.dumps(example, separators=(",", ":")) + "\n")


write_to_file(train_examples, output_jsonl_train_file)
write_to_file(validation_examples, output_jsonl_validation_file)
