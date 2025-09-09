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

# Run as `python resources_servers/stateful_counter/create_examples.py`
import json
from copy import deepcopy

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


queries = [
    ("add 1 then add 2 then get the count", 3),
    ("add 3 then add 4 then get the count", 7),
    ("add 5 then add 6 then get the count", 11),
    ("add 7 then add 8 then get the count", 15),
    ("add 9 then add 10 then get the count", 19),
]

base_dict = {
    "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "user", "content": ""},
        ],
        tools=[
            {
                "type": "function",
                "name": "increment_counter",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "",
                        },
                    },
                    "required": ["count"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_counter_value",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
    ).model_dump(exclude_unset=True),
    "expected_count": None,
}

examples = []
for query, expected_count in queries:
    example = deepcopy(base_dict)
    example["responses_create_params"]["input"][0]["content"] = query
    example["expected_count"] = expected_count

    examples.append(json.dumps(example) + "\n")

with open("resources_servers/stateful_counter/data/example.jsonl", "w") as f:
    f.writelines(examples)
