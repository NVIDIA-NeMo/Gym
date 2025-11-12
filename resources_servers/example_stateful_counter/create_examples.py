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

# Run as `python resources_servers/example_stateful_counter/create_examples.py`
import json
from copy import deepcopy

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


queries = [
    # user query, initial count, expected count.
    ("add 1 then add 2 then get the count", 3, 6),
    ("add 4 then add 5 then get the count", 6, 15),
    ("add 7 then add 8 then get the count", 9, 24),
    ("add 10 then add 11 then get the count", 12, 33),
    ("add 13 then add 14 then get the count", 15, 42),
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
for query, initial_count, expected_count in queries:
    example = deepcopy(base_dict)
    example["responses_create_params"]["input"][0]["content"] = query
    example["initial_count"] = initial_count
    example["expected_count"] = expected_count

    examples.append(json.dumps(example) + "\n")

with open("resources_servers/example_stateful_counter/data/example.jsonl", "w") as f:
    f.writelines(examples)
