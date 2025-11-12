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
import json

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)


queries = [
    "what's it like in sf?",
    "going out in sf tn",
    "humidity in sf",
    "how's the outside?",
    "get the weather for 3 cities in the us",
]
base_response_create_params = NeMoGymResponseCreateParamsNonStreaming(
    input=[
        {
            "role": "developer",
            "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has.",
        },
    ],
    tools=[
        {
            "type": "function",
            "name": "get_weather",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ],
)

example_strs = []
for query in queries:
    example = base_response_create_params.model_copy(
        update={"input": base_response_create_params.input + [NeMoGymEasyInputMessage(role="user", content=query)]}
    )
    example_strs.append(json.dumps({"responses_create_params": example.model_dump(exclude_unset=True)}) + "\n")


with open("resources_servers/example_simple_weather/data/example.jsonl", "w") as f:
    f.writelines(example_strs)
