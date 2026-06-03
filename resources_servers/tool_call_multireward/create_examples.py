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

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)


# (user query, expected city argument)
queries = [
    ("what's the weather in San Francisco?", "San Francisco"),
    ("how hot is it in New York today?", "New York"),
    ("tell me the forecast for Seattle", "Seattle"),
    ("is it raining in London right now?", "London"),
    ("weather in Tokyo please", "Tokyo"),
]

base_response_create_params = NeMoGymResponseCreateParamsNonStreaming(
    input=[
        {
            "role": "developer",
            "content": (
                "You are a helpful assistant. When the user asks about the weather, "
                "respond with exactly one get_weather tool call and no other text."
            ),
        },
    ],
    tools=[
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for.",
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
for query, city in queries:
    example = base_response_create_params.model_copy(
        update={
            "input": base_response_create_params.input
            + [NeMoGymEasyInputMessage(role="user", content=query)]
        }
    )
    row = {
        "responses_create_params": example.model_dump(exclude_unset=True),
        "expected_call": {"name": "get_weather", "arguments": {"city": city}},
    }
    example_strs.append(json.dumps(row) + "\n")


with open("resources_servers/tool_call_multireward/data/example.jsonl", "w") as f:
    f.writelines(example_strs)
