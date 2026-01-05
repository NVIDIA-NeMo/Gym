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
from asyncio import run

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="example_single_tool_call_simple_agent",
    url_path="/v1/responses",
    json=NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "developer",
                "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has.",
            },
            {"role": "user", "content": "going out in sf tn"},
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
    ),
)
result = run(task)
print(json.dumps(run(result.json())["output"], indent=4))
