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
    server_name="stateful_counter_simple_agent",
    url_path="/run",
    json={
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "user", "content": "add 4 then add 3 then get the count"},
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
        "expected_count": 7,
    },
)
result = run(task)
print(json.dumps(run(result.json()), indent=4))
