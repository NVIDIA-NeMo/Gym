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
from asyncio import run

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="simple_agent_stateful",
    url_path="/v1/responses",
    json=NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": "What is the smallest positive integer that leaves a remainder of 4 when divided by 5 and a remainder of 6 when divided by 7?",
            },
        ],
        tools=[
            {
                "type": "function",
                "name": "execute_python",
                "description": "Execute Python code to perform calculations. You have access to numpy, scipy, pandas and basic math operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute for the calculation.",
                        },
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
    ),
)
result = run(task)
print(json.dumps(result.json()["output"], indent=4))
