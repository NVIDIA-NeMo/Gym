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

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="bash_sandbox_agent",
    url_path="/run",
    json=BaseRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "developer",
                    "content": (
                        "You are a capable coding assistant. Use tools when helpful and call finish when done."
                    ),
                },
                {
                    "role": "user",
                    "content": "Run `echo hello` using run_shell, then call finish with a short summary.",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "name": "run_shell",
                    "description": "Execute a shell command in the sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                {
                    "type": "function",
                    "name": "finish",
                    "description": "Mark the task complete.",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            ],
        )
    ),
)
result = run(task)
print(json.dumps(run(result.json()), indent=4))
