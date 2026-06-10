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
import asyncio
import json

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="math_with_judge",
    url_path="/verify",
    json={
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "What is 2 + 2?",
                },
            ]
        },
        "response": {
            "id": "response_1",
            "created_at": 1.0,
            "model": "model_1",
            "object": "response",
            "output": [
                {
                    "id": "message_1",
                    "content": [
                        {
                            "annotations": [],
                            "text": "5",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
        "question": "What is 2 + 2?",
        "expected_answer": "4",
    },
)
result = asyncio.run(task)
print(json.dumps(asyncio.run(result.json()), indent=4))
