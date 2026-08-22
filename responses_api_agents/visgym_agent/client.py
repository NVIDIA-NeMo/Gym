# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Programmatic single-call sample against a running visgym_agent.

Mirrors `responses_api_agents/simple_agent/client.py` for parity. Path B does
not use tools, so we omit the tools field entirely; the agent extracts the
last `\\boxed{...}` from the assistant text.
"""
import json
from asyncio import run

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="visgym_agent",
    url_path="/v1/responses",
    json=NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "developer",
                "content": (
                    "You are a game-playing agent. Read the rules from the "
                    "first user message and write your final action wrapped "
                    "in \\boxed{...} on the last line."
                ),
            },
            {"role": "user", "content": "Move up. Reply with \\boxed{[up]}."},
        ],
    ),
)
result = run(task)
print(json.dumps(run(result.json())["output"], indent=4))
