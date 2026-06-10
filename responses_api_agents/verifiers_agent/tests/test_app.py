# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.verifiers_agent.app import (
    VerifiersAgent,
    VerifiersAgentConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_convert_completion_keeps_tool_outputs_as_response_items(self) -> None:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        agent = VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

        rollout_output = {
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        json.dumps(
                            {
                                "id": "call_1",
                                "name": "python",
                                "arguments": json.dumps({"expr": "2+2"}),
                            }
                        )
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "4"},
                {"role": "assistant", "content": "answer"},
            ],
            "trajectory": [
                {
                    "completion": [
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "python",
                                    "arguments": json.dumps({"expr": "2+2"}),
                                }
                            ],
                        }
                    ],
                    "tokens": {
                        "prompt_ids": [1],
                        "completion_ids": [2],
                        "completion_logprobs": [0.0],
                    },
                },
                {
                    "completion": [{"role": "assistant", "content": "answer"}],
                    "tokens": {
                        "prompt_ids": [3],
                        "completion_ids": [4],
                        "completion_logprobs": [-0.1],
                    },
                },
            ],
        }

        output = agent._convert_trajectory_to_output(rollout_output)

        assert [item["type"] for item in output] == ["function_call", "function_call_output", "message"]
        assert output[0]["call_id"] == "call_1"
        assert output[0]["name"] == "python"
        assert output[0]["arguments"] == json.dumps({"expr": "2+2"})
        assert output[0]["prompt_token_ids"] == [1]
        assert output[1]["call_id"] == "call_1"
        assert output[1]["output"] == "4"
        assert output[2]["content"][0]["text"] == "answer"
        assert output[2]["prompt_token_ids"] == [3]
