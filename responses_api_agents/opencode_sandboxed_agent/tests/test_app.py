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
from pathlib import Path
from typing import Any, Dict

from pytest import fixture

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig


class TestOpenCodeSandboxedAgent:
    def test_sanity(self) -> None:
        OpenCodeSandboxedAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
        )

    @fixture
    def opencode_export_test_data(self) -> Dict[str, Any]:
        test_data_path = Path(__file__).parent / "opencode_export_test_data.json"
        return json.loads(test_data_path.read_text())

    def test_opencode_export_to_output_items(self, opencode_export_test_data: Dict[str, Any]) -> None:
        actual_output_items = OpenCodeSandboxedAgent._opencode_export_to_output_items(None, opencode_export_test_data)
        expected_output_items = [
            NeMoGymEasyInputMessage(content=[{"text": "hello", "type": "input_text"}], role="user", type="message"),
            NeMoGymResponseOutputMessage(
                id="msg_fc5cbdbdf001E8AO55w3vBALYm",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[], text="Hello! How can I help you today?", type="output_text", logprobs=None
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]

        assert expected_output_items == actual_output_items
