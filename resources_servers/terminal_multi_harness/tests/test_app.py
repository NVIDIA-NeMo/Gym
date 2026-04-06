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
from unittest.mock import MagicMock

from pytest import approx, fixture

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_multi_harness.app import (
    TerminalMultiHarnessResourcesServer,
    TerminalMultiHarnessResourcesServerConfig,
    TerminalMultiHarnessVerifyRequest,
)
from resources_servers.terminal_multi_harness.common.verification_utils import (
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
    StepRewardCategory,
    ToolCallComparatorConfig,
)


class TestApp:
    @fixture
    def resources_server(self) -> TerminalMultiHarnessResourcesServer:
        resources_server_config = TerminalMultiHarnessResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="terminal_multi_harness_server",
            tool_call_comparator_config=ToolCallComparatorConfig(
                string_similarity_threshold=0.9,
                ignored_argument_keys_by_tool={"exec_command": ["yield_time_ms"]},
            ),
        )
        return TerminalMultiHarnessResourcesServer(
            config=resources_server_config,
            server_client=MagicMock(spec=ServerClient),
        )

    async def test_verify_function_call_batch(
        self, resources_server: TerminalMultiHarnessResourcesServer
    ) -> None:
        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content="Inspect the repo state.")]
        )
        response = NeMoGymResponse(
            id="resp_batch",
            created_at=1001,
            model="test_model",
            object="response",
            output=[
                NeMoGymResponseFunctionToolCall(
                    call_id="call_1",
                    name="exec_command",
                    arguments='{"cmd": "pwd"}',
                ),
                NeMoGymResponseFunctionToolCall(
                    call_id="call_2",
                    name="exec_command",
                    arguments='{"cmd": "git status --short", "yield_time_ms": 1000}',
                ),
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        expected_action = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "pwd"}'),
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "git status --short"}'),
            ],
        )

        verify_request = TerminalMultiHarnessVerifyRequest(
            responses_create_params=responses_create_params,
            response=response,
            harness="codex",
            expected_action=expected_action,
        )
        verify_response = await resources_server.verify(verify_request)
        assert verify_response.reward == approx(1.0)
        assert verify_response.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

    async def test_verify_message(
        self, resources_server: TerminalMultiHarnessResourcesServer
    ) -> None:
        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content="Say hello.")]
        )
        response = NeMoGymResponse(
            id="resp_message",
            created_at=1001,
            model="test_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_1",
                    content=[NeMoGymResponseOutputText(annotations=[], text="hello there")],
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

        verify_request = TerminalMultiHarnessVerifyRequest(
            responses_create_params=responses_create_params,
            response=response,
            harness="opencode",
            expected_action=MessageAction(type="message", content="hello there"),
        )
        verify_response = await resources_server.verify(verify_request)
        assert verify_response.reward == approx(1.0)
        assert verify_response.category == StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND
