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

from pathlib import Path

import pytest

from resources_servers.conversational_tool_use_simulation import prompt as resource_prompt
from responses_api_agents.conversational_tool_use.simulation import prompt as agent_prompt


RESOURCE_PROMPTS = {
    "agent_conversation_evaluation_system.txt": "Evaluate agent steps: {policy} {tool_definitions} {transfer_instruction} {evaluation_schema}",
    "agent_message_evaluation_system.txt": "Evaluate agent message: {policy} {tool_definitions} {evaluation_schema}",
    "complete_conversation.txt": "<steps>\n{steps}\n</steps>",
    "environment_conversation.txt": "<conversation>\n{conversation}\n</conversation>\n\n",
    "environment_conversation_message.txt": "<{sender}>\n{message}\n</{sender}>",
    "environment_message_evaluation_system.txt": (
        "Evaluate tool result: {policy} {customer_scenario} {tool_definitions} {evaluation_schema}"
    ),
    "environment_simulator_system.txt": "Simulate tool result: {domain_policy} {customer_scenario} {tool_definitions}",
    "environment_user_model_message.txt": (
        "{conversation}Tool to execute: {tool_name}\nArguments for tool execution: {arguments}"
    ),
    "message_conversation.txt": "<previous_steps>{previous_steps}</previous_steps><current_step>{current_step}</current_step>",
    "message_system_prefix.txt": "Evaluate one step under {policy}.",
    "text_message.txt": "<message>Sender: {sender}\nContent: {content}</message>",
    "tool_call_message.txt": (
        "<execute_tool>Execution ID: {execution_id}\nTool name: {tool_name}\nArguments: {arguments}</execute_tool>"
    ),
    "tool_definition.txt": (
        "<tool>Name: {name}\nDocumentation: {documentation}\nParameters: {parameters}\nReturn: {return_type}</tool>"
    ),
    "tool_execution_message.txt": (
        "<tool_result>Execution ID: {execution_id}\nExecution result: {execution_result}</tool_result>"
    ),
    "user_agent_environment_conversation_evaluation_system.txt": (
        "Evaluate conversation: {policy} {tool_definitions} {customer_scenario} "
        "{transfer_instruction} {evaluation_schema}"
    ),
    "user_message_evaluation_system.txt": (
        "Evaluate user message: {policy} {customer_scenario} {complete_indicator} "
        "{transfer_indicator} {evaluation_schema}"
    ),
    "user_simulator_system.txt": ("Simulate customer: {complete_indicator} {transfer_indicator} {customer_scenario}"),
}


@pytest.fixture(autouse=True)
def prepared_prompt_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resource_prompts = tmp_path / "resource_prompts"
    resource_prompts.mkdir()
    for filename, content in RESOURCE_PROMPTS.items():
        (resource_prompts / filename).write_text(content, encoding="utf-8")
    monkeypatch.setattr(resource_prompt, "PROMPTS_DIR", resource_prompts)

    agent_prompts = tmp_path / "agent_prompts"
    agent_prompts.mkdir()
    (agent_prompts / "agent_system.txt").write_text("Agent policy: {domain_policy}", encoding="utf-8")
    (agent_prompts / "agent_parallel_system.txt").write_text(
        "Make one or more tool calls. Parallel agent policy: {domain_policy}", encoding="utf-8"
    )
    monkeypatch.setattr(agent_prompt, "PROMPTS_DIR", agent_prompts)
