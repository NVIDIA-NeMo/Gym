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

"""Lazy prompt-template loading for conversational tool-use simulation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
PREPARE_COMMAND = "python -m resources_servers.conversational_tool_use_simulation.prepare"


@lru_cache(maxsize=None)
def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Conversational tool-use prompts are not prepared. Run `{PREPARE_COMMAND}`; missing {path}."
        )
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class PromptTemplate:
    filename: str

    def read(self) -> str:
        return _read_prompt(PROMPTS_DIR / self.filename)

    def format(self, *args: Any, **kwargs: Any) -> str:
        return self.read().format(*args, **kwargs)


USER_SIMULATOR_SYSTEM_MESSAGE_TEMPLATE = PromptTemplate("user_simulator_system.txt")
ENVIRONMENT_SIMULATOR_SYSTEM_MESSAGE_TEMPLATE = PromptTemplate("environment_simulator_system.txt")
ENVIRONMENT_CONVERSATION_MESSAGE_TEMPLATE = PromptTemplate("environment_conversation_message.txt")
ENVIRONMENT_CONVERSATION_TEMPLATE = PromptTemplate("environment_conversation.txt")
ENVIRONMENT_USER_MODEL_MESSAGE_TEMPLATE = PromptTemplate("environment_user_model_message.txt")
MESSAGE_SYSTEM_MESSAGE_PREFIX = PromptTemplate("message_system_prefix.txt")
USER_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE = PromptTemplate("user_message_evaluation_system.txt")
AGENT_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE = PromptTemplate("agent_message_evaluation_system.txt")
ENVIRONMENT_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE = PromptTemplate("environment_message_evaluation_system.txt")
AGENT_CONVERSATION_EVALUATION_SYSTEM_MESSAGE_TEMPLATE = PromptTemplate("agent_conversation_evaluation_system.txt")
USER_AGENT_ENVIRONMENT_CONVERSATION_EVALUATION_SYSTEM_MESSAGE_TEMPLATE = PromptTemplate(
    "user_agent_environment_conversation_evaluation_system.txt"
)
MESSAGE_CONVERSATION_TEMPLATE = PromptTemplate("message_conversation.txt")
COMPLETE_CONVERSATION_TEMPLATE = PromptTemplate("complete_conversation.txt")
TEXT_MESSAGE_TEMPLATE = PromptTemplate("text_message.txt")
TOOL_CALL_MESSAGE_TEMPLATE = PromptTemplate("tool_call_message.txt")
TOOL_EXECUTION_MESSAGE_TEMPLATE = PromptTemplate("tool_execution_message.txt")
TOOL_DEFINITION_TEMPLATE = PromptTemplate("tool_definition.txt")

PROMPT_FILENAMES = tuple(
    template.filename
    for template in (
        AGENT_CONVERSATION_EVALUATION_SYSTEM_MESSAGE_TEMPLATE,
        AGENT_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE,
        COMPLETE_CONVERSATION_TEMPLATE,
        ENVIRONMENT_CONVERSATION_TEMPLATE,
        ENVIRONMENT_CONVERSATION_MESSAGE_TEMPLATE,
        ENVIRONMENT_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE,
        ENVIRONMENT_SIMULATOR_SYSTEM_MESSAGE_TEMPLATE,
        ENVIRONMENT_USER_MODEL_MESSAGE_TEMPLATE,
        MESSAGE_CONVERSATION_TEMPLATE,
        MESSAGE_SYSTEM_MESSAGE_PREFIX,
        TEXT_MESSAGE_TEMPLATE,
        TOOL_CALL_MESSAGE_TEMPLATE,
        TOOL_DEFINITION_TEMPLATE,
        TOOL_EXECUTION_MESSAGE_TEMPLATE,
        USER_AGENT_ENVIRONMENT_CONVERSATION_EVALUATION_SYSTEM_MESSAGE_TEMPLATE,
        USER_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE,
        USER_SIMULATOR_SYSTEM_MESSAGE_TEMPLATE,
    )
)
