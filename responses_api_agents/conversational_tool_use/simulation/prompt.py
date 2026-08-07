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

"""Canonical policy prompt and tool rendering for conversational tool-use rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
PROMPT_FILENAMES = ("agent_parallel_system.txt", "agent_system.txt")
PREPARE_COMMAND = "python -m resources_servers.conversational_tool_use_simulation.prepare"


def _read_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Conversational tool-use prompts are not prepared. Run `{PREPARE_COMMAND}`; missing {path}."
        )
    return path.read_text(encoding="utf-8")


def agent_system_message(policy: str, *, parallel_tool_calls: bool = False) -> str:
    filename = "agent_parallel_system.txt" if parallel_tool_calls else "agent_system.txt"
    template = _read_prompt(filename)
    return template.format(domain_policy=policy)


def responses_api_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rendered = []
    for tool in tools:
        description = tool.get("doc")
        if description is None:
            description = tool.get("description", "")
        parameters = tool.get("params")
        if parameters is None:
            parameters = tool.get("parameters")
        if parameters is None:
            parameters = {"type": "object", "properties": {}}
        rendered.append(
            {
                "type": "function",
                "name": tool["name"],
                "description": description,
                "parameters": parameters,
                "strict": bool(tool.get("strict", True)),
            }
        )
    return rendered
