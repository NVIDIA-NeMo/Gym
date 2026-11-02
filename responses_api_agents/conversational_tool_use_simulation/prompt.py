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
from typing import Any


AGENT_SYSTEM_MESSAGE_TEMPLATE = """You are a customer service agent that helps the user.  The policy that determines how you should respond to requests from users is described below between the <policy> and </policy> tags.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

<policy>
{domain_policy}
</policy>

Try to be helpful and always follow the policy."""

AGENT_PARALLEL_SYSTEM_MESSAGE_TEMPLATE = """You are a customer service agent that helps the user.  The policy that determines how you should respond to requests from users is described below between the <policy> and </policy> tags.

In each turn you can either:
- Send a message to the user.
- Make one or more tool calls.
You cannot do both at the same time.

<policy>
{domain_policy}
</policy>

Try to be helpful and always follow the policy."""


def agent_system_message(policy: str, *, parallel_tool_calls: bool = False) -> str:
    template = AGENT_PARALLEL_SYSTEM_MESSAGE_TEMPLATE if parallel_tool_calls else AGENT_SYSTEM_MESSAGE_TEMPLATE
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
