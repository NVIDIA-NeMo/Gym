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
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# Adapted from https://github.com/coreweave/inference-cookbook/blob/41f991fef0d45430d20cfd8bc142c4bfc5070369/libs/cwtrain/cwtrain/integrations/gym_tool_calls.py
# Original license: Apache-2.0.

"""Parse the Qwen3-coder XML tool format used by Nemotron 3."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any


_TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _argument(value: str, schema: dict | None) -> Any:
    """A tool argument as its schema type: strings stay text, other types are parsed as JSON."""
    if (schema or {}).get("type", "string") == "string":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_tool_calls(text: str, tools: list | None = None) -> tuple[str, list[dict]]:
    """Split generated text into its content and its Qwen3-coder XML tool calls."""
    schemas = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        schemas[function.get("name")] = (function.get("parameters") or {}).get("properties") or {}
    calls = []
    for match in _TOOL_CALL.finditer(text):
        name = match.group(1).strip()
        properties = schemas.get(name, {})
        arguments = {
            key.strip(): _argument(value, properties.get(key.strip()))
            for key, value in _PARAMETER.findall(match.group(2))
        }
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        )
    if not calls:
        return text, []
    return text[: text.index("<tool_call>")].rstrip(), calls
