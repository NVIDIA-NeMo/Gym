# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Client-side tool-call parsers for the SGLang ``/generate`` path."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


QWEN3_CODER_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
QWEN3_CODER_PARAM_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>",
    re.DOTALL,
)


def _tool_param_types(tools: Optional[List[Dict[str, Any]]], function_name: str) -> Dict[str, str]:
    """Map parameter names to JSON-schema types for one requested tool."""
    for tool in tools or []:
        function = tool.get("function", tool)
        if function.get("name") != function_name:
            continue
        properties = (function.get("parameters") or {}).get("properties") or {}
        return {key: prop.get("type", "string") for key, prop in properties.items() if isinstance(prop, dict)}
    return {}


def _coerce_param_value(raw: str, type_str: str) -> Any:
    """Best-effort conversion of a raw parameter using its declared type."""
    if type_str in ("integer", "number"):
        try:
            return int(raw) if type_str == "integer" else float(raw)
        except ValueError:
            return raw
    if type_str == "boolean":
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        return raw
    if type_str in ("array", "object"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def parse_qwen3_coder_tool_calls(
    text: str, tools: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Parse XML-like tool calls into OpenAI-compatible tool-call mappings."""
    tool_calls: List[Dict[str, Any]] = []
    for match in QWEN3_CODER_TOOL_CALL_PATTERN.finditer(text):
        function_name = match.group(1).strip()
        param_types = _tool_param_types(tools, function_name)
        arguments: Dict[str, Any] = {}
        for param_match in QWEN3_CODER_PARAM_PATTERN.finditer(match.group(2)):
            key = param_match.group(1).strip()
            arguments[key] = _coerce_param_value(
                param_match.group(2),
                param_types.get(key, "string"),
            )
        tool_calls.append(
            {
                "id": f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return tool_calls, QWEN3_CODER_TOOL_CALL_PATTERN.sub("", text).strip()


def normalize_tool_call_arguments(messages: List[Any]) -> List[Any]:
    """Decode JSON-object argument strings for chat-template rendering."""
    normalized: List[Any] = []
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not tool_calls:
            normalized.append(message)
            continue
        new_tool_calls = []
        for tool_call in tool_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                except ValueError:
                    decoded = None
                if isinstance(decoded, dict):
                    tool_call = dict(tool_call, function=dict(function, arguments=decoded))
            new_tool_calls.append(tool_call)
        normalized.append(dict(message, tool_calls=new_tool_calls))
    return normalized
