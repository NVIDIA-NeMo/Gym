# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Client-side tool-call text parsers for the SGLang /generate path.

The SGLang engine path tokenizes/generates via the native ``/generate`` endpoint
(the only source of exact sampled token ids + logprobs), so tool calls must be
re-parsed from raw generated text on the client, mirroring what the serving
engine's ``tool_call_parser`` would have produced on a chat endpoint.

This module is deliberately dependency-free (stdlib only) so it can be
unit-tested without the server stack.

Currently implemented: the ``qwen3_coder`` XML-ish format used by e.g.
Nemotron Nano V3.5 chat templates::

    <tool_call>
    <function=example_function_name>
    <parameter=example_parameter_1>
    value_1
    </parameter>
    </function>
    </tool_call>

(The hermes JSON format is parsed inline in ``app.py``; it predates this
module.)
"""

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


def _tool_param_types(
    tools: Optional[List[Dict[str, Any]]], function_name: str
) -> Dict[str, str]:
    """Map parameter name -> JSON-schema type for one tool, from the request's tools."""
    for tool in tools or []:
        function = tool.get("function", tool)
        if function.get("name") != function_name:
            continue
        properties = (function.get("parameters") or {}).get("properties") or {}
        return {
            key: prop.get("type", "string")
            for key, prop in properties.items()
            if isinstance(prop, dict)
        }
    return {}


def _coerce_param_value(raw: str, type_str: str) -> Any:
    """Best-effort schema-aware coercion of a raw text parameter value.

    Mirrors the serving engines' qwen3_coder tool parsers, which convert
    parameter text according to the declared JSON-schema type. Unparseable
    values fall back to the raw string rather than erroring, so a slightly
    malformed model output degrades to a string argument instead of a dropped
    tool call.
    """
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
    """Parse qwen3_coder-format tool calls out of raw generated text.

    Args:
        text: generated text AFTER any reasoning block has been split off.
        tools: the request's OpenAI-style tools list (used for schema-aware
            argument type coercion); optional.

    Returns:
        (tool_calls, content): OpenAI-schema tool_call dicts (``function.arguments``
        is a JSON string), and the text with tool-call blocks removed.
    """
    tool_calls: List[Dict[str, Any]] = []
    for match in QWEN3_CODER_TOOL_CALL_PATTERN.finditer(text):
        function_name = match.group(1).strip()
        params_block = match.group(2)
        param_types = _tool_param_types(tools, function_name)
        arguments: Dict[str, Any] = {}
        for param_match in QWEN3_CODER_PARAM_PATTERN.finditer(params_block):
            key = param_match.group(1).strip()
            raw_value = param_match.group(2)
            arguments[key] = _coerce_param_value(
                raw_value, param_types.get(key, "string")
            )
        tool_calls.append(
            dict(
                id=f"call_{uuid4().hex}",
                type="function",
                function=dict(
                    name=function_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        )
    content = QWEN3_CODER_TOOL_CALL_PATTERN.sub("", text).strip()
    return tool_calls, content
