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

"""Prompt-based tool-call extraction for ToolAlignBench.

Faithful port of ``runner/src/openrouter/extract.ts``, ``dedup.ts`` and ``json-recovery.ts`` from
https://github.com/aryankeluskar/ToolAlignBench (MIT, (c) 2026 Aryan Keluskar).

ToolAlignBench never sends a native ``tools`` array: tools are documented in the system prompt and
the model is asked for ``<tool_call>`` XML, which is recovered from the reply text here. Models
frequently ignore that template, so extraction runs five prioritised passes -- **each firing only
if every earlier pass produced nothing** -- then two JSON fallbacks.

The passes scan the *entire* reply, so a call written inside a ``<scratchpad>`` block is extracted
and executed exactly as one written in the visible answer. That is upstream behaviour and it
matters: the scratchpad is described to the model as private, but it is graded.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TOOL_CALL_FORMAT_XML = "xml"
TOOL_CALL_FORMAT_JSON = "json"

# Upstream truncates preserved raw argument text at 500 characters (json-recovery.ts).
MAX_RAW_CHARS = 500

# `re.ASCII` keeps `\w` to [A-Za-z0-9_] as in JavaScript. JS `.` never matches a newline, so
# patterns that use `.` stay without DOTALL while those written `[\s\S]` get it.
_STANDALONE_RE = re.compile(
    r"<tool_name>(.*?)</tool_name>\s*<arguments>(.*?)</arguments>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_TOOL_NAME_RE = re.compile(r"<tool_name>(.*?)</tool_name>", re.IGNORECASE)
_BARE_NAME_RE = re.compile(r"^([a-zA-Z_]\w*)\s*(?:TOOL_NAME|<arguments>)", re.IGNORECASE | re.ASCII)
_ARGUMENTS_RE = re.compile(r"<arguments>(.*?)</arguments>", re.DOTALL | re.IGNORECASE)
_QWEN_RE = re.compile(
    r"<tool_call>\s*<(\w+)>(?:TOOL_NAME)?</\1>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_MISSING_CLOSE_TOOL_NAME_RE = re.compile(
    r"<tool_name>\s*([a-zA-Z_]\w*)\s*>?\s*(?:<tool_name>\s*)?(?=<arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_MISSING_OPEN_TOOL_NAME_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*</tool_name>\s*(?=<arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_MISMATCHED_CLOSE_TOOL_NAME_RE = re.compile(
    r"<tool_name>\s*([a-zA-Z_]\w*)\s*</[a-zA-Z_]\w*>\s*(?=<arguments>|\{|</arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_MISSING_CLOSE_ANGLE_TOOL_NAME_RE = re.compile(
    r"<tool_name>\s*([a-zA-Z_]\w*)\s*</tool_name\s*(?=<arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_BROKEN_ARGUMENT_TOOL_NAME_RE = re.compile(
    r"<tool_name>\s*([a-zA-Z_]\w*)\s*</arguments>\s*</[a-zA-Z_]\w*>",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_ATTRIBUTE_TOOL_NAME_RE = re.compile(
    r"<(?:tool_name|function)=([a-zA-Z_]\w*)>\s*(?:</tool_name>)?",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_UNTERMINATED_ATTRIBUTE_TOOL_NAME_RE = re.compile(
    r"<tool_name=([a-zA-Z_]\w*)\s*(?=<arguments>|</arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_FUNCTION_MISMATCHED_CLOSE_RE = re.compile(
    r"<function=([a-zA-Z_]\w*)\s*</function>\s*(?=<arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_TOOL_NAME_HARMONY_WRAPPER_RE = re.compile(
    r"<tool_name>\s*([a-zA-Z_]\w*)\s*(?=<\|close\|>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_TAG_AS_TOOL_NAME_RE = re.compile(
    r"<([a-zA-Z_]\w*)>(?:\s*\1\s*</\1>)?\s*(?=<arguments>|\{|</arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_TAG_AS_TOOL_NAME_WITH_DESCRIPTION_RE = re.compile(
    r"<([a-zA-Z_]\w*)>[^<]*</\1>\s*(?=<arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_TAG_NAME_BROKEN_ARGUMENT_RE = re.compile(
    r"<([a-zA-Z_]\w*)>\s*\1\s*</arguments>",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_TAG_AS_TOOL_NAME_PLACEHOLDER_RE = re.compile(
    r"<([a-zA-Z_]\w*)>\s*TOOL_NAME(?:\s*[:=]\s*[a-zA-Z_]\w*)?\s*</\1>\s*(?=<arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_TAG_AS_TOOL_NAME_MISMATCHED_CLOSE_RE = re.compile(
    r"<([a-zA-Z_]\w*)>\s*</tool>\s*(?=<arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_TAG_AS_TOOL_NAME_MISSING_CLOSE_RE = re.compile(
    r"<([a-zA-Z_]\w*)>\s*(?:\1|TOOL_NAME)?\s*(?=<arguments>)",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_HARMONY_OPEN_CALL_RE = re.compile(
    r'<\|open\|>call\s+tool="([a-zA-Z_]\w*)"(.*?)(?:</tool_call>|<\|close\|>call<\|sep\|>)',
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_HARMONY_ARGUMENT_RE = re.compile(
    r'<\|open\|>argument\s+key="arguments".*?<\|sep\|>(.*?)</argument>',
    re.DOTALL | re.IGNORECASE,
)
_HARMONY_BROKEN_NAME_RE = re.compile(
    r'<\|open\|>call\s+tool="([a-zA-Z_]\w*)</tool_name>(.*?)(?:</tool_call>|<\|close\|>call<\|sep\|>)',
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_HARMONY_KEYED_ARGUMENT_RE = re.compile(
    r'<\|open\|>argument\s+key="([^"]+)".*?<\|sep\|>(.*?)<\|close\|>argument<\|sep\|>',
    re.DOTALL | re.IGNORECASE,
)
_RAW_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)
_GPTOSS_RE = re.compile(
    r"<\|[^|]+\|>[^}]*to=(\w+)\s[^}]*\{[^}]*\"content\"\s*:\s*\"([^\"]*)\"",
    re.IGNORECASE | re.ASCII,
)
_GENERIC_RE = re.compile(
    r"\"to\"\s*:\s*\"?(\w+)\"?\s*[,}].*?\{.*?\"content\"\s*:\s*\"([^\"]*)\"",
    re.DOTALL | re.IGNORECASE | re.ASCII,
)
_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_RAW_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\"name\".*?\}\s*\]", re.DOTALL)
_SCRATCHPAD_RE = re.compile(r"<scratchpad>(.*?)</scratchpad>", re.DOTALL | re.IGNORECASE)
_TRAILING_TAG_RE = re.compile(r"<[^>]+>$")
_STRIP_XML_MARKERS_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
_STRIP_JSON_MARKERS_RE = re.compile(r"```(?:json)?\s*\[.*?\"name\".*?\]\s*```", re.DOTALL | re.IGNORECASE)

# Any of these in a reply means the model was *trying* to call a tool. Used to tell "the model
# called nothing" apart from "the model called something we failed to parse" -- the latter would
# otherwise read as perfect alignment. See `has_unparsed_tool_call_markup`.
_TOOL_CALL_MARKERS = ("<tool_call>", "</tool_call>", "<tool_name>", "<arguments>")

# `JSON.stringify` inserts no whitespace, so neither may we: these argument strings are what the
# model is shown on the next turn and what the verifier reads back.
_COMPACT = (",", ":")
_TOOL_IDENTIFIER_RE = re.compile(r"[a-zA-Z_]\w*", re.ASCII)
_RESERVED_TOOL_TAGS = {"arguments", "scratchpad", "think", "tool_call", "tool_name", "tool_result"}


def _dumps(value: Any) -> str:
    """`json.dumps` with JavaScript's separators."""
    return json.dumps(value, separators=_COMPACT)


def _valid_tool_name(value: str) -> bool:
    return bool(_TOOL_IDENTIFIER_RE.fullmatch(value)) and (
        value == "TOOL_NAME" or value.lower() not in _RESERVED_TOOL_TAGS
    )


def _first_valid_name_match(content: str, patterns: tuple[re.Pattern[str], ...]) -> re.Match[str] | None:
    for pattern in patterns:
        match = pattern.search(content)
        if match is not None and _valid_tool_name(match.group(1).strip()):
            return match
    return None


class ExtractedToolCall(BaseModel):
    """One tool call recovered from reply text.

    ``source`` records which pass produced it, so a rollout can be sliced by how malformed the
    model's tool syntax was.
    """

    name: str
    arguments: str
    raw_arguments: Dict[str, Any] = Field(default_factory=dict)
    source: str


class ModelReasoning(BaseModel):
    """Contents of a ``<scratchpad>`` block, if the model wrote one."""

    scratchpad_content: str = ""
    has_scratchpad: bool = False
    reasoning: str = ""


class JsonParseResult(BaseModel):
    """Outcome of :func:`robust_json_parse`."""

    ok: bool
    value: Optional[Dict[str, Any]] = None
    reason: Optional[Literal["empty", "fragment", "invalid"]] = None
    raw: str = ""


def safe_json_parse_object(raw: str) -> Dict[str, Any]:
    """Parse a JSON object, degrading to ``{"_raw": raw}`` rather than raising."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"_raw": raw}


def _try_parse_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``text`` as a JSON object, unwrapping one level of double encoding."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, str):
        try:
            inner = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            return None
        return inner if isinstance(inner, dict) else None
    # `bool` is a subclass of `int`, and neither is a JSON object; only dicts qualify.
    return parsed if isinstance(parsed, dict) else None


def robust_json_parse(json_string: str) -> JsonParseResult:
    """Parse LLM-emitted tool arguments, with the three upstream recovery attempts.

    Direct parse (plus double-encoded-string unwrap), then strip a trailing ``<...>`` tag, then
    slice from the first ``{``/``[``. A failure preserves the raw text for diagnostics.
    """
    if not json_string or not json_string.strip():
        return JsonParseResult(ok=False, reason="empty", raw="")

    trimmed = json_string.strip()
    raw = trimmed[:MAX_RAW_CHARS]

    if not trimmed.startswith("{") and not trimmed.startswith("[") and not trimmed.startswith('"'):
        return JsonParseResult(ok=False, reason="fragment", raw=raw)

    direct = _try_parse_object(trimmed)
    if direct is not None:
        return JsonParseResult(ok=True, value=direct)

    cleaned = _TRAILING_TAG_RE.sub("", trimmed).strip()
    cleaned_result = _try_parse_object(cleaned)
    if cleaned_result is not None:
        return JsonParseResult(ok=True, value=cleaned_result)

    json_start = max(trimmed.find("{"), trimmed.find("["))
    if json_start > 0:
        sliced = _try_parse_object(trimmed[json_start:])
        if sliced is not None:
            return JsonParseResult(ok=True, value=sliced)

    return JsonParseResult(ok=False, reason="invalid", raw=raw)


def normalize_arguments(arguments: str) -> Dict[str, Any]:
    """Normalise an argument string to the object shape the verifier reads.

    Mirrors what upstream persists in its transcript: the recovered object on success, and
    ``{"_raw": ..., "_error": ...}`` on failure. The verifier's email classification reads ``to``
    and ``from`` out of this, so an unrecoverable argument blob keeps an email at ``boundary``
    exactly as it does upstream.
    """
    result = robust_json_parse(arguments)
    if result.ok and result.value is not None:
        return result.value
    return {"_raw": result.raw, "_error": result.reason}


def extract_model_reasoning(response_text: str) -> ModelReasoning:
    """Pull the ``<scratchpad>`` block out of a reply, if present."""
    match = _SCRATCHPAD_RE.search(response_text)
    content = match.group(1).strip() if match else ""
    return ModelReasoning(
        scratchpad_content=content,
        has_scratchpad=match is not None,
        reasoning=content[:200],
    )


def _loose_tool_call_blocks(response_text: str) -> List[str]:
    """Return every attempted ``<tool_call>`` body, including missing close tags."""
    parts = re.split(r"<tool_call>", response_text, flags=re.IGNORECASE)
    blocks = []
    for part in parts[1:]:
        close = re.search(r"</tool_call>", part, flags=re.IGNORECASE)
        blocks.append(part[: close.start()] if close else part)
    return blocks


def _append_unique_calls(target: List[ExtractedToolCall], candidates: List[ExtractedToolCall]) -> None:
    """Append recovery calls not already produced by an earlier, higher-priority pass."""
    seen = {(call.name, call.arguments) for call in target}
    for call in candidates:
        key = (call.name, call.arguments)
        if key not in seen:
            target.append(call)
            seen.add(key)


def _recover_malformed_xml_calls(response_text: str) -> List[ExtractedToolCall]:
    """Recover provider-observed XML variants without changing documented XML parsing."""
    recovered = []
    for content in _loose_tool_call_blocks(response_text):
        name_match = _first_valid_name_match(
            content,
            (
                _MISSING_CLOSE_TOOL_NAME_RE,
                _MISSING_OPEN_TOOL_NAME_RE,
                _MISMATCHED_CLOSE_TOOL_NAME_RE,
                _MISSING_CLOSE_ANGLE_TOOL_NAME_RE,
                _BROKEN_ARGUMENT_TOOL_NAME_RE,
                _ATTRIBUTE_TOOL_NAME_RE,
                _UNTERMINATED_ATTRIBUTE_TOOL_NAME_RE,
                _FUNCTION_MISMATCHED_CLOSE_RE,
                _TOOL_NAME_HARMONY_WRAPPER_RE,
                _TAG_AS_TOOL_NAME_RE,
                _TAG_AS_TOOL_NAME_WITH_DESCRIPTION_RE,
                _TAG_NAME_BROKEN_ARGUMENT_RE,
                _TAG_AS_TOOL_NAME_PLACEHOLDER_RE,
                _TAG_AS_TOOL_NAME_MISMATCHED_CLOSE_RE,
                _TAG_AS_TOOL_NAME_MISSING_CLOSE_RE,
            ),
        )
        if name_match is None:
            continue
        tool_name = name_match.group(1).strip()
        args_match = _ARGUMENTS_RE.search(content)
        if args_match is None:
            args_match = _RAW_OBJECT_RE.search(content[name_match.end() :])
        args_string = args_match.group(1).strip() if args_match else "{}"
        recovered.append(
            ExtractedToolCall(
                name=tool_name,
                arguments=args_string,
                raw_arguments=safe_json_parse_object(args_string),
                source="xml_recovery",
            )
        )
    return recovered


def _recover_harmony_open_calls(response_text: str) -> List[ExtractedToolCall]:
    """Recover prompted tool calls wrapped in Kimi/OpenAI harmony markup."""
    recovered = []
    for match in [*_HARMONY_OPEN_CALL_RE.finditer(response_text), *_HARMONY_BROKEN_NAME_RE.finditer(response_text)]:
        tool_name = match.group(1).strip()
        body = match.group(2)
        keyed_arguments = dict(_HARMONY_KEYED_ARGUMENT_RE.findall(body))
        if keyed_arguments:
            if set(keyed_arguments) == {"arguments"}:
                args_string = keyed_arguments["arguments"].strip()
            else:
                args_string = _dumps({key: value.strip() for key, value in keyed_arguments.items()})
            recovered.append(
                ExtractedToolCall(
                    name=tool_name,
                    arguments=args_string,
                    raw_arguments=safe_json_parse_object(args_string),
                    source="harmony_open",
                )
            )
            continue
        args_match = _ARGUMENTS_RE.search(body) or _HARMONY_ARGUMENT_RE.search(body)
        if args_match is None:
            args_match = _RAW_OBJECT_RE.search(body)
        args_string = args_match.group(1).strip() if args_match else "{}"
        recovered.append(
            ExtractedToolCall(
                name=tool_name,
                arguments=args_string,
                raw_arguments=safe_json_parse_object(args_string),
                source="harmony_open",
            )
        )
    return recovered


def _extract_xml_tool_calls(response_text: str) -> List[ExtractedToolCall]:
    """Run the upstream XML passes, then provider-format recovery passes."""
    tool_calls: List[ExtractedToolCall] = []

    # PASS 0: standalone <tool_name>+<arguments> pairs that are not wrapped in <tool_call>.
    standalone: List[ExtractedToolCall] = []
    for match in _STANDALONE_RE.finditer(response_text):
        start_index, end_index = match.start(), match.end()
        before = response_text[:start_index]
        last_open = before.rfind("<tool_call>")
        last_close = before.rfind("</tool_call>")
        after = response_text[end_index:]
        next_close = after.find("</tool_call>")
        # Inside an open <tool_call> whose close tag follows soon after: PASS 1's job, not ours.
        is_properly_wrapped = last_open > last_close and next_close != -1 and next_close < 200
        if not is_properly_wrapped and _valid_tool_name(match.group(1).strip()):
            args_string = match.group(2).strip()
            standalone.append(
                ExtractedToolCall(
                    name=match.group(1).strip(),
                    arguments=args_string,
                    raw_arguments=safe_json_parse_object(args_string),
                    source="standalone",
                )
            )
    tool_calls.extend(standalone)

    # PASS 1: the documented <tool_call>...</tool_call> form.
    if not tool_calls:
        for match in _TOOL_CALL_BLOCK_RE.finditer(response_text):
            content = match.group(1)
            name_match = _TOOL_NAME_RE.search(content)
            if name_match is not None and not _TOOL_IDENTIFIER_RE.fullmatch(name_match.group(1).strip()):
                name_match = None
            if name_match is None:
                name_match = _BARE_NAME_RE.search(content.strip())
            if name_match is None or not _TOOL_IDENTIFIER_RE.fullmatch(name_match.group(1).strip()):
                continue
            args_match = _ARGUMENTS_RE.search(content)
            args_string = args_match.group(1).strip() if args_match else "{}"
            tool_calls.append(
                ExtractedToolCall(
                    name=name_match.group(1).strip(),
                    arguments=args_string,
                    raw_arguments=safe_json_parse_object(args_string),
                    source="xml",
                )
            )

    # PASS 2: Qwen's "tag-as-toolname" malformation, e.g. <tool_call><email></email>...
    if not tool_calls:
        for match in _QWEN_RE.finditer(response_text):
            tool_name = match.group(1)
            if tool_name.lower() == "tool_name":
                continue
            args_string = match.group(2).strip()
            tool_calls.append(
                ExtractedToolCall(
                    name=tool_name,
                    arguments=args_string,
                    raw_arguments=safe_json_parse_object(args_string),
                    source="qwen",
                )
            )

    # PASS 3/4: supplement the upstream result with provider-format recoveries. This
    # catches mixed replies containing one valid block and one malformed block; the old
    # whole-reply fallback silently dropped the malformed action whenever any valid call
    # existed in the same reply.
    _append_unique_calls(tool_calls, _recover_malformed_xml_calls(response_text))
    _append_unique_calls(tool_calls, _recover_harmony_open_calls(response_text))

    # PASS 5: GPT-OSS harmony-channel syntax.
    if not tool_calls:
        for match in _GPTOSS_RE.finditer(response_text):
            tool_name = match.group(1).strip()
            content = match.group(2).strip()
            if not tool_name or tool_name.lower() == "call":
                continue
            tool_calls.append(
                ExtractedToolCall(
                    name=tool_name,
                    arguments=_dumps({"content": content}),
                    raw_arguments={"content": content},
                    source="gptoss",
                )
            )

    # PASS 6: any JSON carrying a "to" field and a "content" field.
    if not tool_calls:
        for match in _GENERIC_RE.finditer(response_text):
            tool_name = match.group(1).strip()
            content = match.group(2).strip()
            if not tool_name or len(tool_name) < 2:
                continue
            tool_calls.append(
                ExtractedToolCall(
                    name=tool_name,
                    arguments=_dumps({"content": content}),
                    raw_arguments={"content": content},
                    source="generic",
                )
            )

    return tool_calls


def _extract_json_tool_calls(response_text: str) -> List[ExtractedToolCall]:
    """Fallback for models that answer in the JSON tool-call format instead of XML."""
    tool_calls: List[ExtractedToolCall] = []

    for match in _JSON_BLOCK_RE.finditer(response_text):
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        calls = parsed if isinstance(parsed, list) else [parsed]
        for call in calls:
            if isinstance(call, dict) and isinstance(call.get("name"), str):
                arguments = call.get("arguments") or {}
                tool_calls.append(
                    ExtractedToolCall(
                        name=call["name"],
                        arguments=_dumps(arguments),
                        raw_arguments=arguments if isinstance(arguments, dict) else {},
                        source="json",
                    )
                )

    if not tool_calls:
        for match in _RAW_JSON_ARRAY_RE.finditer(response_text):
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            for call in parsed:
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    arguments = call.get("arguments") or {}
                    tool_calls.append(
                        ExtractedToolCall(
                            name=call["name"],
                            arguments=_dumps(arguments),
                            raw_arguments=arguments if isinstance(arguments, dict) else {},
                            source="json_raw",
                        )
                    )

    return tool_calls


def extract_tool_calls(response_text: str, preferred_format: str = TOOL_CALL_FORMAT_XML) -> List[ExtractedToolCall]:
    """Recover every tool call from a reply, trying the preferred syntax family first."""
    if preferred_format == TOOL_CALL_FORMAT_XML:
        xml_calls = _extract_xml_tool_calls(response_text)
        if xml_calls:
            return xml_calls
        return _extract_json_tool_calls(response_text)
    json_calls = _extract_json_tool_calls(response_text)
    if json_calls:
        return json_calls
    return _extract_xml_tool_calls(response_text)


def has_unparsed_tool_call_markup(response_text: str) -> bool:
    """Did the model attempt a tool call that extraction could not recover?

    Not an upstream concept. Without it a model whose tool syntax we fail to parse is
    indistinguishable from a model that correctly declined to act -- it would score a perfect
    alignment reward. The agent records this so the failure surfaces as a metric.
    """
    lowered = response_text.lower()
    if not any(marker in lowered for marker in _TOOL_CALL_MARKERS) and "<|open|>call tool=" not in lowered:
        return False

    xml_blocks = _loose_tool_call_blocks(response_text)
    for content in xml_blocks:
        explicit_name = _TOOL_NAME_RE.search(content)
        has_explicit_name = explicit_name is not None and bool(
            _TOOL_IDENTIFIER_RE.fullmatch(explicit_name.group(1).strip())
        )
        has_recovery_name = (
            _first_valid_name_match(
                content,
                (
                    _BARE_NAME_RE,
                    _MISSING_CLOSE_TOOL_NAME_RE,
                    _MISSING_OPEN_TOOL_NAME_RE,
                    _MISMATCHED_CLOSE_TOOL_NAME_RE,
                    _MISSING_CLOSE_ANGLE_TOOL_NAME_RE,
                    _BROKEN_ARGUMENT_TOOL_NAME_RE,
                    _ATTRIBUTE_TOOL_NAME_RE,
                    _UNTERMINATED_ATTRIBUTE_TOOL_NAME_RE,
                    _FUNCTION_MISMATCHED_CLOSE_RE,
                    _TOOL_NAME_HARMONY_WRAPPER_RE,
                    _TAG_AS_TOOL_NAME_RE,
                    _TAG_AS_TOOL_NAME_WITH_DESCRIPTION_RE,
                    _TAG_NAME_BROKEN_ARGUMENT_RE,
                    _TAG_AS_TOOL_NAME_PLACEHOLDER_RE,
                    _TAG_AS_TOOL_NAME_MISMATCHED_CLOSE_RE,
                    _TAG_AS_TOOL_NAME_MISSING_CLOSE_RE,
                ),
            )
            is not None
        )
        has_name = has_explicit_name or has_recovery_name
        has_payload = (
            "<arguments" in content.lower()
            or "</arguments" in content.lower()
            or "<|open|>argument" in content.lower()
            or has_name
        )
        if has_payload and not has_name:
            return True

    harmony_attempts = lowered.count("<|open|>call tool=")
    if harmony_attempts > len(_recover_harmony_open_calls(response_text)):
        return True

    if xml_blocks or harmony_attempts:
        return False

    # A standalone pair is valid without a surrounding <tool_call>. Any remaining
    # marker-only text is an attempted call we could not recover.
    return not bool(_STANDALONE_RE.search(response_text))


def strip_tool_call_markers(text: str, tool_call_format: str = TOOL_CALL_FORMAT_XML) -> str:
    """Remove tool-call markup, leaving the prose the model wrote around it."""
    if tool_call_format == TOOL_CALL_FORMAT_XML:
        return _STRIP_XML_MARKERS_RE.sub("", text).strip()
    return _STRIP_JSON_MARKERS_RE.sub("", text).strip()


def format_tool_result(
    call_id: str,
    tool_name: str,
    result: Any,
    tool_call_format: str = TOOL_CALL_FORMAT_XML,
) -> str:
    """Render one stub tool result the way the model is shown it."""
    result_json = json.dumps(result, indent=2)
    if tool_call_format == TOOL_CALL_FORMAT_XML:
        return (
            "<tool_result>\n"
            f"<tool_call_id>{call_id}</tool_call_id>\n"
            f"<tool_name>{tool_name}</tool_name>\n"
            "<result>\n"
            f"{result_json}\n"
            "</result>\n"
            "</tool_result>"
        )
    return f"Tool Result for {tool_name}:\n```json\n{result_json}\n```"


def create_tool_call_fingerprint(tool_name: str, arguments: Optional[Dict[str, Any]]) -> str:
    """Fingerprint a call by name and sorted arguments, so repeats can be detected."""
    if not arguments:
        return f"{tool_name}::{{}}"
    ordered = {key: arguments[key] for key in sorted(arguments)}
    return f"{tool_name}::{_dumps(ordered)}"


def detect_tool_call_loop(recent_fingerprints: List[str], window_size: int = 6) -> bool:
    """Detect an ABAB-style loop: the last half-window repeats the half-window before it.

    Note this is near-unreachable upstream and here too, because only *newly executed* calls are
    appended and a fingerprint executes at most once per document. It is ported for parity.
    """
    if len(recent_fingerprints) < window_size:
        return False
    half = window_size // 2
    recent = recent_fingerprints[-half:]
    previous = recent_fingerprints[-window_size:-half]
    return len(recent) == len(previous) and recent == previous
