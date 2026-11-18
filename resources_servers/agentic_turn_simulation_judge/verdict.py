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
"""Reading a judge verdict the way the agentic_turn_simulation (ATS) judge writes it.

`VERDICT_FIELDS`, `correct_ladder`, the think stripping and the lenient parse are copied from
`resources_servers/agentic_turn_simulation/judge.py` (not imported: that module pulls in the ATS harnesses and
jsonschema), minus the ATS salvage of cut-off output: a verdict that broke off is never read here.
`prompts/evaluation_schema.json` is a copy of the ATS output schema; tests fail when either copy drifts.

Two parse modes:

- `strict`: after stripping whitespace and one optional ```json fence, the text must be exactly one JSON object.
- `lenient`: ATS `parse_verdict` without the salvage (the first object anywhere in the text that holds
  `addresses_concern`, also inside a one-key envelope; non-string verdict fields dropped).

Every failure is named by one of `PARSE_ERRORS`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal


SCHEMA_PATH = Path(__file__).parent / "prompts" / "evaluation_schema.json"

# Copied from ATS judge.py.
VERDICT_FIELDS = ["addresses_concern", "correct", "blocked_by", "request_questionable", "discard", "discard_reason"]
REQUIRED_KEY = "addresses_concern"
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S)

ParseMode = Literal["strict", "lenient"]
# Why an output gave no usable verdict, in the order they are checked.
PARSE_ERRORS = ("empty", "no_json", "multiple_objects", "bad_json", "missing_keys", "bad_enum", "too_long")

# One optional Markdown fence around the whole text (strict mode).
_FENCE_RE = re.compile(r"\A```(?:json)?[ \t]*\n?(.*?)\n?[ \t]*```\Z", re.S | re.I)
# Failure classification tries a JSON decode at no more than this many "{" positions.
_MAX_DECODE_ATTEMPTS = 64


@lru_cache(maxsize=1)
def evaluation_schema() -> dict[str, Any]:
    """The judge's output schema (this env's copy of the ATS file)."""
    return json.loads(SCHEMA_PATH.read_text())


def strip_think(text: str) -> str:
    """Drop `<think>...</think>` blocks and anything before a dangling `</think>` (as ATS does)."""
    if "</think" in text:
        text = _THINK_RE.sub("", text).split("</think>")[-1]
    return text


def correct_ladder(verdict: dict[str, Any] | None) -> str | None:
    """The ATS ladder rule: correctness presupposes engagement, so a turn that did not address the user is not
    correct. None when the verdict does not decide the ladder: `addresses_concern` must be "no", or "yes" with a
    valid `correct`. ATS also requires a schema-valid (`usable`) verdict first; here that check is `schema_error`,
    applied when `validate_schema` is on."""
    if not verdict:
        return None
    if verdict.get("addresses_concern") == "no":
        return "no"
    if verdict.get("addresses_concern") != "yes":
        return None
    value = verdict.get("correct")
    return value if value in ("yes", "no") else None


def _unwrap(value: Any) -> dict[str, Any] | None:
    """Copied from ATS: some providers wrap structured output in a one-key envelope, e.g. {"command": {...}}."""
    if not isinstance(value, dict):
        return None
    if REQUIRED_KEY in value:
        return value
    for inner in value.values():
        if isinstance(inner, dict) and REQUIRED_KEY in inner:
            return inner
    return None


def _outermost_braces(text: str) -> list[str]:
    """What ATS's `re.findall(r"\\{.*\\}", text, re.S)` returns, in linear time (the regex is quadratic on a long
    run of "{" without a "}"): the greedy match runs from the first "{" to the last "}", and nothing follows it."""
    start, end = text.find("{"), text.rfind("}")
    return [text[start : end + 1]] if start != -1 and end > start else []


def _decoded_objects(text: str) -> list[dict[str, Any]]:
    """The JSON objects that can be decoded from `text` left to right, each starting at a "{" outside the
    previous one (bounded work on adversarial text)."""
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    pos, attempts = text.find("{"), 0
    while pos != -1 and attempts < _MAX_DECODE_ATTEMPTS:
        attempts += 1
        try:
            value, end = decoder.raw_decode(text, pos)
        except (ValueError, RecursionError):
            pos = text.find("{", pos + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        pos = text.find("{", end)
    return found


def _failure_kind(text: str) -> str:
    """Why `text` (think-stripped, non-empty) is not one usable JSON object."""
    if "{" not in text:
        return "no_json"
    objects = _decoded_objects(text)
    if len(objects) >= 2:
        return "multiple_objects"
    return "bad_json"


def parse_strict(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """(the verdict object, None), or (None, a PARSE_ERRORS value). Exactly one JSON object, optionally fenced."""
    text = strip_think(text or "").strip()
    if not text:
        return None, "empty"
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
        if not text:
            return None, "empty"
    if not text.startswith("{"):
        return None, _failure_kind(text)
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except (ValueError, RecursionError):
        return None, _failure_kind(text)
    if not isinstance(value, dict):
        return None, "bad_json"
    if text[end:].strip():
        return None, _failure_kind(text)
    return value, None


def parse_lenient(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """ATS `parse_verdict` without its cut-off salvage: (verdict, None) or (None, a PARSE_ERRORS value). On every
    text ATS reads without salvaging, the verdict equals ATS's."""
    if not text:
        return None, "empty"
    text = strip_think(text)
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not text:
        return None, "empty"
    saw_object = False
    for candidate in (text, *_outermost_braces(text)):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, RecursionError):
            continue
        saw_object = saw_object or isinstance(value, dict)
        found = _unwrap(value)
        if found is not None:
            # As ATS: verdict fields are strings (or null); any other value is dropped.
            return {k: v for k, v in found.items() if k not in VERDICT_FIELDS or isinstance(v, str) or v is None}, None
    if saw_object:
        return None, "missing_keys"
    return None, _failure_kind(text)


def schema_error(verdict: dict[str, Any], schema: dict[str, Any] | None = None) -> str | None:
    """The schema error of a parsed verdict, or None: "missing_keys" when a key the schema requires is absent,
    "bad_enum" when a value is outside the schema (an enum value it does not list, or a non-string where it says
    string). Extra keys are allowed."""
    schema = schema or evaluation_schema()
    if any(key not in verdict for key in schema.get("required", [])):
        return "missing_keys"
    for key, spec in schema.get("properties", {}).items():
        if key not in verdict:
            continue
        value = verdict[key]
        if spec.get("type") == "string" and not isinstance(value, str):
            return "bad_enum"
        if "enum" in spec and value not in spec["enum"]:
            return "bad_enum"
    return None


@dataclass(frozen=True)
class ReadVerdict:
    """An output read as a verdict: `ladder` is set iff `error` is None."""

    verdict: dict[str, Any] | None
    ladder: str | None
    error: str | None
    analysis_words: int


def read_verdict(
    text: str | None,
    *,
    parse_mode: ParseMode = "strict",
    validate_schema: bool = True,
    max_analysis_words: int | None = None,
) -> ReadVerdict:
    """Parse, validate and ladder one output. Never raises: an unexpected exception is a "bad_json" output."""
    try:
        verdict, error = (parse_strict if parse_mode == "strict" else parse_lenient)(text)
    except Exception:  # noqa: BLE001 - the reward boundary: no output may crash /verify
        verdict, error = None, "bad_json"
    if verdict is None:
        return ReadVerdict(None, None, error, 0)
    analysis = verdict.get("analysis")
    words = len(analysis.split()) if isinstance(analysis, str) else 0
    if validate_schema:
        error = schema_error(verdict)
    ladder = correct_ladder(verdict) if error is None else None
    if error is None and ladder is None:
        # Only reachable without schema validation: the verdict does not decide the ladder.
        error = "missing_keys" if any(key not in verdict for key in ("addresses_concern", "correct")) else "bad_enum"
    if error is None and max_analysis_words is not None and words > max_analysis_words:
        error, ladder = "too_long", None
    return ReadVerdict(verdict, ladder, error, words)
