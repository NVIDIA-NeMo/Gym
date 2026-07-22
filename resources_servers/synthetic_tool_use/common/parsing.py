# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parsing helpers for model-generated seed artifacts."""

from __future__ import annotations

import json
import re
from typing import Any


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def extract_tag(text: str, tag: str) -> str:
    matches = re.findall(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL | re.IGNORECASE)
    if not matches:
        raise ValueError(f"response is missing <{tag}>...</{tag}>")
    return matches[-1].strip()


def parse_json_value(text: str) -> Any:
    text = strip_json_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
        if not starts:
            raise original_error
        try:
            value, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
            return value
        except json.JSONDecodeError:
            raise original_error


def parse_json_or_jsonl(text: str) -> list[dict[str, Any]]:
    text = strip_json_fence(text)
    try:
        value = parse_json_value(text)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and "tools" in value:
            return value["tools"]
    except json.JSONDecodeError:
        pass
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("response contains no JSON objects")
    return rows


def render_template(template: str, **values: object) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template
