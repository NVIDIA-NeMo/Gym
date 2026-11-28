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
"""The UI code judge without a reference answer.

Copied from ``resources_servers/ui_generation/app.py`` (NVIDIA-NeMo/Gym PR #5774) and adapted: the judge sees the
user prompt, a one-line description of the target stack and the candidate's code, and no reference answer, so the
verdict has no ``reference_differences``. The code is fenced with more backticks than any run inside it, and a reply
may wrap its verdict in a fenced block with text around it (the first fenced JSON object is taken, the language tag in
any case). The validation rules are otherwise unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.verification import JudgeError


SCHEMA_VERSION = "ui-code-judge-v1-noref"
VERDICT_KEYS = frozenset(
    {"schema_version", "prompt_adherence", "quality", "critical_issues", "uncertainty", "execution_verified"}
)
MAX_SCORE = 4
# A fenced block with an optional json tag; the closing fence has as many backticks as the opening one.
_FENCE_RE = re.compile(r"(`{3,})[ \t]*(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*\1(?!`)", re.DOTALL | re.IGNORECASE)
_LEADING_THINK_RE = re.compile(r"^\s*<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)


def validate_verdict(obj: object) -> Optional[str]:
    """Return why ``obj`` is not a rubric verdict, or None when it matches the schema exactly."""
    if not isinstance(obj, dict) or set(obj) != VERDICT_KEYS:
        found = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        return f"keys {found} != {sorted(VERDICT_KEYS)}"
    if obj["schema_version"] != SCHEMA_VERSION:
        return f"schema_version {obj['schema_version']!r}"
    for key in ("prompt_adherence", "quality"):
        part = obj[key]
        if not isinstance(part, dict) or set(part) != {"score", "rationale"}:
            return f"{key} must have exactly score and rationale"
        if type(part["score"]) is not int or not 0 <= part["score"] <= MAX_SCORE:
            return f"{key}.score {part['score']!r} is not an integer in 0..{MAX_SCORE}"
        if not isinstance(part["rationale"], str):
            return f"{key}.rationale is not a string"
    for key in ("critical_issues", "uncertainty"):
        if not isinstance(obj[key], list) or not all(isinstance(x, str) for x in obj[key]):
            return f"{key} must be a list of strings"
    if obj["execution_verified"] is not False:
        return "execution_verified must be false"
    return None


def parse_judge_completion(completion: NeMoGymChatCompletion) -> tuple[dict[str, Any], Optional[str]]:
    """Return (verdict, judge reasoning) from a judge reply; raise ``JudgeError`` when it holds no valid verdict."""
    if not completion.choices:
        raise JudgeError("malformed_response: the judge reply has no choices")
    choice = completion.choices[0]
    if choice.finish_reason != "stop":
        raise JudgeError(f"finish_{choice.finish_reason}: the judge did not finish its verdict")
    content = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
    think = _LEADING_THINK_RE.match(content)
    if think:
        reasoning = reasoning or think.group(1).strip()
        content = content[think.end() :]
    verdict = _verdict_json(content.strip())
    if error := validate_verdict(verdict):
        raise JudgeError(f"invalid_schema: {error}")
    return verdict, reasoning


def _verdict_json(text: str) -> object:
    """The reply as JSON, or else its first fenced block that holds a JSON object; raise ``JudgeError`` if none."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        first_error = error
    for fenced in _FENCE_RE.finditer(text):
        try:
            verdict = json.loads(fenced.group(2))
        except json.JSONDecodeError:
            continue
        if isinstance(verdict, dict):
            return verdict
    raise JudgeError(f"invalid_json: {first_error}") from first_error


def judge_case(prompt: str, stack_description: str, entry_path: str, code: str, language: str) -> str:
    """The judge's user message: the case as JSON, with the candidate's code in one fenced block."""
    # A fence longer than any run of backticks in the code, so the code cannot close it early.
    fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", code)), default=0) + 1)
    candidate = f"### {entry_path}\n{fence}{language}\n{code}\n{fence}"
    return json.dumps(
        {"prompt": prompt, "stack": stack_description, "candidate_answer": candidate}, ensure_ascii=False
    )
