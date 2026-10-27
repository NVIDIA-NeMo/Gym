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
"""Score skill instructions returned in a fresh Harbor ATIF trajectory.

Version 0.2 uses recorded tool results, not independent filesystem observation.
Missing or inconsistent evidence raises instead of becoming a model failure.
"""

import json
import re
from pathlib import PurePosixPath

from harbor.models.trajectories import Trajectory


class EvidenceError(ValueError):
    """The trajectory cannot support a valid score."""


def normalize(text: str, *, javascript: bool = False) -> str:
    """Match the original whitespace and rendered-line-number normalization."""
    if javascript:
        # ECMAScript whitespace and digits differ from Python's Unicode \s/\d.
        space = r"[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
        start = r"(?:^|(?<=[\n\r\u2028\u2029]))"
        text = re.sub(start + space + r"*[0-9]+[|→]" + space + "?", "", text)
        return re.sub(space + "+", " ", text).strip(" ")
    return re.sub(r"\s+", " ", re.sub(r"^\s*\d+[|→]\s?", "", text, flags=re.MULTILINE)).strip()


def skill_body(text: str, *, javascript: bool = False) -> str:
    return normalize(re.sub(r"^---\r?\n[\s\S]*?\r?\n---\r?\n", "", text), javascript=javascript)


def character_units(text: str, *, javascript: bool) -> str:
    """Pi's reference counts UTF-16 units; the other scorers count code points."""
    if not javascript:
        return text
    data = text.encode("utf-16-le", errors="surrogatepass")
    return "".join(chr(int.from_bytes(data[i : i + 2], "little")) for i in range(0, len(data), 2))


def has_excerpt(output: str, source: str, *, competing: list[str], javascript: bool = False) -> bool:
    """Require 120 body characters (or the whole shorter body), excluding shared text."""
    text = character_units(normalize(output, javascript=javascript), javascript=javascript)
    body = character_units(skill_body(source, javascript=javascript), javascript=javascript)
    others = [character_units(skill_body(s, javascript=javascript), javascript=javascript) for s in competing]
    size = min(120, len(body))
    return bool(size) and any(
        body[i : i + size] in text and not any(body[i : i + size] in other for other in others)
        for i in range(len(body) - size + 1)
    )


def text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.text for part in content if getattr(part, "type", None) == "text")
    return ""


def returned_text(text: str, extra: dict) -> str | None:
    """Exclude explicit tool failures; retain unwrapped text emitted by Harbor."""
    if extra.get("is_error") or extra.get("status") in {"failed", "error"}:
        return None
    if re.search(r"Process exited with code [1-9][0-9]*", text):
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return text
    if not isinstance(payload, dict):
        return text
    if payload.get("success") is False or payload.get("error") or payload.get("exit_code", 0) != 0:
        return None
    for key in ("content", "output"):
        if isinstance(payload.get(key), str):
            return payload[key]
    return text


def score_trajectory(trajectory: Trajectory, *, sources: dict[str, str], target: str, harness: str) -> int:
    """Score returned target instructions in the root session, never assistant claims."""
    if target not in sources or not skill_body(sources[target]):
        raise EvidenceError("Missing target instructions")
    if harness not in {"pi", "codex", "codex_guidance", "hermes", "hermes_guidance"}:
        raise EvidenceError("Unknown harness")
    if not trajectory.steps or trajectory.continued_trajectory_ref:
        raise EvidenceError("Expected a complete fresh trajectory")
    calls = {}
    seen = set()
    matched = False
    delegated = False
    for step in trajectory.steps:
        if step.is_copied_context:
            raise EvidenceError("Copied context cannot establish a new skill read")
        for call in step.tool_calls or []:
            if call.tool_call_id in calls:
                raise EvidenceError("Duplicate tool-call ID")
            calls[call.tool_call_id] = call
        for result in step.observation.results if step.observation else []:
            delegated |= bool(result.subagent_trajectory_ref)
            if result.source_call_id not in calls or result.source_call_id in seen:
                raise EvidenceError("Unpaired or duplicate tool result")
            seen.add(result.source_call_id)
            call = calls[result.source_call_id]
            tool = call.function_name.rsplit(".", 1)[-1]
            details = (step.extra or {}).get("tool_call_details", {}).get(result.source_call_id, {})
            extra = {**(step.extra or {}), **details, **(result.extra or {})}
            text = returned_text(text_content(result.content), extra)
            if text is None:
                continue
            direct = False
            if tool in {"read", "read_file"}:
                path = call.arguments.get("path", call.arguments.get("file_path", ""))
                parts = PurePosixPath(path).parts if isinstance(path, str) else ()
                direct = len(parts) >= 3 and parts[-3:] == ("skills", target, "SKILL.md")
                if not direct:
                    continue
            elif tool == "skill_view":
                direct = call.arguments.get("name") == target and not call.arguments.get("file_path")
                if not direct:
                    continue
            elif tool not in {"bash", "terminal", "exec_command", "shell_command", "shell", "write_stdin"}:
                continue
            competitors = [] if direct else [value for name, value in sources.items() if name != target]
            matched |= has_excerpt(text, sources[target], competing=competitors, javascript=harness == "pi")
    if calls.keys() != seen:
        raise EvidenceError("Incomplete tool results")
    if not matched and (delegated or trajectory.subagent_trajectories):
        raise EvidenceError("Unscored: root did not open the skill and delegated traces require separate verification")
    return int(matched)
