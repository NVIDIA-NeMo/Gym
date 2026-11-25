# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
# Adapted source: nemo_rl/experience/loop_detector_v2_1.py (Apache-2.0).
# Scoring behavior is preserved; unused historical extractors are omitted.
# Full reference payload SHA256: a5e4eeba1d6cfd2d624ee6406e0837ddb78097f0f3a29ab046fe40821fe8688b
# Frozen detector payload follows.
import collections
import hashlib
import json
import re
from typing import Any


VERSION = "loop-screen-v2.1-fresh-policy-captures-period4096"
PARAMETERS = {
    "paragraph_min_chars": 80,
    "severe_paragraph_min_copies": 5,
    "severe_paragraph_redundant_chars": 8192,
    "severe_paragraph_fraction": 0.2,
    "warning_paragraph_min_copies": 3,
    "warning_paragraph_redundant_chars": 2048,
    "period_max_chars": 4096,
    "period_anchor_chars": 64,
    "severe_period_min_copies": 16,
    "severe_period_min_chars": 2048,
    "warning_period_min_copies": 8,
    "warning_period_min_chars": 512,
    "tool_cycle_max_period": 4,
    "severe_tool_min_copies": 4,
    "severe_tool_min_calls": 8,
    "warning_tool_min_copies": 3,
    "warning_tool_min_calls": 3,
}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def safe_excerpt(value: object, limit: int = 500) -> str:
    value = str(value)
    value = "\n".join(
        "[credential/environment line omitted]"
        if re.search(r"(API_KEY|TOKEN|SECRET|PASSWORD|AUTHORIZATION|environ|printenv)", line, re.I)
        else line
        for line in value.splitlines()
    )
    value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~-]+", r"\1[redacted]", value)
    return re.sub("[A-Za-z0-9_-]{48,}", "[long token omitted]", value)[:limit]


def periodic_runs(text: str) -> list[dict[str, Any]]:
    """Find exact tandem spans using bounded 64-character anchor searches."""
    anchor_width = PARAMETERS["period_anchor_chars"]
    max_period = PARAMETERS["period_max_chars"]
    text_length = len(text)
    position = 0
    runs = []
    while position + anchor_width < text_length:
        next_match = text.find(
            text[position : position + anchor_width],
            position + 1,
            min(text_length, position + max_period + anchor_width),
        )
        if next_match < 0:
            position += anchor_width
            continue
        period = next_match - position
        start = position
        end = next_match + anchor_width
        while start > 0 and text[start - 1] == text[start + period - 1]:
            start -= 1
        while end < text_length and text[end] == text[end - period]:
            end += 1
        span_chars = end - start
        complete_copies = span_chars // period
        if (
            span_chars >= PARAMETERS["warning_period_min_chars"]
            and complete_copies >= PARAMETERS["warning_period_min_copies"]
        ):
            repeated_unit = text[start : start + period]
            runs.append(
                {
                    "start": start,
                    "end": end,
                    "span_chars": span_chars,
                    "period_chars": period,
                    "complete_copies": complete_copies,
                    "unit_sha256": sha(repeated_unit),
                    "unit_excerpt": safe_excerpt(repeated_unit, 256),
                    "span_excerpt": safe_excerpt(text[start : start + 600], 600),
                }
            )
            position = end
        else:
            position += anchor_width
    return sorted(runs, key=lambda run: (-run["span_chars"], run["start"]))


def tool_cycles(tools: list[dict[str, str]]) -> dict[str, Any] | None:
    """Find the longest exact contiguous cycle of complete tool signatures."""
    signatures = [tool["signature_sha256"] for tool in tools]
    call_count = len(signatures)
    best_cycle = None
    for start in range(call_count):
        for period in range(1, PARAMETERS["tool_cycle_max_period"] + 1):
            end = start + period
            while end < call_count and signatures[end] == signatures[start + (end - start) % period]:
                end += 1
            complete_copies = (end - start) // period
            covered_calls = complete_copies * period
            if complete_copies >= 3 and (
                best_cycle is None
                or (covered_calls, complete_copies) > (best_cycle["covered_calls"], best_cycle["complete_copies"])
            ):
                best_cycle = {
                    "start_call": start,
                    "end_call_exclusive": start + covered_calls,
                    "period_calls": period,
                    "complete_copies": complete_copies,
                    "covered_calls": covered_calls,
                    "cycle_signatures": signatures[start : start + period],
                    "cycle_evidence": tools[start : start + period],
                }
    return best_cycle


def detect_transcript(
    segments: list[dict[str, str]], tools: list[dict[str, str]], metadata: dict[str, bool]
) -> dict[str, Any]:
    """Apply the fixed text and tool-cycle rules with explicit evidence coverage."""
    text = "\n\n".join(segment["text"] for segment in segments)
    normalized_text = re.sub(r"\s+", " ", text).strip()
    paragraphs = [re.sub(r"\s+", " ", paragraph).strip() for paragraph in re.split(r"\n\s*\n", text)]
    paragraph_counts = collections.Counter(
        paragraph for paragraph in paragraphs if len(paragraph) >= PARAMETERS["paragraph_min_chars"]
    )
    redundant_chars = sum((copies - 1) * len(paragraph) for paragraph, copies in paragraph_counts.items())
    max_paragraph_copies = max(paragraph_counts.values(), default=0)
    redundant_fraction = redundant_chars / max(1, len(normalized_text))
    severe_redundant_chars = sum(
        (copies - 1) * len(paragraph) for paragraph, copies in paragraph_counts.items() if copies >= 5
    )
    warning_redundant_chars = sum(
        (copies - 1) * len(paragraph) for paragraph, copies in paragraph_counts.items() if copies >= 3
    )
    severe_redundant_fraction = severe_redundant_chars / max(1, len(normalized_text))
    periodic_matches = periodic_runs(normalized_text)
    tool_cycle = tool_cycles(tools)
    paragraph_severe = (
        max_paragraph_copies >= 5 and severe_redundant_chars >= 8192 and severe_redundant_fraction >= 0.2
    )
    paragraph_warning = max_paragraph_copies >= 3 and warning_redundant_chars >= 2048
    periodic_severe = any(run["complete_copies"] >= 16 and run["span_chars"] >= 2048 for run in periodic_matches)
    periodic_warning = bool(periodic_matches)
    cycle_severe = bool(tool_cycle and tool_cycle["complete_copies"] >= 4 and tool_cycle["covered_calls"] >= 8)
    cycle_warning = bool(tool_cycle and tool_cycle["complete_copies"] >= 3 and tool_cycle["covered_calls"] >= 3)
    text_fallback = False if metadata["text_known"] else None
    text_severe = True if paragraph_severe or periodic_severe else text_fallback
    text_warning = True if paragraph_warning or periodic_warning else text_fallback
    tool_severe = cycle_severe if metadata["tools_known"] else None
    tool_warning = cycle_warning if metadata["tools_known"] else None

    def union(text_flag: bool | None, tool_flag: bool | None) -> bool | None:
        if text_flag is True or tool_flag is True:
            return True
        if text_flag is False and tool_flag is False:
            return False
        return None

    paragraph_evidence = sorted(
        ((paragraph, copies) for paragraph, copies in paragraph_counts.items() if copies >= 3),
        key=lambda item: -(len(item[0]) * (item[1] - 1)),
    )[:3]
    input_signature_counts = collections.Counter(tool["input_signature_sha256"] for tool in tools)
    return {
        "detector_version": VERSION,
        "extraction": metadata,
        "severe": {
            "paragraph": True if paragraph_severe else text_fallback,
            "short_period": True if periodic_severe else text_fallback,
            "text": text_severe,
            "tool_cycle": tool_severe,
            "union": union(text_severe, tool_severe),
        },
        "warning": {
            "paragraph": True if paragraph_warning else text_fallback,
            "short_period": True if periodic_warning else text_fallback,
            "text": text_warning,
            "tool_cycle": tool_warning,
            "union": union(text_warning, tool_warning),
        },
        "metrics": {
            "normalized_text_chars": len(normalized_text),
            "long_paragraph_count": sum(paragraph_counts.values()),
            "max_identical_paragraph_copies": max_paragraph_copies,
            "redundant_paragraph_chars": redundant_chars,
            "redundant_paragraph_fraction": redundant_fraction,
            "severe_qualifying_redundant_paragraph_chars": severe_redundant_chars,
            "severe_qualifying_redundant_paragraph_fraction": severe_redundant_fraction,
            "warning_qualifying_redundant_paragraph_chars": warning_redundant_chars,
            "longest_short_period_chars": max((run["span_chars"] for run in periodic_matches), default=0),
            "max_exact_input_only_copies": max(input_signature_counts.values(), default=0),
            "exact_input_only_extra_occurrences": sum(copies - 1 for copies in input_signature_counts.values()),
        },
        "evidence": {
            "paragraphs": [
                {
                    "copies": copies,
                    "paragraph_chars": len(paragraph),
                    "paragraph_sha256": sha(paragraph),
                    "excerpt": safe_excerpt(paragraph, 600),
                }
                for paragraph, copies in paragraph_evidence
            ],
            "short_periods": periodic_matches[:3],
            "tool_cycle": tool_cycle,
        },
    }
