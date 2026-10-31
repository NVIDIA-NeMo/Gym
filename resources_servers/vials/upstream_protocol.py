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
#
# The prompts, the answer extraction and the judge message below are copied from the VIALS
# reference harness (https://github.com/Handshake-AI-Research/VIALS) at commit
# 002a0d25031f02a760da5d14f2c29bbd26eb06e4: SYSTEM_PROMPT and USER_INSTRUCTION_TAIL from
# vials/pipeline.py; JUDGE_SYSTEM, MULTIMODAL_JUDGE_SYSTEM, lead_bound_block,
# build_judge_user_message, JudgeVerdict and parse_judge_verdict (upstream
# `FreeFormJudgeResponse` and `_salvage_json`) from vials/judge.py; extract_answer_block and its two regexes from
# vials/data.py. They remain under the original MIT terms (reproduced below). Prompt text is
# byte-identical; the code differs in formatting, type hints, names made public (upstream
# `_JUDGE_SYSTEM`, `_lead_bound_block`, ...), and an equivalent linear-time form of the answer
# regex and suffix stripping (upstream's backtracks on long whitespace runs). They are vendored because the
# published scores were produced with exactly this text. Modifications Copyright (c) 2026
# NVIDIA CORPORATION & AFFILIATES and contributors, licensed under the Apache License 2.0
# (SPDX header above).
#
# MIT License
#
# Copyright (c) 2026 The VIALS Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The VIALS reference protocol: policy prompts, answer extraction and judge prompts."""

import re
from typing import Optional

from pydantic import BaseModel, ValidationError


SYSTEM_PROMPT = (
    "You are an expert scientist. You will be shown a scientific image and a "
    "question. Study the image carefully, then respond with your reasoning "
    "and a single final answer."
)

USER_INSTRUCTION_TAIL = (
    "Reason through the image and task, then commit to a single final answer. "
    "Keep the reasoning concise so your response ends with the ANSWER line. "
    "The answer must appear on its own line prefixed by 'ANSWER:' and contain "
    "only the final answer: no LaTeX brackets, no extra explanation, no units "
    "unless the question asks for them."
    "\n\nFormat your response EXACTLY as:\n"
    "REASONING: <your reasoning>\n"
    "ANSWER: <final answer>"
)


# Anchored to line-start so prose like "The answer: 5" never hijacks the
# extraction. Allows optional bold (``**ANSWER:**``), leading whitespace,
# and common bullet markers (``- **ANSWER:**``, ``* ANSWER:``, ``1. ANSWER:``)
# that models emit inside structured final-answer sections.
# Same language as upstream's `^[ \t]*(?:bullet)?[ \t]*\**[ \t]*ANSWER...`, with the adjacent
# whitespace runs merged so a long blank line matches in linear time.
_ANCHORED_ANSWER_RE = re.compile(r"(?mi)^[ \t]*(?:(?:[-*+•·]|\d+[.)])[ \t]*)?(?:\*+[ \t]*)?ANSWER\**[ \t]*:\**[ \t]*")
# Fallback: any ``ANSWER:`` substring. Used only when the anchored form finds nothing.
_LOOSE_ANSWER_RE = re.compile(r"ANSWER[ \t]*:[ \t]*", re.IGNORECASE)


def extract_answer_block(response: Optional[str]) -> Optional[str]:
    """Return the free-form answer after the last ``ANSWER:`` marker, or None.

    Prefers the last line-anchored match and falls back to the loose substring form only
    when no anchored marker exists. Common LaTeX, markdown and code-fence wrappers are
    stripped so the judge sees the raw payload.
    """
    if not response:
        return None

    matches = list(_ANCHORED_ANSWER_RE.finditer(response))
    if not matches:
        matches = list(_LOOSE_ANSWER_RE.finditer(response))
        if not matches:
            return None

    # Upstream strips suffixes with `\s*X$` regexes, which are quadratic on long whitespace runs;
    # the string operations below remove exactly the same suffixes.
    tail = response[matches[-1].end() :].strip()
    tail = re.sub(r"^\\?\[\s*", "", tail)
    if tail.endswith("]"):
        tail = tail[:-1].removesuffix("\\").rstrip()
    tail = re.sub(r"^\$\$?\s*", "", tail)
    if tail.endswith("$"):
        tail = tail[:-2] if tail.endswith("$$") else tail[:-1]
        tail = tail.rstrip()
    tail = re.sub(r"^```(?:\w+)?\s*", "", tail)
    if tail.endswith("```"):
        tail = tail[:-3].rstrip()
    tail = re.sub(r"^\*\*(.+)\*\*$", r"\1", tail.strip())
    return tail.strip() or None


def lead_bound_block(lower: Optional[float], upper: Optional[float]) -> str:
    """Optional inclusive numeric range clause for the judge user message."""
    if lower is None and upper is None:
        return ""
    return (
        f"Lead-assigned acceptable numeric range (inclusive): "
        f"lower={'n/a' if lower is None else lower}, "
        f"upper={'n/a' if upper is None else upper}. "
        f"If the candidate is a numeric answer inside this range, "
        f"mark it equivalent to the GTFA.\n\n"
    )


JUDGE_QUESTION = "Are the candidate and ground-truth final answers equivalent?"
MULTIMODAL_JUDGE_QUESTION = "Using the image only to resolve labelling ambiguities, are the candidate and ground-truth final answers equivalent?"


def build_judge_user_message(
    *, question: str, gtfa: str, answer: str, lower: Optional[float], upper: Optional[float], multimodal: bool
) -> str:
    """The judge's user turn; the image re-grade swaps only the closing question."""
    return (
        f"Question:\n{question}\n\n"
        f"Ground-truth final answer:\n{gtfa}\n\n"
        f"{lead_bound_block(lower, upper)}"
        f"Candidate final answer:\n{answer}\n\n" + (MULTIMODAL_JUDGE_QUESTION if multimodal else JUDGE_QUESTION)
    )


JUDGE_SYSTEM = (
    "You are grading a scientific answer. Given the question, the "
    "ground-truth final answer (GTFA), and a candidate answer, decide "
    "whether the candidate expresses the same intended answer as the GTFA "
    "in the context of the question. You do NOT see the image.\n\n"
    "IMPORTANT: You are grading, not solving. Do NOT attempt to work out "
    "the correct answer to the question yourself and then compare the "
    "candidate to your own derivation. Treat the GTFA as the sole source "
    "of truth. Your only decision is whether the candidate expresses the "
    "same answer as the GTFA.\n\n"
    "Apply the standard of a competent domain scientist. ACCEPT variations "
    "that an expert would recognize as immaterial to the answer:\n"
    "  - Formatting: punctuation, whitespace, dash / hyphen style, "
    "thousands separators, and equivalent scientific notation for the "
    "same underlying value\n"
    "  - Capitalization: when the identity is otherwise unambiguous\n"
    "  - Units: omitted when unambiguous from the question or GTFA — "
    "but only when the numeric value itself matches\n"
    "  - Lead bounds: when an inclusive LeadBoundLower/LeadBoundUpper range "
    "is provided, a numeric candidate inside that range is equivalent "
    "even if it differs from the exact GTFA value\n"
    "  - Order and duplicates: irrelevant when the answer is a set or "
    "unordered list\n"
    "  - Terminology: synonymous scientific terms of the same specificity\n\n"
    "Numeric answers are STRICT. Do NOT invent tolerances, relative-error "
    'allowances, or "same order of magnitude" acceptance.\n'
    "  - The candidate must express the same numeric value as GTFA "
    "(after the formatting/unit rules above), unless LeadBounds apply\n"
    "  - If the question asks for a stated reporting precision, any "
    "difference at that precision is NOT equivalent\n"
    "  - Exact counts of discrete objects/events must match exactly; "
    "off-by-one is NEVER equivalent\n"
    "  - Different coefficients at the same power of ten are NOT "
    "equivalent\n\n"
    "REJECT when the candidate materially disagrees with GTFA on any of:\n"
    "  - Missing or extra items in a list\n"
    "  - The identity, count, direction, mechanism, or sign of a "
    "biological entity\n"
    "  - Any numeric mismatch under the strict numeric rules above "
    "(unless inside LeadBounds)\n"
    "  - A broader or narrower term when the question required a specific "
    "level of detail\n"
    "  - Extra conflicting numeric values or alternate counts that "
    "disagree with GTFA\n\n"
    'Refusals, meta-answers ("the provided image", "cannot determine"), '
    "and empty responses are ALWAYS not equivalent, regardless of GTFA.\n\n"
    "If evaluating equivalence would require seeing the image — for example, "
    "GTFA uses one labelling system (condition names, colour labels) and the "
    "candidate uses another (lane numbers, positional indices), and the "
    "mapping is not in the question text — return equivalent=false with "
    'your reasoning prefixed "UNVERIFIABLE:". Downstream tooling treats '
    "these as needing a multimodal regrade (or human review), not as "
    "confirmed model errors.\n\n"
    "Reply with the JSON schema. Keep the reasoning to ≤3 sentences and "
    "name the specific principle you applied."
)

MULTIMODAL_JUDGE_SYSTEM = JUDGE_SYSTEM.replace(
    "You do NOT see the image.",
    "You are also shown the task image. Use it ONLY to resolve "
    "labelling ambiguities between the candidate and the GTFA. "
    "Do NOT re-solve the question from the image.",
).replace(
    "If evaluating equivalence would require seeing the image — for example, "
    "GTFA uses one labelling system (condition names, colour labels) and the "
    "candidate uses another (lane numbers, positional indices), and the "
    "mapping is not in the question text — return equivalent=false with "
    'your reasoning prefixed "UNVERIFIABLE:". Downstream tooling treats '
    "these as needing a multimodal regrade (or human review), not as "
    "confirmed model errors.\n\n",
    "With the image available, decide equivalence directly. Do NOT tag your reasoning as UNVERIFIABLE.\n\n",
)


class JudgeVerdict(BaseModel):
    """Upstream's `FreeFormJudgeResponse`; its JSON schema is sent as the judge's structured output."""

    reasoning: str
    equivalent: bool


def parse_judge_verdict(raw: str) -> Optional[JudgeVerdict]:
    """Parse the judge's JSON even when wrapped in code fences or prose (upstream `_salvage_json`)."""
    candidates = [raw]
    stripped = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", raw).strip()
    if stripped and stripped != raw:
        candidates.append(stripped)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m and m.group(0) not in candidates:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            return JudgeVerdict.model_validate_json(cand)
        except ValidationError:
            continue
    return None
