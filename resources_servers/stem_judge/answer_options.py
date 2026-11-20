# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose optional answer-judging instructions around the measured v11 guidelines."""

import json
import re
from typing import Literal

from pydantic import BaseModel

from nemo_gym.core import NeMoGymResponse


VerdictMode = Literal["binary", "yes_no_unresolved"]
VerdictFormat = Literal["line", "json"]
GuidelineScope = Literal["stem", "cross_domain"]

_CROSS_DOMAIN = """The question may come from any subject. The named subjects and examples illustrate ways to
evaluate equivalence. Apply only rules relevant to what this question asks. Preserve distinctions
that change the answer, including a mathematical restriction or a legal exception."""
_ABSTENTION = """If the supplied evidence cannot settle correctness, including a material conflict between the
question and reference, choose unresolved even if the candidate repeats the reference."""
_THREE_WAY_LINE = """An answer that is demonstrably wrong earns no.
End with exactly one line: Judgement: 'yes', Judgement: 'no', or Judgement: 'unresolved'."""


def answer_template(
    template: str, *, scope: GuidelineScope, verdict_mode: VerdictMode, verdict_format: VerdictFormat = "line"
) -> str:
    """Keep the default template verbatim and reuse its rule block in opt-in modes."""
    if scope == "stem" and verdict_mode == "binary" and verdict_format == "line":
        return template
    context, sep, body = template.partition("# GUIDELINES #")
    guidelines, close, footer = body.partition("<important>")
    instructions, fields, inputs = footer.partition("**Question**:")
    if not (sep and close and fields):
        raise ValueError("answer guideline and verdict options require the stem_v11 template sections")
    if scope == "cross_domain":
        context = _CROSS_DOMAIN
    if verdict_format == "json":
        labels = '"correct" or "incorrect"'
        if verdict_mode == "yes_no_unresolved":
            labels = '"correct", "incorrect", or "unresolved"'
        instructions = (
            "Return exactly one JSON object with three string fields and no surrounding prose or Markdown. "
            f'Set "verdict" to {labels}. Set "reason_code" to a short machine-readable reason and '
            '"evidence" to a brief, specific justification grounded in the question, reference, and student solution.'
        )
        if verdict_mode == "yes_no_unresolved":
            instructions = (
                "First decide whether the question and reference supply enough information to justify a grade. "
                "If a necessary condition is missing or they materially conflict, return unresolved. "
                "This check takes precedence over the exact-match rule even when the candidate repeats the reference.\n"
                + instructions
            )
    elif verdict_mode == "yes_no_unresolved":
        instructions = _ABSTENTION + " " + _THREE_WAY_LINE
    else:
        _, close, instructions = instructions.partition("</important>")
        if not close:
            raise ValueError("stem_v11 is missing the end of its guidance section")
    return "\n\n".join((context.strip(), sep, guidelines.strip(), instructions.strip(), fields + inputs))


def add_answer_rules(template: str, rules: list[str]) -> str:
    """Append literal rule text to the configured answer template's last rule block."""
    if not rules:
        return template
    before, closing_tag, after = template.rpartition("</rule>")
    if not closing_tag:
        raise ValueError("extra_answer_rules requires an answer template with a </rule> marker")
    blocks = "".join(f"\n\n<rule>\n{rule}\n</rule>" for rule in rules)
    # Only the template's own placeholders should be expanded when rendering the row.
    return before + closing_tag + blocks.replace("{", "{{").replace("}", "}}") + after


def final_verdict(text: str) -> str | None:
    """Only the closing verdict line can decide a three-way judgment."""
    final = text.rsplit("</think>", 1)[-1].strip().splitlines()
    if not final:
        return None
    match = re.fullmatch(r"""Judgement:\s*(['"]?)(yes|no|unresolved)\1\.?""", final[-1].strip(), re.IGNORECASE)
    return match[2].lower() if match else None


class JsonJudgment(BaseModel, extra="forbid", strict=True, str_strip_whitespace=True, str_min_length=1):
    """The model's verdict and supporting evidence for one answer comparison."""

    verdict: Literal["correct", "incorrect", "unresolved"]
    reason_code: str
    evidence: str

    @property
    def label(self) -> str:
        return {"correct": "yes", "incorrect": "no", "unresolved": "unresolved"}[self.verdict]


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("duplicate JSON field")
    return fields


def parse_json_judgment(response: NeMoGymResponse, *, verdict_mode: VerdictMode) -> JsonJudgment | None:
    """Read one complete assistant message and reject ambiguous or incomplete judgments."""
    if response.status not in (None, "completed") or response.incomplete_details is not None or not response.output:
        return None
    *prefix, final = response.output
    if final.type != "message" or final.status != "completed" or any(item.type != "reasoning" for item in prefix):
        return None
    if any(block.type != "output_text" for block in final.content):
        return None
    try:
        text = "".join(block.text for block in final.content)
        judgment = JsonJudgment.model_validate(json.loads(text, object_pairs_hook=_unique_fields))
    except (ValueError, RecursionError):
        return None
    if verdict_mode == "binary" and judgment.verdict == "unresolved":
        return None
    return judgment
