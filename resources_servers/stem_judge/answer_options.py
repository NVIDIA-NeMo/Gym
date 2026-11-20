# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose optional answer-judging instructions around the measured v11 guidelines."""

import re
from typing import Literal


VerdictMode = Literal["binary", "yes_no_unresolved"]
GuidelineScope = Literal["stem", "cross_domain"]

_CROSS_DOMAIN = """The question may come from any subject. The named subjects and examples illustrate ways to
evaluate equivalence. Apply only rules relevant to what this question asks. Preserve distinctions
that change the answer, including a mathematical restriction or a legal exception."""
_THREE_WAY = """If the supplied evidence cannot settle correctness, including a material conflict between the
question and reference, choose unresolved even if the candidate repeats the reference. An answer that is demonstrably wrong earns no.
End with exactly one line: Judgement: 'yes', Judgement: 'no', or Judgement: 'unresolved'."""


def answer_template(template: str, *, scope: GuidelineScope, verdict_mode: VerdictMode) -> str:
    """Keep the default template verbatim and reuse its rule block in opt-in modes."""
    if scope == "stem" and verdict_mode == "binary":
        return template
    context, sep, body = template.partition("# GUIDELINES #")
    guidelines, close, footer = body.partition("<important>")
    instructions, fields, inputs = footer.partition("**Question**:")
    if not (sep and close and fields):
        raise ValueError("answer guideline and verdict options require the stem_v11 template sections")
    if scope == "cross_domain":
        context = _CROSS_DOMAIN
    if verdict_mode == "yes_no_unresolved":
        instructions = _THREE_WAY
    else:
        _, close, instructions = instructions.partition("</important>")
        if not close:
            raise ValueError("stem_v11 is missing the end of its guidance section")
    return "\n\n".join((context.strip(), sep, guidelines.strip(), instructions.strip(), fields + inputs))


def final_verdict(text: str) -> str | None:
    """Only the closing verdict line can decide a three-way judgment."""
    final = text.rsplit("</think>", 1)[-1].strip().splitlines()
    if not final:
        return None
    match = re.fullmatch(r"""Judgement:\s*(['"]?)(yes|no|unresolved)\1\.?""", final[-1].strip(), re.IGNORECASE)
    return match[2].lower() if match else None
