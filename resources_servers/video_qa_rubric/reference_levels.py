# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference-levels protocol: the judge compares the final answer with the row's reference answer under the
row's grading rules and returns one integer level from 0 to ``max_level``; the reward is the level / ``max_level``.

A row of this protocol says ``grading: reference_levels`` in its verifier metadata and carries ``reference_answer``
and a ``rubric`` that may be any JSON object (level rules, accepted phrasings, rejections). The judge's system prompt
and user template are server settings (``ReferenceLevelsSettings``), so a question set graded with its own published
prompt puts that prompt in its config. The template names the row's fields as ``{question}``,
``{reference_answer}``, ``{rubric_json}`` (the rubric as ``json.dumps(rubric, ensure_ascii=False)``) and
``{answer}``; any other brace must be doubled. The reply must be exactly ``{"score": <level>, "reason": <text>}``.
"""

import json
from string import Formatter
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
# The fields a user template must name, each at least once.
TEMPLATE_FIELDS = frozenset({"question", "reference_answer", "rubric_json", "answer"})


class ReferenceLevelsSettings(BaseModel):
    """How the judge grades ``reference_levels`` rows: its messages and the top level of its integer scale."""

    model_config = ConfigDict(extra="forbid")
    system_prompt: Text
    user_template: Text
    max_level: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def template_fields(self) -> Self:
        try:
            names = {name for _, name, _, _ in Formatter().parse(self.user_template) if name is not None}
        except ValueError as error:
            raise ValueError(f"user_template is not a format string: {error}") from error
        if names != TEMPLATE_FIELDS:
            raise ValueError(
                f"user_template must name exactly the fields {sorted(TEMPLATE_FIELDS)}, not {sorted(names)}"
            )
        return self


def judge_messages(
    settings: ReferenceLevelsSettings, *, question: str, reference_answer: str, rubric: dict[str, Any], answer: str
) -> list[dict[str, str]]:
    """The judge's system and user messages for ANSWER; no video and no reasoning."""
    user = settings.user_template.format(
        question=question,
        reference_answer=reference_answer,
        rubric_json=json.dumps(rubric, ensure_ascii=False),
        answer=answer,
    )
    return [{"role": "system", "content": settings.system_prompt}, {"role": "user", "content": user}]


def parse_judgment(text: str, *, max_level: int) -> dict[str, Any]:
    """The judge's reply as ``{"score": level, "reason": text}``; ``ValueError`` unless it is exactly that."""
    judgment = json.loads(text)
    if not isinstance(judgment, dict) or set(judgment) != {"score", "reason"}:
        raise ValueError("The judgment must be a JSON object with exactly score and reason")
    if type(judgment["score"]) is not int or not 0 <= judgment["score"] <= max_level:
        raise ValueError(f"score must be an integer from 0 to {max_level}")
    if not isinstance(judgment["reason"], str) or not judgment["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    return judgment
