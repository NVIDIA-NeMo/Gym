# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IndicIFEval-Trans scoring adapter for the existing IFEval resources server."""

import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from resources_servers.instruction_following.setup_indicifeval import LANGUAGES, IndicLanguage, load_harness


logger = logging.getLogger(__name__)


class IndicIFEvalMetadata(BaseModel):
    """The aligned instruction lists and language required by the Trans harness."""

    model_config = ConfigDict(extra="allow")
    language: IndicLanguage
    prompt: str = Field(min_length=1)
    instruction_id_list: list[str] = Field(min_length=1)
    kwargs: list[dict[str, JsonValue]]
    grading_mode: Literal["binary", "fraction"] = "binary"

    @model_validator(mode="after")
    def aligned_instructions(self) -> "IndicIFEvalMetadata":
        if len(self.instruction_id_list) != len(self.kwargs):
            raise ValueError("instruction_id_list and kwargs must have the same length")
        return self


class IndicIFEvalScores(BaseModel):
    """Per-prompt and per-instruction strict/loose results from the harness."""

    prompt_level_strict_acc: bool
    inst_level_strict_acc: list[bool]
    prompt_level_loose_acc: bool
    inst_level_loose_acc: list[bool]
    instruction_errors: dict[str, str] = Field(default_factory=dict)


def score_response(metadata: IndicIFEvalMetadata, response: str) -> IndicIFEvalScores:
    """Run upstream scoring per instruction, exposing checker errors as failures."""
    harness = load_harness()
    # Reasoning is not part of the assistant's final answer.
    cleaned = re.sub(r"<(think|thinking)>.*?</\1>", "", response, flags=re.DOTALL)
    if cleaned != response:
        response = cleaned.strip()
    strict: list[bool] = []
    loose: list[bool] = []
    errors: dict[str, str] = {}
    for index, (instruction_id, kwargs) in enumerate(zip(metadata.instruction_id_list, metadata.kwargs, strict=True)):
        inp = harness.InputExample(
            key=index, instruction_id_list=[instruction_id], prompt=metadata.prompt, kwargs=[kwargs]
        )
        try:
            strict_result = harness.test_instruction_following_strict(inp, response, metadata.language)
            loose_result = harness.test_instruction_following_loose(inp, response, metadata.language)
            strict.append(strict_result.follow_all_instructions)
            loose.append(loose_result.follow_all_instructions)
        except Exception as exc:
            # Request boundary: malformed responses or upstream checker failures
            # must not crash the server or silently become successful rewards.
            logger.exception("IndicIFEval %s instruction %s failed", metadata.language, instruction_id)
            errors[f"{index}:{instruction_id}"] = f"{type(exc).__name__}: {exc}"
            strict.append(False)
            loose.append(False)
    return IndicIFEvalScores(
        prompt_level_strict_acc=all(strict),
        inst_level_strict_acc=strict,
        prompt_level_loose_acc=all(loose),
        inst_level_loose_acc=loose,
        instruction_errors=errors,
    )


def aggregate_scores(tasks: Sequence[Sequence[Mapping[str, JsonValue]]]) -> dict[str, float | int]:
    """Aggregate instruction micro-accuracy, overall and for each language."""
    buckets: defaultdict[str, list[IndicIFEvalScores]] = defaultdict(list)
    for task in tasks:
        for row in task:
            if "inst_level_strict_acc" not in row:
                continue
            scores = IndicIFEvalScores.model_validate(row)
            buckets[""].append(scores)
            metadata = row.get("verifier_metadata")
            language = metadata.get("language") if isinstance(metadata, dict) else None
            if isinstance(language, str) and language in LANGUAGES:
                buckets[f"language/{language}/"].append(scores)
    metrics: dict[str, float | int] = {}
    for prefix, rows in buckets.items():
        metrics[prefix + "count"] = len(rows)
        metrics[prefix + "checker_error_count"] = sum(len(row.instruction_errors) for row in rows)
        for mode in ("strict", "loose"):
            prompt_key = f"prompt_level_{mode}_acc"
            instruction_key = f"inst_level_{mode}_acc"
            instructions = [passed for row in rows for passed in getattr(row, instruction_key)]
            metrics[prefix + prompt_key] = sum(getattr(row, prompt_key) for row in rows) / len(rows)
            metrics[prefix + instruction_key] = sum(instructions) / len(instructions) if instructions else 0.0
    return metrics
