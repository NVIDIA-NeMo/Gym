# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned IFEval checks with the source's dataset-wide strict/loose ordering."""

import random
from typing import cast

from pydantic import JsonValue

from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.vendor.ifeval import (
    InputExample,
    OutputExample,
    read_prompt_to_response_dict,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from resources_servers.voicebench.vendor.randomness import using_rng


def _example(metadata: dict[str, JsonValue]) -> InputExample:
    prompt = metadata.get("prompt")
    ids = metadata.get("instruction_id_list")
    kwargs = metadata.get("kwargs")
    if not isinstance(prompt, str):
        raise ValueError("VoiceBench IFEval requires the original prompt")
    if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids):
        raise ValueError("VoiceBench IFEval requires a nonempty instruction_id_list")
    if not isinstance(kwargs, list) or len(kwargs) != len(ids) or any(not isinstance(value, dict) for value in kwargs):
        raise ValueError("VoiceBench IFEval requires one kwargs dictionary per instruction")
    return InputExample(
        key=metadata.get("key", 0), instruction_id_list=cast(list[str], ids), prompt=prompt, kwargs=kwargs
    )


def _result(strict: OutputExample, loose: OutputExample, draws: int) -> ScoringResult:
    strict_count = sum(strict.follow_instruction_list)
    loose_count = sum(loose.follow_instruction_list)
    total = len(strict.instruction_id_list)
    reward = (
        float(strict.follow_all_instructions)
        + float(loose.follow_all_instructions)
        + (strict_count + loose_count) / total
    ) / 4
    return ScoringResult(
        reward,
        {
            "strict_prompt_correct": float(strict.follow_all_instructions),
            "loose_prompt_correct": float(loose.follow_all_instructions),
            "prompt_total": 1.0,
            "strict_instruction_correct": float(strict_count),
            "loose_instruction_correct": float(loose_count),
            "instruction_total": float(total),
        },
        {
            "strict_instruction_passes": strict.follow_instruction_list,
            "loose_instruction_passes": loose.follow_instruction_list,
            "strict_prompt_pass": strict.follow_all_instructions,
            "loose_prompt_pass": loose.follow_all_instructions,
            "randomized_default_draws": draws,
        },
    )


def score_dataset(rows: list[dict[str, JsonValue]], *, rng: random.Random) -> list[ScoringResult]:
    """Run all strict checks before all loose checks, including duplicate-prompt lookup."""
    examples = [_example(row) for row in rows]
    responses = read_prompt_to_response_dict(rows)
    strict_draws = []
    strict = []
    loose = []
    loose_draws = []
    with using_rng(rng) as context:
        for example in examples:
            before = context.draws
            strict.append(test_instruction_following_strict(example, responses))
            strict_draws.append(context.draws - before)
        for example in examples:
            before = context.draws
            loose.append(test_instruction_following_loose(example, responses))
            loose_draws.append(context.draws - before)
    return [
        _result(strict_row, loose_row, strict_draw + loose_draw)
        for strict_row, loose_row, strict_draw, loose_draw in zip(
            strict, loose, strict_draws, loose_draws, strict=True
        )
    ]


def score(answer: str, metadata: dict[str, JsonValue], *, rng: random.Random) -> ScoringResult:
    """Score one row; benchmark aggregates replay the full dataset's check ordering."""
    return score_dataset([metadata | {"response": answer}], rng=rng)[0]
