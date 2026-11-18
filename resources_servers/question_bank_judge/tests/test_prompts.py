# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation formatting is shared and independent of expected answers."""

import pytest

from resources_servers.question_bank_judge.prompts import (
    OLD_SOLVER_INSTRUCTION,
    SOLVER_INSTRUCTION,
    build_policy_prompt,
    validate_prompt_overrides,
)


def _build(problem: str, **kwargs) -> str:
    return build_policy_prompt(problem, domain="physics", source="source-a", **kwargs)


def test_shared_instruction_has_exact_requested_wording_and_is_idempotent() -> None:
    assert SOLVER_INSTRUCTION == (
        r"Solve the following problem step by step. Put your final answer, and only your final answer, in \boxed{}."
    )
    first = _build("Compute the speed.")
    assert first == "Compute the speed.\n\n" + SOLVER_INSTRUCTION
    assert _build(first) == first


@pytest.mark.parametrize("instruction", [OLD_SOLVER_INSTRUCTION, SOLVER_INSTRUCTION])
@pytest.mark.parametrize("position", ["prefix", "suffix", "both"])
def test_replaces_only_boundary_boilerplate(instruction: str, position: str) -> None:
    problem = "Compute the speed."
    if position in {"prefix", "both"}:
        problem = instruction + "\n\n" + problem
    if position in {"suffix", "both"}:
        problem = problem + "\n\n" + instruction
    result = _build(problem)
    assert result == "Compute the speed.\n\n" + SOLVER_INSTRUCTION
    assert result.count(SOLVER_INSTRUCTION) == 1


def test_quoted_and_embedded_instructions_are_task_content() -> None:
    problem = 'Analyze the sentence "' + OLD_SOLVER_INSTRUCTION + '" and identify its grammatical mood.'
    assert _build(problem) == problem + "\n\n" + SOLVER_INSTRUCTION
    embedded = "First task.\n\n" + OLD_SOLVER_INSTRUCTION + "\n\nSecond task."
    assert _build(embedded) == embedded + "\n\n" + SOLVER_INSTRUCTION


def test_space_separated_boundary_instruction_is_replaced() -> None:
    assert _build("Task. " + OLD_SOLVER_INSTRUCTION) == "Task.\n\n" + SOLVER_INSTRUCTION
    assert _build(OLD_SOLVER_INSTRUCTION + " Task.") == "Task.\n\n" + SOLVER_INSTRUCTION


def test_domain_and_source_overrides_are_additive_and_exact() -> None:
    overrides = {
        "domains": {"physics": "Include units.", "biology": "Name the species."},
        "sources": {"source-a": "Label each answer part.", "other": "OTHER"},
    }
    result = _build("Task.", overrides=overrides)
    assert result == "Task.\n\nInclude units.\n\nLabel each answer part.\n\n" + SOLVER_INSTRUCTION
    assert "Name the species" not in result and "OTHER" not in result


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"rows": {}},
        {"domains": []},
        {"domains": {"physics": " "}},
        {"sources": {"": "instruction"}},
        {"sources": {" source ": "instruction"}},
        {"domains": {"physics": OLD_SOLVER_INSTRUCTION}},
        {"domains": {"physics": SOLVER_INSTRUCTION}},
    ],
)
def test_invalid_overrides_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        validate_prompt_overrides(value)
