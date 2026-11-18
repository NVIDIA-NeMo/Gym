# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared generation instructions; verifier-only data never enters this interface."""

from __future__ import annotations


POLICY_PROMPT_REVISION = "final-answer-only-box-v1"
SOLVER_INSTRUCTION = (
    r"Solve the following problem step by step. Put your final answer, and only your final answer, in \boxed{}."
)
OLD_SOLVER_INSTRUCTION = r"Solve the following problem step by step. Put your final answer in \boxed{}."


def validate_prompt_overrides(value: object) -> dict[str, dict[str, str]]:
    """Validate additive instructions keyed by exact canonical domain or source."""
    if not isinstance(value, dict) or set(value) - {"domains", "sources"}:
        raise ValueError("prompt overrides allow only domains and sources mappings")
    result: dict[str, dict[str, str]] = {"domains": {}, "sources": {}}
    for group, entries in value.items():
        if not isinstance(entries, dict):
            raise ValueError(f"{group} must be a mapping of names to instructions")
        for name, instruction in entries.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError("override names must be nonempty strings without surrounding whitespace")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("override instructions must be nonempty strings")
            if SOLVER_INSTRUCTION in instruction or OLD_SOLVER_INSTRUCTION in instruction:
                raise ValueError("shared solver instruction is already supplied; do not duplicate it in overrides")
            result[group][name] = instruction.strip()
    return result


def _remove_boundary_instruction(problem: str) -> str:
    # Never replace quoted instructions or matching sentences within the task.
    result = problem
    while True:
        previous = result
        for instruction in (SOLVER_INSTRUCTION, OLD_SOLVER_INSTRUCTION):
            if result == instruction:
                result = ""
            elif result.startswith(instruction) and result[len(instruction)].isspace():
                result = result[len(instruction) :].lstrip()
            elif result.endswith(instruction) and result[-len(instruction) - 1].isspace():
                result = result[: -len(instruction)].rstrip()
        if result == previous:
            return result


def build_policy_prompt(
    problem: str,
    *,
    domain: str,
    source: str,
    overrides: dict[str, dict[str, str]] | None = None,
) -> str:
    """Append shared domain/source instructions and the required final-answer format.

    This creates input for new generations. Historical answer replay must retain
    the actual input that generated each saved answer.
    """
    overrides = validate_prompt_overrides(overrides or {})
    parts = [_remove_boundary_instruction(problem)]
    for group, key in (("domains", domain), ("sources", source)):
        instruction = overrides[group].get(key)
        if instruction:
            parts.append(instruction)
    parts.append(SOLVER_INSTRUCTION)
    return "\n\n".join(part for part in parts if part)
