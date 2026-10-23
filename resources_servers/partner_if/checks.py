# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code-owned deterministic verifier registry. No executable dataset expressions.

Generic checkers and the fixed output checks in `fixed_checks.py` are built in. Other
task-specific checkers live in trusted modules that the host config names in
`task_check_modules`; each exports `CHECKERS`, a mapping from verifier ID to `Checker`,
which `register_modules` adds before any contract is graded.
"""

import importlib
import re
from dataclasses import dataclass
from typing import Callable

from pydantic import model_validator

from resources_servers.partner_if import fixed_checks
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import StrictModel, Text, VerifierSpec


class EmptyParams(StrictModel):
    pass


class LLMParams(StrictModel):
    """Required capture receipts; omission retains conservative all-artifact checks."""

    artifacts: list[Text] | None = None

    @model_validator(mode="after")
    def unique_artifacts(self):
        if self.artifacts is not None and len(set(self.artifacts)) != len(self.artifacts):
            raise ValueError("Artifact dependencies must be unique")
        return self


class TextParams(StrictModel):
    source: Text
    literal: Text

    @model_validator(mode="after")
    def valid_source(self):
        if self.source != "answer" and not (self.source.startswith("artifact:") and len(self.source) > 9):
            raise ValueError("Source must be answer or artifact:<captured-name>")
        return self


@dataclass
class CheckContext:
    evidence: EvidenceBundle


def text_contains(params: TextParams, context: CheckContext) -> tuple[bool, str]:
    text = context.evidence.source_text(params.source)
    return text is not None and params.literal in text, f"Literal {params.literal!r} in {params.source}"


def text_no_uuid(context: CheckContext) -> tuple[bool, str]:
    """Check the answer and every assistant message; tool calls, tool results and context may contain UUIDs."""
    evidence = context.evidence
    text = "\n".join([evidence.answer] + [event.text for event in evidence.trajectory if event.kind == "assistant"])
    found = re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text, re.I)
    return found is None, "Checked visible text for UUID-format identifiers; tool arguments excluded."


@dataclass(frozen=True)
class Checker:
    params: type[StrictModel]
    run: Callable | None


def fixed(check: Callable[[CheckContext], tuple[bool, str]]) -> Checker:
    """Adapt a task-owned function without allowing dataset parameter overrides."""
    return Checker(EmptyParams, lambda _params, context: check(context))


REGISTRY = {
    "llm_judge_v1": Checker(LLMParams, None),
    "text_contains_v1": Checker(TextParams, text_contains),
    "text_no_uuid_v1": fixed(text_no_uuid),
    "scratchpad_comments_v1": fixed(fixed_checks.scratchpad_comments),
    "review_sections_v1": fixed(fixed_checks.review_sections),
    "max_100_words_v1": fixed(fixed_checks.max_100_words),
    "short_headings_v1": fixed(fixed_checks.short_headings),
    "headed_section_v1": fixed(fixed_checks.headed_section),
    "max_five_sections_v1": fixed(fixed_checks.max_five_sections),
    "no_plan_tool_v1": fixed(fixed_checks.no_plan_tool),
    "made_project_edits_v1": fixed(fixed_checks.made_project_edits),
}


def register_modules(names: list[str]) -> None:
    """Add each named module's CHECKERS. Registering again is a no-op; an ID cannot be redefined."""
    for name in names:
        checkers = getattr(importlib.import_module(name), "CHECKERS", None)
        if not isinstance(checkers, dict) or not all(isinstance(checker, Checker) for checker in checkers.values()):
            raise ValueError(f"Task check module {name} must define CHECKERS as a dict of Checker")
        if conflicts := sorted(i for i, checker in checkers.items() if REGISTRY.get(i, checker) is not checker):
            raise ValueError(f"Task check module {name} redefines registered verifiers: {', '.join(conflicts)}")
        REGISTRY.update(checkers)


def resolve(spec: VerifierSpec) -> tuple[Checker, StrictModel]:
    if spec.id not in REGISTRY:
        raise ValueError(f"Unknown registered verifier: {spec.id} (task-specific checkers need task_check_modules)")
    checker = REGISTRY[spec.id]
    return checker, checker.params.model_validate(spec.params)
