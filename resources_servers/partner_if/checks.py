# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code-owned deterministic verifier registry. No executable dataset expressions.

Generic checkers are built in. Task-specific checkers live in trusted modules that the
host config names in `task_check_modules`; each exports `CHECKERS`, a mapping from
verifier ID to `Checker`, which `register_modules` adds before any contract is graded.
"""

import importlib
import math
import re
from dataclasses import dataclass
from typing import Callable

import yaml
from pydantic import JsonValue, model_validator

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


class BoundedLoader(yaml.SafeLoader):
    """Duplicate-key and allocation protections adopted from the native source checker."""

    def __init__(self, stream):
        super().__init__(stream)
        self.depth = 0
        self.nodes = 0

    def compose_node(self, parent, index):
        self.depth += 1
        self.nodes += 1
        try:
            if self.depth > 64 or self.nodes > 50_000:
                raise ValueError("YAML depth/node limit exceeded")
            return super().compose_node(parent, index)
        finally:
            self.depth -= 1

    def construct_mapping(self, node, deep=False):
        if any(key.tag == "tag:yaml.org,2002:merge" for key, _ in node.value):
            raise ValueError("YAML merge keys are not supported")
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("Duplicate or non-string mapping key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def parse_yaml(text: str) -> dict[str, JsonValue]:
    if len(text) > 1_000_000:
        raise ValueError("YAML character limit exceeded")
    value = yaml.load(text, Loader=BoundedLoader)
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping")
    remaining = 50_000

    def check(node, ancestors=frozenset(), depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 64:
            raise ValueError("Expanded YAML limit exceeded")
        if isinstance(node, (dict, list)):
            if id(node) in ancestors:
                raise ValueError("Cyclic YAML alias")
            for child in node.values() if isinstance(node, dict) else node:
                check(child, ancestors | {id(node)}, depth + 1)
        elif type(node) not in (str, int, float, bool, type(None)):
            raise ValueError("YAML values must be JSON compatible")
        elif type(node) is float and not math.isfinite(node):
            raise ValueError("YAML number must be finite")

    check(value)
    return value


@dataclass
class CheckContext:
    evidence: EvidenceBundle
    parsed: dict[str, dict[str, JsonValue] | ValueError]

    def yaml(self, name: str) -> dict[str, JsonValue]:
        if name not in self.parsed:
            text = self.evidence.artifact_text(name)
            try:
                if text is None:
                    raise ValueError("Required artifact was not produced")
                self.parsed[name] = parse_yaml(text)
            except (ValueError, yaml.YAMLError, RecursionError) as error:
                self.parsed[name] = ValueError(str(error))
        result = self.parsed[name]
        if isinstance(result, ValueError):
            raise result
        return result


def text_contains(params: TextParams, context: CheckContext) -> tuple[bool, str]:
    text = context.evidence.source_text(params.source)
    return text is not None and params.literal in text, f"Literal {params.literal!r} in {params.source}"


def text_no_uuid(context: CheckContext) -> tuple[bool, str]:
    """Check captured user-facing text only; tool arguments may contain UUIDs."""
    found = re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", context.evidence.answer, re.I)
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
