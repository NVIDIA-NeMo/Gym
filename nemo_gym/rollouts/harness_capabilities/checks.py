# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed check definitions; execution state and reporting live in Results."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

from jsonschema import Draft202012Validator

from nemo_gym.health.types import Verdict


PriorityTier = Literal["P0", "P1", "P2"]
Kind = Literal["schema", "semantic", "behavioral", "health"]


@dataclass(frozen=True)
class Evaluation:
    """A predicate outcome and any specific failure locations, without payloads."""

    passed: bool
    locations: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Check(ABC):
    """One rule bound to the input being inspected. TE labels do not set priority."""

    id: str
    tier: PriorityTier
    evidence: tuple[str, ...]
    location: str
    reason: str
    depends_on: tuple[str, ...] = ()
    applies: bool = True
    available: bool = True
    kind: ClassVar[Kind]

    def __post_init__(self) -> None:
        if self.tier not in ("P0", "P1", "P2"):
            raise ValueError(f"invalid priority tier for check {self.id}: {self.tier}")

    @abstractmethod
    def evaluate(self) -> Evaluation:
        """Evaluate only after applicability, availability and prerequisites pass."""


@dataclass(frozen=True, kw_only=True)
class SchemaCheck(Check):
    """Validate the selected input against this check's JSON Schema."""

    schema: dict
    value: object
    kind: ClassVar[Kind] = "schema"

    def evaluate(self) -> Evaluation:
        errors = list(Draft202012Validator(self.schema).iter_errors(self.value))
        # jsonschema messages can contain payloads. Retain only their JSON paths.
        locations = tuple(
            self.location
            + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path)
            for error in errors
        )
        return Evaluation(not errors, locations)


@dataclass(frozen=True, kw_only=True)
class _PredicateCheck(Check):
    predicate: Callable[[], bool]

    def evaluate(self) -> Evaluation:
        return Evaluation(self.predicate())


@dataclass(frozen=True, kw_only=True)
class SemanticCheck(_PredicateCheck):
    """Evaluate a relationship between retained evidence records."""

    kind: ClassVar[Kind] = "semantic"


@dataclass(frozen=True, kw_only=True)
class BehavioralCheck(_PredicateCheck):
    """Evaluate behavior or evidence against independent controlled observations."""

    kind: ClassVar[Kind] = "behavioral"


@dataclass(frozen=True, kw_only=True)
class HealthCheck(Check):
    """Match an individual Gym health result against the scenario's expectation."""

    expected: Verdict
    actual: Verdict | None
    kind: ClassVar[Kind] = "health"

    def evaluate(self) -> Evaluation:
        return Evaluation(self.actual is not None and self.actual == self.expected)
