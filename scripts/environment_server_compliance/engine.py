# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rules over one server class, and the order and outcomes in which they are judged.

A rule makes one claim about a class and decides it in `check`, by whatever means: reading the class, or running
it. Rules name the rules that must hold before they are judged. This module is standard library only.
"""

import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


AGENT_SERVER = "agent_server"
RESOURCES_SERVER = "resources_server"
Kind = Literal["agent_server", "resources_server"]

KINDS: dict[str, Kind] = {"responses_api_agents": AGENT_SERVER}


class Unavailable(Exception):
    """A rule cannot be judged, because what it needs to see could not be produced."""


@dataclass
class Target:
    """The server class under judgment, and the facts about it that rules share.

    `component` is the directory the class runs from, or None when it cannot be run.
    """

    server: type
    component: Path | None = None
    _facts: dict[str, tuple[Any, Unavailable | None]] = field(default_factory=dict, repr=False)

    def once(self, name: str, produce: Callable[[], Any]) -> Any:
        """The fact `name`, produced on first use; a failure to produce it is raised again for every rule."""
        if name not in self._facts:
            try:
                self._facts[name] = (produce(), None)
            except Unavailable as error:
                self._facts[name] = (None, error)
        value, error = self._facts[name]
        if error is not None:
            raise error
        return value


@dataclass(frozen=True)
class Rule:
    """A requirement on one kind of component. `message` states the expected pattern.

    `check` returns the offending function or class, which locates the warning, or None when the rule holds. It
    raises `Unavailable` when it cannot tell. `requires` lists ids of earlier rules that must hold first.
    """

    id: str
    kind: Kind
    message: str
    check: Callable[[Target], object | None]
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outcome:
    """How one rule fared: `held`, `violated` at `location`, or `skipped`.

    A skip has a `reason` when the rule could not be judged, and none when a required rule did not hold.
    """

    rule: Rule
    status: Literal["held", "violated", "skipped"]
    location: object | None = None
    reason: str | None = None


def validate(rules: tuple[Rule, ...]) -> None:
    """Require every rule to name only earlier rules of its own kind, so the order given is a valid run order."""
    seen: dict[str, Rule] = {}
    for rule in rules:
        if rule.id in seen:
            raise ValueError(f"Duplicate rule id: {rule.id!r}")
        for required in rule.requires:
            if required not in seen or seen[required].kind != rule.kind:
                raise ValueError(f"Rule {rule.id} requires {required!r}, which is not an earlier rule of its kind")
        seen[rule.id] = rule


def evaluate(rules: tuple[Rule, ...], target: Target) -> list[Outcome]:
    """Judge `rules`, in order, on `target`."""
    held: set[str] = set()
    outcomes = []
    for rule in rules:
        if not set(rule.requires) <= held:
            outcomes.append(Outcome(rule, "skipped"))
            continue
        try:
            location = rule.check(target)
        except Unavailable as error:
            outcomes.append(Outcome(rule, "skipped", reason=str(error)))
            continue
        if location is None:
            held.add(rule.id)
            outcomes.append(Outcome(rule, "held"))
        else:
            outcomes.append(Outcome(rule, "violated", location))
    return outcomes


def finding(outcome: Outcome, path: str) -> dict:
    line = inspect.getsourcelines(inspect.unwrap(outcome.location))[1]
    return {"rule": outcome.rule.id, "path": path, "line": line, "message": outcome.rule.message}


def findings(path: str, classes: list[type], rules: tuple[Rule, ...], *, component: Path | None = None) -> list[dict]:
    """Judge the server classes defined in the file `path`; the ones that cannot be judged are noted on stderr."""
    results = []
    for cls in classes:
        for outcome in evaluate(rules, Target(cls, component)):
            if outcome.status == "violated":
                results.append(finding(outcome, path))
            elif outcome.reason:
                print(f"{path}: {cls.__name__}: {outcome.rule.id} not applicable: {outcome.reason}", file=sys.stderr)
    return results
