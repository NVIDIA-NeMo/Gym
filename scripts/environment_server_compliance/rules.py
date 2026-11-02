# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rules for the server classes that a component's entry point defines."""

from .engine import AGENT_SERVER, Kind, Rule, Target, validate


def defined_run(target: Target) -> object | None:
    return vars(target.server).get("run")


def lacking_agent_sessions(target: Target) -> object | None:
    return None if target.server.supports_agent_sessions() else target.server


RULES: tuple[Rule, ...] = (
    Rule(
        "A1",
        AGENT_SERVER,
        "Agent servers should not define `run()`; implement agent sessions instead.",
        defined_run,
    ),
    Rule(
        "A2",
        AGENT_SERVER,
        "Agent servers should implement agent sessions: seed and close.",
        lacking_agent_sessions,
    ),
)
validate(RULES)


def rules_for(kind: Kind) -> tuple[Rule, ...]:
    return tuple(rule for rule in RULES if rule.kind == kind)
