# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rules for the server classes that a component's entry point defines."""

from .engine import AGENT_SERVER, Kind, Rule, Target, Unavailable, validate
from .probe import observe_session


def defined_run(target: Target) -> object | None:
    return vars(target.server).get("run")


def lacking_agent_sessions(target: Target) -> object | None:
    return None if target.server.supports_agent_sessions() else target.server


def defined_aggregate_metrics(target: Target) -> object | None:
    return vars(target.server).get("aggregate_metrics")


def failed_sandbox_access(target: Target) -> object | None:
    if observe_session(target).sandbox_access != "failed":
        return None
    return vars(target.server).get("_seed_agent_session_state") or target.server


def stopped_lent_sandbox(target: Target) -> object | None:
    session = observe_session(target)
    if session.sandbox_access == "refused":
        return None
    if session.session_close != "accepted":
        raise Unavailable(f"the session close was {session.session_close}")
    if session.sandbox_killed is None:
        raise Unavailable("whether the lent sandbox survived the close could not be determined")
    if not session.sandbox_killed:
        return None
    return vars(target.server).get("_close_agent_session_state") or target.server


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
    Rule(
        "A3",
        AGENT_SERVER,
        "Agent servers should not override `aggregate_metrics`; aggregation belongs to the environment server.",
        defined_aggregate_metrics,
    ),
    Rule(
        "A4",
        AGENT_SERVER,
        "Agent servers should accept a sandbox lent in the session seed, or refuse it with a 422 saying 'does not support sandbox access'.",
        failed_sandbox_access,
        requires=("A2",),
    ),
    Rule(
        "A5",
        AGENT_SERVER,
        "Agent servers should leave a borrowed sandbox running when their session closes; its owner stops it.",
        stopped_lent_sandbox,
        requires=("A4",),
    ),
)
validate(RULES)


def rules_for(kind: Kind) -> tuple[Rule, ...]:
    return tuple(rule for rule in RULES if rule.kind == kind)
