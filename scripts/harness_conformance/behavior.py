# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks against controlled observations; independent of artifact validation and process exit."""

import json
from collections import Counter
from collections.abc import Callable

from nemo_gym.harness_capabilities.results import CheckResult, Results

from .scenarios import Scenario


def tool_checks(record: dict | None, witnessed: list[dict], *, applies: bool = True) -> list[dict]:
    results = Results()
    trajectory = (record or {}).get("ng_trajectory") or {}
    observations = ((record or {}).get("ng_agent_observations") or {}).get("records", [])
    tools = trajectory.get("tool_calls") or [r for r in observations if r.get("kind") == "tool_call"]
    invocations = trajectory.get("invocations") or [r for r in observations if r.get("kind") == "agent_invocation"]
    wanted = Counter(t.get("id") for t in witnessed)
    identities = wanted == Counter(t.get("tool_call_id") for t in tools) and all(n == 1 for n in wanted.values())
    results.check(
        "tools.witness_ids",
        "behavioral",
        identities,
        evidence=("TE-5",),
        applies=applies,
        available=record is not None,
        location="$.ng_trajectory.tool_calls",
        reason="retained tool identities differ from the independent tool witness",
    )
    values = {"join": [], "request": [], "status": [], "output": []}
    if identities and record is not None:
        by_id = {t["tool_call_id"]: t for t in tools}
        for expected in witnessed:
            tool = by_id[expected["id"]]
            items = [
                item
                for inv in invocations
                if inv.get("invocation_id") == tool.get("invocation_id")
                for item in inv.get("conversation", [])
                if item.get("call_id") == expected["id"]
            ]
            requests = [i for i in items if i.get("type") == "function_call"]
            outputs = [i for i in items if i.get("type") == "function_call_output"]
            values["join"].append(len(requests) == len(outputs) == 1)
            if len(requests) != 1 or len(outputs) != 1:
                continue
            request = requests[0]
            try:
                arguments = json.loads(request["arguments"])
            except (KeyError, TypeError, ValueError):
                arguments = None
            name = request.get("name")
            if request.get("namespace"):
                name = f"{request['namespace']}__{name}"
            values["request"].append(
                name == expected["name"]
                and arguments == expected["arguments"]
                and tool.get("tool_name") == request.get("name")
            )
            values["status"].append(tool.get("status") == ("failed" if expected["exit_code"] else "completed"))
            observed = expected.get("outputs", [])
            values["output"].append(
                bool(observed)
                and all(o == outputs[0].get("output") for o in observed)
                and (tool.get("output") is None or tool["output"] == outputs[0].get("output"))
            )
    for name, description in (
        ("join", "request/result does not join uniquely"),
        ("request", "name or arguments differ"),
        ("status", "status differs"),
        ("output", "output differs"),
    ):
        results.check(
            "tools.witness_" + name,
            "behavioral",
            lambda n=name: all(values[n]),
            evidence=("TE-5",),
            applies=applies,
            available=bool(values[name]),
            depends_on=("tools.witness_ids",),
            reason="retained tool " + description + " from the independent tool witness",
        )
    return results.dump()


def inspect_behavior(
    scenario: Scenario, witness: dict | None, record: dict | None, *, fingerprint: Callable
) -> list[dict]:
    results = Results()
    attempts = (witness or {}).get("attempts", [])
    statuses = [a["status_code"] for a in attempts]
    tools = (witness or {}).get("tool_calls", [])
    available = witness is not None

    def check(key, condition, *, evidence=(), applies=True, present=available, reason=""):
        results.check(
            key, "behavioral", condition, evidence=evidence, applies=applies, available=present, reason=reason
        )

    check(
        "episode.initialized", (witness or {}).get("seeded") == 1, reason="expected one fresh episode initialization"
    )
    check(
        "model.reached",
        bool(attempts),
        evidence=("TE-1",),
        reason="the harness never reached the controlled model endpoint",
    )
    check(
        "model.protocol",
        not (witness or {}).get("violations"),
        reason="scripted provider reported protocol violations",
    )
    check(
        "model.error_sequence",
        statuses[: len(scenario.http_errors)] == list(scenario.http_errors),
        evidence=("TE-1",),
        applies=bool(scenario.http_errors),
        reason="not all prescribed model failures were observed",
    )
    # Preserve the original probe's comparison scope: repeated error attempts, not recovery requests.
    check(
        "model.retry_request",
        lambda: all(a["request"] == attempts[0]["request"] for a in attempts[1 : len(scenario.http_errors)]),
        evidence=("TE-1",),
        applies=len(scenario.http_errors) > 1,
        present=len(attempts) >= len(scenario.http_errors),
        reason="retry scenario did not repeat the same request body",
    )
    check(
        "model.terminal_error",
        bool(statuses) and all(s == scenario.http_errors[-1] for s in statuses) if scenario.terminal_error else True,
        evidence=("TE-1",),
        applies=scenario.terminal_error,
        reason="terminal model failure was not observed",
    )
    check(
        "model.finished",
        (witness or {}).get("finished"),
        applies=not scenario.terminal_error,
        reason="the harness did not finish the scripted model exchange",
    )
    check(
        "tools.executed",
        len(tools) == scenario.tool_steps and all(t.get("executed") and t.get("result_seen") for t in tools),
        evidence=("TE-4", "TE-5"),
        applies=scenario.tool_steps > 0,
        reason="prescribed tool executions and their returned results were not all witnessed",
    )
    verifications = (witness or {}).get("verifications", [])
    check(
        "verifier.outcome",
        len(verifications) == 1 and verifications[0]["reward"] == scenario.expected_reward,
        evidence=("TE-6",),
        applies=not scenario.terminal_error,
        reason="expected verifier outcome was not observed",
    )
    check(
        "verifier.answer",
        len(verifications) == 1 and verifications[0].get("answer_seen"),
        evidence=("TE-6",),
        applies=not scenario.terminal_error,
        reason="the verifier did not receive the scripted final answer",
    )
    check(
        "verifier.saved_reward",
        lambda: record.get("reward") == scenario.expected_reward,
        evidence=("TE-6",),
        applies=not scenario.terminal_error,
        present=record is not None,
        reason="rollout reward differs from the verifier witness",
    )
    calls = ((record or {}).get("ng_model_call_capture") or {}).get("calls", [])
    expected = Counter(fingerprint(a["request"], a["status_code"], a["response"]) for a in attempts)
    observed = Counter(fingerprint(c.get("request"), c.get("status_code"), c.get("response")) for c in calls)
    check(
        "model.saved_exchanges",
        expected == observed,
        evidence=("TE-1", "TE-2", "TE-4", "TE-7"),
        present=record is not None and available,
        reason="retained model attempts differ from the independent endpoint witness",
    )
    for row in tool_checks(record, tools, applies=scenario.tool_steps > 0):
        results.rows[row["id"]] = CheckResult(**row)
    return results.dump()
