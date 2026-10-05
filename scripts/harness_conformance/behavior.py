# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks against controlled observations; independent of artifact validation and process exit."""

import json
from collections import Counter
from collections.abc import Callable

from nemo_gym.harness_capabilities.results import CheckResult, Results

from .scenarios import Scenario


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def tool_checks(
    record: dict | None, witnessed: list[dict], *, applies: bool = True, available: bool = True
) -> list[dict]:
    results = Results()
    trajectory = _mapping((record or {}).get("ng_trajectory"))
    tools = _objects(trajectory.get("tool_calls"))
    invocations = _objects(trajectory.get("invocations"))
    wanted = Counter(t.get("id") for t in witnessed)
    identities = (
        all(isinstance(t.get("tool_call_id"), str) for t in tools)
        and wanted == Counter(t.get("tool_call_id") for t in tools)
        and all(n == 1 for n in wanted.values())
    )
    results.check(
        "tools.witness_ids",
        "behavioral",
        identities,
        evidence=("TE-5",),
        applies=applies,
        available=record is not None and available,
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
                for item in _objects(inv.get("conversation"))
                if item.get("call_id") == expected["id"]
            ]
            requests = [i for i in items if i.get("type") == "function_call"]
            values["join"].append(len(requests) == 1)
            values["status"].append(tool.get("status") == ("failed" if expected["exit_code"] else "completed"))
            observed = expected.get("outputs", [])
            values["output"].append(bool(observed) and all(o == tool.get("output") for o in observed))
            if len(requests) != 1:
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
    for name, description in (
        ("join", "request does not join uniquely"),
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
            depends_on=("tools.witness_ids", "tools.witness_join") if name == "request" else ("tools.witness_ids",),
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
        present=record is not None and available,
        reason="rollout reward differs from the verifier witness",
    )
    for row in model_checks(record, attempts, fingerprint=fingerprint, available=available):
        results.rows[row["id"]] = CheckResult(**row)
    for row in tool_checks(record, tools, applies=scenario.tool_steps > 0, available=available):
        results.rows[row["id"]] = CheckResult(**row)
    return results.dump()


def model_checks(
    record: dict | None, attempts: list[dict], *, fingerprint: Callable, available: bool = True
) -> list[dict]:
    """Compare canonical fields to the independently scripted endpoint exchanges."""
    calls = _objects(_mapping((record or {}).get("ng_trajectory")).get("model_calls"))
    expected_exchanges, observed_exchanges = Counter(), Counter()
    expected_metadata, observed_metadata = Counter(), Counter()
    expected_tokens, observed_tokens = Counter(), Counter()
    token_fields = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens")
    for attempt in attempts:
        response = attempt["response"] or {}
        exchange = fingerprint(attempt["request"], attempt["status_code"], response)
        expected_exchanges[exchange] += 1
        chat = "messages" in attempt["request"]
        choices = response.get("choices") or []
        finish = choices[0].get("finish_reason") if chat and choices else None
        expected_metadata[
            (exchange, json.dumps(response.get("id")), json.dumps(response.get("status")), json.dumps(finish))
        ] += 1
        usage = response.get("usage") or {}
        counts = (
            usage.get("prompt_tokens" if chat else "input_tokens"),
            usage.get("completion_tokens" if chat else "output_tokens"),
            (usage.get("completion_tokens_details" if chat else "output_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            usage.get("total_tokens"),
            (usage.get("prompt_tokens_details" if chat else "input_tokens_details") or {}).get("cached_tokens"),
        )
        expected_tokens[(exchange, json.dumps(counts))] += 1
    for call in calls:
        metadata = _mapping(call.get("response_metadata"))
        exchange = fingerprint(call.get("request"), metadata.get("status_code"), call.get("response"))
        observed_exchanges[exchange] += 1
        observed_metadata[
            (
                exchange,
                json.dumps(metadata.get("response_id")),
                json.dumps(metadata.get("response_status")),
                json.dumps(metadata.get("finish_reason")),
            )
        ] += 1
        stats = _mapping(call.get("token_stats"))
        observed_tokens[(exchange, json.dumps([stats.get(field) for field in token_fields]))] += 1
    result = Results()
    for key, expected, observed, evidence in (
        ("model.saved_exchanges", expected_exchanges, observed_exchanges, ("TE-1", "TE-4", "TE-7")),
        ("model.response_metadata", expected_metadata, observed_metadata, ("TE-1",)),
        ("model.token_counts", expected_tokens, observed_tokens, ("TE-2",)),
    ):
        result.check(
            key,
            "behavioral",
            expected == observed,
            evidence=evidence,
            available=record is not None and available,
            reason={
                "model.saved_exchanges": "canonical exchanges differ from the independent endpoint witness",
                "model.response_metadata": "canonical response metadata differs from the independent endpoint witness",
                "model.token_counts": "canonical token counts differ from the independent endpoint witness",
            }[key],
        )
    return result.dump()
