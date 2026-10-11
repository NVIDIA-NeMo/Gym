# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks against controlled observations; independent of artifact validation and process exit."""

import json
from collections import Counter
from collections.abc import Callable

from .checks import BehavioralCheck
from .results import Results


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
    results.run(
        BehavioralCheck(
            id="tools.witness_ids",
            tier="P0",
            evidence=("TE-5",),
            location="$.ng_trajectory.tool_calls",
            reason="retained tool identities differ from the independent tool witness",
            predicate=lambda: identities,
            available=record is not None and available,
            applies=applies,
        )
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
    results.run(
        BehavioralCheck(
            id="tools.witness_join",
            tier="P0",
            evidence=("TE-5",),
            location="witness.json",
            reason="retained tool request does not join uniquely from the independent tool witness",
            predicate=lambda: all(values["join"]),
            available=bool(values["join"]),
            applies=applies,
            depends_on=("tools.witness_ids",),
        )
    )
    results.run(
        BehavioralCheck(
            id="tools.witness_request",
            tier="P0",
            evidence=("TE-5",),
            location="witness.json",
            reason="retained tool name or arguments differ from the independent tool witness",
            predicate=lambda: all(values["request"]),
            available=bool(values["request"]),
            applies=applies,
            depends_on=("tools.witness_ids", "tools.witness_join"),
        )
    )
    results.run(
        BehavioralCheck(
            id="tools.witness_status",
            tier="P0",
            evidence=("TE-5",),
            location="witness.json",
            reason="retained tool status differs from the independent tool witness",
            predicate=lambda: all(values["status"]),
            available=bool(values["status"]),
            applies=applies,
            depends_on=("tools.witness_ids",),
        )
    )
    results.run(
        BehavioralCheck(
            id="tools.witness_output",
            tier="P0",
            evidence=("TE-5",),
            location="witness.json",
            reason="retained tool output differs from the independent tool witness",
            predicate=lambda: all(values["output"]),
            available=bool(values["output"]),
            applies=applies,
            depends_on=("tools.witness_ids",),
        )
    )
    return results.dump()


def inspect_behavior(
    witness: dict | None,
    record: dict | None,
    *,
    http_errors: tuple[int, ...],
    terminal_error: bool,
    tool_steps: int,
    expected_reward: float,
    fingerprint: Callable,
) -> list[dict]:
    """Check saved evidence and observations against explicit test expectations."""
    results = Results()
    attempts = (witness or {}).get("attempts", [])
    statuses = [a["status_code"] for a in attempts]
    tools = (witness or {}).get("tool_calls", [])
    available = witness is not None

    results.run(
        BehavioralCheck(
            id="episode.initialized",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason="expected one fresh episode initialization",
            predicate=lambda: (witness or {}).get("seeded") == 1,
            available=available,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.reached",
            tier="P0",
            evidence=("TE-1",),
            location="witness.json",
            reason="the harness never reached the controlled model endpoint",
            predicate=lambda: bool(attempts),
            available=available,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.protocol",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason="scripted provider reported protocol violations",
            predicate=lambda: not (witness or {}).get("violations"),
            available=available,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.error_sequence",
            tier="P0",
            evidence=("TE-1",),
            location="witness.json",
            reason="not all prescribed model failures were observed",
            predicate=lambda: statuses[: len(http_errors)] == list(http_errors),
            available=available,
            applies=bool(http_errors),
        )
    )
    # Preserve the original probe's comparison scope: repeated error attempts, not recovery requests.
    results.run(
        BehavioralCheck(
            id="model.retry_request",
            tier="P0",
            evidence=("TE-1",),
            location="witness.json",
            reason="retry scenario did not repeat the same request body",
            predicate=lambda: all((a["request"] == attempts[0]["request"] for a in attempts[1 : len(http_errors)])),
            available=len(attempts) >= len(http_errors),
            applies=len(http_errors) > 1,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.terminal_error",
            tier="P0",
            evidence=("TE-1",),
            location="witness.json",
            reason="terminal model failure was not observed",
            predicate=lambda: bool(statuses) and all((s == http_errors[-1] for s in statuses))
            if terminal_error
            else True,
            available=available,
            applies=terminal_error,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.finished",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason="the harness did not finish the scripted model exchange",
            predicate=lambda: bool((witness or {}).get("finished")),
            available=available,
            applies=not terminal_error,
        )
    )
    results.run(
        BehavioralCheck(
            id="tools.executed",
            tier="P0",
            evidence=("TE-4", "TE-5"),
            location="witness.json",
            reason="prescribed tool executions and their returned results were not all witnessed",
            predicate=lambda: len(tools) == tool_steps
            and all((t.get("executed") and t.get("result_seen") for t in tools)),
            available=available,
            applies=tool_steps > 0,
        )
    )
    verifications = (witness or {}).get("verifications", [])
    results.run(
        BehavioralCheck(
            id="verifier.outcome",
            tier="P0",
            evidence=("TE-6",),
            location="witness.json",
            reason="expected verifier outcome was not observed",
            predicate=lambda: len(verifications) == 1 and verifications[0]["reward"] == expected_reward,
            available=available,
            applies=not terminal_error,
        )
    )
    results.run(
        BehavioralCheck(
            id="verifier.answer",
            tier="P0",
            evidence=("TE-6",),
            location="witness.json",
            reason="the verifier did not receive the scripted final answer",
            predicate=lambda: len(verifications) == 1 and verifications[0].get("answer_seen"),
            available=available,
            applies=not terminal_error,
        )
    )
    results.run(
        BehavioralCheck(
            id="verifier.saved_reward",
            tier="P0",
            evidence=("TE-6",),
            location="$.reward",
            reason="rollout reward differs from the verifier witness",
            predicate=lambda: record.get("reward") == expected_reward,
            available=record is not None and available,
            applies=not terminal_error,
        )
    )
    return (
        results.dump()
        + model_checks(record, attempts, fingerprint=fingerprint, available=available)
        + tool_checks(record, tools, applies=tool_steps > 0, available=available)
    )


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
    result.run(
        BehavioralCheck(
            id="model.saved_exchanges",
            tier="P0",
            evidence=("TE-1", "TE-4", "TE-7"),
            location="$.ng_trajectory.model_calls",
            reason="canonical exchanges differ from the independent endpoint witness",
            predicate=lambda: expected_exchanges == observed_exchanges,
            available=record is not None and available,
        )
    )
    result.run(
        BehavioralCheck(
            id="model.response_metadata",
            tier="P0",
            evidence=("TE-1",),
            location="$.ng_trajectory.model_calls",
            reason="canonical response metadata differs from the independent endpoint witness",
            predicate=lambda: expected_metadata == observed_metadata,
            available=record is not None and available,
        )
    )
    result.run(
        BehavioralCheck(
            id="model.token_counts",
            tier="P0",
            evidence=("TE-2",),
            location="$.ng_trajectory.model_calls",
            reason="canonical token counts differ from the independent endpoint witness",
            predicate=lambda: expected_tokens == observed_tokens,
            available=record is not None and available,
        )
    )
    return result.dump()
