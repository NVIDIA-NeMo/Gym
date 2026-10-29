# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compaction evidence compared with a controlled provider's independent witness."""

from collections import Counter

from nemo_gym.responses_converter import message_content_to_text
from nemo_gym.rollout_observability import ContextCompactionObservation

from .checker import _mapping, _objects, _resolve
from .checks import BehavioralCheck, SchemaCheck
from .results import Results


def inspect_compaction(
    witness: dict | None, record: dict | None, *, expected: bool, policy_responses: int
) -> list[dict]:
    """Inspect the declared fixture; absent compaction evidence never implies an exemption."""
    results = Results()
    raw_records = _mapping((record or {}).get("ng_agent_observations")).get("records")
    events = [r for r in _objects(raw_records) if r.get("kind") == "context_compaction"]
    event_schema = ContextCompactionObservation.model_json_schema()
    definitions = event_schema.pop("$defs", {})
    results.run(
        SchemaCheck(
            id="compaction.schema",
            tier="P1",
            evidence=("TE-13",),
            location="$.ng_agent_observations.records",
            reason="compaction events must satisfy ContextCompactionObservation",
            schema={"type": "array", "items": event_schema, "$defs": definitions},
            value=events,
            available=isinstance(raw_records, list),
        )
    )
    fixture = _mapping((witness or {}).get("compaction"))
    helper_ids = fixture.get("response_ids")
    summary = fixture.get("summary")
    attempts = _objects((witness or {}).get("attempts"))
    expected_ids = [_mapping(a.get("response")).get("id") for a in attempts]
    available = (
        record is not None
        and isinstance(helper_ids, list)
        and all(isinstance(value, str) and value for value in helper_ids)
        and isinstance(summary, str)
        and bool(summary)
        and bool(attempts)
        and all(isinstance(value, str) and value for value in expected_ids)
    )
    helper_ids = helper_ids if available else []
    policy_ids = [response_id for response_id in expected_ids if response_id not in helper_ids]
    trajectory = _mapping((record or {}).get("ng_trajectory"))
    calls = _objects(trajectory.get("model_calls"))
    invocations = _objects(trajectory.get("invocations"))

    def check(name, predicate, location, reason, *, applies=True, depends_on=()):
        results.run(
            BehavioralCheck(
                id=f"compaction.{name}",
                tier="P1",
                evidence=("TE-13",),
                location=location,
                reason=reason,
                predicate=predicate,
                available=available,
                applies=applies,
                depends_on=depends_on,
            )
        )

    check(
        "event",
        lambda: len(events) == len(helper_ids) == int(expected)
        and all(e.get("outcome") == "completed" and e.get("trigger") == "automatic" for e in events),
        "$.ng_agent_observations.records",
        "expected one completed automatic compaction, or none in the control",
        depends_on=("compaction.schema",),
    )

    def helper_references() -> bool:
        referenced = []
        for event in events:
            for ref in _objects(event.get("model_calls")):
                matches = _resolve(ref, calls)
                if len(matches) != 1:
                    return False
                target = matches[0]
                owners = [
                    inv.get("invocation_id")
                    for inv in invocations
                    if any(target in _resolve(owned, calls) for owned in _objects(inv.get("model_calls")))
                ]
                if owners != [event.get("invocation_id")]:
                    return False
                referenced.append(_mapping(calls[target].get("response_metadata")).get("response_id"))
        return Counter(referenced) == Counter(helper_ids)

    check(
        "helper_references",
        helper_references,
        "$.ng_agent_observations.records[*].model_calls",
        "compaction must reference the witnessed helper, retained under the same invocation",
        depends_on=("compaction.event",),
    )

    def summary_reused() -> bool:
        if events[0].get("summary") != summary or helper_ids[0] not in expected_ids:
            return False
        following = expected_ids[expected_ids.index(helper_ids[0]) + 1 :]
        if not following:
            return False
        resumed = [
            call for call in calls if _mapping(call.get("response_metadata")).get("response_id") == following[0]
        ]
        if len(resumed) != 1:
            return False
        request = _mapping(resumed[0].get("request"))
        return any(
            summary in message_content_to_text(message.get("content"))
            for message in _objects(request.get("messages", request.get("input")))
        )

    check(
        "summary_reuse",
        summary_reused,
        "$.ng_agent_observations.records[*].summary; $.ng_trajectory.model_calls[*].request",
        "the saved summary and the next policy request must contain the witnessed compaction summary",
        applies=expected,
        depends_on=("compaction.event", "compaction.helper_references"),
    )

    def policy_turns() -> bool:
        turns = _objects(trajectory.get("turns"))
        if len(turns) != len(policy_ids) or len(policy_ids) != policy_responses:
            return False
        referenced = []
        for turn in turns:
            refs = _objects(turn.get("model_calls"))
            if len(refs) != 1:
                return False
            matches = _resolve(refs[0], calls)
            if len(matches) != 1:
                return False
            referenced.append(_mapping(calls[matches[0]].get("response_metadata")).get("response_id"))
        return Counter(referenced) == Counter(policy_ids)

    check(
        "policy_turns",
        policy_turns,
        "$.ng_trajectory.turns",
        "saved turns must contain exactly the witnessed policy responses, excluding compaction helpers",
    )

    def usage_retained() -> bool:
        usages = [_mapping(_mapping(a.get("response")).get("usage")) for a in attempts]
        counts = [
            (u.get("prompt_tokens", u.get("input_tokens")), u.get("completion_tokens", u.get("output_tokens")))
            for u in usages
        ]
        if any(type(value) is not int or value < 0 for pair in counts for value in pair):
            return False
        prompt = sum(pair[0] for pair in counts)
        completion = sum(pair[1] for pair in counts)
        usage = _mapping(_mapping((record or {}).get("response")).get("usage"))
        return (usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")) == (
            prompt,
            completion,
            prompt + completion,
        )

    check(
        "usage_retained",
        usage_retained,
        "$.response.usage",
        "aggregate usage must include every witnessed policy and compaction call exactly once",
    )
    return results.dump()
