# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map a rollout's segment rewards to the captured model calls they target.

A verifier scores the rollout with ``reward`` and may score parts of it with ``segment_rewards``. An entry targets
the model call that produced an item of the scored response, or every model call of a subagent. A trainer that keeps
every captured call trains each segment of the call forest on the entry of the calls it trains, and the rest on
``reward``. This lives outside ``nemo_gym.models.token_capture``, which must stay free of the observation models'
imports.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_gym.models.token_capture.completion import output_item_fingerprint
from nemo_gym.models.token_capture.staging.records import CallRecord
from nemo_gym.server_runtime.base_resources_server import SegmentReward
from nemo_gym.trajectory.trajectory_record import (
    NG_AGENT_OBSERVATIONS_KEY,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
)


logger = logging.getLogger(__name__)


def resolve_segment_rewards(records: Sequence[CallRecord], result: Mapping[str, Any]) -> dict[str, float]:
    """Return the reward each targeted captured call gets, keyed by ``model_call_id``.

    ``records`` are the rollout's capture manifest rows and ``result`` is its ``/run`` result. Calls that no entry
    targets are absent and keep ``reward``. Raises ``ValueError`` when an entry targets nothing in the capture, when
    an item could have come from more than one call, or when two entries give one call different rewards.
    """
    entries = [SegmentReward.model_validate(entry) for entry in result.get("segment_rewards") or []]
    targeted: dict[str, float] = {}

    def assign(record: CallRecord, reward: float) -> None:
        if targeted.setdefault(record.model_call_id, reward) != reward:
            raise ValueError(f"segment rewards give captured call {record.model_call_id} two different rewards")

    output = (result.get("response") or {}).get("output") or []
    launches: dict[str, float] = {}
    for entry in entries:
        if entry.output_index is not None:
            assign(_producing_call(records, output, entry.output_index), entry.reward)
        elif launches.setdefault(entry.launched_by, entry.reward) != entry.reward:
            raise ValueError(f"segment rewards give the subagent launched by {entry.launched_by} two rewards")
    for record, reward in _subagent_calls(records, result, launches):
        assign(record, reward)
    return targeted


def _producing_call(records: Sequence[CallRecord], output: list[dict], index: int) -> CallRecord:
    """The captured call that produced ``output[index]``: by its served item id, else by its content."""
    if index >= len(output):
        raise ValueError(f"output_index {index} is outside the scored response, which has {len(output)} items")
    item = output[index]
    item_id = item.get("id")
    matches = [r for r in records if item_id and any(e.id == item_id for e in r.output_items or [])]
    if not matches:
        # Raises for an item the model did not write, such as a tool result.
        fingerprint = output_item_fingerprint(item)
        matches = [r for r in records if any(e.fingerprint == fingerprint for e in r.output_items or [])]
    if len(matches) != 1:
        raise ValueError(f"output_index {index} matches {len(matches)} captured calls, not exactly one")
    return matches[0]


def _subagent_calls(
    records: Sequence[CallRecord], result: Mapping[str, Any], launches: dict[str, float]
) -> list[tuple[CallRecord, float]]:
    """Each captured call of a launched subagent, with the reward of the nearest launch above it that has one."""
    if not launches:
        return []
    if result.get(NG_AGENT_OBSERVATIONS_KEY) is None:
        raise ValueError(f"launched_by segment rewards require the agent's {NG_AGENT_OBSERVATIONS_KEY}")
    bundle = AgentObservationBundle.model_validate(result[NG_AGENT_OBSERVATIONS_KEY])
    invocations = {record.invocation_id: record for record in bundle.records if isinstance(record, AgentInvocation)}
    unmatched = set(launches) - {invocation.spawned_by_tool_call_id for invocation in invocations.values()}
    if unmatched:
        raise ValueError(f"launched_by names tool calls that launched no observed subagent: {sorted(unmatched)}")

    def deciding_launch(invocation_id: str | None) -> str | None:
        # The bundle has already rejected parent cycles.
        while invocation_id in invocations:
            invocation = invocations[invocation_id]
            if invocation.spawned_by_tool_call_id in launches:
                return invocation.spawned_by_tool_call_id
            invocation_id = invocation.parent_invocation_id
        return None

    by_call_id = {record.model_call_id: record for record in records}
    by_response_id = {record.response_id: record for record in records}
    calls: list[tuple[CallRecord, float]] = []
    used: set[str] = set()
    # A compaction's summary call belongs to the invocation whose context it compacted.
    owners = [*invocations.values(), *(r for r in bundle.records if isinstance(r, ContextCompactionObservation))]
    for owner in owners:
        launch = deciding_launch(owner.invocation_id)
        if launch is None:
            continue
        for reference in owner.model_calls:
            record = by_call_id.get(reference.model_call_id) or by_response_id.get(reference.response_id)
            if record is not None:
                used.add(launch)
                calls.append((record, launches[launch]))
    if unused := set(launches) - used:
        # The agent observed these subagents but reported none of their captured calls.
        logger.warning("launched_by %s matched no captured call; those calls keep reward", sorted(unused))
    return calls
