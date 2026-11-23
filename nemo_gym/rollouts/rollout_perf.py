# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-rollout performance numbers the collector attaches to each row, and the dispatch latency tracker."""

import bisect
import logging
from collections import Counter
from typing import Any, List, Optional

from nemo_gym.trajectory.trajectory_record import (
    NG_TRAJECTORY_KEY,
    ModelCallRef,
    TrajectoryRecord,
)


logger = logging.getLogger(__name__)


NG_PERF_KEY = "ng_perf"


def _build_ng_perf(result: dict[str, Any], *, rollout_latency_ms: Optional[float]) -> Optional[dict[str, Any]]:
    """Assemble the per-rollout ``ng_perf`` summary from ``ng_trajectory``.

    Returns ``None`` (``ng_perf`` stays absent) unless at least one reasoning turn was
    observed: per-turn evidence is needed rather than just raw model-call capture,
    so a rollout collected with observability disabled produces no ``ng_perf`` at all.

    Token fields are summed over every model call referenced by a reasoning-turn
    ``AgentInvocation``. This includes compaction calls whenever the harness also lists
    them in ``AgentInvocation.model_calls``.

    ``num_turns`` counts reasoning turns summed across all invocations (an ``AgentInvocation``
    is one root-agent or subagent conversation that may span many turns). Each invocation
    contributes its explicit ``TrajectoryTurn`` count when the harness emits turn records,
    falling back to its owned model-call count (one assistant response per turn), then to 1
    (an invocation that ran had at least one turn) -- so hybrid trajectories where only some
    invocations report turns still count every conversation.

    ``token_observability_coverage`` reports what fraction of those turns actually resolved to a
    captured call: a turn whose ``ModelCallRef`` was unmatched or ambiguous silently loses its
    tokens from the sums below, and this is the only signal that it happened.
    """
    raw_trajectory = result.get(NG_TRAJECTORY_KEY)
    if not isinstance(raw_trajectory, dict):
        return None
    try:
        trajectory = TrajectoryRecord.model_validate(raw_trajectory)
    except Exception:
        return None
    if not trajectory.invocations:
        return None

    # Index captured calls by both identities ModelCallRef supports, mirroring
    # join_agent_model_calls: a ref may carry model_call_id, or the exact
    # (model_ref, response_id) pair.
    calls_by_id: dict[str, list[int]] = {}
    calls_by_response: dict[tuple[str, str, str], list[int]] = {}
    for index, call in enumerate(trajectory.model_calls):
        if call.model_call_id:
            calls_by_id.setdefault(call.model_call_id, []).append(index)
        call_model_ref = call.response_metadata.model_ref
        call_response_id = call.response_metadata.response_id
        if call_model_ref is not None and call_response_id:
            calls_by_response.setdefault((call_model_ref.type, call_model_ref.name, call_response_id), []).append(
                index
            )

    def _match_call_index(ref: ModelCallRef) -> Optional[int]:
        if ref.model_call_id:
            candidates = [
                index
                for index in calls_by_id.get(ref.model_call_id, [])
                if (
                    ref.model_ref is None or ref.model_ref == trajectory.model_calls[index].response_metadata.model_ref
                )
                and (
                    ref.response_id is None
                    or ref.response_id == trajectory.model_calls[index].response_metadata.response_id
                )
            ]
        elif ref.model_ref is not None and ref.response_id:
            candidates = calls_by_response.get((ref.model_ref.type, ref.model_ref.name, ref.response_id), [])
        else:
            candidates = []
        return candidates[0] if len(candidates) == 1 else None

    # De-duplicate by matched call *position* rather than model_call_id; the id is
    # optional because a (model_ref, response_id)-only match has none. One physical call
    # can't be claimed twice even if two invocations' refs both resolve to it (e.g. a
    # join_agent_model_calls conflict that left a ref attached to its losing
    # invocation, unremoved, just gap-flagged).
    seen_positions: set[int] = set()
    owned_calls = []
    owned_calls_by_invocation: Counter = Counter()
    for invocation in trajectory.invocations:
        for ref in invocation.model_calls:
            index = _match_call_index(ref)
            if index is None or index in seen_positions:
                continue
            seen_positions.add(index)
            owned_calls.append(trajectory.model_calls[index])
            owned_calls_by_invocation[invocation.invocation_id] += 1
    tool_calls_by_invocation = Counter(tool.invocation_id for tool in trajectory.tool_calls)
    num_tool_calls = sum(
        tool_calls_by_invocation.get(invocation.invocation_id, 0) for invocation in trajectory.invocations
    )

    def _sum_tokens(attr: str) -> Optional[int]:
        values = [
            getattr(call.token_stats, attr) for call in owned_calls if getattr(call.token_stats, attr) is not None
        ]
        return sum(values) if values else None

    turns_by_invocation = Counter(turn.invocation_id for turn in trajectory.turns)
    num_turns = sum(
        turns_by_invocation.get(invocation.invocation_id, 0)
        or owned_calls_by_invocation.get(invocation.invocation_id, 0)
        or 1
        for invocation in trajectory.invocations
    )
    ng_perf: dict[str, Any] = {
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "token_observability_coverage": min(1.0, len(owned_calls) / num_turns),
    }
    for ng_perf_key, token_stats_attr in (
        ("prompt_tokens", "prompt_tokens"),
        ("cached_prompt_tokens", "cached_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("reasoning_tokens", "reasoning_tokens"),
    ):
        value = _sum_tokens(token_stats_attr)
        if value is not None:
            ng_perf[ng_perf_key] = value

    if isinstance(rollout_latency_ms, (int, float)):
        ng_perf["total_latency_ms"] = rollout_latency_ms

    return ng_perf


def _attach_ng_perf(
    result: dict[str, Any], *, observability_enabled: bool, rollout_latency_ms: Optional[float] = None
) -> None:
    if not observability_enabled:
        # ng_perf stays absent entirely when observability is off (OQ4): a caller who
        # disabled it only wants the final score, not partial/best-effort perf evidence.
        return
    try:
        ng_perf = _build_ng_perf(result, rollout_latency_ms=rollout_latency_ms)
    except Exception:
        logger.warning("Could not assemble ng_perf for a rollout.", exc_info=True)
        return
    if ng_perf is not None:
        result[NG_PERF_KEY] = ng_perf


class DispatchLatencyTracker:
    """Per-task latency observed by the dispatcher, and the drain margin from it.

    Rollouts/hr is the number operators watch, and on its own it is misleading:
    raising concurrency raises aggregate throughput while making every individual
    task slower, so the run looks healthier right up until tasks start breaching
    their per-task timeout. Reporting the latency percentiles next to the rate
    makes that trade visible while there is still time to react to it.
    """

    def __init__(self) -> None:
        # Kept sorted: the adaptive drain margin reads a quantile before every dispatch.
        self._durations: List[float] = []
        self._drained = 0

    def record(self, seconds: float) -> None:
        if seconds > 0:
            bisect.insort(self._durations, seconds)

    def record_failure(self, seconds: float) -> None:
        """Record a failed attempt only when it ran at least as long as the median so far.

        A task that timed out or died late shows how long tasks can run, so it can raise the
        drain margin. One that failed at once says nothing about duration and would drag the
        p75 toward zero, so it is ignored, as is any failure before a real completion exists.
        """
        median = self.quantile(0.5)
        if median is not None and seconds >= median:
            self.record(seconds)

    def record_drained(self) -> None:
        self._drained += 1

    @property
    def drained(self) -> int:
        return self._drained

    def quantile(self, q: float) -> Optional[float]:
        if not self._durations:
            return None
        ordered = self._durations
        pos = (len(ordered) - 1) * q
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] * (1 - (pos - low)) + ordered[high] * (pos - low)

    def drain_margin(self, configured: Optional[float]) -> Optional[float]:
        """Seconds of headroom a task needs before it is worth starting.

        An explicit value wins. Otherwise adapt to this run's own p75, once
        enough tasks have finished for that to mean anything.
        """
        if configured is not None:
            return configured
        if len(self._durations) < 5:
            return None
        return self.quantile(0.75)

    def summary(self) -> str:
        if not self._durations:
            lines = ["No completed rollouts to report latency for."]
        else:
            total = sum(self._durations)
            lines = [
                f"Per-task latency over {len(self._durations)} finished attempt(s): "
                f"median {self.quantile(0.5) / 60:.1f} min, "
                f"p90 {self.quantile(0.9) / 60:.1f} min, "
                f"p99 {self.quantile(0.99) / 60:.1f} min, "
                f"max {max(self._durations) / 60:.1f} min",
                f"Task-time observed: {total / 3600:.1f} task-hours",
            ]
        if self._drained:
            lines.append(
                f"Drained (not dispatched, no time left in the budget): {self._drained}. "
                "They are not attempts and will be re-dispatched on resume."
            )
        return "\n".join(lines)
