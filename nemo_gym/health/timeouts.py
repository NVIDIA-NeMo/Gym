# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``run.stats.timeouts``: which rollouts ran out of time, and where their time went.

The section is a report. It changes no row, reward, score or health verdict, and
``gym eval health-check`` recomputes it from the saved rows.

A rollout counts as timed out on one of two kinds of evidence:

- ``typed``: a :class:`TimeoutEvent` in the row's ``timeouts`` list, added by the component that enforces
  the limit. The operation records in ``ng_trajectory`` have no field for it, so the event goes on the row.
- ``legacy``: a field that a Gym component writes today when its own deadline fires. Each one is listed in
  :data:`LEGACY_TIMEOUT_SIGNALS` together with the component that writes it.

How long a rollout took never makes it count as timed out.

For the timed-out rollouts, the section also splits their time between the policy model, other model
servers (a judge, a user simulator) and tools, and reports the policy's serving speed. A run that times out
because the deployment is too slow can then be retuned.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError


if TYPE_CHECKING:
    from nemo_gym.health.types import HealthCheckResult

logger = logging.getLogger(__name__)

# Rollout collection stamps each rollout's `/run` wall clock, success or failure, under this key.
ELAPSED_S_KEY = "_ng_elapsed_s"

# Calls to any other model server, such as a judge or a user simulator, count as other-model time.
POLICY_MODEL_SERVER = "policy_model"

Evidence = Literal["typed", "legacy"]


class TimeoutEvent(BaseModel):
    """One confirmed firing of one time limit, in the ``timeouts`` list of the rollout's row.

    An event listed twice keeps its ``event_id``, so one firing counts once.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    limit: Optional[str] = Field(default=None, min_length=1)
    enforcer: str = Field(min_length=1)
    evidence: Literal["internal", "external"]
    configured_s: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    elapsed_s: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    fired_at: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)


class LegacySignal(NamedTuple):
    """A row field that a Gym component sets when its own deadline fires; a dotted name reaches a nested field.

    ``value`` is the field's value on a timeout, or None for a flag that is set when nonzero. ``written_as``
    is the source text in ``producer`` that writes it; the tests check that it is still there.
    """

    field: str
    value: Optional[str]
    producer: str
    written_as: str

    def matches(self, row: Mapping[str, Any]) -> bool:
        *parents, field = self.field.split(".")
        for parent in parents:
            row = _mapping(row.get(parent))
        found = row.get(field)
        return bool(found) if self.value is None else found == self.value

    @property
    def name(self) -> str:
        return self.field if self.value is None else f"{self.field}={self.value}"


LEGACY_TIMEOUT_SIGNALS = (
    LegacySignal(
        "_ng_failure_class", "agent_timeout", "resources_servers/gdpval/app.py", "NG_FAILURE_CLASS_KEY: AGENT_TIMEOUT"
    ),
    LegacySignal(
        "_ng_failure_class", "timeout_exceeded", "responses_api_agents/pinchbench/app.py", 'return "timeout_exceeded"'
    ),
    LegacySignal(
        "_ng_failure_message",
        "Episode timed out",
        "nemo_gym/server_runtime/base_environment_server.py",
        'failure_reason="Episode timed out"',
    ),
    LegacySignal("agent_timed_out", None, "responses_api_agents/codex_agent/app.py", '"agent_timed_out": timed_out}'),
    LegacySignal("agent_timeout_error", None, "responses_api_agents/harbor_agent/app.py", "agent_timeout_error=int("),
    LegacySignal(
        "eval_timed_out", None, "resources_servers/swebench_pro/app.py", '"eval_timed_out": result.timed_out'
    ),
    # BrowseComp's per-rollout `rollout_timeout_s`.
    LegacySignal("timed_out", None, "responses_api_agents/browsecomp_agent/app.py", 'update={"timed_out": int(bool('),
    # Terminus 2's run budget (the task's `[agent].timeout_sec`), recorded on the response it returns.
    LegacySignal(
        "response.metadata.terminus2_outcome",
        "timeout",
        "responses_api_agents/terminus_2_sandboxed_agent/app.py",
        'outcome = "timeout"',
    ),
    # Terminal-Bench's agent termination record, which the row carries.
    LegacySignal(
        "termination.reason",
        "timeout",
        "resources_servers/terminal_bench_4/app.py",
        'session.termination = AgentTermination(reason="timeout"',
    ),
)


class TimedOut(NamedTuple):
    evidence: Evidence
    limit: Optional[str]
    enforcer: Optional[str]


@dataclass(frozen=True)
class RolloutTiming:
    """What one rollout contributes to the section, read from its final record.

    A busy time is None when the rollout does not measure it: a call without timing, or no tool timing.
    ``active_s`` is how long a call of any kind ran, so a stretch in which two kinds overlap counts once.
    """

    failed: bool
    timed_out: Optional[TimedOut]
    event_ids: tuple[str, ...]
    invalid_events: int
    elapsed_s: Optional[float]
    has_model_calls: bool
    policy_s: Optional[float]
    other_model_s: Optional[float]
    tool_s: Optional[float]
    active_s: Optional[float]
    policy_calls: int
    policy_tokens_per_s: tuple[float, ...]


@dataclass(frozen=True)
class _Span:
    """When a call ran. A call that reports only its duration has no start or end."""

    start: Optional[float]
    end: Optional[float]
    duration_s: Optional[float]

    @property
    def is_timed(self) -> bool:
        # A call that ends before it starts, as a clock stepped back can record, or lasts less than nothing has
        # no usable timing.
        if self.start is not None and self.end is not None:
            return self.start <= self.end
        return self.duration_s is not None and self.duration_s >= 0


# The collector's join attaches one of these when it read a capture only in part.
_PARTIAL_CAPTURE_GAPS = frozenset(
    {"model_call_capture_records_unreadable", "model_call_capture_pending", "model_call_capture_incomplete"}
)
# A trajectory with one of these did not observe its tool calls, so its tool list is not a measured history.
_TOOL_TIMING_GAPS = frozenset({"tool_timing_unavailable", "agent_artifact_unavailable", "observation_capture_failed"})
# A transcript that could not be read leaves an empty tool list unmeasured. Tools read from another source, as Codex
# reads its native transcript when its CLI events are missing, stay measured.
_TRANSCRIPT_GAPS = frozenset({"agent_transcript_unavailable"})


def read_rollout_timing(row: Mapping[str, Any]) -> RolloutTiming:
    """Everything the section needs from one rollout's final record.

    A malformed record never raises: a value of the wrong type reads as not measured, and the record checks
    report the record itself. A capture read only in part, or with a call record that is not an object, does not
    measure the model time, so its calls are left out.
    """
    row = _mapping(row)
    trajectory = _mapping(row.get("ng_trajectory"))
    capture = _mapping(row.get("ng_model_call_capture"))
    partial = not _PARTIAL_CAPTURE_GAPS.isdisjoint(_gap_codes(capture.get("gaps")))
    calls = capture.get("calls")
    whole = isinstance(calls, list) and all(isinstance(call, Mapping) for call in calls)
    model_calls = list(calls) if whole and not partial else []
    # Collection records which server the agent used as its policy; records without it follow the convention.
    policy_server = row.get("ng_policy_model_server")
    policy_server = policy_server if isinstance(policy_server, str) and policy_server else POLICY_MODEL_SERVER
    policy_calls = [call for call in model_calls if _is_policy_call(call, policy_server)]
    other_calls = [call for call in model_calls if not _is_policy_call(call, policy_server)]
    events, invalid_events = _timeout_events(row)
    speeds = (_tokens_per_s(call) for call in policy_calls)
    tool_spans = _tool_spans(trajectory)
    elapsed_s = _number(row.get(ELAPSED_S_KEY))
    return RolloutTiming(
        failed=row.get("_ng_failure_class") is not None,
        timed_out=classify_timeout(row),
        event_ids=tuple(event.event_id for event in events),
        invalid_events=invalid_events,
        elapsed_s=elapsed_s if elapsed_s and elapsed_s > 0 else None,
        has_model_calls=bool(model_calls),
        policy_s=_busy_seconds([_model_call_span(call) for call in policy_calls]),
        other_model_s=_busy_seconds([_model_call_span(call) for call in other_calls]),
        tool_s=None if tool_spans is None else _busy_seconds(tool_spans),
        active_s=None if tool_spans is None else _busy_seconds([*map(_model_call_span, model_calls), *tool_spans]),
        policy_calls=len(policy_calls),
        policy_tokens_per_s=tuple(speed for speed in speeds if speed is not None),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    """A JSON object, or an empty one in place of any other value."""
    return value if isinstance(value, Mapping) else {}


def _objects(value: Any) -> list[Mapping[str, Any]]:
    """The objects in a JSON list. Any other value reads as an empty list, and an item that is not an object
    is skipped."""
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _gap_codes(value: Any) -> set[str]:
    """The codes in a JSON list of gaps. A code that is not a string names no gap."""
    return {code for gap in _objects(value) if isinstance(code := gap.get("code"), str)}


def _number(value: Any) -> Optional[float]:
    """A JSON number, or None in place of any other value. A boolean is not a number here."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _timeout_events(row: Mapping[str, Any]) -> tuple[list[TimeoutEvent], int]:
    """The valid events in the row's ``timeouts`` list, once per ``event_id``, and how many entries were not
    valid events. Only the row carries events: the trajectory's operation records have no field for them, and
    collection drops a trajectory that adds one."""
    raw_events = row.get("timeouts")
    if raw_events is None:
        return [], 0
    if not isinstance(raw_events, list):
        return [], 1
    events: dict[str, TimeoutEvent] = {}
    invalid = 0
    for raw_event in raw_events:
        try:
            event = TimeoutEvent.model_validate(raw_event)
        except ValidationError:
            invalid += 1
        else:
            events.setdefault(event.event_id, event)
    return list(events.values()), invalid


def classify_timeout(row: Mapping[str, Any]) -> Optional[TimedOut]:
    """Why a rollout's record counts as timed out, or None: the one classifier for this section and the run timing
    table. A typed event wins over a legacy signal; of several events, the first one names the limit."""
    events, _ = _timeout_events(row)
    if events:
        return TimedOut("typed", events[0].limit, events[0].enforcer)
    for signal in LEGACY_TIMEOUT_SIGNALS:
        if signal.matches(row):
            return TimedOut("legacy", signal.name, None)
    return None


def _is_policy_call(call: Mapping[str, Any], policy_model_server: str = POLICY_MODEL_SERVER) -> bool:
    model_ref = call.get("model_ref")
    # Captures made before `model_ref` existed held only policy calls.
    return model_ref is None or _mapping(model_ref).get("name") == policy_model_server


def _model_call_span(call: Mapping[str, Any]) -> _Span:
    latency_ms = _number(call.get("latency_total_ms"))
    return _Span(
        start=_number(call.get("started_at")),
        end=_number(call.get("completed_at")),
        duration_s=latency_ms / 1000 if latency_ms is not None else None,
    )


def _tokens_per_s(call: Mapping[str, Any]) -> Optional[float]:
    """End-to-end output speed of one call, so time in the server queue and in prefill counts too. A call that
    produced no output is measured at zero; one without a token count or a positive latency is not measured."""
    tokens, latency_ms = _number(call.get("tokens_out")), _number(call.get("latency_total_ms"))
    if tokens is None or tokens < 0 or latency_ms is None or latency_ms <= 0:
        return None
    return tokens / (latency_ms / 1000)


def _tool_spans(trajectory: Mapping[str, Any]) -> Optional[list[_Span]]:
    """When the tool calls ran, or None when the trajectory does not measure it.

    An empty ``tool_calls`` list is a measured zero. A missing or malformed list leaves the time unknown, and so
    do a tool call without timing (see ``_busy_seconds``) and a gap saying the tool calls were not observed:
    ``tool_timing_unavailable``, ``agent_artifact_unavailable`` or ``observation_capture_failed``. An
    ``agent_transcript_unavailable`` gap leaves an empty list unknown too, but not tools read from another source.
    """
    tool_calls = trajectory.get("tool_calls")
    if not isinstance(tool_calls, list) or not all(isinstance(call, Mapping) for call in tool_calls):
        return None
    codes = _gap_codes(trajectory.get("gaps"))
    if not _TOOL_TIMING_GAPS.isdisjoint(codes) or (not tool_calls and not _TRANSCRIPT_GAPS.isdisjoint(codes)):
        return None
    return [_tool_call_span(call) for call in tool_calls]


def _tool_call_span(call: Mapping[str, Any]) -> _Span:
    duration_ms = _number(call.get("duration_ms"))
    return _Span(
        start=_number(call.get("started_at")),
        end=_number(call.get("completed_at")),
        duration_s=duration_ms / 1000 if duration_ms is not None else None,
    )


def _busy_seconds(spans: list[_Span]) -> Optional[float]:
    """How long at least one of the calls was running, or None when a call has no timing.

    Overlapping calls, such as parallel subagents, count once. A call that reports only its duration cannot be
    placed against other calls, so it is measured alone and leaves a union with other calls unknown.
    """
    if not all(span.is_timed for span in spans):
        return None
    if len(spans) > 1 and any(span.start is None or span.end is None for span in spans):
        return None
    busy = sum(span.duration_s for span in spans if span.start is None or span.end is None)
    covered_until = None
    for start, end in sorted(
        (span.start, span.end) for span in spans if span.start is not None and span.end is not None
    ):
        if covered_until is None or start > covered_until:
            busy += end - start
            covered_until = end
        elif end > covered_until:
            busy += end - covered_until
            covered_until = end
    return busy


class TimeSplit(BaseModel):
    """How the elapsed time of the timed-out rollouts with captured model calls divides.

    A share is that kind's busy time over the rollouts' summed elapsed time, or None when one of the
    rollouts does not measure it. The kinds can overlap, as when a tool calls the policy model, so
    ``untracked`` is the share in which no measured call ran, not one minus the shares.
    """

    rollouts_with_captures: int = 0
    policy_model: Optional[float] = None
    other_model: Optional[float] = None
    tool: Optional[float] = None
    untracked: Optional[float] = None
    # Serving speed matters at its slow end, so the 5th percentile is reported with the median.
    policy_output_tokens_per_s: dict[str, Optional[float]] = {"p50": None, "p5": None}
    policy_calls_per_rollout: dict[str, Optional[float]] = {"p50": None, "p95": None}


class TimeoutsSection(BaseModel):
    """The ``run.stats.timeouts`` section of the run health summary."""

    rollouts: int
    timed_out: int
    by_evidence: dict[Evidence, int]
    by_limit: dict[str, int]
    by_enforcer: dict[str, int]
    events: int
    invalid_events: int
    # Failures that no evidence ties to a time limit: not known to be timeouts, and not known not to be.
    failed_without_signal: int
    elapsed_s: dict[str, Optional[float]]
    time_split: TimeSplit


def summarize_timeouts(timings: Iterable[RolloutTiming], attempts: Iterable[RolloutTiming] = ()) -> TimeoutsSection:
    """The section for a run, from the timing of every rollout's final record.

    A limit or enforcer named only by ``attempts``, the run's earlier failed attempts, is counted 0, as the failure
    export counts each failure class the sidecars name: a reopened tracking run keeps a counter's last value until
    it is sent.
    """
    timings = list(timings)
    timed_out = [timing for timing in timings if timing.timed_out]
    named = [timing.timed_out for timing in attempts if timing.timed_out]
    by_limit = Counter(dict.fromkeys((t.limit or "unknown" for t in named), 0))
    by_limit.update(t.timed_out.limit or "unknown" for t in timed_out)
    by_enforcer = Counter(dict.fromkeys((t.enforcer for t in named if t.enforcer), 0))
    by_enforcer.update(t.timed_out.enforcer for t in timed_out if t.timed_out.enforcer)
    return TimeoutsSection(
        rollouts=len(timings),
        timed_out=len(timed_out),
        by_evidence={
            evidence: sum(t.timed_out.evidence == evidence for t in timed_out) for evidence in ("typed", "legacy")
        },
        by_limit=dict(sorted(by_limit.items())),
        by_enforcer=dict(sorted(by_enforcer.items())),
        events=sum(len(t.event_ids) for t in timed_out),
        invalid_events=sum(t.invalid_events for t in timings),
        failed_without_signal=sum(t.failed and not t.timed_out for t in timings),
        elapsed_s=_percentiles([t.elapsed_s for t in timed_out if t.elapsed_s], p50=50, p95=95),
        time_split=_time_split(timed_out),
    )


def _time_split(timed_out: list[RolloutTiming]) -> TimeSplit:
    measured = [t for t in timed_out if t.elapsed_s and t.has_model_calls]
    if not measured:
        return TimeSplit()
    policy = _share([t.policy_s for t in measured], measured)
    other_model = _share([t.other_model_s for t in measured], measured)
    tool = _share([t.tool_s for t in measured], measured)
    active = _share([t.active_s for t in measured], measured)
    return TimeSplit(
        rollouts_with_captures=len(measured),
        policy_model=policy,
        other_model=other_model,
        tool=tool,
        untracked=None if active is None else round(max(0.0, 1 - active), 4),
        policy_output_tokens_per_s=_percentiles(
            [speed for t in measured for speed in t.policy_tokens_per_s], p50=50, p5=5
        ),
        policy_calls_per_rollout=_percentiles([t.policy_calls for t in measured], p50=50, p95=95),
    )


def _share(busy_times: list[Optional[float]], rollouts: list[RolloutTiming]) -> Optional[float]:
    """Summed busy time over summed elapsed time, or None when a rollout does not measure it.

    A rollout's busy time is capped at its elapsed time, so clock skew between servers cannot push a share
    past what the rollout took.
    """
    if None in busy_times:
        return None
    busy = sum(min(busy_s, rollout.elapsed_s) for busy_s, rollout in zip(busy_times, rollouts))
    return round(busy / sum(rollout.elapsed_s for rollout in rollouts), 4)


def _percentiles(values: list[float], **percentiles: int) -> dict[str, Optional[float]]:
    if not values:
        return dict.fromkeys(percentiles)
    return {name: round(float(np.percentile(values, q)), 4) for name, q in percentiles.items()}


def export_keys(section: Mapping[str, Any]) -> dict[str, float]:
    """The section's numbers as flat exporter keys, such as ``health/timeouts/by_evidence/typed``.

    A null value is left out rather than exported as 0, so an unknown stays unknown on the tracking server.
    """
    return _flatten(section, "health/timeouts")


def export_run_health_timeouts(health: "HealthCheckResult") -> None:
    """Send the run's timeouts section to the configured exporters (MLflow, W&B).

    ``gym eval run`` and ``gym eval aggregate`` call this wherever they compute the run's health. A failure,
    in reading the section or in exporting it, is logged and never stops the run.
    """
    # Imported here: health-check workers load this module and need none of the exporters.
    from nemo_gym.artifacts.exporters import export_metrics

    try:
        export_metrics(export_keys(health.summary["run"]["stats"]["timeouts"]))
    except Exception:
        logger.exception("Could not export the run health timeouts.")


def _flatten(values: Mapping[str, Any], prefix: str) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = value
    return flat
