"""Dataclass contracts for the trajectory-health spike.

Trimmed from evalpipeline's backend-agnostic contracts (backends/base.py):
only the step/health/capture surfaces the health verdict reads survive;
the ATIF projection (Agent, Extra, from_atif/to_atif) is deliberately
absent. Field names and None-vs-zero semantics are preserved verbatim so
per-rollout verdicts compare field-by-field against the evalpipeline
ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Metrics:
    """Per-agent-step token metrics.

    Optional token counts stay None when the producer did not report them —
    None always means unknown, never zero.

    Fields:
        prompt_tokens: Full prompt tokens processed by that model turn.
        completion_tokens: Completion tokens emitted by that model turn.
        total_tokens: Provider-reported total, cross-checks prompt+completion.
        cached_tokens: Prompt tokens served from a prefix cache.
        reasoning_tokens: Hidden chain-of-thought tokens, billed but unseen.
        extra: Non-token per-call signals (latency_ms, latency_ttft_ms,
            finish_reason, cache_creation_tokens, cache_hit).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCall:
    """Tool call requested by an agent step.

    Fields:
        tool_call_id: Stable tool-call identifier from the producer.
        function_name: Tool or function name requested by the agent.
        arguments: JSON-compatible function arguments, possibly empty.
        extra: Producer residue, e.g. the raw un-normalized argument
            string as transmitted.
    """

    tool_call_id: str
    function_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class ObservationResult:
    """One observed result attached to a step.

    Fields:
        source_call_id: The tool_call_id this result answers.
        content: Observed payload, when carried inline.
        is_error: True when the producer marked the result as an error;
            None means unknown — never synthesized.
        extra: Producer residue.
    """

    source_call_id: str | None = None
    content: Any = None
    is_error: bool | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class Observation:
    """Results observed by a step.

    Fields:
        results: Ordered observed results.
    """

    results: tuple[ObservationResult, ...] = ()


@dataclass(frozen=True)
class Step:
    """Trajectory step in the causal projection.

    A tool result is never its own step: it lands in the issuing agent
    step's observation, correlated by tool_call_id.

    Fields:
        step_id: Sequential step identifier assigned by the projection.
        source: Step source: system, user, or agent.
        message: Message payload passed through from the rollout record.
        timestamp: ISO-8601 timestamp, when known.
        model_name: Served model for this turn (agent steps).
        reasoning_content: Explicit chain-of-thought text (agent steps).
        metrics: Optional token metrics, allowed only for agent steps.
        tool_calls: Tool calls, allowed only for agent steps.
        observation: Observed results, allowed on agent and system steps.
        llm_call_count: Model calls behind this step: 0 = deterministic
            dispatch (no model), 1 = one inference, >1 = aggregated,
            None = unknown.
        extra: Producer residue and the capture-binding block
            (model_call_id, call_index, dialect, status_code, ...).
    """

    step_id: int
    source: str
    message: Any
    timestamp: str | None = None
    model_name: str | None = None
    reasoning_content: str | None = None
    metrics: Metrics | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    observation: Observation | None = None
    llm_call_count: int | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class Identity:
    """Rollout identity attached from the rollout row.

    Fields:
        dataset: Grouping scope for the run (the artifacts directory
            name here; evalpipeline uses the config stem).
        task: Stable problem key within the dataset scope (Gym task index).
        trial: Repeat identifier (Gym rollout index).
        attempt: Retry index of the same trial; absent (0) on a clean run.
    """

    dataset: Any
    task: Any
    trial: Any
    attempt: Any = 0


@dataclass(frozen=True)
class Health:
    """Health verdict computed over agent steps, plus the optional
    step-to-capture reconciliation against the ground-truth capture (the
    transport-side record of every model call). Every signal is a flag
    for later triage, never a gate beyond is_clean; capture fields stay
    None when no capture was ingested.

    Field semantics are identical to evalpipeline's Health (see
    backends/base.py in nvidia-eval-factory-benchmarking).
    """

    agent_step_count: int
    tool_call_count: int
    total_prompt_tokens: int
    zero_token_turn_count: int
    missed_metrics_count: int
    has_no_agent_steps: bool
    has_zero_token_turns: bool
    is_fully_zero: bool
    is_clean: bool
    anomalous_zero_token_count: int = 0
    unknown_zero_token_count: int = 0
    deterministic_step_count: int = 0
    hollow_step_count: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int | None = None
    total_cached_tokens: int | None = None
    capture_found: bool = False
    unique_capture_calls: int | None = None
    step_capture_delta: int | None = None
    duplicate_captured_call_count: int | None = None
    duplicate_captured_call_all_count: int | None = None
    capture_prompt_tokens: int | None = None
    capture_completion_tokens: int | None = None
    token_sums_match: bool | None = None
    retry_sessions: int | None = None
    retry_dropped_calls: int | None = None
    non_policy_calls: int | None = None
    failed_captured_calls: int | None = None
    last_captured_call_non_200: bool | None = None
    binding_disagreement: bool | None = None
    discriminator_disagreement: bool | None = None
    capture_integrity_mismatch: bool | None = None


@dataclass(frozen=True)
class Trajectory:
    """One rollout's causal step projection plus identity.

    Fields:
        session_id: Stable rollout identifier (capture rollout_id or
            "<task>-<trial>").
        steps: Ordered causal steps for the rollout.
        final_metrics: Producer-reported trajectory totals; no
            policy-scoped final_metrics exists in Gym (the in-row capture
            aggregate sums policy AND user-sim calls), so this stays None
            (two-way token reconciliation).
        identity: Rollout identity for capture-sidecar lookup.
    """

    session_id: str
    steps: tuple[Step, ...]
    final_metrics: dict[str, Any] | None = None
    identity: Identity | None = None


@dataclass(frozen=True)
class WireCall:
    """One upstream model call from the ground-truth wire capture.

    Fields:
        call_id: Stable call identifier (Gym model_call_id).
        status_code: HTTP status; None means the call never got a response.
        request_hash: Dedup key over the request body; byte-identical
            replays share it. None when the producer recorded no body.
        prompt_tokens: Provider-reported prompt tokens; None unknown.
        completion_tokens: Provider-reported completion tokens; None unknown.
        tool_call_ids: Tool-call ids the model requested in this call;
            binds the call to the agent step that carries them.
    """

    call_id: str
    status_code: int | None = None
    request_hash: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class WireRecord:
    """Selected ground-truth wire calls for one rollout, plus selection
    residue. Loading removes retries (attempt selection) and non-policy
    calls before returning; success filtering, dedup, binding, and
    counting are neutral and live in checks.py.

    Fields:
        found: True when a capture existed for this rollout.
        calls: Policy calls of the selected attempt, in capture order.
        retry_sessions: Attempt files dropped by selection.
        dropped_calls: Calls dropped with those retries.
        non_policy_calls: Calls excluded as non-policy (e.g. user-sim).
        discriminator_disagreement: Policy discriminators disagreed.
        capture_integrity_mismatch: Reserved; reads False today.
    """

    found: bool = False
    calls: tuple[WireCall, ...] = ()
    retry_sessions: int = 0
    dropped_calls: int = 0
    non_policy_calls: int = 0
    discriminator_disagreement: bool = False
    capture_integrity_mismatch: bool = False


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            yield line_number, row
