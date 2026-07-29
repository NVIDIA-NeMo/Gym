"""Backend-neutral trajectory health verdict.

Ports evalpipeline's verify()/_scan_agent_steps/_reconcile
(artifacts.py) unchanged. Every signal is a flag for later triage —
a flagged rollout is suspicious, never automatically wrong — and no
flag beyond the clean-rollout verdict gates anything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from verify_spike.models import Health, Step, Trajectory, WireCall, WireRecord


@dataclass(frozen=True)
class _StepScan:
    """Zero-token classification over ordered agent steps.

    Fields:
        anomalous_zero: Zero-token steps whose llm_call_count > 0.
        unknown_zero: Zero-token steps with unknown llm_call_count.
        deterministic: Steps with llm_call_count == 0 (no model called).
        hollow: Zero-token steps with empty message and no tool calls.
        missed_metrics: Non-deterministic steps with no metrics object.
        flaggable: Steps eligible for the zero-token test.
    """

    anomalous_zero: int
    unknown_zero: int
    deterministic: int
    hollow: int
    missed_metrics: int
    flaggable: int


def verify(
    trajectory: Trajectory,
    *,
    token_semantics: str = "unknown",
    capture: WireRecord | None = None,
) -> Health:
    """Compute the recall-first health verdict for one trajectory.

    token_semantics: "per_call" tests zero-token steps on absolute
    per-step tokens; "unknown" uses the prompt-delta body that also
    handles cumulative-reporting solvers. When a ground-truth capture
    (the transport-side record of the rollout's model calls) is
    supplied, the step-to-capture reconciliation fields are populated;
    capture signals are flags for triage; they never gate is_clean.
    """
    agent_steps = tuple(
        step for step in trajectory.steps if step.source == "agent"
    )
    scan = _scan_agent_steps(agent_steps, token_semantics)
    zero_token_turn_count = scan.anomalous_zero + scan.unknown_zero
    # Deterministic-dispatch steps (llm_call_count == 0, e.g. a scripted
    # seed greeting) witness no model activity: a rollout whose only
    # agent steps are deterministic did no agent work.
    has_no_agent_steps = not any(
        step.llm_call_count != 0 for step in agent_steps
    )
    has_zero_token_turns = zero_token_turn_count > 0
    health = Health(
        agent_step_count=len(agent_steps),
        tool_call_count=sum(len(step.tool_calls) for step in trajectory.steps),
        total_prompt_tokens=sum(
            step.metrics.prompt_tokens
            for step in agent_steps
            if step.metrics is not None
        ),
        total_completion_tokens=sum(
            step.metrics.completion_tokens
            for step in agent_steps
            if step.metrics is not None
        ),
        total_reasoning_tokens=_sum_known(
            step.metrics.reasoning_tokens
            for step in agent_steps
            if step.metrics is not None
        ),
        total_cached_tokens=_sum_known(
            step.metrics.cached_tokens
            for step in agent_steps
            if step.metrics is not None
        ),
        zero_token_turn_count=zero_token_turn_count,
        missed_metrics_count=scan.missed_metrics,
        has_no_agent_steps=has_no_agent_steps,
        has_zero_token_turns=has_zero_token_turns,
        is_fully_zero=scan.flaggable > 0
        and zero_token_turn_count == scan.flaggable,
        is_clean=(
            not has_no_agent_steps
            and not has_zero_token_turns
            and scan.missed_metrics == 0
        ),
        anomalous_zero_token_count=scan.anomalous_zero,
        unknown_zero_token_count=scan.unknown_zero,
        deterministic_step_count=scan.deterministic,
        hollow_step_count=scan.hollow,
    )
    if capture is None or not capture.found:
        return health
    return replace(health, **_reconcile(trajectory, agent_steps, capture))


def _scan_agent_steps(
    agent_steps: tuple[Step, ...],
    token_semantics: str,
) -> _StepScan:
    if token_semantics not in ("per_call", "unknown"):
        raise ValueError(f"unknown token_semantics: {token_semantics}")
    previous_prompt: int | None = None
    anomalous = unknown = deterministic = hollow = missed = flaggable = 0
    for step in agent_steps:
        if step.llm_call_count == 0:
            # Deterministic dispatch: no model was called; zero tokens
            # and absent metrics are both valid here.
            deterministic += 1
        elif step.metrics is None:
            missed += 1
        else:
            flaggable += 1
            prompt_tokens = step.metrics.prompt_tokens
            completion_tokens = step.metrics.completion_tokens
            if token_semantics == "per_call":
                input_tokens = prompt_tokens
            else:
                if previous_prompt is None:
                    input_tokens = prompt_tokens
                else:
                    input_tokens = prompt_tokens - previous_prompt
                    if input_tokens < 0:
                        # Negative delta means context was freed
                        # (compaction/reset); fall back to absolute.
                        input_tokens = prompt_tokens
                previous_prompt = prompt_tokens
            if input_tokens == 0 and completion_tokens == 0:
                if step.llm_call_count is not None and step.llm_call_count > 0:
                    anomalous += 1
                else:
                    unknown += 1
                if _is_hollow(step):
                    hollow += 1
    return _StepScan(
        anomalous_zero=anomalous,
        unknown_zero=unknown,
        deterministic=deterministic,
        hollow=hollow,
        missed_metrics=missed,
        flaggable=flaggable,
    )


def _is_hollow(step: Step) -> bool:
    return step.message in (None, "", [], ()) and not step.tool_calls


def _reconcile(
    trajectory: Trajectory,
    agent_steps: tuple[Step, ...],
    capture: WireRecord,
) -> dict[str, Any]:
    """Backend-neutral step-to-capture reconciliation.

    The capture is ground truth. Ordering is load-bearing: the
    success filter runs BEFORE dedup, else a failed call and its
    byte-identical successful retry collapse onto the failed row.
    """
    successful = tuple(call for call in capture.calls if _is_successful(call))
    unique = _dedup(successful)
    countable = tuple(
        step
        for step in agent_steps
        if step.llm_call_count is None or step.llm_call_count >= 1
    )
    capture_prompt = _sum_known(call.prompt_tokens for call in unique)
    capture_completion = _sum_known(call.completion_tokens for call in unique)
    return {
        "capture_found": True,
        "unique_capture_calls": len(unique),
        "step_capture_delta": len(unique) - len(countable),
        "duplicate_captured_call_count": len(successful) - len(unique),
        "duplicate_captured_call_all_count": _duplicate_all(capture.calls),
        "capture_prompt_tokens": capture_prompt,
        "capture_completion_tokens": capture_completion,
        "token_sums_match": _token_sums_match(
            countable,
            capture_prompt,
            capture_completion,
            trajectory.final_metrics,
        ),
        "retry_sessions": capture.retry_sessions,
        "retry_dropped_calls": capture.dropped_calls,
        "non_policy_calls": capture.non_policy_calls,
        "failed_captured_calls": len(capture.calls) - len(successful),
        "last_captured_call_non_200": bool(capture.calls)
        and not _is_successful(capture.calls[-1]),
        "binding_disagreement": _binding_disagreement(countable, unique),
        "discriminator_disagreement": capture.discriminator_disagreement,
        "capture_integrity_mismatch": capture.capture_integrity_mismatch,
    }


def _is_successful(call: WireCall) -> bool:
    return call.status_code is not None and 200 <= call.status_code < 400


def _dedup(calls: tuple[WireCall, ...]) -> tuple[WireCall, ...]:
    seen: set = set()
    unique: list[WireCall] = []
    for index, call in enumerate(calls):
        key: Any = (
            call.request_hash
            if call.request_hash is not None
            else ("_no_hash", index)
        )
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return tuple(unique)


def _duplicate_all(calls: tuple[WireCall, ...]) -> int:
    hashes = [
        call.request_hash for call in calls if call.request_hash is not None
    ]
    return len(hashes) - len(set(hashes))


def _sum_known(values: Iterable) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _token_sums_match(
    countable: tuple[Step, ...],
    capture_prompt: int | None,
    capture_completion: int | None,
    final_metrics: dict[str, Any] | None,
) -> bool | None:
    step_prompt = sum(
        step.metrics.prompt_tokens
        for step in countable
        if step.metrics is not None
    )
    step_completion = sum(
        step.metrics.completion_tokens
        for step in countable
        if step.metrics is not None
    )
    comparisons: list[bool] = []
    if capture_prompt is not None and max(step_prompt, capture_prompt) > 0:
        comparisons.append(step_prompt == capture_prompt)
    if (
        capture_completion is not None
        and max(step_completion, capture_completion) > 0
    ):
        comparisons.append(step_completion == capture_completion)
    totals = final_metrics or {}
    for key, step_sum in (
        ("total_prompt_tokens", step_prompt),
        ("total_completion_tokens", step_completion),
    ):
        reported = totals.get(key)
        if (
            isinstance(reported, int)
            and not isinstance(reported, bool)
            and max(step_sum, reported) > 0
        ):
            comparisons.append(step_sum == reported)
    return all(comparisons) if comparisons else None


def _binding_disagreement(
    countable: tuple[Step, ...],
    unique: tuple[WireCall, ...],
) -> bool:
    disagreement = False
    for step, call in zip(countable, unique):
        if call.tool_call_ids:
            step_ids = {tool.tool_call_id for tool in step.tool_calls}
            if set(call.tool_call_ids) != step_ids:
                disagreement = True
        if (
            step.metrics is not None
            and call.prompt_tokens is not None
            and call.completion_tokens is not None
            and (
                step.metrics.prompt_tokens,
                step.metrics.completion_tokens,
            )
            != (call.prompt_tokens, call.completion_tokens)
        ):
            # The per-position token pair is the second binding witness;
            # a shift across consecutive no-tool-call calls shows here.
            disagreement = True
    return disagreement
