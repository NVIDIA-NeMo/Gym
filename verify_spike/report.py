"""Run-level summary and the human-readable health report.

Ports evalpipeline's summarize()/health_report() (artifacts.py)
verbatim — including the historical
``rollouts_with_duplicate_captured_callsd_calls_all`` key, kept so
summaries diff field-by-field against evalpipeline ground truth.
"""

from __future__ import annotations

from typing import Iterable

from verify_spike.checks import _sum_known
from verify_spike.models import Health


def summarize(
    verifications: "list[Health] | tuple[Health, ...]",
) -> dict:
    return {
        "total_trajectories": len(verifications),
        "total_agent_steps": sum(
            item.agent_step_count for item in verifications
        ),
        "total_tool_calls": sum(
            item.tool_call_count for item in verifications
        ),
        "total_prompt_tokens": sum(
            item.total_prompt_tokens for item in verifications
        ),
        "total_completion_tokens": sum(
            item.total_completion_tokens for item in verifications
        ),
        # None means no rollout reported the bucket (unknown) — never
        # conflated with a genuine zero.
        "total_reasoning_tokens": _sum_known(
            item.total_reasoning_tokens for item in verifications
        ),
        "total_cached_tokens": _sum_known(
            item.total_cached_tokens for item in verifications
        ),
        "total_zero_token_turns": sum(
            item.zero_token_turn_count for item in verifications
        ),
        "total_missed_metrics": sum(
            item.missed_metrics_count for item in verifications
        ),
        "clean_rollouts": sum(1 for item in verifications if item.is_clean),
        "dirty_rollouts": sum(
            1 for item in verifications if not item.is_clean
        ),
        "fully_zero_rollouts": sum(
            1 for item in verifications if item.is_fully_zero
        ),
        "rollouts_with_zero_token_turns": sum(
            1 for item in verifications if item.has_zero_token_turns
        ),
        "rollouts_with_no_agent_steps": sum(
            1 for item in verifications if item.has_no_agent_steps
        ),
        "rollouts_with_missed_metrics": sum(
            1 for item in verifications if item.missed_metrics_count > 0
        ),
        "total_anomalous_zero_token_turns": sum(
            item.anomalous_zero_token_count for item in verifications
        ),
        "total_unknown_zero_token_turns": sum(
            item.unknown_zero_token_count for item in verifications
        ),
        "total_deterministic_steps": sum(
            item.deterministic_step_count for item in verifications
        ),
        "total_hollow_steps": sum(
            item.hollow_step_count for item in verifications
        ),
        "rollouts_with_hollow_steps": sum(
            1 for item in verifications if item.hollow_step_count > 0
        ),
        "rollouts_with_capture": sum(
            1 for item in verifications if item.capture_found
        ),
        "total_unique_capture_calls": sum(
            item.unique_capture_calls
            for item in verifications
            if item.unique_capture_calls is not None
        ),
        "total_non_policy_calls": sum(
            item.non_policy_calls
            for item in verifications
            if item.non_policy_calls is not None
        ),
        "rollouts_with_more_captured_calls_than_steps": sum(
            1
            for item in verifications
            if item.step_capture_delta is not None
            and item.step_capture_delta > 0
        ),
        "rollouts_with_fewer_captured_calls_than_steps": sum(
            1
            for item in verifications
            if item.step_capture_delta is not None
            and item.step_capture_delta < 0
        ),
        "rollouts_with_duplicate_captured_calls": sum(
            1
            for item in verifications
            if item.duplicate_captured_call_count is not None
            and item.duplicate_captured_call_count > 0
        ),
        "rollouts_with_duplicate_captured_callsd_calls_all": sum(
            1
            for item in verifications
            if item.duplicate_captured_call_all_count is not None
            and item.duplicate_captured_call_all_count > 0
        ),
        "rollouts_with_no_capture_calls": sum(
            1
            for item in verifications
            if item.capture_found and item.unique_capture_calls == 0
        ),
        "rollouts_with_token_sum_mismatch": sum(
            1 for item in verifications if item.token_sums_match is False
        ),
        "rollouts_with_retries": sum(
            1
            for item in verifications
            if item.retry_sessions is not None and item.retry_sessions > 0
        ),
        "total_retry_dropped_calls": sum(
            item.retry_dropped_calls
            for item in verifications
            if item.retry_dropped_calls is not None
        ),
        "total_failed_captured_calls": sum(
            item.failed_captured_calls
            for item in verifications
            if item.failed_captured_calls is not None
        ),
        "rollouts_with_failed_captured_calls": sum(
            1
            for item in verifications
            if item.failed_captured_calls is not None
            and item.failed_captured_calls > 0
        ),
        "total_capture_prompt_tokens": sum(
            item.capture_prompt_tokens
            for item in verifications
            if item.capture_prompt_tokens is not None
        ),
        "total_capture_completion_tokens": sum(
            item.capture_completion_tokens
            for item in verifications
            if item.capture_completion_tokens is not None
        ),
        "total_calls_without_steps": sum(
            item.step_capture_delta
            for item in verifications
            if item.step_capture_delta is not None
            and item.step_capture_delta > 0
        ),
        "total_steps_without_calls": sum(
            -item.step_capture_delta
            for item in verifications
            if item.step_capture_delta is not None
            and item.step_capture_delta < 0
        ),
        "rollouts_with_last_captured_call_non_200": sum(
            1
            for item in verifications
            if item.last_captured_call_non_200 is True
        ),
        "rollouts_with_binding_disagreement": sum(
            1 for item in verifications if item.binding_disagreement is True
        ),
        "rollouts_with_discriminator_disagreement": sum(
            1
            for item in verifications
            if item.discriminator_disagreement is True
        ),
        "rollouts_with_capture_integrity_mismatch": sum(
            1
            for item in verifications
            if item.capture_integrity_mismatch is True
        ),
    }


def _known(value) -> str:
    return "unknown" if value is None else str(value)


def health_report(summary: dict) -> str:
    """Human-readable health report: the three core checks up top, the
    secondary signals below. Every signal is a flag for later triage —
    a flagged rollout is suspicious, never automatically wrong."""
    total = summary["total_trajectories"]
    capture_rollouts = summary["rollouts_with_capture"]
    lines = [
        "# Trajectory health report",
        "",
        "Recall-first flags: each check nets anomalies for a later deep",
        "dive. A flagged rollout is suspicious, not automatically wrong,",
        "and no flag beyond the clean-rollout verdict gates anything.",
        "",
        "## Core health checks",
        "",
        "| # | Check | Result | What it means |",
        "|---|---|---|---|",
        (
            "| 1 | Zero-token agent steps | "
            f"{summary['total_zero_token_turns']} flagged steps in "
            f"{summary['rollouts_with_zero_token_turns']}/{total} rollouts "
            f"({summary['total_anomalous_zero_token_turns']} anomalous, "
            f"{summary['total_unknown_zero_token_turns']} to investigate; "
            f"{summary['total_hollow_steps']} hollow) | "
            "An agent step normally spends tokens on a model call. Zero "
            "input and zero output means the step never reached the model "
            "(dropped call, recovery stub, mis-serialized turn). Steps "
            "marked as deterministic dispatch (llm_call_count=0) are "
            "exempt. Hollow = empty message, no tool calls, no tokens. |"
        ),
        (
            "| 2 | No-agent rollouts | "
            f"{summary['rollouts_with_no_agent_steps']}/{total} rollouts | "
            "Every rollout of an agent benchmark should contain agent "
            "steps. A rollout row with none means the agent died before "
            "its first step (dead endpoint, throttling, crash) or the "
            "harness failed to record the trajectory. |"
        ),
        (
            "| 3 | Steps vs captured calls | "
            f"capture on {capture_rollouts}/{total} rollouts: "
            f"{summary['rollouts_with_more_captured_calls_than_steps']} "
            "with more calls than steps "
            f"({summary['total_calls_without_steps']} calls unrecorded), "
            f"{summary['rollouts_with_fewer_captured_calls_than_steps']} "
            "with fewer calls than steps "
            f"({summary['total_steps_without_calls']} steps without "
            "calls), "
            f"{summary['rollouts_with_duplicate_captured_calls']} with "
            "replayed calls, "
            f"{summary['rollouts_with_token_sum_mismatch']} token-mismatch | "
            "The capture — the transport-side record of every model call "
            "— is ground truth. After removing retries and non-policy "
            "calls, each recorded agent step should match exactly one "
            "successful unique call, and token sums should agree. More "
            "captured calls than steps = calls the trajectory never "
            "recorded; fewer = phantom or duplicated steps. Replayed = "
            "byte-identical captured calls (same request body sent "
            "twice). |"
        ),
        "",
        "## Secondary signals",
        "",
        "| Signal | Count | Meaning |",
        "|---|---|---|",
        (
            f"| Missed metrics | {summary['total_missed_metrics']} steps in "
            f"{summary['rollouts_with_missed_metrics']} rollouts | Agent "
            "steps that made a model call but carry no token metrics — the "
            "cost of those calls is unaccounted. |"
        ),
        (
            f"| Fully-zero rollouts | {summary['fully_zero_rollouts']} | "
            "Every flaggable step in the rollout is zero-token — the whole "
            "rollout likely never reached the model. |"
        ),
        (
            f"| Deterministic steps | {summary['total_deterministic_steps']} "
            "| Steps the producer marked llm_call_count=0 (scripted "
            "dispatch, e.g. a seeded greeting) — zero tokens are valid "
            "there. |"
        ),
        (
            f"| Retried rollouts | {summary['rollouts_with_retries']} "
            f"rollouts, {summary['total_retry_dropped_calls']} retry calls "
            "dropped | Rollouts whose capture contained earlier retry "
            "sessions or attempt files; only the last attempt is compared, "
            "and the dropped calls' work (and cost) is not in the capture "
            "sums. |"
        ),
        (
            "| Non-policy captured calls | "
            f"{summary['total_non_policy_calls']} | Captured calls excluded "
            "from the comparison because another model (e.g. a user "
            "simulator) shares the model server. |"
        ),
        (
            "| Failed captured calls | "
            f"{summary['total_failed_captured_calls']} calls in "
            f"{summary['rollouts_with_failed_captured_calls']} rollouts | "
            "Policy calls that got no 2xx response (timeouts, 4xx/5xx). "
            "This does NOT mean the rollout failed — agents usually "
            "retry and "
            "recover, and such rollouts stay clean; it measures transport "
            "turbulence (and possibly unaccounted cost), never rollout "
            "outcomes. |"
        ),
        (
            "| Last captured call failed | "
            f"{summary['rollouts_with_last_captured_call_non_200']} "
            "rollouts | The trial's FINAL model call errored, so the "
            "rollout may have ended on a silent failure — worth a look, "
            "but on its own "
            "still not a failed rollout. |"
        ),
        (
            "| Tokens accounted | steps: "
            f"{summary['total_prompt_tokens']} prompt / "
            f"{summary['total_completion_tokens']} completion · capture: "
            f"{summary['total_capture_prompt_tokens']} / "
            f"{summary['total_capture_completion_tokens']} · reasoning "
            f"{_known(summary['total_reasoning_tokens'])}, cached "
            f"{_known(summary['total_cached_tokens'])} | Step-side sums "
            "come from the trajectory, the captured-call side from "
            "provider usage on unique policy calls — run-level agreement "
            "complements the per-rollout token flag. Reasoning = billed hidden "
            "chain-of-thought; cached = prompt tokens served from a "
            "prefix cache. 'unknown' = no call reported that bucket "
            "(distinct from a real 0). |"
        ),
        (
            "| Binding disagreement | "
            f"{summary['rollouts_with_binding_disagreement']} rollouts | "
            "Step-to-call matching cross-checks disagreed; the mapping "
            "for these rollouts needs a manual look. |"
        ),
        (
            "| Discriminator disagreement | "
            f"{summary['rollouts_with_discriminator_disagreement']} "
            "rollouts | The policy-vs-user-simulator classifiers "
            "disagreed; the non-policy exclusion may be off. |"
        ),
        (
            "| Rollouts with no captured calls | "
            f"{summary['rollouts_with_no_capture_calls']} | A capture "
            "existed but held zero usable calls for the trial — the run "
            "recorded a rollout the model never served. |"
        ),
        (
            f"| Clean rollouts | {summary['clean_rollouts']}/{total} | "
            "Rollouts with agent steps, no flagged zero-token steps, and "
            "no missed metrics — eligible for scavenge selection. |"
        ),
        "",
    ]
    return "\n".join(lines)
