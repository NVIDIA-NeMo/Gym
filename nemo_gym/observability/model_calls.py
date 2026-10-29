# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read saved model exchanges, resolve ownership, and attach evidence to rollouts."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, Field

from nemo_gym.base_responses_api_model import CaptureStore, ModelCallCaptureConfig
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.observability.verifiers import merge_verifier_call_observation
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_observability import (
    NG_AGENT_OBSERVATIONS_KEY,
    AgentObservationBundle,
    ObservationGap,
    TrajectoryRecord,
    VerifierInvocation,
    exclude_verifier_model_calls,
    join_agent_model_calls,
    join_assistant_message_calls,
)


logger = logging.getLogger(__name__)


# --- Observability records derived from captured exchanges ---


def _token_count(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


def _usage_detail_token(
    usage: Mapping[str, Any], detail_groups: tuple[str, ...], field_names: tuple[str, ...]
) -> Optional[int]:
    """Return the first valid token count across equivalent provider detail shapes."""
    for group_name in detail_groups:
        details = usage.get(group_name)
        if not isinstance(details, Mapping):
            continue
        for field_name in field_names:
            value = _token_count(details.get(field_name))
            if value is not None:
                return value
    return None


def extract_token_stats(usage: Any) -> dict[str, Optional[int]]:
    """Normalize token totals across Responses, Chat Completions, and Anthropic Messages usage.

    ``tokens_in`` is the whole prompt size (see ``_prompt_size``); cache creation is also reported
    separately as ``cache_creation_tokens``. A count the provider did not report stays ``None``, never 0.

    ``tokens_in`` is a prompt-*size* metric, not a cost proxy: providers price cache-read (~0.1x) and
    cache-creation (~1.25x) differently from base input, so cost-accurate consumers should weight
    ``cached_tokens`` and ``cache_creation_tokens`` separately rather than summing ``tokens_in``.
    """
    if not isinstance(usage, Mapping):
        return {
            "tokens_in": None,
            "tokens_out": None,
            "tokens_reasoning": None,
            "tokens_total": None,
            "cache_creation_tokens": None,
        }
    tokens_in = _prompt_size(usage)
    tokens_out = _token_count(usage.get("output_tokens"))
    if tokens_out is None:
        tokens_out = _token_count(usage.get("completion_tokens"))
    cache_creation = _token_count(usage.get("cache_creation_input_tokens"))
    tokens_total = _token_count(usage.get("total_tokens"))
    if tokens_total is None and tokens_in is not None and tokens_out is not None:
        tokens_total = tokens_in + tokens_out
    tokens_reasoning = _usage_detail_token(
        usage, ("output_tokens_details", "completion_tokens_details"), ("reasoning_tokens",)
    )
    if tokens_reasoning is None:
        tokens_reasoning = _token_count(usage.get("reasoning_output_tokens"))
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_reasoning": tokens_reasoning,
        "tokens_total": tokens_total,
        "cache_creation_tokens": cache_creation,
    }


def _cache_signal(usage: Any) -> tuple[Optional[bool], Optional[int]]:
    """Cache hit/miss + cached-token count, from usage cache fields (OpenAI / Anthropic)."""
    if not isinstance(usage, Mapping):
        return None, None
    cached = _usage_detail_token(
        usage,
        ("prompt_tokens_details", "input_tokens_details"),
        ("cached_tokens", "cached_input_tokens"),
    )
    if cached is None:
        cached = _token_count(usage.get("cache_read_input_tokens"))  # Anthropic
    if cached is None:
        cached = _token_count(usage.get("cached_input_tokens"))
    if cached is None:
        return None, None
    return cached > 0, cached


def _as_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return {"_raw": arguments}
    return {}


def _tool_calls_and_reasoning(response: dict[str, Any]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Structured tool calls (name, arguments, call_id) and reasoning text, across all three shapes."""
    tool_calls: list[dict[str, Any]] = []
    reasoning: list[str] = []

    output = response.get("output")
    if isinstance(output, list):  # Responses
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                tool_calls.append(
                    {
                        "call_id": item.get("call_id") or item.get("id"),
                        "name": item.get("name"),
                        "arguments": _as_arguments(item.get("arguments")),
                    }
                )
            elif item.get("type") == "custom_tool_call":
                # A free-form tool takes one raw string; record it the way a function stand-in takes it.
                tool_calls.append(
                    {
                        "call_id": item.get("call_id") or item.get("id"),
                        "name": item.get("name"),
                        "arguments": {"input": item.get("input")},
                    }
                )
            elif item.get("type") == "reasoning":
                for summary in item.get("summary") or []:
                    text = summary.get("text") if isinstance(summary, dict) else None
                    if isinstance(text, str) and text:
                        reasoning.append(text)
        return tool_calls, ("\n".join(reasoning) or None)

    choices = response.get("choices")
    if isinstance(choices, list):  # Chat Completions
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, Mapping):
                continue
            for tc in message.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                if not isinstance(fn, Mapping):
                    fn = {}
                tool_calls.append(
                    {"call_id": tc.get("id"), "name": fn.get("name"), "arguments": _as_arguments(fn.get("arguments"))}
                )
            # vLLM and newer OpenAI-compatible servers emit `reasoning`; `reasoning_content` is the
            # older field. Accept either (reasoning_content wins when both are present).
            reasoning_text = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning_text, str) and reasoning_text:
                reasoning.append(reasoning_text)
        return tool_calls, ("\n".join(reasoning) or None)

    content = response.get("content")
    if isinstance(content, list):  # Anthropic Messages
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {"call_id": block.get("id"), "name": block.get("name"), "arguments": block.get("input") or {}}
                )
            elif block.get("type") in ("thinking", "redacted_thinking") and isinstance(block.get("thinking"), str):
                reasoning.append(block["thinking"])
        return tool_calls, ("\n".join(reasoning) or None)

    return tool_calls, None


class ModelCallRecord(BaseModel):
    """Observability record derived from one captured model-server exchange."""

    # Unique server-generated identity for each persisted call.
    model_call_id: Optional[str] = None
    response_id: Optional[str] = None
    client_session_id: Optional[str] = Field(
        default=None,
        description="Client-declared correlation identifier; evidence only, not a security boundary.",
    )
    client_assistant_message_id: Optional[str] = Field(
        default=None,
        description="Persisted client assistant-message ID; retries may share it, model_call_id remains per attempt.",
    )

    # Durable append order, not a causal or semantic order for concurrent calls.
    call_index: int
    model_ref: Optional[ModelServerRef] = None
    model: Optional[str] = None
    dialect: Optional[str] = None
    status_code: Optional[int] = None
    response_status: Optional[str] = None
    finish_reason: Optional[str] = None
    upstream_attempted: Optional[bool] = None
    response_source: Optional[Literal["upstream", "local"]] = None
    upstream_status_code: Optional[int] = None
    local_response_reason: Optional[str] = None

    # Wall-clock bounds around the downstream ASGI invocation, as UTC Unix timestamps. These are
    # for external trace correlation; durations use the monotonic latency fields below.
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Token accounting. tokens_reasoning is OpenAI/Responses-only
    # (sourced from *_tokens_details.reasoning_tokens); Anthropic does not expose it, so it is null
    # there -- consumers must treat null as "unknown", not 0.
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    tokens_reasoning: Optional[int] = None
    tokens_total: Optional[int] = None

    # Model-call record.
    request: Optional[dict[str, Any]] = None
    response: Optional[dict[str, Any]] = None
    request_raw: Optional[str] = None
    response_raw: Optional[str] = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    # Structured reasoning (not flattened into the response text).
    reasoning_content: Optional[str] = None

    # Cache visibility. cached_tokens is the cache-read count; cache_creation_tokens is the
    # Anthropic cache-write count (also folded into tokens_in for the true prompt size).
    cache_hit: Optional[bool] = None
    cached_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None

    # Error classification.
    error_category: Optional[str] = None

    # Latency.
    latency_total_ms: Optional[float] = None
    latency_ttft_ms: Optional[float] = None


def build_model_call_record(exchange: dict[str, Any], *, call_index: int) -> ModelCallRecord:
    """Map one captured exchange and its transport metadata into an observability record."""
    raw_response = exchange.get("response")
    response = raw_response if isinstance(raw_response, dict) else {}
    tokens = extract_token_stats(response.get("usage"))
    cache_hit, cached_tokens = _cache_signal(response.get("usage"))
    tool_calls, reasoning_content = _tool_calls_and_reasoning(response)
    raw_request = exchange.get("request")
    request = raw_request if isinstance(raw_request, dict) else {}
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    incomplete_details = response.get("incomplete_details")
    if not isinstance(incomplete_details, dict):
        incomplete_details = {}
    finish_reason = next(
        (
            value
            for value in (
                response.get("stop_reason"),
                first_choice.get("finish_reason"),
                incomplete_details.get("reason"),
            )
            if isinstance(value, str)
        ),
        None,
    )
    model = response.get("model") or request.get("model")
    return ModelCallRecord(
        model_call_id=exchange.get("model_call_id"),
        response_id=response.get("id") if isinstance(response.get("id"), str) else None,
        client_session_id=(
            exchange.get("client_session_id") if isinstance(exchange.get("client_session_id"), str) else None
        ),
        client_assistant_message_id=(
            exchange.get("client_assistant_message_id")
            if isinstance(exchange.get("client_assistant_message_id"), str)
            else None
        ),
        call_index=call_index,
        model_ref=exchange.get("model_ref"),
        model=model if isinstance(model, str) else None,
        dialect=exchange.get("dialect"),
        status_code=exchange.get("status_code"),
        response_status=response.get("status") if isinstance(response.get("status"), str) else None,
        finish_reason=finish_reason,
        upstream_attempted=exchange.get("upstream_attempted"),
        response_source=exchange.get("response_source"),
        upstream_status_code=exchange.get("upstream_status_code"),
        local_response_reason=exchange.get("local_response_reason"),
        started_at=exchange.get("started_at"),
        completed_at=exchange.get("completed_at"),
        request=raw_request if isinstance(raw_request, dict) else None,
        response=raw_response if isinstance(raw_response, dict) else None,
        request_raw=exchange.get("request_raw") if isinstance(exchange.get("request_raw"), str) else None,
        response_raw=exchange.get("response_raw") if isinstance(exchange.get("response_raw"), str) else None,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        cache_hit=cache_hit,
        cached_tokens=cached_tokens,
        error_category=exchange.get("error_category"),
        latency_total_ms=exchange.get("latency_ms"),
        latency_ttft_ms=exchange.get("latency_ttft_ms"),
        **tokens,
    )


def read_model_call_records(store: CaptureStore, rollout_id: str) -> list[ModelCallRecord]:
    """Read captured exchanges in durable append order."""
    return [
        build_model_call_record(exchange, call_index=index) for index, exchange in enumerate(store.read(rollout_id))
    ]


def read_available_model_call_records(store: CaptureStore, rollout_id: str) -> tuple[list[ModelCallRecord], int]:
    """Read valid call records and count damaged records."""
    calls, _, invalid_count = _read_capture_records(store, rollout_id)
    return calls, invalid_count


def _read_capture_records(
    store: CaptureStore, rollout_id: str
) -> tuple[list[ModelCallRecord], dict[str, "VerifierInvocation"], int]:
    exchanges, invalid_count = store.read_available(rollout_id)
    calls = []
    verifier_invocations: dict[str, "VerifierInvocation"] = {}
    for index, exchange in exchanges:
        try:
            call = build_model_call_record(exchange, call_index=index)
            calls.append(call)
            merge_verifier_call_observation(verifier_invocations, exchange, model_call_id=call.model_call_id)
        except Exception:
            invalid_count += 1
    return calls, verifier_invocations, invalid_count


def aggregate_model_call_records(calls: list[ModelCallRecord]) -> dict[str, Any]:
    """Aggregate token and latency values from model-call records."""

    def _sum(attr: str) -> Optional[float]:
        values = [getattr(call, attr) for call in calls if getattr(call, attr) is not None]
        return sum(values) if values else None

    return {
        "tokens_in": _sum("tokens_in"),
        "tokens_out": _sum("tokens_out"),
        "tokens_reasoning": _sum("tokens_reasoning"),
        "tokens_total": _sum("tokens_total"),
        "cached_tokens": _sum("cached_tokens"),
        "latency_total_ms": _sum("latency_total_ms"),
        "num_calls": len(calls),
    }


def aggregate_model_call_metrics(store: CaptureStore, rollout_id: str) -> dict[str, Any]:
    """Aggregate model-call metrics for one rollout id."""
    return aggregate_model_call_records(read_model_call_records(store, rollout_id))


# --- Run-level capture helpers (rollout-collection side) ---


def model_call_capture_dirs_from_config(global_config_dict: Any) -> list[Path]:
    """Return the single run-wide capture directory when capture is enabled."""
    config = ModelCallCaptureConfig.model_validate(global_config_dict)
    if not config.observability_enabled:
        return []
    assert config.model_call_capture_dir is not None  # enforced by ModelCallCaptureConfig
    return [config.model_call_capture_dir]


def observability_enabled_from_config(global_config_dict: Any) -> bool:
    """Return the run-wide ``observability_enabled`` flag directly, without going through capture dirs."""
    return ModelCallCaptureConfig.model_validate(global_config_dict).observability_enabled


# How long a reader waits for writes already claimed by the model server to land. A claimed
# write is one append + fsync, so this only ever elapses when the writer died holding a claim.
CAPTURE_DRAIN_TIMEOUT_S = 10.0
_CAPTURE_DRAIN_POLL_S = 0.05


def _store_for_rollout(rollout_id: str, capture_dirs: list[Path]) -> Optional[CaptureStore]:
    for directory in capture_dirs:
        store = CaptureStore(directory)
        if store.path_for(rollout_id).exists() or store.is_incomplete(rollout_id) or store.has_pending(rollout_id):
            return store
    return None


async def await_model_call_capture_quiescence(
    rollout_id: str,
    capture_dirs: list[Path],
    *,
    timeout_s: float = CAPTURE_DRAIN_TIMEOUT_S,
    poll_s: float = _CAPTURE_DRAIN_POLL_S,
) -> bool:
    """Wait until no claimed capture write for this rollout is outstanding.

    A cancelled model call finishes its capture write detached (see the shield in
    ``_CaptureMiddleware``), so the row can land after the rollout record is snapshotted and be
    missing from ``ng_model_call_capture`` while present on disk. Returns False on timeout, which
    the merge turns into a ``model_call_capture_pending`` gap rather than a silent omission.
    """
    if not capture_dirs:
        return True
    stores = [CaptureStore(directory) for directory in capture_dirs]
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if not any(store.has_pending(rollout_id) for store in stores):
                return True
        except OSError:
            logger.warning("Could not poll model-call capture claims for rollout %s.", rollout_id, exc_info=True)
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "Model-call capture for rollout %s still has claimed writes after %.1fs; snapshotting anyway.",
                rollout_id,
                timeout_s,
            )
            return False
        await asyncio.sleep(poll_s)


def clear_model_call_captures_for_rollouts(records: list[Any], capture_dirs: list[Path]) -> None:
    """Remove stale per-rollout capture files for these records before dispatch.

    Capture files are keyed by a deterministic rollout id (task-rollout-attempt), so without this a
    fresh run or a kill-shaped retry would append onto the previous attempt's capture for the same
    id. The caller passes only rows about to be dispatched, after assigning any retry suffix.
    """
    if not capture_dirs:
        return
    for directory in capture_dirs:
        store = CaptureStore(directory)
        for record in records:
            rollout_id = maybe_rollout_id_from_run_body(record)
            if rollout_id:
                store.path_for(rollout_id).unlink(missing_ok=True)
                store.incomplete_path_for(rollout_id).unlink(missing_ok=True)
                store.clear_all_pending(rollout_id)


def merge_model_call_capture_into_record(
    record: dict[str, Any], capture_dirs: list[Path], *, include_payloads: bool = False
) -> dict[str, Any]:
    """Attach captured model-call observability data to a rollout record in place.

    Keyed by the rollout id derived from the record's task/rollout/attempt indices, so the attached
    shape is identical for every agent harness. Adds
    ``ng_model_call_capture = {rollout_id, metrics, calls}`` where ``calls`` are derived observability
    records. Raw request and response payloads remain in the capture store and are omitted from the
    attachment unless ``include_payloads`` is true. Capture/read/join failures are attached as
    ``gaps``. The harness output and reward are not modified.
    """
    if not capture_dirs:
        return record
    rollout_id = maybe_rollout_id_from_run_body(record)
    if rollout_id is None:
        return record
    gaps: list[ObservationGap] = []
    verifier_invocations: dict[str, "VerifierInvocation"] = {}
    store = _store_for_rollout(rollout_id, capture_dirs)
    if store is None:
        calls = []
        gaps.append(ObservationGap(code="model_call_capture_no_records"))
    else:
        try:
            calls, verifier_invocations, invalid_count = _read_capture_records(store, rollout_id)
            if invalid_count:
                gaps.append(
                    ObservationGap(
                        code="model_call_capture_records_unreadable",
                        detail=f"count={invalid_count}",
                    )
                )
            if store.has_pending(rollout_id):
                gaps.append(ObservationGap(code="model_call_capture_pending"))
            if store.is_incomplete(rollout_id):
                gaps.append(ObservationGap(code="model_call_capture_incomplete"))
            elif not calls and not invalid_count:
                gaps.append(ObservationGap(code="model_call_capture_no_records"))
        except Exception:
            logger.warning("Could not read model-call capture for rollout %s.", rollout_id, exc_info=True)
            calls = []
            gaps.append(ObservationGap(code="model_call_capture_unreadable"))
    observations = record.get(NG_AGENT_OBSERVATIONS_KEY)
    if observations is not None:
        try:
            bundle = AgentObservationBundle.model_validate(observations)
            if bundle.source == "claude_code":
                try:
                    from responses_api_agents.claude_code_agent.observability import (
                        associate_claude_code_compaction_calls,
                    )

                    bundle = associate_claude_code_compaction_calls(bundle, calls)
                except Exception:
                    logger.warning(
                        "Could not associate Claude Code compaction calls for rollout %s.",
                        rollout_id,
                        exc_info=True,
                    )
                    bundle.gaps.append(ObservationGap(code="compaction_model_call_join_failed"))
            agent_calls = exclude_verifier_model_calls(calls, verifier_invocations.values())
            bundle = join_agent_model_calls(bundle, agent_calls)
            if any(call.client_assistant_message_id for call in calls):
                try:
                    raw_trajectory = record.get("ng_trajectory")
                    trajectory = (
                        TrajectoryRecord.model_validate(raw_trajectory) if raw_trajectory is not None else None
                    )
                    if trajectory is not None and trajectory.rollout_id != rollout_id:
                        bundle.gaps.append(ObservationGap(code="assistant_message_call_rollout_mismatch"))
                    else:
                        bundle, trajectory = join_assistant_message_calls(bundle, trajectory, agent_calls)
                        if trajectory is not None:
                            record["ng_trajectory"] = trajectory.model_dump(mode="json")
                except Exception:
                    logger.warning(
                        "Could not associate assistant message calls for rollout %s.", rollout_id, exc_info=True
                    )
                    bundle.gaps.append(ObservationGap(code="assistant_message_call_join_failed"))
            record[NG_AGENT_OBSERVATIONS_KEY] = bundle.model_dump(mode="json")
        except Exception:
            logger.warning("Could not join agent observations for rollout %s.", rollout_id, exc_info=True)
            gaps.append(ObservationGap(code="agent_observation_join_failed"))
    exclude = None if include_payloads else {"request", "response", "request_raw", "response_raw"}
    capture = {
        "rollout_id": rollout_id,
        "metrics": aggregate_model_call_records(calls),
        "calls": [call.model_dump(exclude=exclude) for call in calls],
    }
    if verifier_invocations:
        capture["verifier_invocations"] = [inv.model_dump(mode="json") for inv in verifier_invocations.values()]
    if gaps:
        capture["gaps"] = [gap.model_dump(mode="json", exclude_none=True) for gap in gaps]
    record["ng_model_call_capture"] = capture
    return record


def _prompt_size(usage: Mapping[str, Any]) -> Optional[int]:
    """Return the whole prompt size, with cached tokens counted exactly once.

    Usage formats disagree on whether the prompt count includes the cache:

    - ``prompt_tokens`` (Chat Completions) includes it. It wins whenever present, also when a gateway
      adds Anthropic-style ``cache_*`` fields beside it.
    - ``input_tokens`` with top-level ``cache_read_input_tokens`` or ``cache_creation_input_tokens``
      (Anthropic Messages) excludes it, so the cache counts are added.
    - ``input_tokens`` alone (Responses) includes it.
    """
    prompt_tokens = _token_count(usage.get("prompt_tokens"))
    if prompt_tokens is not None:
        return prompt_tokens
    input_tokens = _token_count(usage.get("input_tokens"))
    cache_read = _token_count(usage.get("cache_read_input_tokens"))
    cache_creation = _token_count(usage.get("cache_creation_input_tokens"))
    if cache_read is None and cache_creation is None:
        return input_tokens
    cache_total = (cache_read or 0) + (cache_creation or 0)
    # A fully cached Anthropic response may omit input_tokens; zero cache counts alone prove no prompt size.
    if input_tokens is None and cache_total == 0:
        return None
    return (input_tokens or 0) + cache_total
