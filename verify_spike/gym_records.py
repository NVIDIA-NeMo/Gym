"""NeMo-Gym rollout record -> causal step projection.

Ports the evalpipeline Gym backend projection (backends/gym.py) minus the
ATIF emission and config plumbing. The causal projection reads two row
surfaces:

- ``responses_create_params.input`` seeds the prologue steps (system
  prompt, scripted assistant greeting, first user turn) that the
  Responses-shape transcript replays but never re-emits;
- ``response.output`` is the transcript body. A contiguous run of
  assistant-emitted items (``reasoning``, assistant ``message``,
  ``function_call``) folds into ONE agent step; only a
  ``function_call_output`` or a user/system message closes the run — one
  model call may emit message text AND a function_call as adjacent items.
  Tool results attach to the issuing agent step's observation, correlated
  by call_id, never as their own step.

Per-step metrics bind from the in-row ``ng_model_call_capture.calls[]``
(the surface carrying tokens and call_index) — by call_id threading
first, positionally as fallback, flagging disagreement. Token semantics
are per-call (proven, never cumulative).

SENSITIVITY: ``model_ref.name`` values may identify unreleased models.
They are stored on steps for equality comparison and MUST never appear
in exception messages or reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from verify_spike.models import (
    Identity,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

# Naming: "call" = in-row ng_model_call_capture.calls[] entry;
# "record" = a model_calls/ sidecar JSONL row.

# In-row per-call tokens equal the sidecar per-call usage (proven);
# the cumulative-delta transform must never apply to Gym steps.
TOKEN_SEMANTICS = "per_call"


def map_rollout(row: dict[str, Any], *, dataset: Any) -> Trajectory:
    """Project one evaluator_rollouts.jsonl row into a causal Trajectory."""
    calls = _in_row_calls(row)
    transcript = _transcript(row)
    if calls is None and transcript is None:
        raise ValueError(
            "rollout row carries neither a response.output transcript "
            "nor ng_model_call_capture; enable Gym observability "
            "(PR #1715) so rollout records carry the per-rollout "
            "model-call capture"
        )
    drafts = _seed_steps(row)
    if transcript is not None:
        drafts.extend(_transcript_steps(transcript))
    if calls is not None:
        _bind_metrics(drafts, calls)
    steps = tuple(
        _freeze(step_id, draft) for step_id, draft in enumerate(drafts)
    )
    # No policy-scoped final_metrics exists in Gym: the in-row capture
    # aggregate sums policy AND user-sim calls, so final_metrics stays
    # absent (two-way token reconciliation).
    return Trajectory(
        session_id=_session_id(row),
        steps=steps,
        identity=_identity(row, dataset),
    )


@dataclass
class _DraftStep:
    """Mutable step under construction while folding the transcript."""

    source: str
    message: str = ""
    timestamp: str | None = None
    model_name: str | None = None
    reasoning_content: str | None = None
    metrics: Metrics | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[ObservationResult] = field(default_factory=list)
    llm_call_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _freeze(step_id: int, draft: _DraftStep) -> Step:
    observation = (
        Observation(results=tuple(draft.results)) if draft.results else None
    )
    return Step(
        step_id=step_id,
        source=draft.source,
        message=draft.message,
        timestamp=draft.timestamp,
        model_name=draft.model_name,
        reasoning_content=draft.reasoning_content,
        metrics=draft.metrics,
        tool_calls=tuple(draft.tool_calls),
        observation=observation,
        llm_call_count=draft.llm_call_count,
        extra=draft.extra or None,
    )


def _in_row_calls(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    capture = row.get("ng_model_call_capture")
    if capture is None:
        return None
    if not isinstance(capture, dict):
        raise ValueError("ng_model_call_capture must be a mapping")
    calls = capture.get("calls")
    if not isinstance(calls, list):
        raise ValueError("ng_model_call_capture.calls must be a list")
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError(
                "ng_model_call_capture.calls entries must be mappings"
            )
    return calls


def _transcript(row: dict[str, Any]) -> list[Any] | None:
    response = row.get("response")
    if response is None:
        return None
    if not isinstance(response, dict):
        raise ValueError("response must be a mapping")
    output = response.get("output")
    if output is None:
        return None
    if not isinstance(output, list):
        raise ValueError("response.output must be a list")
    return output


_SEED_SOURCES = {"system": "system", "assistant": "agent", "user": "user"}


def _seed_steps(row: dict[str, Any]) -> list[_DraftStep]:
    params = row.get("responses_create_params")
    if not isinstance(params, dict):
        raise ValueError(
            "responses_create_params must be a mapping; the causal "
            "projection needs the seed prologue (system prompt, scripted "
            "greeting, first user turn)"
        )
    items = params.get("input")
    if not isinstance(items, list):
        raise ValueError("responses_create_params.input must be a list")
    return [_seed_step(item) for item in items]


def _seed_step(item: Any) -> _DraftStep:
    if not isinstance(item, dict):
        raise ValueError(
            "responses_create_params.input items must be mappings"
        )
    role = item.get("role")
    if role not in _SEED_SOURCES:
        raise ValueError(f"unsupported seed input role: {role!r}")
    draft = _DraftStep(
        source=_SEED_SOURCES[role],
        message=_text(item.get("content")),
    )
    draft.extra["from_seed"] = True
    if draft.source == "agent":
        # Scripted greeting: no model call happened in this rollout.
        draft.llm_call_count = 0
    return draft


def _transcript_steps(items: list[Any]) -> list[_DraftStep]:
    drafts: list[_DraftStep] = []
    open_calls: dict[str, _DraftStep] = {}
    current: _DraftStep | None = None
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("response.output items must be mappings")
        current = _apply_item(item, drafts, open_calls, current)
    return drafts


def _apply_item(
    item: dict[str, Any],
    drafts: list[_DraftStep],
    open_calls: dict[str, _DraftStep],
    current: _DraftStep | None,
) -> _DraftStep | None:
    kind = item.get("type")
    if kind == "function_call":
        step = _open_agent(drafts, current)
        tool_call = _function_tool_call(item)
        step.tool_calls.append(tool_call)
        open_calls[tool_call.tool_call_id] = step
        return step
    if kind == "function_call_output":
        _attach_output(item, open_calls)
        # The tool result is the causal boundary: it closes the run.
        return None
    if kind == "reasoning":
        step = _open_agent(drafts, current)
        step.reasoning_content = (
            step.reasoning_content or ""
        ) + _summary_text(item)
        return step
    if kind == "message":
        return _apply_message(item, drafts, current)
    raise ValueError(f"unsupported transcript item type: {kind!r}")


def _apply_message(
    item: dict[str, Any],
    drafts: list[_DraftStep],
    current: _DraftStep | None,
) -> _DraftStep | None:
    role = item.get("role")
    text = _text(item.get("content"))
    if role == "assistant":
        step = _open_agent(drafts, current)
        step.message = step.message + text if step.message else text
        # An assistant message does NOT close the run: one model call
        # may emit message text AND a function_call as adjacent items.
        return step
    if role in ("user", "system"):
        drafts.append(_DraftStep(source=role, message=text))
        return None
    raise ValueError(f"unsupported transcript message role: {role!r}")


def _open_agent(
    drafts: list[_DraftStep],
    current: _DraftStep | None,
) -> _DraftStep:
    if current is not None:
        return current
    draft = _DraftStep(source="agent")
    drafts.append(draft)
    return draft


def _attach_output(
    item: dict[str, Any],
    open_calls: dict[str, _DraftStep],
) -> None:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or call_id not in open_calls:
        raise ValueError(
            f"function_call_output {call_id!r} matches no open tool "
            "call; the transcript is mis-threaded"
        )
    open_calls[call_id].results.append(
        ObservationResult(source_call_id=call_id, content=item.get("output"))
    )


def _function_tool_call(item: dict[str, Any]) -> ToolCall:
    call_id = item.get("call_id")
    name = item.get("name")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise ValueError("function_call items need a string call_id and name")
    raw = item.get("arguments")
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"function_call {call_id!r} arguments are not valid "
                f"JSON: {exc}"
            ) from exc
        # Keep the raw argument string only when parsing changed it.
        extra = (
            None if json.dumps(arguments) == raw else {"raw_arguments": raw}
        )
    else:
        arguments, extra = raw, None
    if not isinstance(arguments, dict):
        raise ValueError(
            f"function_call {call_id!r} arguments must decode to a mapping"
        )
    return ToolCall(
        tool_call_id=call_id,
        function_name=name,
        arguments=arguments,
        extra=extra,
    )


def _summary_text(item: dict[str, Any]) -> str:
    parts = item.get("summary") or []
    if not isinstance(parts, list):
        raise ValueError("reasoning summary must be a list")
    texts = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("reasoning summary parts must be mappings")
        texts.append(str(part.get("text") or ""))
    return "".join(texts)


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_part_text(part) for part in content)
    raise ValueError(
        f"unsupported message content shape: {type(content).__name__}"
    )


def _part_text(part: Any) -> str:
    if not isinstance(part, dict):
        raise ValueError("message content parts must be mappings")
    text = part.get("text") or part.get("refusal") or ""
    return text if isinstance(text, str) else ""


def _bind_metrics(
    drafts: list[_DraftStep],
    calls: list[dict[str, Any]],
) -> None:
    """Attach per-call metrics to non-seed agent steps.

    call_id threading is primary (a call whose tool_calls carry one of
    the step's tool_call ids); positional binding over the policy-side
    calls is the fallback and the cross-check — a disagreement binds by
    threading and flags the step.
    """
    body_steps = [
        draft
        for draft in drafts
        if draft.source == "agent" and not draft.extra.get("from_seed")
    ]
    positional = _positional_calls(calls, _step_tool_call_ids(body_steps))
    for index, step in enumerate(body_steps):
        threaded = _threaded_call(step, calls)
        expected = positional[index] if index < len(positional) else None
        chosen = threaded if threaded is not None else expected
        if chosen is not None:
            if (
                threaded is not None
                and expected is not None
                and expected is not threaded
            ):
                step.extra["binding_disagreement"] = True
            _apply_call(step, chosen)


def _positional_calls(
    calls: list[dict[str, Any]],
    step_tool_call_ids: set,
) -> list[dict[str, Any]]:
    """Calls eligible for positional binding, in call_index order.

    With multiple model_ref.name groups, the policy side is the unique
    group threaded to the transcript's tool_call ids; when that is not
    decidable in-row, positional binding runs over all calls.
    """
    groups: dict[Any, list[dict[str, Any]]] = {}
    for call in calls:
        groups.setdefault(_call_model_name(call), []).append(call)
    policy = calls
    if len(groups) > 1:
        threaded = [
            group
            for group in groups.values()
            if _group_tool_call_ids(group) & step_tool_call_ids
        ]
        # Only a uniquely-threaded group is safe to bind positionally.
        # Anything else leaves the steps unbound so missed-metrics flags
        # them — never silently misattribute a user-simulator call to an
        # agent step (recall over precision).
        policy = threaded[0] if len(threaded) == 1 else []
    return sorted(policy, key=_call_index)


def _threaded_call(
    step: _DraftStep,
    calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    step_ids = {tool_call.tool_call_id for tool_call in step.tool_calls}
    if not step_ids:
        return None
    matches = [call for call in calls if _call_tool_call_ids(call) & step_ids]
    return matches[0] if matches else None


def _apply_call(step: _DraftStep, call: dict[str, Any]) -> None:
    step.metrics = _call_metrics(call)
    name = _call_model_name(call)
    step.model_name = name if isinstance(name, str) else None
    step.llm_call_count = 1
    started = call.get("started_at")
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        step.timestamp = datetime.fromtimestamp(
            started, tz=timezone.utc
        ).isoformat()
    step.extra.update(
        model_call_id=call.get("model_call_id"),
        call_index=call.get("call_index"),
        dialect=call.get("dialect"),
        status_code=call.get("status_code"),
    )


_METRICS_EXTRA_FIELDS = (
    ("latency_total_ms", "latency_ms"),
    ("latency_ttft_ms", "latency_ttft_ms"),
    ("cache_creation_tokens", "cache_creation_tokens"),
    ("cache_hit", "cache_hit"),
    ("finish_reason", "finish_reason"),
    ("error_category", "error_category"),
)


def _call_metrics(call: dict[str, Any]) -> Metrics:
    # Missing token counts stay exactly as the source reports them
    # (0 for the required fields, None for the optional ones) — the
    # health verdict flags the zero, the mapper never invents a value.
    extra: dict[str, Any] = {}
    for source_key, target_key in _METRICS_EXTRA_FIELDS:
        value = call.get(source_key)
        if value is not None:
            extra[target_key] = value
    return Metrics(
        prompt_tokens=_int_or_zero(call.get("tokens_in")),
        completion_tokens=_int_or_zero(call.get("tokens_out")),
        total_tokens=_int_or_none(call.get("tokens_total")),
        cached_tokens=_int_or_none(call.get("cached_tokens")),
        reasoning_tokens=_int_or_none(call.get("tokens_reasoning")),
        extra=extra or None,
    )


def _step_tool_call_ids(steps: list[_DraftStep]) -> set:
    return {
        tool_call.tool_call_id
        for step in steps
        for tool_call in step.tool_calls
    }


def _call_tool_call_ids(call: dict[str, Any]) -> set:
    raw = call.get("tool_calls") or []
    if not isinstance(raw, list):
        raise ValueError(
            "ng_model_call_capture call tool_calls must be a list"
        )
    ids = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "ng_model_call_capture call tool_calls entries "
                "must be mappings"
            )
        call_id = item.get("call_id")
        if isinstance(call_id, str):
            ids.add(call_id)
    return ids


def _group_tool_call_ids(group: list[dict[str, Any]]) -> set:
    return {call_id for call in group for call_id in _call_tool_call_ids(call)}


def _call_model_name(call: dict[str, Any]) -> Any:
    model_ref = call.get("model_ref")
    return model_ref.get("name") if isinstance(model_ref, dict) else None


def _call_index(call: dict[str, Any]) -> int:
    value = call.get("call_index")
    return value if _is_int(value) else 0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_or_none(value: Any) -> int | None:
    return value if _is_int(value) else None


def _int_or_zero(value: Any) -> int:
    return value if _is_int(value) else 0


def _session_id(row: dict[str, Any]) -> str:
    capture = row.get("ng_model_call_capture")
    rollout_id = (
        capture.get("rollout_id") if isinstance(capture, dict) else None
    )
    return str(rollout_id or _rollout_id(row))


def _identity(row: dict[str, Any], dataset: Any) -> Identity:
    # dataset is the run's grouping scope (the artifacts directory name
    # here); task and trial are Gym's task/rollout indices; attempt is
    # the retry index, absent (0) on a clean run.
    return Identity(
        dataset=dataset,
        task=row["_ng_task_index"],
        trial=row["_ng_rollout_index"],
        attempt=row.get("_ng_attempt_index", 0),
    )


def _rollout_id(row: dict[str, Any]) -> str:
    return f"{row.get('_ng_task_index')}-{row.get('_ng_rollout_index')}"
