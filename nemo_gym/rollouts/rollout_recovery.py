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
"""Identity checks for resuming saved evaluation results.

The manifest stores digests, not configuration values or credentials. Recovery reuses
completed rollouts; it does not checkpoint an agent's conversation or remote sandbox.
"""

import errno
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import socket
import stat
import warnings
from collections.abc import Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal, cast, get_args
from uuid import uuid4

from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError, OmegaConfBaseException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.artifacts.rollout_rows import is_no_generation_placeholder
from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.secret_utils import mask_url_passwords
from nemo_gym.core.failure_kinds import (
    AGENT_ROUTE_FAILED,
    AGENT_TIMEOUT,
    COLLECTION_RESUME_REFUSED,
    JUDGE_FAILED,
    JUDGE_FAILURE_KINDS,
    JUDGE_UNPARSEABLE,
    MODEL_SERVER_ERROR,
    MODEL_TRANSPORT_ERROR,
    PROVIDER_FAILURE_KINDS,
    SESSION_EXPIRED,
    SESSION_LOST,
    TRANSPORT_PEER_DROP,
    TRANSPORT_UNREACHABLE,
    FailureClass,
    FailureClassificationConfig,
    FailureStage,
    classify_failure,
    is_namespaced,
    is_registered,
    is_server_status,
)
from nemo_gym.core.outcome_policy import OutcomeAction
from nemo_gym.utils.path_utils import (
    attempt_extension_path_for,
    failures_path_for,
    rerun_masked_journal_path_for,
    resubmit_offset_path_for,
)


def has_saved_generation(row: Mapping[str, Any]) -> bool:
    """Recognize retained response data, including a completed response with empty output.

    A diagnostic error object or an empty mapping is not generation. A typed response counts when it has at least
    one output item or a status: a producer finished it, even with no assistant text. A response object with no
    output and no status is a placeholder, for example from an agent that never ran. So is a response an agent
    marked with ``NG_NO_GENERATION_METADATA``: its empty assistant message was built without any generation.
    """
    record = row.get("_ng_failure_record")
    failure = record.get("failure") if isinstance(record, Mapping) else None
    stage = failure.get("stage") if isinstance(failure, Mapping) else row.get("_ng_failure_stage")
    for key in ("response", "_ng_failure_response", "_ng_failure_partial_response"):
        answer = row.get(key)
        if not isinstance(answer, Mapping):
            continue
        # The agent outcome policy already rejected this failed response as an answer. Retain its
        # diagnostic output without vetoing RETRY; cleanup/verification responses keep their protection.
        if stage == "agent" and answer.get("status") == "failed":
            continue
        if is_no_generation_placeholder(answer):
            continue
        output = answer.get("output")
        if not isinstance(output, list):
            continue
        if (
            answer.get("object") == "response"
            and isinstance(answer.get("id"), str)
            and answer["id"]
            and (output or (answer.get("error") is None and answer.get("status") not in (None, "failed")))
        ):
            return True
        if answer.get("status") == "completed" and any(
            isinstance(item, Mapping) and item.get("type") == "message" and item.get("role") == "assistant"
            for item in output
        ):
            return True
    return False


def keeps_partial_response(row: Mapping[str, Any]) -> bool:
    """Whether the row keeps the agent's partial response from a native cleanup, even one without output.

    The agent ran, so its partial response stays a kept answer and is never regenerated. An agent-stage response
    the outcome policy rejected as failed is diagnostic only, as in ``has_saved_generation``.
    """
    answer = row.get("_ng_failure_partial_response")
    if not isinstance(answer, Mapping):
        return False
    record = row.get("_ng_failure_record")
    failure = record.get("failure") if isinstance(record, Mapping) else None
    stage = failure.get("stage") if isinstance(failure, Mapping) else row.get("_ng_failure_stage")
    return not (stage == "agent" and answer.get("status") == "failed")


def has_recorded_grade(row: Mapping[str, Any]) -> bool:
    """Whether the verifier finished grading this failed attempt, so it is a graded result and a rerun would replace it.

    Collection keeps no reward on a failure row (a placeholder score is never a result), so the completed
    evaluation is the grade's durable record. Only a row with neither a kept answer nor a recorded grade may run fresh.
    """
    return row.get("evaluation_completed") is True


def unsettled_masked_note(coverage: Mapping[str, Any]) -> str:
    """What keeps masked results from counting as settled, for an incomplete-run message; empty when none does.

    Names the kinds a producer masked with ``fail`` (final: fix or drop those rows, a larger attempt cap does not
    reopen them) and the infrastructure-masked results that spent their attempt budget (reopened by
    ``--extend-attempts``).
    """
    notes = []
    if coverage.get("masked_failed"):
        by_kind = sorted(coverage.get("masked_failed_kinds", {}).items())
        kinds = ", ".join(f"{kind} ({count})" for kind, count in by_kind)
        notes.append(
            f"{coverage['masked_failed']} masked result(s) were failed by their producer and are not retried "
            f"(masked_failed_kinds: {kinds}): fix or drop those rows."
        )
    if coverage.get("masked_exhausted"):
        notes.append(
            f"{coverage['masked_exhausted']} infrastructure-masked result(s) spent their attempt budget "
            "(masked_exhausted): continue with gym eval resubmit --recover --extend-attempts N."
        )
    return " ".join(notes)


def _row_failure_kind(row: Mapping[str, Any]) -> Any:
    """The selected failure event's kind when the row selects one (None when that event is missing), else the row's."""
    events, selected_id = row.get("_ng_failure_events"), row.get("_ng_failure_id")
    selected = events.get(selected_id) if isinstance(events, Mapping) and isinstance(selected_id, str) else None
    if isinstance(selected, Mapping):
        return selected.get("failure_kind")
    if "_ng_failure_id" in row:
        return None
    return row.get("failure_kind") if row.get("failure_kind") is not None else row.get("_ng_failure_class")


def is_answered_agent_deadline(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> bool:
    """Whether the agent ran out of its own deadline and kept its answer: the evaluated model's final result.

    The deadline belongs to the agent, so what it kept is its result, scored by the benchmark's rule (a recorded
    grade, else a settled zero). Recovery classifies it as final, and no resume or in-process retry runs it again:
    that would replace the model's answer. The kind is the selected failure event's when the row selects one, else
    the row's own. A timeout that kept no answer keeps its retry policy, and so does one whose window held a recorded
    server-side fault (``is_infrastructure_agent_timeout``): what it kept was cut short by us, not by its deadline.
    """
    return (
        _row_failure_kind(row) == AGENT_TIMEOUT
        and (has_saved_generation(row) or keeps_partial_response(row))
        and not is_infrastructure_agent_timeout(row, classification=classification)
    )


# The collector's minimal projection of a saved row's captured model calls (``COLLECT_SCRIPT`` in
# ``nemo_gym.orchestration.recovery`` writes it in ``captured_model_calls``'s normalized shape).
COLLECTED_MODEL_CALLS_KEY = "_ng_collected_model_calls"
# Faulted model calls own an agent's deadline when they took at least this share of its window.
AGENT_TIMEOUT_FAULT_SHARE = 0.25
# Capture error categories of a server-side failure: a 5xx class reply or an upstream error event, and a connection
# that failed. ``timeout``, ``cancelled``, ``stream_truncated`` and client errors are not: a slow model is the model's
# result under its time limit, and the agent's own deadline cancels the call it was waiting on.
SERVER_FAULT_CATEGORIES = frozenset({"upstream_error", "connection"})
# Evidence eligible for the timeout-materiality test; responsibility still comes from the shared classifier.
# ``transport_timeout`` is left out for the same reason as the capture timeout category.
_MATERIAL_CALL_FAILURE_KINDS = frozenset(
    {MODEL_TRANSPORT_ERROR, MODEL_SERVER_ERROR, TRANSPORT_UNREACHABLE, TRANSPORT_PEER_DROP}
)


def captured_model_calls(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The row's captured model calls in capture order, each reduced to the facts a server fault is read from.

    ``status_code``, ``upstream_status_code``, ``error_category``, ``failure_events`` (each failure-report event's
    ``failure_kind`` and ``http_status``), ``started_at`` and ``duration_ms``. Read from the collector's projection
    when the row is one, else from ``ng_model_call_capture.calls``, else from ``ng_trajectory.model_calls``.
    """
    collected = row.get(COLLECTED_MODEL_CALLS_KEY)
    if isinstance(collected, list):
        return [call for call in collected if isinstance(call, dict)]
    capture = row.get("ng_model_call_capture")
    raw = capture.get("calls") if isinstance(capture, Mapping) else None
    if not isinstance(raw, list):
        trajectory = row.get("ng_trajectory")
        raw = trajectory.get("model_calls") if isinstance(trajectory, Mapping) else None
    calls = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, Mapping):
            continue
        meta = item.get("response_metadata")
        meta = meta if isinstance(meta, Mapping) else item
        report = meta.get("failure_report")
        events = report.get("events") if isinstance(report, Mapping) else None
        duration = item.get("latency_total_ms", item.get("duration_ms"))
        calls.append(
            {
                "status_code": meta.get("status_code"),
                "upstream_status_code": meta.get("upstream_status_code"),
                "error_category": meta.get("error_category"),
                "failure_events": [
                    {
                        "failure_kind": event.get("failure_kind"),
                        "http_status": event.get("http_status"),
                        "stage": event.get("stage"),
                    }
                    for event in (events.values() if isinstance(events, Mapping) else ())
                    if isinstance(event, Mapping)
                ],
                "started_at": item.get("started_at"),
                "duration_ms": duration,
            }
        )
    return calls


def saved_failure_classification(output: Path) -> FailureClassificationConfig:
    """Read only the owning launcher's saved classification in its conventional artifact layout.

    Standalone outputs have no launcher policy and use shared defaults. A malformed saved policy is never
    silently replaced with defaults. Applied classes on individual results still take precedence.
    """
    if output.name != "rollouts.jsonl" or output.parent.name != "artifacts":
        return FailureClassificationConfig()
    job = output.parent.parent
    resolved = job.parent / "resolved-config.yaml"
    if not resolved.is_file():
        return FailureClassificationConfig()
    try:
        config = OmegaConf.to_container(OmegaConf.load(resolved), resolve=False)
        benchmark = config["driver"]["benchmarks"][job.name]
        resumable = benchmark.get("resumable")
        if resumable is None:
            resumable = {}
        if not isinstance(resumable, Mapping):
            raise ValueError("resumable must be a mapping")
        policy = resumable.get("recovery_outcome_policy")
        if policy is None:
            policy = {}
        if not isinstance(policy, Mapping):
            raise ValueError("recovery_outcome_policy must be a mapping")
        return FailureClassificationConfig.model_validate(policy.get("classification", {}))
    except (OSError, KeyError, TypeError, AttributeError, ValueError, OmegaConfBaseException) as error:
        raise ConfigError(f"Cannot read saved failure classification from {resolved}: {error}") from error


def applied_failure_class(row: Mapping[str, Any]) -> FailureClass | None:
    """Read an applied result class, preserving explicit unknown over later configuration."""
    action, attribution = row.get("failure_outcome_action"), row.get("failure_outcome_attribution")
    if (
        isinstance(action, str)
        and action in OutcomeAction
        and isinstance(attribution, str)
        and attribution in FailureClass
    ):
        return FailureClass(attribution)
    return None


def failure_stage(row: Mapping[str, Any]) -> FailureStage | None:
    """Read selected final context; recovered observations never supply the final stage."""
    events, selected_id = row.get("_ng_failure_events"), row.get("_ng_failure_id")
    if isinstance(selected_id, str):
        selected = events.get(selected_id) if isinstance(events, Mapping) else None
        stage = selected.get("stage") if isinstance(selected, Mapping) else None
    else:
        record = row.get("_ng_failure_record")
        failure = record.get("failure") if isinstance(record, Mapping) else None
        stage = failure.get("stage") if isinstance(failure, Mapping) else row.get("_ng_failure_stage")
    return cast(FailureStage, stage) if stage in get_args(FailureStage) else None


def _evaluation_event(event: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None) -> bool:
    stage = event.get("stage")
    return (
        classify_failure(
            failure_kind=event.get("failure_kind"),
            stage=cast(FailureStage, stage) if stage in get_args(FailureStage) else None,
            http_status=event.get("http_status"),
            config=classification,
        )
        == FailureClass.EVALUATION
    )


def is_server_fault_call(
    call: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> bool:
    """Whether eligible recorded model-call failures classify as evaluation-side.

    Temporal materiality excludes timeout, cancellation and truncated-stream evidence: a slow model or an
    agent cancelling its current call does not establish time spent on infrastructure. HTTP evidence and the
    eligible transport errors use the shared classifier; classification alone does not establish materiality.
    """
    if any(
        classify_failure(failure_kind=None, http_status=call.get(key), config=classification)
        == FailureClass.EVALUATION
        for key in ("status_code", "upstream_status_code")
    ):
        return True
    category = call.get("error_category")
    if category in SERVER_FAULT_CATEGORIES:
        kind = MODEL_TRANSPORT_ERROR if category == "connection" else MODEL_SERVER_ERROR
        if classify_failure(failure_kind=kind, config=classification) == FailureClass.EVALUATION:
            return True
    events = call.get("failure_events")
    return any(
        isinstance(event, Mapping)
        and (event.get("failure_kind") in _MATERIAL_CALL_FAILURE_KINDS or is_server_status(event.get("http_status")))
        and _evaluation_event(event, classification=classification)
        for event in (events if isinstance(events, list) else ())
    )


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _material_model_call_faults(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> bool:
    """Whether server-faulted model calls own the window: the agent's last call faulted, or faulted calls took at
    least ``AGENT_TIMEOUT_FAULT_SHARE`` of it. The window is the attempt's recorded elapsed time, else the span of its
    captured calls; an early fault that a retry recovered within a long window is the model's time, not ours."""
    calls = captured_model_calls(row)
    if not calls:
        return False
    starts = [_number(call.get("started_at")) for call in calls]
    if all(start is not None for start in starts):
        # Capture order is append order; when every call recorded its start, the last started is the last call.
        order = sorted(range(len(calls)), key=lambda index: (starts[index], index))
        calls, starts = [calls[index] for index in order], [starts[index] for index in order]
    if is_server_fault_call(calls[-1], classification=classification):
        return True
    faulted = sum(
        _number(call.get("duration_ms")) or 0.0
        for call in calls
        if is_server_fault_call(call, classification=classification)
    )
    if not faulted:
        return False
    elapsed = _number(row.get("_ng_elapsed_s")) or observed_elapsed(dict(row))
    if elapsed:
        window_ms = elapsed * 1000
    else:
        spans = [
            (start, start + (_number(call.get("duration_ms")) or 0.0) / 1000)
            for start, call in zip(starts, calls)
            if start is not None
        ]
        window_ms = (max(end for _, end in spans) - min(start for start, _ in spans)) * 1000 if spans else 0.0
    return window_ms > 0 and faulted >= AGENT_TIMEOUT_FAULT_SHARE * window_ms


def _sandbox_or_tool_fault(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> bool:
    """Eligible sandbox/session or tool-server evidence classified by the shared mapping.

    Cleanup and verification lie outside the agent's execution window. Registered provider/session kinds
    identify candidate observations; prefixes do not assign responsibility. For example agent OOM remains
    model-side while seed OOM is evaluation-side when its originating stage is available.
    """
    events = row.get("_ng_failure_events")
    for event in events.values() if isinstance(events, Mapping) else ():
        if not isinstance(event, Mapping) or not isinstance(event.get("failure_kind"), str):
            continue
        kind = event["failure_kind"]
        if event.get("stage") in ("cleanup", "verification"):
            continue
        if kind in PROVIDER_FAILURE_KINDS | {SESSION_LOST, SESSION_EXPIRED} and _evaluation_event(
            event, classification=classification
        ):
            return True
        if (
            event.get("boundary") == "resources_server"
            and event.get("operation") != "verify"
            and (kind in _MATERIAL_CALL_FAILURE_KINDS or is_server_status(event.get("http_status")))
            and _evaluation_event(event, classification=classification)
        ):
            return True
    return False


def is_infrastructure_agent_timeout(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> bool:
    """Whether an ``agent_timeout`` row's window held a recorded server-side fault: the evaluation's failure.

    The row (a failure row, or a graded result such as Harbor's) names ``agent_timeout`` and either recorded a sandbox
    or tool-server failure event (``_sandbox_or_tool_fault``), or its captured model calls show a server fault
    (``is_server_fault_call``) that owns the window (``_material_model_call_faults``). A clean agent timeout stays the
    model's result under the benchmark's time limit. Read from saved rows alone, so it applies to existing runs.
    """
    applied = applied_failure_class(row)
    if applied is not None:
        if row.get("failure_outcome_action") in (OutcomeAction.COUNT_AS_ZERO, OutcomeAction.FAIL):
            return False
        return _row_failure_kind(row) == AGENT_TIMEOUT and applied == FailureClass.EVALUATION
    return _row_failure_kind(row) == AGENT_TIMEOUT and (
        _sandbox_or_tool_fault(row, classification=classification)
        or _material_model_call_faults(row, classification=classification)
    )


# Masked kinds whose saved answer is graded again (judge) or whose verify call may have been lost in transport: their
# recovery reads the saved answer and never reruns the rollout, so they are not rerun here either.
_MASKED_REGRADE_KINDS = frozenset({JUDGE_FAILED, JUDGE_UNPARSEABLE, AGENT_ROUTE_FAILED})


def _masked_evaluation_side_kind(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> str | None:
    """The evaluation-side kind of a masked result row that is not set aside for another owner; else None.

    The rules ``masked_infrastructure_kind`` and ``masked_kept_answer_kind`` share, all but the saved answer: the row
    is masked by the Environment Server that graded it, is no failure row, omitted stand-in or re-judgment, and its
    ``failure_kind`` is registered, or namespaced with the ``retry`` action its owner recorded and no model
    attribution. The kind is evaluation-side through the shared classifier (or carries a recorded ``retry``), not one
    whose saved answer is graded again (``_MASKED_REGRADE_KINDS``) and not a ``transport_*`` verify call.
    """
    if (
        row.get("mask_sample") is not True
        or row.get("failure_outcome_action") == OutcomeAction.FAIL
        or row.get("_ng_failure_class") is not None
        or row.get("_ng_omitted_rollout") is True
        or row.get("_ng_rejudge_operation_id") is not None
    ):
        return None
    kind = row.get("failure_kind")
    selected_class = applied_failure_class(row)
    if selected_class is None:
        events, selected_id = row.get("_ng_failure_events"), row.get("_ng_failure_id")
        selected = events.get(selected_id) if isinstance(events, Mapping) and isinstance(selected_id, str) else None
        if isinstance(selected, Mapping):
            kind, status = selected.get("failure_kind"), selected.get("http_status")
        else:
            status = row.get("_ng_failure_http_status")
        selected_class = classify_failure(
            failure_kind=kind if isinstance(kind, str) else None,
            stage=failure_stage(row),
            http_status=status,
            config=classification,
        )
    retry_declared = row.get("failure_outcome_action") == OutcomeAction.RETRY
    namespaced_retry = retry_declared and row.get("failure_outcome_attribution") != FailureClass.MODEL
    if (
        not isinstance(kind, str)
        or not (is_registered(kind) or (is_namespaced(kind) and namespaced_retry))
        or (selected_class != FailureClass.EVALUATION and not retry_declared)
        or kind in _MASKED_REGRADE_KINDS
        or kind.startswith("transport_")
    ):
        return None
    return kind


def masked_infrastructure_kind(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> str | None:
    """The kind of a masked result that infrastructure, not the evaluated model, left unmeasured; else None.

    A result row is masked (``mask_sample``) by the Environment Server that graded it, so its ``failure_kind`` was
    reported at that boundary: a lost session, a failed provider, an infrastructure error, a verifier that could not
    run. When the kind is evaluation-side through the shared classifier, or its owner recorded the ``retry`` action,
    the rollout is retried like an infrastructure failure, within its attempt cap: the masked row counts one failed
    attempt and a later attempt supersedes it. A recorded ``retry`` is honored for a registered kind whatever its
    attribution, but for a namespaced kind only when it is not attributed to the model. A model-class kind, no kind,
    an unregistered kind, a namespaced kind without a recorded ``retry`` or attributed to the model, and the kinds
    whose saved answer is graded again (``_MASKED_REGRADE_KINDS``, including a ``transport_*`` verify call) keep the
    masked row as the result. So does a saved answer (``_keeps_saved_answer``): it is never sampled again, whatever its
    verifier could grade of it.
    """
    kind = _masked_evaluation_side_kind(row, classification=classification)
    return None if kind is None or _keeps_saved_answer(row) else kind


def masked_kept_answer_kind(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> str | None:
    """The kind of a masked result that ``masked_infrastructure_kind`` rejects only for its saved answer and that has
    no recorded grade (``has_recorded_grade``); else None.

    The part of ``masked_without_artifact_kind`` that the Store can decide: the planner adds that no re-verification
    cause applies. A graded result is a result and stays one.
    """
    kind = _masked_evaluation_side_kind(row, classification=classification)
    return kind if kind is not None and _keeps_saved_answer(row) and not has_recorded_grade(row) else None


def _keeps_saved_answer(row: Mapping[str, Any]) -> bool:
    """Whether the masked result holds a model answer a rerun would replace: never sampled again.

    Same facts as recovery's saved-answer test: a collector projection's own flag, else the retained generation of the
    response (``has_saved_generation``, which leaves out a no-generation placeholder) or of a native partial response.
    A recorded grade (``has_recorded_grade``) always keeps it. An agent-stage fault whose owner recorded ``retry`` and
    has no grade keeps no answer: its response is the failed attempt's partial transcript, whatever its status.
    """
    if has_recorded_grade(row):
        return True  # A graded result stays the result, whatever the collector's flag or the stage's retry says.
    flag = row.get("_ng_collected_saved_answer")
    if isinstance(flag, bool):
        return flag
    if failure_stage(row) == "agent" and row.get("failure_outcome_action") == OutcomeAction.RETRY:
        return False
    answer = row.get("response") if "response" in row else row.get("_ng_failure_partial_response")
    return has_saved_generation({"response": answer}) or keeps_partial_response(row)


def holds_response(row: Mapping[str, Any]) -> bool:
    """Whether the row keeps a full or partial response object, usable as an answer or not."""
    return any(
        isinstance(row.get(key), Mapping)
        for key in ("response", "_ng_failure_response", "_ng_failure_partial_response")
    )


def has_unclassified_execution_failure(row: Mapping[str, Any]) -> bool:
    """A selected route failure names no owned cause that permits running the agent again.

    A request that was possibly delivered permits a fresh run only while the row keeps no full or partial
    response: then no saved answer can be duplicated or replaced.
    """
    if matches_failure_class(row, JUDGE_FAILURE_KINDS):
        return True  # Verification recovery may reuse an answer; it never reruns the agent.
    record = row.get("_ng_failure_record")
    failure = record.get("failure") if isinstance(record, Mapping) else None
    not_sent = isinstance(record, Mapping) and (
        record.get("delivery") == "not_sent" or isinstance(failure, Mapping) and failure.get("stage") == "admission"
    )
    possibly_delivered = (
        isinstance(record, Mapping)
        and record.get("source") == "collector"
        and record.get("delivery") == "possibly_delivered"
        and not not_sent
    )
    if (
        row.get("_ng_unrouted_run_error") is True
        and not not_sent
        and not (possibly_delivered and not holds_response(row))
    ):
        # An aborted collection retains this attempt as diagnostic history, not a replacement outcome.
        return True
    if possibly_delivered and holds_response(row):
        return True
    if row.get("_ng_failure_events_omitted"):
        # The transport's size bound omitted the report, so its events say nothing either way: the row's own failure
        # record and kind decide, as for a row without a report. The answer gates above (and a saved generation,
        # recorded grade or partial response, which every caller checks beside this) still keep a saved answer.
        return False
    events = row.get("_ng_failure_events")
    selected = row.get("_ng_failure_id")
    if "_ng_failure_id" not in row:
        return False  # Legacy producer fields remain usable without a selected event report.
    if not isinstance(events, Mapping) or not isinstance(selected, str) or not selected:
        return True
    seen = set()
    while selected is not None:
        if not isinstance(selected, str):
            return True
        event = events.get(selected)
        if not isinstance(event, Mapping) or selected in seen:
            # Store resume and legacy collection also use this gate, before the recovery reader runs.
            return True
        seen.add(selected)
        status = event.get("http_status")
        if (
            event.get("boundary") == "model"
            and event.get("failure_kind") == "model_request_error"
            and event.get("operation") == "model_request"
            and type(status) is int
            and 200 <= status < 300
        ):
            # The provider sent successful headers, but the model operation failed
            # before its body was usable. Running the agent again could regenerate.
            return True
        if (
            event.get("boundary") == "environment"
            and event.get("failure_kind") == "environment_server_failed"
            and event.get("operation") == "episode.deadline"
        ):
            # The relay stopped waiting; the agent may already have generated an answer.
            return True
        cause = event.get("cause_id")
        if cause is None:
            # A route failure with no deeper cause: the agent run returned no answer. It runs again fresh, within
            # the attempt cap, unless the row kept an answer; a legacy /run error other than 5xx stays refused.
            if event.get("failure_kind") == "agent_route_failed":
                return has_saved_generation(row)
            if (
                event.get("failure_kind") == "collection_dispatch_failed"
                and row.get("_ng_failure_class") == "agent_run_error"
            ):
                status = row.get("_ng_failure_http_status")
                return not (type(status) is int and 500 <= status <= 599) or has_saved_generation(row)
            return False
        selected = cause
    return False


logger = logging.getLogger(__name__)


# Collection/output choices can change between invocations without changing a task.
_COLLECTION_OPTIONS = frozenset(
    {
        "resume_from_cache",
        "resubmit",
        "checkpoint_dir",
        "checkpoint_every_s",
        "checkpoint_timeout_s",
        "dispatch_budget_s",
        "drain_margin_s",
        "dispatch_longest_first",
        "retry_terminal_timeouts",
        "allow_unsafe_resume",
        "output_jsonl_fpath",
        "input_jsonl_fpath",
        "num_samples_in_parallel",
        "max_resident_rollout_tasks",
        "retain_results_in_memory",
        "require_complete",
        "disable_aggregation",
        "disable_health_check",
        "health_check_workers",
        "health_check_ignored_checks",
        "health_check_max_not_final_failure_fraction",
        "health_check_max_final_failure_fraction",
        "upload_rollouts",
        "count_failure_classes_as_zero",
        "count_missing_rollouts_as_zero",
        "route_failures_to_sidecar",
        # How long a refused /run is resent: dispatch tuning, not a task setting. It is in ``_ADDED_COLLECTION_DEFAULTS``
        # too, so a run saved with it in its digest still resumes.
        "dispatch_connect_grace_s",
        # Timing/queue reports observe the run; they do not change task execution.
        "dispatch_start_report_within_s",
        "dispatch_start_report_min_fraction",
        "dispatch_all_started_within_s",
        "server_queue_metrics_urls",
        "server_queue_max_waiting_ratio",
        "server_queue_sample_interval_s",
        "long_rollout_s",
        "long_rollout_max_fraction",
        "timing_summary_interval_s",
        "timed_out_max_fraction",
    }
)
# Server addresses and credentials are resolved anew when a job restarts. Keep model
# names, prompts, generation settings and environment configuration in the digest.
_SERVER_RUNTIME_OPTIONS = frozenset(
    {
        "host",
        "port",
        "api_key",
        "openai_api_key",
        "policy_api_key",
        "judge_api_key",
        "tavily_api_key",
        "anthropic_api_key",
        "model_api_key",
        "switchyard_api_key",
        "hf_token",
        "num_workers",
        "uvicorn_reuse_port",
        "base_url",
        "openai_base_url",
        "policy_base_url",
        "judge_base_url",
        "sandbox_model_base_url",
        "model_base_url",
        "vllm_base_url",
        "switchyard_base_url",
        "head_server",
        "disallowed_ports",
        "port_range_low",
        "port_range_high",
        "model_call_capture_dir",
    }
)
_SERVER_KINDS = ("responses_api_agents", "responses_api_models", "resources_servers", "environment_servers")


_DEFAULT_MAX_ROLLOUT_ATTEMPTS = 3


def _get_max_rollout_attempts() -> int:
    """Read ``NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`` (positive int) or default to 3."""
    raw = os.environ.get("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS")
    if raw is None or raw == "":
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS
    try:
        n = int(raw)
        if n < 1:
            raise ValueError(f"must be >= 1, got {n}")
        return n
    except (TypeError, ValueError) as e:
        print(
            f"WARNING: could not parse NEMO_GYM_MAX_ROLLOUT_ATTEMPTS={raw!r} ({e}); "
            f"falling back to default {_DEFAULT_MAX_ROLLOUT_ATTEMPTS}.",
            flush=True,
        )
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS


def load_resubmit_offset(output_fpath: str | Path) -> int:
    """Read the number of sidecar lines excluded from the current failure budget."""
    path = resubmit_offset_path_for(Path(output_fpath))
    if not path.exists():
        return 0
    try:
        offset = int(path.read_text())
        if offset < 0:
            raise ValueError("negative offset")
    except (ValueError, OSError) as error:
        raise ConfigError(f"Cannot read resubmit budget offset {path}: {error}") from error
    return offset


def load_attempt_extension(output_fpath: str | Path) -> int:
    """Attempts an explicit recovery added to every rollout's saved attempt cap (`--extend-attempts`); 0 when none.

    A rollout's cap is the run's saved ``max_rollout_attempts`` plus this. An unreadable record stops the run rather
    than guess its cap.
    """
    path = attempt_extension_path_for(Path(output_fpath))
    if not path.exists():
        return 0
    try:
        text = path.read_text().strip()
        if not (text.isascii() and text.isdecimal()):
            raise ValueError(f"not a whole number: {text[:40]!r}")
        return int(text)
    except (ValueError, OSError) as error:
        raise ConfigError(f"Cannot read the recorded attempt extension {path}: {error}") from error


def load_rerun_masked_approvals(output_fpath: str | Path) -> frozenset[tuple[int, int, int]]:
    """The masked results an operator approved to run again (`--rerun-masked-without-artifact`): each approved
    attempt as (task index, rollout index, attempt index); empty when none.

    An approval names the attempt it covered, so a later attempt of the same rollout is never reopened by it. A line
    or entry that cannot be read approves nothing: the default is that a saved answer is never sampled again.
    """
    path = rerun_masked_journal_path_for(Path(output_fpath))
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return frozenset()
    approved = set()
    for line in lines:
        try:
            entries = json.loads(line)["identities"]
        except (ValueError, KeyError, TypeError):
            continue
        for entry in entries if isinstance(entries, list) else ():
            identity = entry.get("identity") if isinstance(entry, dict) else None
            attempt = entry.get("attempt") if isinstance(entry, dict) else None
            if (
                isinstance(identity, list)
                and len(identity) == 2
                and all(type(value) is int and value >= 0 for value in (*identity, attempt))
            ):
                approved.add((identity[0], identity[1], attempt))
    return frozenset(approved)


def record_resubmit_offset(output_fpath: str | Path) -> int:
    """Reset the failure budget without altering outcomes or execution identities.

    The caller must hold the run lock and validate/repair history first. The
    line-based format is shared with GDPVal's existing resubmit implementation.
    """
    output = Path(output_fpath)
    failures = failures_path_for(output)
    offset = 0
    if failures.exists():
        with failures.open("rb") as file:
            offset = sum(1 for _ in file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_file(resubmit_offset_path_for(output)) as file:
        file.write(f"{offset}\n".encode())
    return offset


def _plain(value: Any) -> Any:
    """Pydantic's Python dump may still contain nested Hydra containers."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def manifest_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_manifest.json")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _without_runtime_options(value: Any) -> Any:
    # Exclude only known configuration boundaries. Recursively dropping names such
    # as "port" would also erase meaningful tool schemas or environment parameters.
    result = {k: v for k, v in value.items() if k not in _SERVER_RUNTIME_OPTIONS and k not in _COLLECTION_OPTIONS}

    def settings_identity(settings):
        if not isinstance(settings, dict):
            return settings
        settings = {k: v for k, v in settings.items() if k not in _SERVER_RUNTIME_OPTIONS}
        # Only authentication headers at known HTTP settings boundaries are
        # operational. Other headers may change task data or model behavior.
        for name in ("headers", "default_headers", "openai_default_headers", "artifact_request_headers"):
            if isinstance(settings.get(name), dict):
                settings[name] = {
                    key: value
                    for key, value in settings[name].items()
                    if key.lower() not in {"authorization", "proxy-authorization", "x-api-key", "api-key"}
                }
        if isinstance(settings.get("token_id_capture"), dict):
            settings["token_id_capture"] = {k: v for k, v in settings["token_id_capture"].items() if k != "dir"}
        return settings

    for name, block in list(result.items()):
        if not isinstance(block, dict):
            continue
        updated = dict(block)
        for kind in _SERVER_KINDS:
            if kind in block and isinstance(block[kind], dict):
                updated[kind] = {
                    implementation: settings_identity(settings) for implementation, settings in block[kind].items()
                }
        result[name] = updated
    return result


def _normalize_runtime_references(value: Any) -> Any:
    """Keep recognized connection references stable wherever clients nest them.

    Literal nested task data is untouched, including fields named port or api_key.
    Preserve the reference name and any surrounding URL template in the identity.
    """
    if isinstance(value, dict):
        return {key: _normalize_runtime_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_runtime_references(item) for item in value]
    if isinstance(value, str):

        def normalize(match: re.Match[str]) -> str:
            reference = match[1]
            parts = reference.split(".")
            operational = reference in _SERVER_RUNTIME_OPTIONS or (
                len(parts) == 4 and parts[1] in _SERVER_KINDS and parts[3] in _SERVER_RUNTIME_OPTIONS
            )
            return f"<runtime:{reference}>" if operational else match[0]

        return re.sub(r"\$\{([^{}]+)\}", normalize, value)
    return value


# A setting holding a credential, matched on its name lower-cased with `-`, `.`, spaces and case changes as `_`:
# `exa_api_key`, `X-Api-Key`, `service_key`, `Authorization`, `X-Session`, `X_TOKEN`, `my_service_pw`, but not
# `max_tokens`, `author`, `reference_key` or `record_terminal_session`. A `..._key` counts only after a word that names
# an account or a service, since other `..._key` settings (`metric_key`, `reference_key`) select what is measured.
_CREDENTIAL_NAME = re.compile(
    r"api_?key|secret|passw(or)?d|credential|private_?key|access_?key|cookie"
    r"|(^|_)(token|pw|pwd|pass|bearer|o?auth)$|(^|_)o?auth(orization|entication)?(_|$)"
    r"|(^|_)(service|client|signing|license|master|account|subscription|encryption|session|app)_key$"
    r"|^(x_)?session(_id)?$"
)
_REDACTED = "<redacted>"
# An environment variable or config reference by name, a `NAME=value` assignment, and an HTTP credential scheme.
_ENV_REFERENCE = re.compile(r"\$\{oc\.env:\s*([A-Za-z_][A-Za-z0-9_]*)[^{}]*\}")
_CONFIG_REFERENCE = re.compile(r"\$\{([A-Za-z_][\w.\-]*)\}")
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$", re.DOTALL)
_HTTP_CREDENTIAL = re.compile(r"(?i)^(bearer|basic)\s+\S+$")


def _is_credential_name(name: object) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return bool(_CREDENTIAL_NAME.search(re.sub(r"[\s.\-]", "_", normalized).lower()))


def _credential_safe(value: Any, *, secret: bool = False) -> Any:
    """The value with every credential replaced by a fixed marker; it is what a run persists for its identity.

    A credential is any value below a credential-named key (every value of a ``secrets`` mapping, an ``Authorization``
    header), the ``value`` of a ``{name: SOME_TOKEN, value: ...}`` entry, the value of a ``SOME_TOKEN=...`` string, an
    HTTP ``Bearer``/``Basic`` credential, a URL password, and an interpolation of a credential-named environment variable
    or setting. Applied before interpolations are resolved, so such a credential is never read from the environment.
    """
    if isinstance(value, dict):
        named = value.get("name", value.get("key"))
        pair = isinstance(named, str) and _is_credential_name(named)
        return {
            key: _credential_safe(item, secret=secret or _is_credential_name(key) or (pair and key == "value"))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_credential_safe(item, secret=secret) for item in value]
    if value is None:
        return None
    if secret:
        return _REDACTED
    if not isinstance(value, str):
        return value
    value = _ENV_REFERENCE.sub(lambda m: _REDACTED if _is_credential_name(m[1]) else m[0], value)
    value = _CONFIG_REFERENCE.sub(lambda m: _REDACTED if _is_credential_name(m[1].split(".")[-1]) else m[0], value)
    assignment = _ASSIGNMENT.match(value)
    if assignment and _is_credential_name(assignment[1]):
        return f"{assignment[1]}={_REDACTED}"
    credential = _HTTP_CREDENTIAL.match(value)
    return f"{credential[1]} {_REDACTED}" if credential else mask_url_passwords(value)


def _configuration_identity(
    config: dict, global_config: dict, rows: list[dict], *, credential_safe: bool = True
) -> dict:
    """The configuration a run's results depend on, compared on resume.

    Credentials are redacted (see ``_credential_safe``): they do not change what a run measures, and the identity is
    persisted beside the manifest. ``credential_safe=False`` is the form an earlier Gym took its digest from; it is only
    ever compared in memory, to resume such a run.
    """
    # Resolve only the servers reachable from the materialized agents. Keep the
    # original root as interpolation context, but do not evaluate unused servers
    # (which may require credentials/environment variables unavailable here).
    safe = _credential_safe if credential_safe else lambda value: value
    raw = safe(_normalize_runtime_references(_plain(global_config)))
    context = OmegaConf.create(raw)
    filtered = _without_runtime_options(raw)
    servers = {
        name for name, block in raw.items() if isinstance(block, dict) and any(k in block for k in _SERVER_KINDS)
    }
    pending = {(row.get("agent_ref") or {}).get("name") for row in rows} - {None}
    pending.update(row["_ng_environment_server"] for row in rows if row.get("_ng_environment_server"))
    # Agent-routed legacy rows are wrapped by an Environment Server at dispatch.
    # Its orchestration settings are part of the task even without a row stamp.
    agents = {(row.get("agent_ref") or {}).get("name") for row in rows if not row.get("_ng_environment_server")} - {
        None
    }
    if agents:
        for name in servers:
            for settings in context[name].get("environment_servers", {}).values():
                if not OmegaConf.is_dict(settings):
                    continue
                # Match environment_server_agent_refs: any typed top-level agent
                # reference, or agent_server without a type. Resolve mapping/name
                # interpolations, but defer unavailable settings on unused servers.
                # A selected server is resolved in full below, so its errors remain fatal.
                for field in settings:
                    try:
                        reference = settings[field]
                        if not OmegaConf.is_dict(reference):
                            continue
                        kind = reference.get("type")
                        if kind == "responses_api_agents" or (field == "agent_server" and kind is None):
                            if reference.get("name") in agents:
                                pending.add(name)
                    except InterpolationResolutionError:
                        continue
    missing = pending - servers
    if missing:
        raise ConfigError(f"Cannot verify run identity: running configuration is missing servers {sorted(missing)!r}.")
    pending.update(
        row["task_source"] for row in rows if isinstance(row.get("task_source"), str) and row["task_source"] in servers
    )
    if not pending:
        pending = set(servers)
    selected = {}

    def references(value, *, server_reference: bool = False):
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str):
                if (server_reference or value.get("type") in _SERVER_KINDS) and name not in servers:
                    raise ConfigError(f"Cannot verify run identity: running configuration is missing server {name!r}.")
                if name in servers:
                    pending.add(name)
            for key, item in value.items():
                references(
                    item,
                    server_reference=key in {"model_server", "resources_server", "agent_server", "environment_server"},
                )
        elif isinstance(value, list):
            for item in value:
                references(item, server_reference=server_reference)

    while pending:
        name = pending.pop()
        if name in selected or name not in servers:
            continue
        context[name] = filtered[name]
        selected[name] = OmegaConf.to_container(context[name], resolve=True)
        references(selected[name])
    # Collection overrides may themselves contain DictConfig/ListConfig values.
    collection = {key: _plain(value) for key, value in config.items() if key not in _COLLECTION_OPTIONS}
    context["_collection_identity"] = safe(_normalize_runtime_references(collection))
    capture = raw.get("token_id_capture", {})
    context["_capture_identity"] = {key: value for key, value in capture.items() if key != "dir"}
    # Again after resolution: a resolved value can carry a credential its reference did not name (a URL password).
    return safe(
        {
            "collection": OmegaConf.to_container(context._collection_identity, resolve=True),
            "servers": selected,
            "token_id_capture": OmegaConf.to_container(context._capture_identity, resolve=True),
        }
    )


# Null defaults a server config once declared and later dropped. Null and absent resolve alike, so a run saved while a
# config still declared them resumes on the config without them; any other difference keeps the identities apart.
# These are the settings whose default is null, so only here a null and an absent setting are the same identity.
_REMOVED_NULL_DEFAULTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("responses_api_models", "vllm_model"): ("chat_template_kwargs", "sampling_overrides", "extra_body"),
}


def _server_settings(identity: dict) -> Iterator[tuple[str, str, str, dict]]:
    """Each server block's settings with its concrete path: server name, kind and implementation."""
    for name, block in identity["servers"].items():
        for kind in _SERVER_KINDS:
            implementations = block.get(kind) if isinstance(block, dict) else None
            for implementation, settings in (implementations or {}).items():
                if isinstance(settings, dict):
                    yield name, kind, implementation, settings


def _without_null_defaults(identity: dict) -> dict:
    """The identity without null settings whose default is null: there a null setting and an absent one are the same.

    Elsewhere null is a value of its own (claude_code_agent's ``max_turns: null`` is unlimited, its omission 30
    turns), so it stays in the identity.
    """
    normalized = json.loads(json.dumps(identity))
    for _, kind, implementation, settings in _server_settings(normalized):
        for key in _REMOVED_NULL_DEFAULTS.get((kind, implementation), ()):
            if key in settings and settings[key] is None:
                del settings[key]
    return normalized


# Each configuration digest this process recorded, mapped to the digests an earlier Gym recorded for the same
# configuration, each with the server settings it had as removed null defaults. Deterministic: configurations that
# share a digest differ only by null settings.
_LEGACY_CONFIG_DIGESTS: dict[str, dict[str, tuple[str, ...]]] = {}
# Above this many absent defaults, only the variant with all of them restored is tried, not every subset.
_LEGACY_SUBSET_LIMIT = 10


# Collection settings added after a run could have been saved, with the default each had on release. They are left out
# of the digest (``_COLLECTION_OPTIONS``), but a Gym that had them recorded them, so a run saved by it carries them in
# its digest. A setting with a default must not change the resume fingerprint of a run saved before it existed.
_ADDED_COLLECTION_DEFAULTS: dict[str, float] = {"dispatch_connect_grace_s": 60.0}


def _earlier_collection_forms(identity: dict, config: Mapping[str, Any]) -> Iterator[dict]:
    """The identity as the Gyms before and after each added collection setting recorded it.

    First the identity as it is (a Gym before the setting existed), then with the setting restored at its default and at
    the value this configuration sets. A run saved with another value is still compared on its saved identity, if any.
    """
    yield identity
    for key, default in _ADDED_COLLECTION_DEFAULTS.items():
        values = [default]
        configured = config.get(key)
        if isinstance(configured, (int, float)) and not isinstance(configured, bool) and configured != default:
            values.append(configured)
        for value in values:
            variant = json.loads(json.dumps(identity))
            variant.setdefault("collection", {})[key] = value
            yield variant


def _legacy_config_digests(identity: dict, config: Mapping[str, Any] | None = None) -> dict[str, tuple[str, ...]]:
    """Digests an earlier Gym recorded for this configuration, with the settings each restored as null defaults.

    Restoration is per concrete server path: a removed default is added only to a server that lacks it, never over a
    value another server of the same implementation sets. Each form is also tried with the collection settings added
    since restored (``_earlier_collection_forms``).
    """
    digests: dict[str, tuple[str, ...]] = {}
    for base in _earlier_collection_forms(identity, config or {}):
        for digest, restored in _null_default_digests(base).items():
            digests.setdefault(digest, restored)
    return digests


def _null_default_digests(identity: dict) -> dict[str, tuple[str, ...]]:
    digests: dict[str, tuple[str, ...]] = {_digest(identity): ()}
    absent = sorted(
        {
            (name, kind, implementation, key)
            for name, kind, implementation, settings in _server_settings(identity)
            for key in _REMOVED_NULL_DEFAULTS.get((kind, implementation), ())
            if key not in settings
        }
    )
    sizes = range(1, len(absent) + 1) if len(absent) <= _LEGACY_SUBSET_LIMIT else (len(absent),)
    for size in sizes:
        for restored in combinations(absent, size):
            variant = json.loads(json.dumps(identity))
            for name, kind, implementation, settings in _server_settings(variant):
                for server, k, i, key in restored:
                    if (server, k, i) == (name, kind, implementation) and key not in settings:
                        settings[key] = None
            digests.setdefault(
                _digest(variant), tuple(f"servers.{server}.{k}.{i}.{key}" for server, k, i, key in restored)
            )
    return digests


# The normalized identity behind each configuration digest this process recorded (deterministic: digest of identity).
_CONFIG_IDENTITIES: dict[str, dict] = {}


def config_identity_path_for(manifest_path: Path) -> Path:
    """The normalized configuration identity a manifest's ``config_digest`` was taken from, saved beside it."""
    return manifest_path.with_suffix(".identity.json")


def resume_drift_path_for(manifest_path: Path) -> Path:
    """One line per resume that proceeded across operational configuration drift, naming the keys."""
    return manifest_path.with_suffix(".resume-drift.jsonl")


# Settings that change what is measured. A difference in any of them, anywhere in the identity, refuses a resume.
_SEMANTIC_KEYS = frozenset(
    {
        # dataset, split and task limits
        "split",
        "dataset",
        "datasets",
        "limit",
        "task_ids",
        "num_repeats",
        "num_repeats_add_seed",
        # agent and policy model identity, and its served window
        "agent_name",
        "agent_map",
        "fan_out",
        "model",
        "model_name",
        "served_model_name",
        "max_model_len",
        "max_context_length",
        "context_window",
        # sampling
        "responses_create_params",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "max_output_tokens",
        "max_tokens",
        "max_completion_tokens",
        "max_new_tokens",
        "seed",
        "sampling_overrides",
        "extra_body",
        "chat_template_kwargs",
    }
)
# A setting named like this is operational (timeouts, retries, transport, ports, concurrency, health, paths, tracking),
# unless a segment above already made its key semantic. Every other setting is semantic: an unknown name could change
# what the run produces (an agent's step budget, which tools it executes), so a resume across it is refused by name.
_OPERATIONAL_SETTING = re.compile(
    r"(^|_)(timeout|timeouts|deadline|retry|retries|attempts|backoff|wait|interval|port|ports|host|hostname|url|urls"
    r"|endpoint|concurrency|concurrent|parallel|parallelism|workers|semaphore|pool|transport|version|log|logging"
    r"|verbose|debug"
    r"|dir|path|fpath|cache|health|heartbeat|keepalive|tracking|mlflow|wandb|otel|telemetry|canary)(_|$)|_(s|ms)$"
)
# Any other key under a segment naming grading or the model is semantic: a judge, grader, verifier, reward, rubric,
# reference-answer or prompt setting.
_SEMANTIC_PARTS = (
    "judge",
    "grad",
    "verif",
    "reward",
    "scor",
    "rubric",
    "metric",
    "prompt",
    "model",
    "sampling",
    "eval_",
    "answer",
    "reference",
)
# A path to a prompt, template or rubric names what the model or verifier reads, not where something is kept: two
# shipped grading templates are different rubrics. The generic path rule does not make such a path operational.
_SEMANTIC_ASSET_PATH = re.compile(r"(^|_)(prompt|prompts|template|templates|rubric|rubrics)_(.+_)?(dir|path|fpath)$")


def _is_semantic(key: str) -> bool:
    segments = key.split(".")
    # A credential is redacted, so only its presence can differ; the key used does not change what is measured.
    if any(_is_credential_name(segment) for segment in segments):
        return False
    # Task data (task_parameters and the like) is part of the experiment whatever its keys are called.
    if any(segment in _SEMANTIC_KEYS or segment.startswith("task") for segment in segments):
        return True
    if _SEMANTIC_ASSET_PATH.search(segments[-1]):
        return True
    if any(part in segment for segment in segments[:-1] for part in _SEMANTIC_PARTS):
        # Under a grading or model segment only operational-named leaves (a judge's timeout) stay operational.
        return not _OPERATIONAL_SETTING.search(segments[-1])
    return not _OPERATIONAL_SETTING.search(segments[-1])


def _differing_keys(saved: Any, current: Any, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two identities; a list is one value."""
    if isinstance(saved, dict) and isinstance(current, dict):
        keys = []
        for key in sorted(set(saved) | set(current), key=str):
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved or key not in current:
                keys.append(dotted)
            else:
                keys.extend(_differing_keys(saved[key], current[key], dotted))
        return keys
    return [] if saved == current else [prefix or "<root>"]


def _configuration_drift(
    path: Path, saved: "RunManifest", current: "RunManifest"
) -> tuple[list[str], list[str]] | None:
    """The semantic and operational keys that differ from the saved identity, or None when it was not saved.

    The saved identity counts only when it is the one the manifest's ``config_digest`` was taken from.
    """
    try:
        identity = json.loads(config_identity_path_for(path).read_bytes())
    except (OSError, ValueError):
        return None
    if current.config_digest not in _CONFIG_IDENTITIES or _digest(identity) != saved.config_digest:
        return None
    # An identity an earlier Gym saved may hold credentials: compare on its credential-safe form.
    keys = _differing_keys(_credential_safe(identity), _CONFIG_IDENTITIES[current.config_digest])
    return [key for key in keys if _is_semantic(key)], [key for key in keys if not _is_semantic(key)]


def _record_resume_drift(path: Path, saved: "RunManifest", current: "RunManifest", keys: list[str]) -> None:
    from datetime import datetime, timezone

    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": saved.run_id,
        "saved_config_digest": saved.config_digest,
        "current_config_digest": current.config_digest,
        "operational_keys": keys,
    }
    with resume_drift_path_for(path).open("a", encoding="utf-8") as journal:
        journal.write(json.dumps(entry, sort_keys=True) + "\n")
    if not keys:
        return
    message = (
        f"Resuming across operational configuration drift in {', '.join(keys)}; "
        f"recorded in {resume_drift_path_for(path).name}."
    )
    logger.warning(message)
    warnings.warn(message, stacklevel=3)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_output_file(path: Path, mode: int | None = None) -> Iterator[BinaryIO]:
    """Replace a file after a successful write, retaining its sharing permissions unless ``mode`` sets them."""
    temporary = None
    try:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        # Exclusive creation honors the caller's umask. Rewrites retain existing
        # sharing permissions instead of inheriting NamedTemporaryFile's 0600.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(descriptor, "wb") as file:
            if mode is not None:
                os.fchmod(file.fileno(), mode)
            elif path.exists():
                os.fchmod(file.fileno(), stat.S_IMODE(path.stat().st_mode))
            yield file
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict, mode: int | None = None) -> None:
    """Publish JSON metadata without exposing a partial rewrite to readers."""
    with atomic_output_file(path, mode) as file:
        file.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


class PendingRejudge(BaseModel):
    """A judge-only re-judgment admitted before its judge call; it closes when its outcome is saved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    predecessor_failure_id: Annotated[str, Field(min_length=1)]
    predecessor_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    admitted_at: Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]  # UTC epoch seconds


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    selection_policy: Literal["latest_allocated"] = "latest_allocated"
    # Controller-owned execution numbers. Reserving a number is not a failure.
    next_attempt: dict[str, Annotated[int, Field(ge=1, strict=True)]] = Field(default_factory=dict)
    max_rollout_attempts: Annotated[int, Field(ge=1, strict=True)] | None = None
    retry_terminal_timeouts: bool = False
    run_id: str
    source_digest: str
    materialized_digest: str
    config_digest: str
    legacy_import: bool = False
    identity_overridden: bool = False
    # Admitted judge-only re-judgments whose outcome is not saved yet, by logical rollout id.
    pending_rejudge: dict[str, PendingRejudge] = Field(default_factory=dict)

    @classmethod
    def create(cls, source: Path, rows: list[dict], config: dict, global_config: dict) -> "RunManifest":
        identity = _configuration_identity(config, global_config, rows)
        config_digest = _digest(_without_null_defaults(identity))
        legacy = _legacy_config_digests(identity, config)
        try:
            # The digest an earlier Gym took over resolved credentials, compared in memory only, so its runs resume.
            earlier = _configuration_identity(config, global_config, rows, credential_safe=False)
        except (ConfigError, OmegaConfBaseException):
            earlier = None
        if earlier is not None:
            legacy = {_digest(_without_null_defaults(earlier)): ()} | _legacy_config_digests(earlier, config) | legacy
        _LEGACY_CONFIG_DIGESTS[config_digest] = _LEGACY_CONFIG_DIGESTS.get(config_digest, {}) | legacy
        manifest = cls(
            max_rollout_attempts=_get_max_rollout_attempts(),
            run_id=uuid4().hex,
            retry_terminal_timeouts=config.get("retry_terminal_timeouts", False),
            source_digest=_file_digest(source),
            materialized_digest=_digest(rows),
            config_digest=config_digest,
        )
        _CONFIG_IDENTITIES[config_digest] = _without_null_defaults(identity)
        return manifest

    def matches_config(self, saved_digest: str) -> bool:
        """Whether a saved configuration digest names this configuration, in this or an earlier recorded form."""
        return saved_digest == self.config_digest or saved_digest in _LEGACY_CONFIG_DIGESTS.get(self.config_digest, {})

    def write(self, path: Path) -> None:
        # Publish a complete manifest, or leave the previous one intact. An empty pending_rejudge is omitted so a
        # manifest without admissions keeps the version-2 shape that readers without the field accept.
        value = self.model_dump(mode="json")
        if not value["pending_rejudge"]:
            del value["pending_rejudge"]
        atomic_write_json(path, value)
        # The identity behind config_digest, so a later resume can name the settings that differ. Readers trust it only
        # while its digest is the manifest's config_digest.
        if self.config_digest in _CONFIG_IDENTITIES:
            atomic_write_json(config_identity_path_for(path), _CONFIG_IDENTITIES[self.config_digest], mode=0o600)

    @classmethod
    def import_legacy(cls, rows: list[dict]) -> "RunManifest":
        return cls(
            max_rollout_attempts=_get_max_rollout_attempts(),
            run_id=uuid4().hex,
            source_digest="unverified",
            materialized_digest=_digest(rows),
            config_digest="unverified",
            legacy_import=True,
        )


class ResumeRefusedError(ConfigError):
    """Saved rollouts whose identity does not verify against the current run are not resumed."""

    failure_kind = COLLECTION_RESUME_REFUSED


def validate_resume(
    path: Path,
    current: RunManifest | None,
    materialized_path: Path,
    *,
    allow_unsafe: bool = False,
) -> RunManifest | None:
    """Reject unknown or incompatible saved work before any output is modified.

    A configuration that differs from the saved one is compared on the identity saved beside the manifest: a
    difference only in operational settings (timeouts, retries, transport, ports, concurrency, health, paths,
    tracking) resumes, with a warning and a line in the resume-drift journal; a difference in a semantic setting
    (dataset, task limits, repeats, agent and model identity, sampling, judge and grading) refuses and names it. A
    manifest saved without that identity can only be compared by digest, as before.

    ``allow_unsafe`` is an explicit compatibility escape hatch for legacy artifacts or
    deliberate identity overrides. Unsupported/corrupt manifests remain errors.
    """
    if not path.exists():
        problem = "Saved rollouts have no run manifest; their input/configuration identity cannot be verified."
        saved = None
    else:
        try:
            saved = RunManifest.model_validate_json(path.read_bytes())
            with materialized_path.open(encoding="utf-8") as file:
                saved_rows = [json.loads(line) for line in file if line.strip()]
        except (ValidationError, ValueError, OSError) as error:
            raise ConfigError(f"Cannot read recovery manifest {path} or materialized inputs: {error}") from error
        mismatches = []
        operational_drift: list[str] = []
        if saved.legacy_import:
            mismatches.append("unverified legacy run identity")
        if saved.identity_overridden:
            mismatches.append("previously overridden run identity")
        if saved.materialized_digest != _digest(saved_rows):
            mismatches.append("saved materialized inputs")
        if current is None:
            mismatches.append("current input/configuration identity")
        else:
            for field, description in (
                ("source_digest", "source dataset"),
                ("materialized_digest", "current materialized inputs"),
            ):
                if getattr(saved, field) != getattr(current, field):
                    mismatches.append(description)
            if not current.matches_config(saved.config_digest):
                drift = _configuration_drift(path, saved, current)
                if drift is None:
                    mismatches.append(
                        "resolved configuration (this run saved only a digest of its configuration, so the "
                        "settings that differ cannot be named)"
                    )
                elif drift[0]:
                    mismatches.append(f"resolved configuration (semantic settings differ: {', '.join(drift[0])})")
                else:
                    operational_drift = drift[1]
            elif current.config_digest != saved.config_digest:
                # Matched as an earlier Gym recorded it: name the removed null defaults it still had, if any.
                operational_drift = list(
                    _LEGACY_CONFIG_DIGESTS.get(current.config_digest, {}).get(saved.config_digest, ())
                )
            elif (
                current.config_digest == saved.config_digest
                and current.config_digest in _CONFIG_IDENTITIES
                and not config_identity_path_for(path).exists()
            ):
                # The same configuration: save its identity now, so a later resume of this run can name any drift.
                atomic_write_json(
                    config_identity_path_for(path), _CONFIG_IDENTITIES[current.config_digest], mode=0o600
                )
        if not mismatches:
            if operational_drift or resume_drift_path_for(path).exists():
                # Once a run has drifted, every later resume adds a line, so the last one is the latest resume's.
                _record_resume_drift(path, saved, current, operational_drift)
            return saved
        problem = "Cannot resume with incompatible " + ", ".join(mismatches) + "."
    if not allow_unsafe:
        raise ResumeRefusedError(problem + " Start a fresh run or explicitly set allow_unsafe_resume=true.")
    warnings.warn(problem + " Continuing because allow_unsafe_resume=true; saved inputs will be reused.", stacklevel=2)
    return saved.model_copy(update={"identity_overridden": True}) if saved is not None else None


def observed_elapsed(record: dict[str, Any]) -> float | None:
    """Best-effort per-rollout wallclock from a result/failure row."""
    # Native environments own their result schema; response and metadata need
    # not be mappings. Optional scheduling hints must never invalidate an answer.
    response = record.get("response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    for candidate in (
        record.get("elapsed_seconds"),
        metadata.get("elapsed_seconds") if isinstance(metadata, Mapping) else None,
    ):
        try:
            if candidate is not None:
                value = float(candidate)
                if value > 0:
                    return value
        except (TypeError, ValueError):
            continue
    return None


def matches_failure_class(record: Mapping[str, Any], classes: Collection[str]) -> bool:
    """Match shared failure names and retained producer routing names alike."""
    return record.get("_ng_failure_class") in classes or record.get("_ng_failure_legacy_class") in classes


def is_terminal_failure(record: Mapping[str, Any], *, retry_terminal_timeouts: bool = False) -> bool:
    """Whether a persisted failure must be gated on resume.

    By default a row is terminal iff it is stamped ``_ng_failure_terminal``, so
    an agent that marks its timeouts terminal on purpose keeps them gated.

    ``retry_terminal_timeouts`` is for agents whose older builds incorrectly
    stamped per-attempt timeouts terminal. A timeout reflects the
    load/remaining walltime of that attempt, so it remains retryable (up to the
    normal max-attempt cap).

    ``timeout_exceeded`` is an ``_ng_failure_class`` label that agents already
    write to the failures sidecar (pinchbench, nooa). It predates
    ``nemo_gym.core.failure_kinds`` and is not registered there; ``agent_timeout`` is
    the shared name.

    As ``failure_kinds`` requires, this function decides retryability for the
    occurrence, under an explicit caller opt-in; the name carries no retry meaning.
    """
    if not retry_terminal_timeouts:
        return bool(record.get("_ng_failure_terminal"))
    if matches_failure_class(record, {"timeout_exceeded"}):
        return False
    return bool(record.get("_ng_failure_terminal"))


def run_lock_path_for(output: Path) -> Path:
    """Stable lock file; never unlink it while another process may hold it open."""
    output = output.resolve()
    return output.with_name(output.stem + "_run.lock")


@contextmanager
def run_lock(output: Path, *, checkpoint_dir: Path | None = None) -> Iterator[None]:
    """Own a run before preparing inputs, restoring state, or writing artifacts.

    Collection currently acquires its own lock in run_from_config; do not nest
    it inside this context. Checkpoint integration must extend that acquisition to
    cover restore and supply checkpoint_dir, rather than acquire a second lock.
    When supplied, the directory lock prevents two outputs sharing checkpoints.
    The filesystem must support flock across all writer nodes. Kernel locks are released
    on process death; the presence of an old lock file does not block recovery.
    """
    paths = {run_lock_path_for(output)}
    if checkpoint_dir is not None:
        paths.add(checkpoint_dir.resolve() / ".collector.lock")
    with ExitStack() as files:
        for path in sorted(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            file = files.enter_context(path.open("a+b"))
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                file.seek(0)
                holder = file.read(4096).decode("utf-8", errors="replace").strip() or "owner details unavailable"
                raise ConfigError(
                    f"Another collector owns this run ({path}; holder: {holder}). Wait for it to stop before resuming."
                ) from error
            except OSError as error:
                if error.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOLCK}:
                    raise ConfigError(
                        f"Cannot lock run at {path}: the filesystem must support flock across all writer nodes. "
                        "Use a filesystem with distributed flock support; Lustre noflock disables it and "
                        "localflock only locks within one node."
                    ) from error
                raise
            file.seek(0)
            file.truncate()
            file.write(
                json.dumps(
                    {"pid": os.getpid(), "host": socket.gethostname(), "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID")}
                ).encode()
                + b"\n"
            )
            file.flush()
        yield


def has_only_intentional_gaps(*, expected: int, completed: int, intentionally_omitted: int) -> bool:
    """Accept omissions only alongside completed tasks, without inventing scores."""
    return completed > 0 and intentionally_omitted > 0 and completed + intentionally_omitted == expected


class IncompleteEvaluationError(RuntimeError):
    """Saved partial results are valid, but the evaluation still has unfinished work."""

    exit_code = 75

    def __init__(
        self, message: str, *, retryable: bool = True, made_progress: bool = True, failure_kind: str | None = None
    ) -> None:
        if retryable and not made_progress:
            message += " No retry progress was made; automatic restarts use the infrastructure retry limit."
        super().__init__(message)
        # 75 continues productive work; an empty invocation must spend a bounded
        # infrastructure retry. 76 stops when another attempt cannot help.
        self.exit_code = (75 if made_progress else 1) if retryable else 76
        # The registered kind that ended the run, left for the next segment's guard (see record_failure_kind).
        self.failure_kind = failure_kind
