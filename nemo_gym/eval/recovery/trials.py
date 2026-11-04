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
"""The trial gate's observations: whether each expected trial was observed, and its counted failures."""

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from nemo_gym.core.failure_kinds import (
    AGENT_TIMEOUT_INFRASTRUCTURE,
    EMPTY_ANSWER,
    INVALID_OUTPUT_BUDGET,
    INVALID_REQUEST_PARAMETER,
    JUDGE_UNPARSEABLE,
    MALFORMED_ANSWER,
    MODEL_CONTEXT_LIMIT,
    MODEL_FAULT_KINDS,
    MODEL_POLICY_RESPONSE_TIMEOUT,
    MODEL_REQUEST_ERROR,
    MODEL_SERVER_ERROR,
    NON_POLICY_MODEL_REJECTION,
    SERVER_FAULT_KINDS,
    TRUNCATED_ANSWER,
    is_server_status,
)
from nemo_gym.eval.recovery.decision import (
    FAILURE_CLASS,
    FAILURE_EVENTS,
    FAILURE_ID,
    SCORED_UNPARSEABLE_REASON,
    IdentityState,
    RecoveryDecision,
    _known_kind,
)
from nemo_gym.eval.recovery.evidence import (
    CollectedEvidence,
    Identity,
    _identity,
)
from nemo_gym.eval.recovery.grading import (
    _bound_answer,
    _empty_answer,
    _incomplete_answer,
    _judge_unparseable,
    _model_failure,
    _reward,
    _unknown_grading_outcome,
    _unparsed_judge_verdict,
    grading_artifact,
)
from nemo_gym.eval.recovery.policy import (
    _policy_evidence,
)


def _percent(limit: float) -> str:
    return f"{limit * 100:g}%"


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _clip_text(quality: Mapping[str, Any]) -> str:
    counts = quality.get("context_window_exceeded")
    fields = ("true", "false", "absent")
    if isinstance(counts, dict) and all(_count(counts.get(k)) is not None for k in fields):
        return str({k: counts[k] for k in fields})
    return "not reported or malformed"


def trial_boundary(quality: Mapping[str, Any] | None) -> str:
    """Why trial failures outside the sidecar cannot be ruled out, from the run's own persisted facts.

    Every persisted value is type-checked: a malformed report is named as malformed, never read as a fact.
    """
    quality = quality if isinstance(quality, Mapping) else {}
    provenance = quality.get("provenance")
    servers = provenance.get("servers") if isinstance(provenance, dict) else None
    if not isinstance(provenance, dict) or provenance.get("resolved_config") != "read":
        state = provenance.get("resolved_config") if isinstance(provenance, dict) else "not reported"
        facts = f"run provenance {state if state in ('absent', 'unreadable', 'not reported') else 'malformed'}"
    elif not isinstance(servers, dict) or not all(isinstance(f, dict) for f in servers.values()):
        facts = "run provenance malformed"
    else:
        agents = {
            str(name): (fact.get("implementation"), fact.get("resources_server"), fact.get("skip_verification"))
            for name, fact in servers.items()
            if fact.get("kind") == "responses_api_agents"
        }
        environments = sorted(str(n) for n, f in servers.items() if f.get("kind") == "environment_servers")
        route = provenance.get("route_failures_to_sidecar")
        facts = (
            f"agents (implementation, resources server, skip_verification) {agents}, environment servers "
            f"{environments}, route_failures_to_sidecar {route if isinstance(route, bool) else None}"
        )
    runtime = quality.get("observed_runtime")
    runtime_count = _count(runtime.get("invocation_observations")) if isinstance(runtime, dict) else None
    runtime_valid = runtime_count is not None and all(
        isinstance(runtime.get(key), dict)
        and all(_count(runtime[key].get(k)) is not None for k in ("true", "false", "absent"))
        and sum(runtime[key][k] for k in ("true", "false", "absent")) == runtime_count
        for key in ("in_session", "verification_skipped")
    )
    if runtime_valid:
        session_text = f"observed invocation in_session {runtime['in_session']}; runtime verification_skipped {runtime['verification_skipped']}; these observations do not certify the full trial population"
    else:
        session_text = "agent session (in_session) runtime observations are absent or malformed, so it is unknown"
    ended = _count(quality.get("ended_on_failed_model_call"))
    skipped = quality.get("verification_skipped")
    fields = ("true", "false", "absent")
    if isinstance(skipped, dict) and all(_count(skipped.get(k)) is not None for k in fields):
        skipped_text = str({k: skipped[k] for k in fields})
    else:
        skipped_text = "not reported or malformed"
    # The producer sets ended_on_error only where the rollout's model calls were captured, so the count is a floor.
    ended_text = f"at least {ended} (counted where calls were captured)" if ended is not None else "unknown"
    return (
        f"{facts}, rollouts ended on a failed model call {ended_text}, rows by verification_skipped {skipped_text}, "
        f"rows whose step_usage records context_window_exceeded {_clip_text(quality)} (a signal from a best-effort "
        "error-text match or model metadata, not proof that the input was clipped); "
        f"{session_text}; model-generated "
        "code that fails, raises or times out is scored by the benchmark protocol, and whether it counts as a trial "
        "failure is open for the benchmark PIC"
    )


def _cut_judge_reply(call: Mapping[str, Any]) -> bool:
    """Whether a captured judge reply was cut short: an incomplete Responses reply, or a chat completion that stopped
    on its output cap or a content filter."""
    meta = call.get("response_metadata") if isinstance(call, Mapping) else None
    return isinstance(meta, Mapping) and (
        (meta.get("dialect") == "responses" and meta.get("response_status") == "incomplete")
        or (
            meta.get("dialect") == "chat"
            and meta.get("response_status") is None
            and meta.get("finish_reason") in ("length", "content_filter")
        )
    )


def _observed_trial_operations(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None,
    *,
    judge_recovery: dict[str, int] | None = None,
    grading_failures: list[str] | None = None,
    policy_recovery: dict[str, int] | None = None,
    verifier_recorded: list[Identity] | None = None,
    artifact_recorded: list[Identity] | None = None,
    trajectory_incomplete: dict[Identity, list[str]] | None = None,
    agent_failed_graded: list[Identity] | None = None,
) -> bool:
    """Whether every expected trial that is not a counted failure is observed: its saved answer is bound to its grading.

    Observed means a recorded grading bound to the saved answer and score: one completed verifier's grading operation
    bound to the row, its own grading record, a saved grading artifact (``GRADING_ARTIFACTS``) with the collector's
    binding, or the model's own failed answer. Only that decides coverage. Each trial's model/tool trajectory is
    joined too (its one agent invocation, turns, calls, tools and failure events), but that is observability
    evidence: a trial whose trajectory does not join is reported in ``trajectory_incomplete`` with the kinds of its
    missing evidence (``_trajectory_kinds``), and it is still observed.

    Join observed model/tool operations and the outer verifier invocation for every expected trial.

    A trial whose verifier recorded no grading operation and no grading record of its own is observed through a
    saved grading artifact (``GRADING_ARTIFACTS``) or through the model's own failed answer the benchmark's rule
    scored (``_model_failure``); those trials go to ``artifact_recorded``, and the verifier's captured calls are its
    own. An environment reward record stands in for a missing verifier invocation.

    A source-reviewed programmatic grading operation must bind the saved response and score. A verifier that
    recorded no grading operation and owns no model call is observed through its own grading record instead
    (``_verifier_recorded``); those trials go to ``verifier_recorded``. Empty captured model calls or a reward
    alone never establish grading coverage; old or uninstrumented verifiers remain unknown.
    Recovered grading attempts are reported separately and do not count toward the trial-health threshold;
    every failed attempt must join the same completed, result-bound grading dependency. A policy request retried
    inside one completed operation is recovered the same way (``policy_recovery`` receives its counts): each failed
    attempt is an event of that operation and of the row, and the answer is the final attempt's.
    """
    decision = _health_view(decision)
    from nemo_gym.core.episode_context import maybe_rollout_id_from_run_body

    # A settled model failure (MODEL_FINAL) or a rollout that ended on a failed policy request is a counted trial
    # failure with its recorded kind; only the other trials need their operations and grading observed. A row an
    # accepted partial stage omitted is no trial (``trial_population``), so it needs no observation.
    if (
        collected is None
        or not collected.quality
        or collected.quality.get("source_matches") is not True
        or decision.counts.get(IdentityState.SCORED, 0) + decision.counts.get(IdentityState.MODEL_FINAL, 0)
        != trial_population(decision)
    ):
        return False
    rows = collected.results
    expected = set(decision.identities.get(IdentityState.SCORED.value, []))
    if rows is None or len(rows) != len(expected) or {_identity(row) for row in rows} != expected:
        return False
    final_zero = final_zero_trials(decision, collected)
    provenance = collected.quality.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("resolved_config") != "read":
        return False
    servers = provenance.get("servers")
    if not isinstance(servers, dict):
        return False
    input_task_ids = {_identity(item): item.get("_ng_collected_task_id") for item in collected.materialized or []}
    populations = (set(), set(), set(), set(), set())
    grading_ids = set()
    recovered = {
        "judge_recovered_attempts": 0,
        "judge_recovered_operations": 0,
        "observed_grading_attempt_errors": 0,
        "observed_grading_failed_operations": 0,
    }
    recovered_policy = {"policy_recovered_attempts": 0, "policy_recovered_operations": 0}
    recovered_events = {}
    invalid_grades = []
    recorded_trials = []
    artifact_trials = []
    incomplete: dict[Identity, list[str]] = {}
    agent_failed: list[Identity] = []

    def nonempty(value):
        return isinstance(value, str) and bool(value.strip())

    def matches(ref, call):
        meta = call["response_metadata"]
        return (
            isinstance(ref, dict)
            and (
                ref.get("model_call_id") == call["model_call_id"]
                if ref.get("model_call_id")
                else ref.get("model_ref") == meta["model_ref"] and ref.get("response_id") == meta["response_id"]
            )
            and (ref.get("model_ref") is None or ref["model_ref"] == meta["model_ref"])
            and (ref.get("response_id") is None or ref["response_id"] == meta["response_id"])
        )

    def completed_once(call, *, judge=False, valid_grade=True, truncated=False):
        """One successful HTTP response with one completed physical attempt, ids unique in the run.

        ``truncated``: the policy's final answer that ran into its output cap (a length finish on an incomplete or
        chat response); the model answered, so it is a model fault the trial gate counts, not missing evidence.
        """
        meta = call["response_metadata"]
        operations, report = meta["model_operations"], meta["failure_report"]
        if truncated and not (
            meta.get("finish_reason") == "length"
            and meta.get("error_category") in (None, "incomplete")
            and (
                meta.get("response_status") == "incomplete"
                or (meta.get("dialect") == "chat" and meta.get("response_status") is None)
            )
        ):
            return False
        if (
            type(meta["status_code"]) is not int
            or not 200 <= meta["status_code"] < 300
            or (meta.get("error_category") is not None and not truncated)
            or not (
                truncated
                or meta.get("response_status") == "completed"
                or (
                    meta.get("dialect") == "chat"
                    and meta.get("response_status") is None
                    and meta.get("finish_reason") == "stop"
                )
                or (
                    judge
                    and not valid_grade
                    and (
                        (meta.get("dialect") == "responses" and meta.get("response_status") == "incomplete")
                        or (
                            meta.get("dialect") == "chat"
                            and meta.get("response_status") is None
                            and meta.get("finish_reason") in ("length", "content_filter")
                        )
                    )
                )
            )
            or not isinstance(operations, list)
            or not operations
            or not isinstance(report["events"], dict)
            or report["omitted"] is not False
            or report["failure_id"] is not None
        ):
            return False
        if judge and (len(operations) > 1 or report["events"]):
            events = report["events"]
            used = set()
            failed_operations = 0
            for number, operation in enumerate(operations):
                final = number == len(operations) - 1
                op_id, attempt_id = operation.get("operation_id"), operation.get("final_attempt_id")
                attempts = operation.get("http_attempts")
                transport = operation.get("transport_failures")
                if (
                    not nonempty(op_id)
                    or not nonempty(attempt_id)
                    or op_id in populations[2]
                    or attempt_id in populations[4]
                    or operation.get("outcome") != ("completed" if final else "failed")
                    or type(attempts) is not int
                    or attempts < 1
                    or type(transport) is not int
                    or not 0 <= transport <= attempts
                    or any(
                        type(operation.get(key)) not in (int, float)
                        or not math.isfinite(operation[key])
                        or operation[key] < 0
                        for key in ("retry_wait_s", "request_elapsed_ms")
                    )
                ):
                    return False
                owned = {
                    key: event
                    for key, event in events.items()
                    if isinstance(event, dict) and event.get("operation_id") == op_id
                }
                # A failed HTTP attempt or a transport failure (no HTTP status) before the final attempt is its own
                # recorded event of this operation, as for a policy request.
                if len(owned) != attempts - int(final) + transport or transport != sum(
                    event.get("http_status") is None for event in owned.values()
                ):
                    return False
                ids = set()
                for key, event in owned.items():
                    eid = event.get("attempt_id")
                    if (
                        key != event.get("event_id")
                        or not nonempty(key)
                        or not nonempty(eid)
                        or eid in ids
                        or eid in populations[4]
                        or key in used
                        or key in recovered_events
                        or event.get("boundary") != "model"
                        or event.get("operation") != "model_request"
                        or event.get("failure_kind") != "model_request_error"
                        or event.get("cause_id") is not None
                        or not (
                            event.get("http_status") is None
                            or (type(event["http_status"]) is int and 400 <= event["http_status"] <= 599)
                        )
                    ):
                        return False
                    ids.add(eid)
                    used.add(key)
                if final:
                    if operation.get("failure_id") is not None or attempt_id in ids:
                        return False
                elif (
                    operation.get("failure_id") not in owned
                    or owned[operation["failure_id"]].get("attempt_id") != attempt_id
                ):
                    return False
                if owned:
                    failed_operations += 1
                populations[2].add(op_id)
                populations[4].update(ids | {attempt_id})
            if used != set(events):
                return False
            if not all(
                nonempty(value)
                for value in (
                    call["model_call_id"],
                    meta["model_ref"]["type"],
                    meta["model_ref"]["name"],
                    meta["response_id"],
                )
            ):
                return False
            identity = (meta["model_ref"]["type"], meta["model_ref"]["name"], meta["response_id"])
            if call["model_call_id"] in populations[0] or identity in populations[1]:
                return False
            populations[0].add(call["model_call_id"])
            populations[1].add(identity)
            recovered_events.update(events)
            recovered["observed_grading_attempt_errors"] += len(used)
            recovered["observed_grading_failed_operations"] += failed_operations
            if valid_grade:
                recovered["judge_recovered_attempts"] += len(used)
                recovered["judge_recovered_operations"] += failed_operations
            return True
        if len(operations) != 1:
            return False
        operation = operations[0]
        events = report["events"]
        attempts, transport = operation["http_attempts"], operation["transport_failures"]
        if (
            operation["outcome"] != "completed"
            or operation["failure_id"] is not None
            or type(attempts) is not int
            or attempts < 1
            or type(transport) is not int
            or transport < 0
            # Every failed attempt before the final one is its own recorded event of this operation.
            or len(events) != attempts - 1 + transport
            or not all(
                nonempty(value)
                for value in (
                    call["model_call_id"],
                    meta["model_ref"]["type"],
                    meta["model_ref"]["name"],
                    meta["response_id"],
                    operation["operation_id"],
                    operation["final_attempt_id"],
                )
            )
        ):
            return False
        # A policy request retried inside one completed operation (a timeout, a reset or a 5xx, then an answer) is a
        # recovered attempt, not a trial failure: each failed attempt must join this operation exactly once.
        failed_ids = set()
        for key, event in events.items():
            status = event.get("http_status") if isinstance(event, dict) else None
            if (
                not isinstance(event, dict)
                or key != event.get("event_id")
                or not nonempty(event.get("attempt_id"))
                or event["attempt_id"] in failed_ids
                or event["attempt_id"] in populations[4]
                or event["attempt_id"] == operation["final_attempt_id"]
                or key in recovered_events
                or event.get("boundary") != "model"
                or event.get("operation") != "model_request"
                or event.get("operation_id") != operation["operation_id"]
                or event.get("failure_kind") != "model_request_error"
                or event.get("cause_id") is not None
                or not (status is None or (type(status) is int and 400 <= status <= 599))
            ):
                return False
            failed_ids.add(event["attempt_id"])
        if sum(event.get("http_status") is None for event in events.values()) != transport:
            return False
        if operation["final_attempt_id"] in populations[4]:
            return False
        populations[4].add(operation["final_attempt_id"])
        values = (
            call["model_call_id"],
            (meta["model_ref"]["type"], meta["model_ref"]["name"], meta["response_id"]),
            operation["operation_id"],
        )
        if any(value in seen for value, seen in zip(values, populations)):
            return False
        for value, seen in zip(values, populations):
            seen.add(value)
        populations[4].update(failed_ids)
        recovered_events.update(events)
        recovered_policy["policy_recovered_attempts"] += len(events)
        recovered_policy["policy_recovered_operations"] += bool(events)
        return True

    def trajectory_complete(row, observed, calls, artifact, truncated):
        """The trial's model/tool trajectory joins its one agent invocation, turn by turn, call by call."""
        try:
            invocations, turns, tools = observed["invocations"], observed["turns"], observed["tool_calls"]
            if (
                not all(isinstance(items, list) for items in (invocations, calls, turns, tools))
                or len(invocations) != 1
                or not calls
                or len(turns) != len(calls)
                or type(observed["tool_call_count"]) is not int
                or observed["tool_call_count"] != len(tools)
                or not isinstance(observed["gaps"], list)
                or any(gap.get("code") != "resolution_unavailable" for gap in observed["gaps"])
            ):
                return False
            invocation = invocations[0]
            session = invocation["in_session"]
            task_id = row.get("materialized_task_id") if session else input_task_ids.get(_identity(row))
            if (
                type(session) is not bool
                or (session and (collected.checkpoint_enabled is not False or collected.managed_selection is None))
                or not nonempty(task_id)
                or observed["task_id"] != task_id
                or observed["rollout_id"] != maybe_rollout_id_from_run_body(row)
                or not nonempty(invocation["invocation_id"])
                or invocation["status"] != "completed"
                or invocation.get("parent_invocation_id") is not None
                or invocation.get("error_type") is not None
                or invocation.get("stop_reason") is not None
                or invocation["verification_skipped"] is not (None if session else False)
                or (session and invocation["invocation_id"] != "root")
                or len(invocation["model_calls"]) != len(calls)
            ):
                return False
            ordered = []
            for number, turn in enumerate(turns, 1):
                if (
                    turn["invocation_id"] != invocation["invocation_id"]
                    or turn["task_id"] != task_id
                    or turn["rollout_id"] != observed["rollout_id"]
                    or type(turn["turn_no"]) is not int
                    or turn["turn_no"] != number
                    or not isinstance(turn["model_calls"], list)
                    or len(turn["model_calls"]) != 1
                ):
                    return False
                joined = [call for call in calls if matches(turn["model_calls"][0], call)]
                if len(joined) != 1 or not matches(invocation["model_calls"][number - 1], joined[0]):
                    return False
                ordered.append(joined[0])
            requested = {}
            tool_by_id = {tool["tool_call_id"]: tool for tool in tools}
            if len(tool_by_id) != len(tools):
                return False
            for number, (call, turn) in enumerate(zip(ordered, turns), 1):
                if not completed_once(call, truncated=truncated and number == len(ordered)):
                    return False
                inputs, outputs = call["input_items"], call["output_items"]
                if not isinstance(inputs, list) or not isinstance(outputs, list) or not outputs:
                    return False
                # This proof joins client-executed function tools. Hosted/computer/custom operations
                # need their own runtime evidence; an assistant message beside them does not certify them.
                if any(item.get("type") not in {"message", "reasoning", "function_call"} for item in outputs) or any(
                    item.get("type") not in {None, "message", "reasoning", "function_call", "function_call_output"}
                    for item in inputs
                ):
                    return False
                consumed = [item["call_id"] for item in inputs if item.get("type") == "function_call_output"]
                if len(consumed) != len(set(consumed)) or set(consumed) != set(requested):
                    return False
                emitted = [item for item in outputs if item.get("type") == "function_call"]
                for item in emitted:
                    tool_id, name = item["call_id"], item["name"]
                    if not nonempty(tool_id) or not nonempty(name) or tool_id in requested:
                        return False
                    tool = tool_by_id.get(tool_id)
                    if (
                        tool is None
                        or tool["invocation_id"] != invocation["invocation_id"]
                        or tool["tool_name"] != name
                        or tool["status"] != "completed"
                        or tool["error_type"] is not None
                    ):
                        return False
                    requested[tool_id] = name
                # step_count is producer-specific (before/after a tool execution); the explicit tool
                # request/execution/result joins above establish coverage without reinterpreting that counter.
                if type(turn["step_count"]) is not int or turn["step_count"] < 0:
                    return False
                if number == len(ordered):
                    # A truncated or empty final answer may hold only reasoning or a cut-off message.
                    if (
                        emitted
                        or not (truncated or artifact == "model_failure")
                        and not any(
                            item.get("type") == "message"
                            and item.get("role") == "assistant"
                            and item.get("status") == "completed"
                            for item in outputs
                        )
                    ):
                        return False
                elif not emitted:
                    return False
            saved_response = row["response"]
            return (
                set(requested) == set(tool_by_id)
                and saved_response.get("id") == ordered[-1]["response_metadata"]["response_id"]
                and saved_response.get("status") == ("incomplete" if truncated else "completed")
            )
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            return False

    try:
        for row in rows:
            if _identity(row) in final_zero:
                continue
            if _scored_kind(row) is not None:
                continue  # A counted failure with its own recorded kind (``_scored_fault``), like a final zero.
            if _answer_fault(row, collected):
                # A counted model failure (``_answer_fault``), like a final zero: only its answer must be observed.
                if not _observed_incomplete_answer(row):
                    return False
                continue
            if _unparsed_judge_verdict(row):
                continue  # A counted grading fault (``_scored_fault``): its reward is no grade to observe.
            truncated = _incomplete_answer(row)
            prior_recovered_events = set(recovered_events)
            agent = servers[row["agent_ref"]["name"]]
            owner = _verification_owner(row["agent_ref"]["name"], agent)
            # An agent without a resources server grades its own rollouts (verifiers_agent); it has no resources
            # server to look up, and only its own grading record can observe such a trial (below).
            resource = servers[owner] if owner == agent.get("resources_server") else None
            environment = servers[row["_ng_environment_server"]]
            if (
                agent.get("kind") != "responses_api_agents"
                or (resource is not None and resource.get("kind") != "resources_servers")
                or environment.get("kind") != "environment_servers"
                or row.get("_ng_result_type") != environment.get("implementation")
                or row.get(FAILURE_CLASS) is not None
                or row.get("failure_kind") is not None
                or row.get(FAILURE_ID) is not None
                or row.get("_ng_failure_events_omitted", False) is not False
                or not isinstance(row.get(FAILURE_EVENTS, {}), dict)
            ):
                return False
            observed = row["ng_trial_observations"]
            verifiers, calls = observed["verifier_invocations"], observed.get("model_calls")
            judge_calls = None
            single = verifiers[0] if isinstance(verifiers, list) and len(verifiers) == 1 else None
            graded = single.get("grading_operations") if isinstance(single, dict) else None
            # Without a grading operation, a saved grading artifact or the model's own failed answer records the
            # grading; the verifier's captured calls (a judge's) are then its own, never the policy's.
            artifact = None
            recorded = single is not None and _verifier_recorded(single, row, owner)
            if not graded and not recorded:
                artifact = grading_artifact(row) or ("model_failure" if _model_failure(row) else None)
            # An agent without a resources server is observed through its own grading record, or, when no verifier
            # ran, through a saved grading artifact (the environment's reward record, or the agent rubric's own saved
            # score); its own failed answer alone does not observe it.
            if resource is None and not recorded and not (artifact not in (None, "model_failure") and verifiers == []):
                return False
            if artifact is not None and single is not None and isinstance(calls, list):
                owned = single.get("model_calls")
                if not isinstance(owned, list):
                    return False
                judge_calls = []
                for ref in owned:
                    hit = [call for call in calls if matches(ref, call)]
                    if len(hit) != 1 or any(hit[0] is seen for seen in judge_calls):
                        return False
                    judge_calls.append(hit[0])
                calls = [call for call in calls if not any(call is judge for judge in judge_calls)]
            if isinstance(graded, list) and len(graded) == 1 and (graded[0] or {}).get("kind") == "llm_judge":
                # Every dependency is exactly one captured call owned by this verifier invocation, every owned call is
                # a dependency, and only the remaining calls are the policy's.
                dependencies, owned = graded[0].get("dependencies"), single.get("model_calls")
                if not dependencies or not isinstance(dependencies, list) or not isinstance(owned, list):
                    return False
                if not isinstance(calls, list):
                    return False
                judge_calls = []
                for ref in dependencies:
                    hit = [call for call in calls if matches(ref, call)]
                    if len(hit) != 1 or any(hit[0] is seen for seen in judge_calls):
                        return False
                    if sum(1 for item in owned if matches(item, hit[0])) != 1:
                        return False
                    judge_calls.append(hit[0])
                if len(owned) != len(judge_calls):
                    return False
                calls = [call for call in calls if not any(call is judge for judge in judge_calls)]
            # Grading coverage: the trial's one verifier (none only beside an environment reward) and how it graded.
            if not isinstance(verifiers, list) or len(verifiers) != (
                0 if not verifiers and (artifact == "environment_reward" or resource is None) else 1
            ):
                return False
            verifier = verifiers[0] if verifiers else None
            if verifier is not None and (
                verifier["status"] != "completed"
                or verifier["failure_kind"] is not None
                or _grading_sandbox_failed(verifier.get("sandbox_observations"))
                or (judge_calls is None and verifier["model_calls"] != [])
                or not nonempty(verifier["invocation_id"])
                or verifier["invocation_id"] in populations[3]
            ):
                return False
            grading = verifier.get("grading_operations") if verifier is not None else None
            if artifact is not None:
                # A saved grading artifact must be bound to the saved answer and score; the model's own failed
                # answer is a counted model failure instead.
                if artifact != "model_failure" and not _bound_answer(row):
                    return False
                # The verifier's own judge replies are part of that grading: each completed once.
                if judge_calls and not all(completed_once(call, judge=True) for call in judge_calls):
                    return False
                grade = None
                artifact_trials.append(_identity(row))
            elif judge_calls is None and _verifier_recorded(verifier, row, owner):
                grade = None
                recorded_trials.append(_identity(row))
            elif not isinstance(grading, list) or len(grading) != 1 or not isinstance(grading[0], dict):
                return False
            else:
                grade = grading[0]
            binding = row.get("_ng_collected_grading_binding")
            if grade is not None and (
                grade.get("kind") != ("programmatic_calculation" if judge_calls is None else "llm_judge")
                or grade.get("outcome")
                not in (
                    ("completed", "unknown")
                    if judge_calls is not None and grading_failures is not None
                    else ("completed",)
                )
                or (judge_calls is None and grade.get("dependencies") != [])
                or not nonempty(grade.get("operation_id"))
                or grade["operation_id"] in grading_ids
                or not isinstance(binding, str)
                or len(binding) != 64
                or grade.get("result_binding_sha256") != binding
            ):
                return False
            if grade is not None:
                grading_ids.add(grade["operation_id"])
            if verifier is not None:
                populations[3].add(verifier["invocation_id"])
            # A completed grade that rests on a judge reply cut by its output cap or a content filter has no valid
            # verdict behind it: where the caller counts those, it is an observed grading without a valid verdict.
            cut_reply = (
                grade is not None
                and judge_calls is not None
                and grade["outcome"] == "completed"
                and grading_failures is not None
                and any(_cut_judge_reply(call) for call in judge_calls)
            )
            if (
                grade is not None
                and judge_calls is not None
                and not all(
                    completed_once(call, judge=True, valid_grade=grade["outcome"] == "completed" and not cut_reply)
                    for call in judge_calls
                )
            ):
                return False
            if cut_reply:
                invalid_grades.append(_identity(row))
            if (
                grade is not None
                and judge_calls is not None
                and grade["outcome"] == "unknown"
                and not _unknown_grading_outcome(row)  # counted once, as ``judge_unparseable`` (``_scored_fault``)
            ):
                invalid_grades.append(_identity(row))
            # An agent invocation that says verification was skipped contradicts any recorded grading.
            if any(
                isinstance(invocation, dict) and invocation.get("verification_skipped") is True
                for invocation in (observed.get("invocations") or [])
            ):
                return False
            # A judge's call is the verifier's own: the agent never asked for it.
            agent_refs = [
                ref
                for invocation in (observed.get("invocations") or [])
                if isinstance(invocation, dict)
                for ref in (invocation.get("model_calls") or [])
            ]
            if judge_calls and any(matches(ref, call) for ref in agent_refs for call in judge_calls):
                return False
            saved_response = row.get("response")
            if not isinstance(saved_response, dict) or saved_response.get("error") is not None:
                return False
            # The agent's own run failed, yet its saved answer was graded: observed, and reported as a fact.
            if any(
                isinstance(invocation, dict) and invocation.get("status") == "failed"
                for invocation in (observed.get("invocations") or [])
            ):
                agent_failed.append(_identity(row))
            # Trajectory evidence: reported, never a coverage condition.
            if not trajectory_complete(row, observed, calls, artifact, truncated):
                incomplete[_identity(row)] = _trajectory_kinds(observed) or ["operation_join_incomplete"]
            # Every recorded model failure is accounted for: a failure is never hidden behind a graded answer.
            fresh = {key: value for key, value in recovered_events.items() if key not in prior_recovered_events}
            # A graded row whose rollout ended on the model's own failed request (its context limit, say) is a model
            # result at any reward: the verifier graded the state the model reached. That request's failure is
            # accounted for; one of an evaluation-side kind is not.
            ended = _ended_on_failed_request(row) if _reward(row) not in (None, 0) else None
            if ended is not None and ended[0] in MODEL_FAULT_KINDS:
                report = (ended[1].get("response_metadata") or {}).get("failure_report")
                events = report.get("events") if isinstance(report, dict) else None
                if isinstance(events, dict):
                    fresh |= {key: event for key, event in events.items() if isinstance(event, dict)}
            if _unaccounted_failures(row, fresh):
                return False
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    if judge_recovery is not None:
        judge_recovery.update(recovered)
    if grading_failures is not None:
        grading_failures.extend(invalid_grades)
    if policy_recovery is not None:
        policy_recovery.update(recovered_policy)
    if verifier_recorded is not None:
        verifier_recorded.extend(recorded_trials)
    if artifact_recorded is not None:
        artifact_recorded.extend(artifact_trials)
    if trajectory_incomplete is not None:
        trajectory_incomplete.update(incomplete)
    if agent_failed_graded is not None:
        agent_failed_graded.extend(agent_failed)
    return True


# A sandbox outcome that reports the sandbox did not finish its work; "unknown" means the provider did not report one.
_SANDBOX_FAILURES = frozenset({"failed", "timeout", "oom", "sandbox_error", "cancelled"})


def _teardown_failure_only(item: Mapping) -> bool:
    """Whether a sandbox observation's only failure is its teardown: the producer records a failed close or release
    after the work as outcome ``sandbox_error`` or ``unknown`` with that cleanup error as its ``error_type``."""
    errors = item.get("cleanup_errors")
    return (
        isinstance(errors, list)
        and bool(errors)
        and item.get("outcome") in ("unknown", "sandbox_error")
        and item.get("error_type") == errors[0]
    )


def _grading_sandbox_failed(observations: Any) -> bool:
    """Whether a verifier's sandbox observations report that a grading sandbox failed, or are malformed.

    They are lifecycle evidence of the sandboxes the verification used, not a grading record: a completed or
    unreported outcome, an unconfirmed or failed teardown, a failing test command's exit code and the agent's own
    sandbox outcome do not leave a bound grading unobserved. A verifier sandbox whose outcome is a failure, or that
    records an error other than its teardown's, does.
    """
    if observations is None:
        return False
    if not isinstance(observations, list):
        return True
    return any(
        not isinstance(item, Mapping)
        or (
            item.get("role") == "verifier"
            and not _teardown_failure_only(item)
            and (item.get("outcome") in _SANDBOX_FAILURES or item.get("error_type") is not None)
        )
        for item in observations
    )


def _failed_request(call: Any) -> tuple[str, Any] | None:
    """The failure kind and HTTP status of a captured call whose final model operation failed with its own recorded
    request event, or None."""
    meta = call.get("response_metadata") if isinstance(call, dict) else None
    operations = meta.get("model_operations") if isinstance(meta, dict) else None
    report = meta.get("failure_report") if isinstance(meta, dict) else None
    events = report.get("events") if isinstance(report, dict) else None
    if not isinstance(operations, list) or not operations or not isinstance(operations[-1], dict):
        return None
    final = operations[-1]
    event = events.get(final.get("failure_id")) if isinstance(events, dict) else None
    if (
        final.get("outcome") == "failed"
        and isinstance(event, dict)
        and event.get("boundary") == "model"
        and event.get("operation") == "model_request"
        and event.get("operation_id") == final.get("operation_id")
        and isinstance(event.get("failure_kind"), str)
    ):
        return event["failure_kind"], event.get("http_status")
    return None


# The kind of a request the model server rejected, by the reason it recorded: the conversation outgrew the context
# window (the model's own limit), the request's output budget alone exceeds it, or a request parameter carries a value
# the deployment does not support (both setups the deployment cannot serve, evaluation-side).
LENGTH_REFUSAL_KINDS = {
    "context_length_exceeded": MODEL_CONTEXT_LIMIT,
    "invalid_output_token_parameter": INVALID_OUTPUT_BUDGET,
    "invalid_request_parameter": INVALID_REQUEST_PARAMETER,
}


def _length_refusal_kind(meta: Any) -> str | None:
    """The kind of a captured call rejected with HTTP 4xx on length (``LENGTH_REFUSAL_KINDS``), or None.

    The reason is the call's ``error_category`` or its ``local_response_reason``; the status is the call's own, its
    upstream's, or that of a failure event it reported. A 5xx among them is the server's fault, and an event of another
    kind than a rejected request (a transport failure) is its own cause, never the length's.
    """
    if not isinstance(meta, dict):
        return None
    kinds = {
        LENGTH_REFUSAL_KINDS[reason]
        for reason in (meta.get("error_category"), meta.get("local_response_reason"))
        if isinstance(reason, str) and reason in LENGTH_REFUSAL_KINDS
    }
    if len(kinds) != 1:
        return None
    report = meta.get("failure_report")
    events = report.get("events") if isinstance(report, dict) else None
    events = [event for event in events.values() if isinstance(event, dict)] if isinstance(events, dict) else []
    if any(event.get("failure_kind") not in (None, MODEL_REQUEST_ERROR) for event in events):
        return None
    statuses = [
        status
        for status in [
            meta.get("status_code"),
            meta.get("upstream_status_code"),
            *(event.get("http_status") for event in events),
        ]
        if type(status) is int and status >= 400
    ]
    return kinds.pop() if statuses and all(status <= 499 for status in statuses) else None


def _policy_call(row: Mapping[str, Any], call: Any) -> bool:
    """Whether a captured call is the evaluated policy's: its model is the row's policy server, or none is named."""
    policy = row.get("ng_policy_model_server")
    meta = call.get("response_metadata") if isinstance(call, dict) else None
    model = meta.get("model_ref") if isinstance(meta, dict) else None
    return not isinstance(policy, str) or (isinstance(model, dict) and model.get("name") == policy)


def _last_captured_call(calls: list) -> Any:
    """The call a rollout ended on, for an agent whose invocations reference none of its calls.

    The highest capture position (``call_index``, the durable append order). When every call recorded its start
    time, that call must also be the latest started; otherwise the end is ambiguous and there is none.
    """
    if not calls:
        return None
    last = calls[-1]
    starts = [call.get("started_at") if isinstance(call, dict) else None for call in calls]
    if all(type(start) in (int, float) for start in starts) and starts[-1] < max(starts):
        return None
    return last


def infrastructure_timeouts(decision: RecoveryDecision, collected: CollectedEvidence | None) -> set[Identity]:
    """Scored identities whose graded result is an agent timeout with a server-side fault in its window.

    A Harbor graded zero a run without a Store cannot reopen (``is_infrastructure_agent_timeout``): it stays in the
    results, and the trial gate counts it once, as ``AGENT_TIMEOUT_INFRASTRUCTURE``, an evaluation-side fault.
    """
    from nemo_gym.rollouts.rollout_recovery import is_infrastructure_agent_timeout

    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    return {
        _identity(row)
        for row in ((collected.results or []) if collected is not None else [])
        if _identity(row) in scored and is_infrastructure_agent_timeout(row)
    }


def final_zero_trials(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[Identity, str]:
    """Scored identities whose rollout ended on a failed policy request and was scored zero, with the failure kind.

    The last policy call of the row's one invocation ends in a failed model operation whose failure event the call
    recorded, and the row's reward is zero. Such a trial is settled as a failure under the benchmark protocol: it
    has no selectable response, and it is counted once, in the trial family. An agent that records no call of its own
    (it has no invocation, or its invocations reference none) ended on a failed request when its last captured call (``_last_captured_call``)
    failed that way; an earlier failed call is no end and stays unaccounted. The kind is keyed on the failure's HTTP
    status (``status_fault_kind``): a 5xx reply is the server's fault, and another model's 4xx rejection is
    evaluation-side. A policy whose own last call was rejected on length (``_length_refusal_kind``) ended there,
    whatever its event's kind: ``model_context_limit`` for the context window, ``invalid_output_budget`` for an output
    budget the window cannot hold. The policy's last call is its last captured one when its invocations reference
    none, since another model's capture may follow.
    """
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    found: dict[Identity, str] = {}
    infrastructure = infrastructure_timeouts(decision, collected)
    for row in (collected.results or []) if collected is not None else []:
        reward = row.get("reward")
        if _identity(row) not in scored or type(reward) not in (int, float) or reward != 0:
            continue
        if _identity(row) in infrastructure:
            continue  # Counted once, as its agent timeout's server-side fault.
        ended = _ended_on_failed_request(row)
        if ended is not None:
            found[_identity(row)] = ended[0]
    return found


def _ended_on_failed_request(row: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]] | None:
    """The failure kind of the request a row's rollout ended on, with that call, or None (``final_zero_trials``).

    The last policy call of the row's one invocation ended in a failed model operation whose failure event the call
    recorded, or, for an agent that recorded no call of its own, its last captured call failed that way; a policy whose
    own last call was rejected on length ended there, whatever its event's kind.
    """
    observed = row.get("ng_trial_observations")
    invocations = observed.get("invocations") if isinstance(observed, dict) else None
    # No invocation names a call: an agent that recorded no tool exchange has none at all.
    unreferenced = isinstance(invocations, list) and all(
        isinstance(item, dict) and item.get("model_calls") == [] for item in invocations
    )
    if unreferenced and isinstance(observed.get("model_calls"), list):
        own = _last_captured_call([call for call in observed["model_calls"] if _policy_call(row, call)])
        if own is not None and (refused := _length_refusal_kind(own.get("response_metadata"))):
            return refused, own
        last = _last_captured_call(observed["model_calls"])
        failed = _failed_request(last)
        if failed is not None:
            return status_fault_kind(failed[0], failed[1], policy=_policy_call(row, last)), last
        return None
    if not isinstance(invocations, list) or len(invocations) != 1 or not isinstance(invocations[0], dict):
        return None
    refs, calls = invocations[0].get("model_calls"), observed.get("model_calls")
    if not isinstance(refs, list) or not refs or not isinstance(refs[-1], dict) or not isinstance(calls, list):
        return None
    ref = refs[-1]
    last = [
        call
        for call in calls
        if isinstance(call, dict)
        and isinstance(call.get("response_metadata"), dict)
        and (
            call.get("model_call_id") == ref["model_call_id"]
            if ref.get("model_call_id")
            else ref.get("response_id") is not None
            and call["response_metadata"].get("response_id") == ref["response_id"]
            and call["response_metadata"].get("model_ref") == ref.get("model_ref")
        )
    ]
    meta = last[0]["response_metadata"] if len(last) == 1 else None
    if meta is not None and _policy_call(row, last[0]) and (refused := _length_refusal_kind(meta)):
        return refused, last[0]
    operations = meta.get("model_operations") if isinstance(meta, dict) else None
    report = meta.get("failure_report") if isinstance(meta, dict) else None
    if not isinstance(operations, list) or not operations or not isinstance(operations[-1], dict):
        return None
    final = operations[-1]
    events = report.get("events") if isinstance(report, dict) else None
    event = events.get(final.get("failure_id")) if isinstance(events, dict) else None
    if (
        final.get("outcome") == "failed"
        and isinstance(event, dict)
        and event.get("boundary") == "model"
        and event.get("operation") == "model_request"
        and event.get("operation_id") == final.get("operation_id")
        and isinstance(event.get("failure_kind"), str)
    ):
        return status_fault_kind(event["failure_kind"], event.get("http_status")), last[0]
    return None


def _graded(row: Mapping[str, Any]) -> bool:
    """Whether the row's one verifier recorded grading operations for it that all completed with a valid verdict.

    An operation without a valid verdict does not resolve the trial; the gate's join decides whether it is a fault.
    """
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, dict) else None
    operations = (
        verifiers[0].get("grading_operations") if isinstance(verifiers, list) and len(verifiers) == 1 else None
    )
    return (
        isinstance(operations, list)
        and bool(operations)
        and all(isinstance(item, dict) and item.get("outcome") == "completed" for item in operations)
    )


def _bound_grading(row: Mapping[str, Any]) -> bool:
    """Whether the row is ``_graded`` and every grading operation binds its saved answer and score, as the gate
    requires (``_ng_collected_grading_binding``)."""
    observed = row.get("ng_trial_observations")
    binding = row.get("_ng_collected_grading_binding")
    return (
        _graded(row)
        and _bound_answer(row)
        and all(
            item.get("result_binding_sha256") == binding
            for item in observed["verifier_invocations"][0]["grading_operations"]
        )
    )


def _bound_artifact(row: Mapping[str, Any]) -> bool:
    """Whether a saved grading artifact or the model's own failed answer resolves the row, as the gate requires: an
    artifact other than the model's failed answer must bind the saved answer and score."""
    if grading_artifact(row) is not None and _bound_answer(row):
        return True
    return grading_artifact(row) is None and _model_failure(row)


def _verification_owner(agent_name: Any, agent: Mapping[str, Any]) -> Any:
    """Who verifies an agent's rollouts: its resources server, or the agent itself when it names none.

    An agent that runs its own rubric (verifiers_agent) has no resources server; its grading record names the agent.
    """
    server = agent.get("resources_server")
    return server if server is not None else agent_name


def _verifier_recorded(verifier: Any, row: Mapping[str, Any], resources_server: Any) -> bool:
    """Whether the row's verifier saved its own grading record for it, in place of a recorded grading operation.

    The shared verify path records it only for a verify execution that completed without asking any model. It must
    name the row's resources server (or the agent, for an agent that verifies its own rollouts), carry the saved score with a ``scored`` verdict and bind the saved response and
    score. A verifier that recorded a grading operation or owns a model call is judged by those, never by the record.
    """
    record = verifier.get("grading_record") if isinstance(verifier, dict) else None
    binding, reward = row.get("_ng_collected_grading_binding"), row.get("reward")
    return (
        isinstance(record, dict)
        and verifier.get("status") == "completed"
        and verifier.get("failure_kind") is None
        and verifier.get("grading_operations") is None
        and verifier.get("model_calls") == []
        and isinstance(resources_server, str)
        and bool(resources_server)
        and record.get("verifier") == resources_server
        and record.get("verdict") == "scored"
        and type(record.get("score")) in (int, float)
        and type(reward) in (int, float)
        and record["score"] == reward
        and isinstance(binding, str)
        and len(binding) == 64
        and record.get("result_binding_sha256") == binding
    )


def _recorded_by_verifier(row: Mapping[str, Any], collected: CollectedEvidence) -> bool:
    """Whether the row's one verifier invocation saved its own grading record for it (``_verifier_recorded``)."""
    provenance = (collected.quality or {}).get("provenance")
    servers = provenance.get("servers") if isinstance(provenance, dict) else None
    agent_ref = row.get("agent_ref")
    agent = servers.get(agent_ref.get("name")) if isinstance(servers, dict) and isinstance(agent_ref, dict) else None
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, dict) else None
    return (
        isinstance(agent, dict)
        and isinstance(verifiers, list)
        and len(verifiers) == 1
        and _verifier_recorded(verifiers[0], row, _verification_owner(agent_ref.get("name"), agent))
    )


def _verifying_servers(collected: CollectedEvidence) -> str:
    """The resources servers the run's agents name, from the run provenance, for an unobserved-trial reason."""
    provenance = (collected.quality or {}).get("provenance")
    servers = provenance.get("servers") if isinstance(provenance, dict) else None
    names = sorted(
        {
            str(fact["resources_server"])
            for fact in (servers.values() if isinstance(servers, dict) else ())
            if isinstance(fact, dict) and fact.get("kind") == "responses_api_agents" and fact.get("resources_server")
        }
    )
    return ", ".join(names) if names else "a resources server the run provenance does not name"


def _health_view(decision: RecoveryDecision) -> RecoveryDecision:
    """``decision`` as the trial gate counts it: a scored row planned for a re-judge because its judge gave no verdict
    (``SCORED_UNPARSEABLE_REASON``) is a scored trial whose fault, ``judge_unparseable``, is evaluation-side. The
    recovery action is unchanged; only health reads the row as it stands until the re-judge settles it."""
    planned = {
        item.identity
        for item in decision.evidence
        if item.state == IdentityState.JUDGE_RECOVERABLE and item.reason == SCORED_UNPARSEABLE_REASON
    }
    if not planned:
        return decision
    evidence = [
        item.model_copy(update={"state": IdentityState.SCORED, "failure_kind": None, "reason": None})
        if item.identity in planned and item.state == IdentityState.JUDGE_RECOVERABLE
        else item
        for item in decision.evidence
    ]
    identities: dict[str, list[Identity]] = {}
    for item in evidence:
        identities.setdefault(item.state.value, []).append(item.identity)
    counts = {state: len(ids) for state, ids in identities.items()}
    return decision.model_copy(update={"evidence": evidence, "identities": identities, "counts": counts})


def trial_population(decision: RecoveryDecision) -> int:
    """The trial denominator: every expected trial, including repeats, failed and missing ones, less the rows an
    accepted partial stage omitted (no trial: reported separately, never in a numerator or denominator)."""
    return decision.expected - decision.counts.get(IdentityState.OMITTED, 0)


def trial_counts(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, Any] | None:
    """Acceptance's trial counts: observed healthy, unhealthy and unobserved, over the expected trials.

    A saved grading artifact or the model's own failed answer (``grading_artifact``, ``_model_failure``) resolves a
    trial too. ``unhealthy`` are the failures the trial gate counts (``trial_faults``, model and evaluation-side). When the
    gate's observation join covers every other trial, its observed gradings without a valid verdict are unhealthy
    too and nothing is unobserved. Otherwise ``unobserved`` are the other scored trials whose verifier recorded
    neither completed grading operations bound to the saved row nor its own grading record: nothing resolved their
    grading, so they are unverified, never healthy. ``observed_healthy`` are the rest; ``verifier_recorded`` of them
    are observed only through the verifier's grading record. None until the run is complete.
    """
    decision = _health_view(decision)
    trials = trial_population(decision)
    if not decision.complete or trials <= 0 or collected is None:
        return None
    unhealthy = sum(trial_faults(decision, collected).values())
    invalid_grades: list[Identity] = []
    covered: list[Identity] = []
    if _observed_trial_operations(decision, collected, grading_failures=invalid_grades, verifier_recorded=covered):
        unhealthy += len(invalid_grades)
        return {
            "observed_healthy": trials - unhealthy,
            "unhealthy": unhealthy,
            "unobserved": 0,
            "verifier_recorded": len(covered),
        }
    final_zero = final_zero_trials(decision, collected)
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    rows = [
        row
        for row in collected.results or []
        if _identity(row) in scored and _identity(row) not in final_zero and not _scored_fault(row, collected)
    ]
    # The same per-trial evidence the gate validates: a grading bound to the saved row, the verifier's own record, or
    # a bound grading artifact. A mismatched or missing binding stays unobserved, never healthy.
    recorded = sum(1 for row in rows if not _bound_grading(row) and _recorded_by_verifier(row, collected))
    artifacts = sum(
        1
        for row in rows
        if not _bound_grading(row) and not _recorded_by_verifier(row, collected) and _bound_artifact(row)
    )
    unobserved = sum(1 for row in rows if not _bound_grading(row)) - recorded - artifacts
    counts: dict[str, Any] = {
        "observed_healthy": trials - unhealthy - unobserved,
        "unhealthy": unhealthy,
        "unobserved": unobserved,
        "verifier_recorded": recorded,
    }
    if unobserved:
        counts["unobserved_reason"] = (
            f"no completed grading operation or grading record was recorded by the verifier "
            f"({_verifying_servers(collected)})"
        )
    return counts


def _predates_operation_observations(decision: RecoveryDecision, collected: CollectedEvidence | None) -> bool:
    """Whether every scored row was written by a Gym pin from before operation observations were recorded.

    Current pins write ``response_metadata.model_operations`` on every captured model call (a list, or null), and
    the collector reports whether the key was there (``model_operations_recorded``). Only an explicit False on every
    call of every scored row counts. A row without captured calls, or read by a collector that does not report the
    key, proves nothing, so the strict rule applies to it.
    """
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    rows = [row for row in (collected.results or []) if _identity(row) in scored] if collected is not None else []
    if not scored or len(rows) != len(scored):
        return False
    for row in rows:
        observed = row.get("ng_trial_observations")
        calls = observed.get("model_calls") if isinstance(observed, dict) else None
        if not isinstance(calls, list) or not calls:
            return False
        for call in calls:
            meta = call.get("response_metadata") if isinstance(call, dict) else None
            if not isinstance(meta, dict) or meta.get("model_operations_recorded") is not False:
                return False
    return True


# The trial gate's fault classes are the shared default (``MODEL_FAULT_KINDS``, ``SERVER_FAULT_KINDS`` in
# ``nemo_gym.core.failure_kinds``), keyed on a failure's recorded HTTP status first (``status_fault_kind``) and on the
# run's ``policy_timeout_counts_as_model`` setting (``fault_class``). The gate-derived classes are named there too.
GRADING_WITHOUT_VERDICT = "grading_without_verdict"


def status_fault_kind(kind: Any, http_status: Any, *, policy: bool = True) -> Any:
    """A counted failure's kind for the trial gate's classes, keyed on the HTTP status the failure recorded.

    A model-server reply with a 5xx status is the server's fault (``MODEL_SERVER_ERROR``) whatever the kind says. A
    4xx rejection keeps its kind when the evaluated policy made the request (its context or request is the model's
    own), and is ``NON_POLICY_MODEL_REJECTION`` (evaluation-side) for another model's request. Without a recorded
    status the kind decides, as before.
    """
    if type(http_status) is not int:
        return kind
    if is_server_status(http_status):
        return MODEL_SERVER_ERROR
    if 400 <= http_status <= 499 and not policy:
        return NON_POLICY_MODEL_REJECTION
    return kind


UNVERIFIED_NOTE = "unverified: this withholds a pass and is no health failure"


def fault_class(kind: Any, *, policy_timeout_counts_as_model: bool = True) -> str | None:
    """A counted trial failure's class: ``model``, ``server``, or None for another evaluation-side kind.

    A policy-model response timeout is the model's fault by default (``GateCriteria.policy_timeout_counts_as_model``)
    and the server's when a run sets that to false.
    """
    if kind == MODEL_POLICY_RESPONSE_TIMEOUT and not policy_timeout_counts_as_model:
        return "server"
    if kind in MODEL_FAULT_KINDS:
        return "model"
    if kind in SERVER_FAULT_KINDS or (isinstance(kind, str) and kind.startswith("provider_")):
        return "server"
    return None


# The reasons that say an answer ran into its output cap: a chat ``length`` finish, or a responses-API incomplete
# reason (also recorded as the call's finish reason).
CUTOFF_REASONS = frozenset({"length", "max_output_tokens", "max_tokens"})


def _answer_cut_off(response: Mapping[str, Any]) -> bool:
    """Whether the saved answer itself says it was cut off: its ``incomplete_details.reason`` is ``max_output_tokens``
    or ``max_tokens``, or its output tokens reached its ``max_output_tokens``."""
    details = response.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason") in CUTOFF_REASONS - {"length"}:
        return True
    usage, cap = response.get("usage"), response.get("max_output_tokens")
    tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    return type(cap) is int and cap > 0 and type(tokens) is int and tokens >= cap


def _cut_off(row: Mapping[str, Any]) -> bool:
    """Whether the saved answer carries an actual cutoff signal, not just an incomplete status.

    The answer's ``incomplete_details.reason`` is ``max_output_tokens`` or ``max_tokens``; or the captured call that
    returned it finished for ``length`` or one of those reasons; or its output tokens reached its ``max_output_tokens``.
    """
    response = row.get("response")
    if not isinstance(response, dict):
        return False
    if _answer_cut_off(response):
        return True
    observed = row.get("ng_trial_observations")
    calls = observed.get("model_calls") if isinstance(observed, dict) else None
    return any(
        isinstance(call, dict)
        and isinstance(call.get("response_metadata"), dict)
        and call["response_metadata"].get("response_id") == response.get("id")
        and call["response_metadata"].get("finish_reason") in CUTOFF_REASONS
        for call in (calls if isinstance(calls, list) else [])
    )


def _recorded_grade(row: Mapping[str, Any], collected: CollectedEvidence | None) -> bool:
    """Whether the row's reward came from a recorded grade: completed grading operations, the verifier's own grading
    record, or a saved grading artifact (``GRADING_ARTIFACTS``). The model's own failed answer is not one."""
    return (
        _graded(row)
        or (collected is not None and _recorded_by_verifier(row, collected))
        or grading_artifact(row) is not None
    )


def _answer_fault(row: Mapping[str, Any], collected: CollectedEvidence | None) -> str | None:
    """The model fault of a scored-zero incomplete answer.

    ``TRUNCATED_ANSWER`` when the answer carries a cutoff signal (``_cut_off``), whether the benchmark's rule or a
    recorded grade scored it zero: the cutoff is why it scored zero. Without a cutoff signal, a row whose reward came
    from a recorded grade (``_recorded_grade``) has none, as its score is its grade; otherwise ``EMPTY_ANSWER`` when it
    holds no assistant message with text (``_empty_answer``) and ``MALFORMED_ANSWER`` when it does.
    """
    if type(row.get("reward")) not in (int, float) or row["reward"] != 0 or not _incomplete_answer(row):
        return None
    if _cut_off(row):
        return TRUNCATED_ANSWER
    if _recorded_grade(row, collected):
        return None
    return EMPTY_ANSWER if _empty_answer(row) else MALFORMED_ANSWER


def _scored_kind(row: Mapping[str, Any]) -> str | None:
    """The registered failure kind a scored result row records as its own outcome, keyed on its recorded HTTP status
    (``status_fault_kind``), or None. For example Harbor's graded ``agent_timeout`` is a scored main result."""
    kind, status = _policy_evidence(row)
    return status_fault_kind(kind, status) if _known_kind(kind) else None


def _scored_fault(row: Mapping[str, Any], collected: CollectedEvidence | None) -> str | None:
    """A scored row's one counted fault: its own recorded kind (``_scored_kind``), else its answer's
    (``_answer_fault``), else ``judge_unparseable`` when its saved judge evaluation recorded no verdict
    (``_unparsed_judge_verdict``)."""
    fault = _scored_kind(row) or _answer_fault(row, collected)
    if fault is None and _judge_unparseable(row):
        return JUDGE_UNPARSEABLE
    return fault


def _trajectory_kinds(observed: Any) -> list[str]:
    """The kinds of trajectory evidence a trial lacks: its recorded gaps, and its agent invocations' own shape.

    Recorded gap codes other than ``resolution_unavailable`` (for example ``sandbox_lifecycle_timing_unavailable``,
    ``conversation_unavailable``, ``model_call_ownership_unavailable``), then ``invocation_unavailable`` or
    ``multiple_invocations``, ``in_session_unrecorded`` and ``invocation_<status>`` for an invocation that did not
    record a completed status. Empty when none of these is named.
    """
    if not isinstance(observed, dict):
        return ["trajectory_unavailable"]
    gaps = observed.get("gaps")
    kinds = sorted(
        {
            gap["code"]
            for gap in (gaps if isinstance(gaps, list) else [])
            if isinstance(gap, dict) and isinstance(gap.get("code"), str) and gap["code"] != "resolution_unavailable"
        }
    )
    invocations = observed.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        return kinds + ["invocation_unavailable"]
    if len(invocations) > 1:
        kinds.append("multiple_invocations")
    statuses = sorted(
        {
            str(invocation.get("status"))
            for invocation in invocations
            if isinstance(invocation, dict) and invocation.get("status") != "completed"
        }
    )
    if any(
        isinstance(invocation, dict) and type(invocation.get("in_session")) is not bool for invocation in invocations
    ):
        kinds.append("in_session_unrecorded")
    return kinds + [f"invocation_{status}" for status in statuses]


def trajectory_evidence(incomplete: Mapping[Identity, list[str]], trials: int) -> dict[str, Any]:
    """Trials whose trajectory evidence is incomplete, over the trials, with the count of each missing kind."""
    kinds = Counter(kind for found in incomplete.values() for kind in found)
    return {"incomplete": len(incomplete), "trials": trials, "kinds": dict(sorted(kinds.items()))}


def _trajectory_text(evidence: Mapping[str, Any]) -> str:
    """``trajectory evidence incomplete: N of M (kinds: kind n, ...)``, reported and never a condition."""
    kinds = ", ".join(f"{kind} {n}" for kind, n in evidence["kinds"].items())
    return (
        f"trajectory evidence incomplete: {evidence['incomplete']} of {evidence['trials']} (kinds: {kinds}); "
        "reported, not a coverage or health condition"
    )


def _observed_incomplete_answer(row: Mapping[str, Any]) -> bool:
    """Whether a counted incomplete answer (``_answer_fault``) is the model's own observed answer.

    The trial gate counts such a row as a model failure and reports it as a response failure, so its grading is not
    joined. Its answer must still be observed: the row recorded no failure, its one agent
    invocation ended completed or incomplete (an agent marks a cut-off answer incomplete) without an error, its last
    model call is exactly one captured call that answered in one completed operation, and that call's response is
    the saved answer, or, for an agent that saves its own answer envelope (its own response id), the envelope says it
    was cut off (``_answer_cut_off``) and that call itself finished on a cutoff (``CUTOFF_REASONS``): two independent
    signals that the model's own last reply was cut. A verifier that ran must have completed: a verifier error is never
    read as the model's fault.
    """
    observed = row.get("ng_trial_observations")
    if (
        not isinstance(observed, dict)
        or row.get(FAILURE_CLASS) is not None
        or row.get("failure_kind") is not None
        or row.get(FAILURE_ID) is not None
    ):
        return False
    invocations, verifiers, calls = (
        observed.get(key) for key in ("invocations", "verifier_invocations", "model_calls")
    )
    if (
        not isinstance(invocations, list)
        or len(invocations) != 1
        or not isinstance(invocations[0], dict)
        or not isinstance(calls, list)
        or not isinstance(verifiers, list)
        or len(verifiers) > 1
    ):
        return False
    invocation = invocations[0]
    refs = invocation.get("model_calls")
    if (
        invocation.get("status") not in ("completed", "incomplete")
        or invocation.get("error_type") is not None
        or invocation.get("stop_reason") is not None
        or invocation.get("parent_invocation_id") is not None
        or not isinstance(refs, list)
        or not refs
        or not isinstance(refs[-1], dict)
    ):
        return False
    if verifiers and (
        not isinstance(verifiers[0], dict)
        or verifiers[0].get("status") != "completed"
        or verifiers[0].get("failure_kind") is not None
    ):
        return False
    ref = refs[-1]
    last = [
        call
        for call in calls
        if isinstance(call, dict)
        and isinstance(call.get("response_metadata"), dict)
        and (
            call.get("model_call_id") == ref["model_call_id"]
            if ref.get("model_call_id")
            else ref.get("response_id") is not None
            and call["response_metadata"].get("response_id") == ref["response_id"]
            and call["response_metadata"].get("model_ref") == ref.get("model_ref")
        )
    ]
    if len(last) != 1:
        return False
    meta = last[0]["response_metadata"]
    operations = meta.get("model_operations")
    status = meta.get("status_code")
    return (
        type(status) is int
        and 200 <= status < 300
        and isinstance(operations, list)
        and len(operations) == 1
        and isinstance(operations[0], dict)
        and operations[0].get("outcome") == "completed"
        and operations[0].get("failure_id") is None
        and isinstance(meta.get("response_id"), str)
        and bool(meta["response_id"])
        and (
            meta["response_id"] == (row.get("response") or {}).get("id")
            or (meta.get("finish_reason") in CUTOFF_REASONS and _answer_cut_off(row.get("response") or {}))
        )
    )


def _row_recovered_attempts(row: Mapping[str, Any]) -> list[tuple[str, dict[str, dict]]]:
    """The row's recovered model attempts, by the completed operation that recovered them (``recovered_attempt_joins``).

    Each entry is an operation id and its failure events, keyed by event id, from one answered call of the row.
    """

    def nonempty(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    observed = row.get("ng_trial_observations")
    calls = observed.get("model_calls") if isinstance(observed, dict) else None
    saved = row.get(FAILURE_EVENTS)
    saved = saved if isinstance(saved, dict) else {}
    found: list[tuple[str, dict[str, dict]]] = []
    seen: set[str] = set()
    for call in calls if isinstance(calls, list) else []:
        meta = call.get("response_metadata") if isinstance(call, dict) else None
        report = meta.get("failure_report") if isinstance(meta, dict) else None
        events = report.get("events") if isinstance(report, dict) else None
        operations = meta.get("model_operations") if isinstance(meta, dict) else None
        status = meta.get("status_code") if isinstance(meta, dict) else None
        # The health counter's own test of an answered call: a success status and no failure recorded on it.
        if (
            not isinstance(events, dict)
            or not events
            or not isinstance(operations, list)
            or type(status) is not int
            or not 200 <= status < 400
            or meta.get("error_category")
            or meta.get("response_status") in ("failed", "error", "cancelled")
        ):
            continue
        for operation in operations:
            op_id = operation.get("operation_id") if isinstance(operation, dict) else None
            if not nonempty(op_id) or operation.get("outcome") != "completed" or op_id in seen:
                continue
            owned = {
                key: event
                for key, event in events.items()
                if isinstance(event, dict) and event.get("operation_id") == op_id
            }
            tries, transport = operation.get("http_attempts"), operation.get("transport_failures")
            answer = operation.get("final_attempt_id")
            ids = {event.get("attempt_id") for event in owned.values()}
            if (
                not owned
                or type(tries) is not int
                or type(transport) is not int
                or tries < 1
                or transport < 0
                or len(owned) != tries - 1 + transport
                or sum(event.get("http_status") is None for event in owned.values()) != transport
                or not nonempty(answer)
                or answer in ids
                or len(ids) != len(owned)
                or operation.get("failure_id") is not None
                or any(
                    key != event.get("event_id")
                    or not nonempty(event.get("attempt_id"))
                    or event.get("boundary") != "model"
                    or event.get("operation") != "model_request"
                    or event.get("failure_kind") != MODEL_REQUEST_ERROR
                    or event.get("cause_id") is not None
                    or not (
                        event.get("http_status") is None
                        or (type(event["http_status"]) is int and 400 <= event["http_status"] <= 599)
                    )
                    or (
                        key in saved
                        and (
                            not isinstance(saved[key], dict)
                            or any(saved[key].get(name) != value for name, value in event.items())
                        )
                    )
                    for key, event in owned.items()
                )
            ):
                continue
            seen.add(op_id)
            found.append((op_id, owned))
    return found


def _unaccounted_failures(row: Mapping[str, Any], recovered: Mapping[str, Any]) -> bool:
    """Whether the row recorded a model failure that no recovery accounts for.

    Every failure event the row saved or its captured calls reported must be a recovered attempt (``recovered``: the
    events the coverage join joined, with the row's own ``_row_recovered_attempts``), a saved copy must agree with its
    call's, and every failed model operation must be one whose failure such a recovery accounts for. A trial that
    ended on its failed request is counted separately (``final_zero_trials``) and is never asked this.
    """
    joined = dict(recovered)
    for _, owned in _row_recovered_attempts(row):
        joined.update(owned)
    saved = row.get(FAILURE_EVENTS, {})
    observed = row.get("ng_trial_observations")
    calls = observed.get("model_calls") if isinstance(observed, dict) else None
    if not isinstance(saved, dict) or (calls is not None and not isinstance(calls, list)):
        return True
    for event_id, event in saved.items():
        if (
            event_id not in joined
            or not isinstance(event, dict)
            or any(event.get(key) != value for key, value in joined[event_id].items())
        ):
            return True
    for call in calls or []:
        meta = call.get("response_metadata") if isinstance(call, dict) else None
        if not isinstance(meta, dict):
            continue
        report = meta.get("failure_report")
        events = report.get("events") if isinstance(report, dict) else None
        if isinstance(events, dict) and any(event_id not in joined for event_id in events):
            return True
        for operation in meta.get("model_operations") or []:
            if isinstance(operation, dict) and operation.get("outcome") == "failed":
                if operation.get("failure_id") not in joined:
                    return True
    return False


def recovered_attempt_joins(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, int]:
    """Failed model attempts each joined to the completed operation that recovered them, in a scored trial's own call.

    Counted like the run's observed upstream recovery (``observed_upstream_recovery``): a model-request failure event
    whose operation completed in an answered call. Here the event must be in the failure report of the same call
    whose one completed operation it names, as a failed attempt other than the answer, and the operation's attempts
    must account for every failed attempt of it (``http_attempts - 1 + transport_failures``, the HTTP-less ones being
    the transport failures). A copy the row saved must agree, and no event, attempt or operation joins twice. The
    join holds for any model the trial called (its policy, a judge or a simulated user). It says nothing about how
    the trial was graded: whether the final answer is bound to its grading is the coverage join's question
    (``_observed_trial_operations``).
    """
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    events_seen: set[str] = set()
    attempts_seen: set[str] = set()
    operations_seen: set[str] = set()
    joined = {"attempts": 0, "operations": 0}
    for row in (collected.results or []) if collected is not None else []:
        if _identity(row) not in scored:
            continue
        for op_id, owned in _row_recovered_attempts(row):
            ids = {event["attempt_id"] for event in owned.values()}
            if op_id in operations_seen or events_seen & set(owned) or attempts_seen & ids:
                continue
            events_seen.update(owned)
            attempts_seen.update(ids)
            operations_seen.add(op_id)
            joined["attempts"] += len(owned)
            joined["operations"] += 1
    return joined


def _recovered_note(joined: Mapping[str, int]) -> str:
    """``N failed model attempt(s) were retried and completed in the same operation (...)``, or empty."""
    if not joined["attempts"]:
        return ""
    return (
        f"{joined['attempts']} failed model attempt(s) in {joined['operations']} operation(s) were retried and "
        "completed in the same operation (recovered, no trial failure); "
    )


def trial_faults(decision: RecoveryDecision, collected: CollectedEvidence | None) -> Counter:
    """The trial gate's counted failures by kind.

    Settled model failures (MODEL_FINAL), scored rows whose rollout ended on a failed policy request
    (``final_zero_trials``), scored-zero incomplete answers the benchmark's rule scored without a recorded grade
    (``_answer_fault``: truncated with a cutoff signal, otherwise empty or malformed), and scored rows whose saved
    judge evaluation recorded no verdict (``judge_unparseable``) or that record their own failure kind, each counted
    once (``_scored_fault``).
    """
    decision = _health_view(decision)
    final_zero = final_zero_trials(decision, collected)
    infrastructure = infrastructure_timeouts(decision, collected)
    kinds = Counter(
        status_fault_kind(e.failure_kind, e.http_status)
        for e in decision.evidence
        if e.state == IdentityState.MODEL_FINAL
    )
    kinds.update(final_zero.values())
    kinds[AGENT_TIMEOUT_INFRASTRUCTURE] += len(infrastructure)
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    kinds.update(
        fault
        for row in (collected.results or [] if collected is not None else [])
        if _identity(row) in scored
        and _identity(row) not in final_zero
        and _identity(row) not in infrastructure
        and (fault := _scored_fault(row, collected)) is not None
    )
    return +kinds


def fault_report(kinds: Counter, trials: int, *, policy_timeout_counts_as_model: bool = True) -> dict[str, Any]:
    """Model and evaluation-side fault counts, fractions and kinds over the trials; ``server`` counts server faults."""

    def side(found: dict) -> dict[str, Any]:
        count = sum(found.values())
        return {"count": count, "fraction": count / trials, "kinds": dict(sorted(found.items(), key=str))}

    def class_of(kind):
        return fault_class(kind, policy_timeout_counts_as_model=policy_timeout_counts_as_model)

    model = {kind: n for kind, n in kinds.items() if class_of(kind) == "model"}
    evaluation = {kind: n for kind, n in kinds.items() if class_of(kind) != "model"}
    server = sum(n for kind, n in evaluation.items() if class_of(kind) == "server")
    report = {"trials": trials, "evaluation": side(evaluation) | {"server": server}, "model": side(model)}
    return report if policy_timeout_counts_as_model else report | {"policy_timeout_counts_as_model": False}


def _fault_text(report: dict[str, Any], limit: float) -> str:
    """``evaluation-side faults: N of M (x%) is [not] below L (server S: kind n; kind n); model faults: ...``."""
    trials, evaluation, model = report["trials"], report["evaluation"], report["model"]
    flag = report.get("policy_timeout_counts_as_model", True)
    server = [
        f"{kind} {n}"
        for kind, n in evaluation["kinds"].items()
        if fault_class(kind, policy_timeout_counts_as_model=flag) == "server"
    ]
    other = [
        f"{kind} {n}"
        for kind, n in evaluation["kinds"].items()
        if fault_class(kind, policy_timeout_counts_as_model=flag) != "server"
    ]
    named = ([f"server {evaluation['server']}: {', '.join(server)}"] if server else []) + other
    below = "is below" if evaluation["fraction"] < limit else "is not below"
    return (
        f"evaluation-side faults: {evaluation['count']} of {trials} ({evaluation['fraction']:.2%}) {below} "
        f"{_percent(limit)}" + (f" ({'; '.join(named)})" if named else "") + f"; model faults: {model['count']} of "
        f"{trials} ({model['fraction']:.2%})"
        + (f" ({', '.join(f'{kind} {n}' for kind, n in model['kinds'].items())})" if model["kinds"] else "")
    )
