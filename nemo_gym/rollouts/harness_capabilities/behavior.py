# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks against controlled observations; independent of artifact validation and process exit."""

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from nemo_gym.agents.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.artifacts.rollout_rows import NG_FAILURE_CLASS_KEY
from nemo_gym.core.failure_kinds import MODEL_REQUEST_ERROR, MODEL_REQUEST_REJECTED
from nemo_gym.models.chat_completions_converter import message_content_to_text
from nemo_gym.server_runtime.base_environment_server import is_retryable_dependency_status

from .checks import BehavioralCheck
from .results import Results


EnvironmentAction = Literal["verify", "retry", "fail", "count_as_zero"]


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


@dataclass(frozen=True)
class ExpectedOutcome:
    """One response the outcome ladder may report for a run, and the Environment Server's action on it.

    ``failure_kind`` and ``status_code`` are the response's ``ng_agent_outcome``. Only a failed response carries one;
    the ladder removes it from a completed or incomplete response.
    """

    status: Literal["completed", "incomplete", "failed"]
    action: EnvironmentAction
    failure_kind: str | None = None
    status_code: int | None = None

    def __post_init__(self) -> None:
        if (self.status == "failed") != (self.failure_kind is not None):
            raise ValueError("a failed outcome names its failure kind; a completed or incomplete one names none")
        if self.failure_kind is not None:
            # Accept only an outcome the agent could report.
            AgentOutcome(failure_kind=self.failure_kind, status_code=self.status_code)

    def describe(self) -> str:
        outcome = "" if self.failure_kind is None else f" {self.failure_kind}"
        status = "" if self.status_code is None else f" {self.status_code}"
        return f"{self.status}{outcome}{status} ({self.action})"

    def matches(self, response: dict) -> bool:
        """Whether an agent response has this status and exactly this ``ng_agent_outcome``."""
        if response.get("status") != self.status:
            return False
        raw = _mapping(response.get("metadata")).get(AGENT_OUTCOME_METADATA_KEY)
        if self.failure_kind is None or raw is None:
            return self.failure_kind is None and raw is None
        try:
            outcome = AgentOutcome.model_validate_json(raw)
        except ValidationError:
            return False
        return outcome == AgentOutcome(failure_kind=self.failure_kind, status_code=self.status_code)


def environment_action(record: dict | None) -> EnvironmentAction | None:
    """The Environment Server's action on an episode, read from the record collection kept for it.

    A graded rollout is ``verify``. A failure row is ``fail`` when ``_ng_failure_terminal`` is set (not retried) and
    ``retry`` when it is not; a row the policy counted as zero is ``count_as_zero``. ``None`` without a record.
    """
    if record is None:
        return None
    if record.get(NG_FAILURE_CLASS_KEY) is None:
        return "verify"
    if record.get("_ng_failure_counts_as_zero"):
        return "count_as_zero"
    return "fail" if record.get("_ng_failure_terminal") else "retry"


def _terminal_outcome(status: int) -> AgentOutcome:
    """The outcome ladder's report of a run whose every model request failed with ``status``.

    The model never replied, so a 4xx other than 408, 425 and 429 is a request the run's setup made invalid
    (``model_request_rejected``); any other status stays ``model_request_error`` with that status.
    """
    if status < 500 and not is_retryable_dependency_status(status):
        return AgentOutcome(failure_kind=MODEL_REQUEST_REJECTED)
    return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)


def _terminal_error_reported(attempts: list[dict], responses: list[dict], expected_status: int) -> bool:
    """Compare the scripted provider error with the response received by the EnvironmentServer."""
    if not attempts or any(attempt["status_code"] != expected_status for attempt in attempts):
        return False
    if len(responses) != 1 or responses[0].get("status") != "failed":
        return False
    raw = _mapping(responses[0].get("metadata")).get(AGENT_OUTCOME_METADATA_KEY)
    try:
        outcome = AgentOutcome.model_validate_json(raw)
    except ValidationError:
        return False
    if outcome != _terminal_outcome(expected_status):
        return False
    message = _mapping(_mapping(attempts[-1].get("response")).get("error")).get("message")
    reported = _mapping(responses[0].get("error")).get("message")
    return isinstance(message, str) and bool(message) and isinstance(reported, str) and message in reported


def tool_checks(
    record: dict | None,
    witnessed: list[dict],
    *,
    finish_id: str | None = None,
    applies: bool = True,
    available: bool = True,
) -> list[dict]:
    results = Results()
    trajectory = _mapping((record or {}).get("ng_trajectory"))
    # The scripted finish call ends the episode; it is witnessed by the model exchange, not by a shell marker.
    tools = [
        t for t in _objects(trajectory.get("tool_calls")) if finish_id is None or t.get("tool_call_id") != finish_id
    ]
    invocations = _objects(trajectory.get("invocations"))
    wanted = Counter(t.get("id") for t in witnessed)
    identities = (
        all(isinstance(t.get("tool_call_id"), str) for t in tools)
        and wanted == Counter(t.get("tool_call_id") for t in tools)
        and all(n == 1 for n in wanted.values())
    )
    results.run(
        BehavioralCheck(
            id="tools.witness_ids",
            tier="P0",
            evidence=("TE-5",),
            location="$.ng_trajectory.tool_calls",
            reason="retained tool identities differ from the independent tool witness",
            predicate=lambda: identities,
            available=record is not None and available,
            applies=applies,
        )
    )
    values = {"join": [], "request": [], "status": [], "output": []}
    if identities and record is not None:
        by_id = {t["tool_call_id"]: t for t in tools}
        for expected in witnessed:
            tool = by_id[expected["id"]]
            items = [
                item
                for inv in invocations
                if inv.get("invocation_id") == tool.get("invocation_id")
                for item in _objects(inv.get("conversation"))
                if item.get("call_id") == expected["id"]
            ]
            requests = [i for i in items if i.get("type") == "function_call"]
            values["join"].append(len(requests) == 1)
            values["status"].append(tool.get("status") == ("failed" if expected["exit_code"] else "completed"))
            observed = expected.get("outputs", [])
            values["output"].append(
                bool(observed)
                and all(message_content_to_text(o) == message_content_to_text(tool.get("output")) for o in observed)
            )
            if len(requests) != 1:
                continue
            request = requests[0]
            try:
                arguments = json.loads(request["arguments"])
            except (KeyError, TypeError, ValueError):
                arguments = None
            name = request.get("name")
            if request.get("namespace"):
                name = f"{request['namespace']}__{name}"
            values["request"].append(
                name == expected["name"]
                and arguments == expected["arguments"]
                and tool.get("tool_name") == request.get("name")
            )
    results.run(
        BehavioralCheck(
            id="tools.witness_join",
            tier="P0",
            evidence=("TE-5",),
            location="witness.json",
            reason="retained tool request does not join uniquely from the independent tool witness",
            predicate=lambda: all(values["join"]),
            available=bool(values["join"]),
            applies=applies,
            depends_on=("tools.witness_ids",),
        )
    )
    results.run(
        BehavioralCheck(
            id="tools.witness_request",
            tier="P0",
            evidence=("TE-5",),
            location="witness.json",
            reason="retained tool name or arguments differ from the independent tool witness",
            predicate=lambda: all(values["request"]),
            available=bool(values["request"]),
            applies=applies,
            depends_on=("tools.witness_ids", "tools.witness_join"),
        )
    )
    results.run(
        BehavioralCheck(
            id="tools.witness_status",
            tier="P0",
            evidence=("TE-5",),
            location="witness.json",
            reason="retained tool status differs from the independent tool witness",
            predicate=lambda: all(values["status"]),
            available=bool(values["status"]),
            applies=applies,
            depends_on=("tools.witness_ids",),
        )
    )
    results.run(
        BehavioralCheck(
            id="tools.witness_output",
            tier="P0",
            evidence=("TE-5",),
            location="witness.json",
            reason="retained tool output differs from the independent tool witness",
            predicate=lambda: all(values["output"]),
            available=bool(values["output"]),
            applies=applies,
            depends_on=("tools.witness_ids",),
        )
    )
    return results.dump()


def inspect_behavior(
    witness: dict | None,
    record: dict | None,
    *,
    http_errors: tuple[int, ...],
    terminal_error: bool,
    tool_steps: int,
    expected_reward: float,
    fingerprint: Callable,
    judge_verifier: bool = False,
    final_answer: bool = True,
    tool_results: bool = True,
    outcomes: Sequence[ExpectedOutcome] = (),
    target_outcome: ExpectedOutcome | None = None,
    retry_budget_seconds: float | None = None,
) -> list[dict]:
    """Check saved evidence and observations against explicit test expectations.

    ``final_answer`` is false when the script never gives the final answer, and ``tool_results`` is false when a
    scripted tool call ends the harness, so no result returns to the model. ``outcomes`` are the responses the outcome
    ladder may report for the run, each with the Environment Server's action on it; the run's response must be one of
    them (P0). ``target_outcome`` is the one the ladder reports once its transitional rules are gone (P1).
    ``retry_budget_seconds`` bounds the time from the first model request to the agent's reply, for a harness that
    retries a failing model itself.
    """
    results = Results()
    attempts = (witness or {}).get("attempts", [])
    statuses = [a["status_code"] for a in attempts]
    agent_responses = (witness or {}).get("agent_responses", [])
    tools = (witness or {}).get("tool_calls", [])
    available = witness is not None
    action = environment_action(record)
    # A run whose outcome may go ungraded is verified only when the Environment Server graded it.
    graded = action == "verify" if outcomes else not terminal_error

    results.run(
        BehavioralCheck(
            id="episode.initialized",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason="expected one fresh episode initialization",
            predicate=lambda: (witness or {}).get("seeded") == 1,
            available=available,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.reached",
            tier="P0",
            evidence=("TE-1",),
            location="witness.json",
            reason="the harness never reached the controlled model endpoint",
            predicate=lambda: bool(attempts),
            available=available,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.protocol",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason="scripted provider reported protocol violations",
            predicate=lambda: not (witness or {}).get("violations"),
            available=available,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.error_sequence",
            tier="P0",
            evidence=("TE-1",),
            location="witness.json",
            reason="not all prescribed model failures were observed",
            predicate=lambda: statuses[: len(http_errors)] == list(http_errors),
            available=available,
            applies=bool(http_errors),
        )
    )
    # Preserve the original probe's comparison scope: repeated error attempts, not recovery requests.
    results.run(
        BehavioralCheck(
            id="model.retry_request",
            tier="P0",
            evidence=("TE-1",),
            location="witness.json",
            reason="retry scenario did not repeat the same request body",
            predicate=lambda: all((a["request"] == attempts[0]["request"] for a in attempts[1 : len(http_errors)])),
            available=len(attempts) >= len(http_errors),
            applies=len(http_errors) > 1,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.terminal_error",
            tier="P0",
            evidence=("TE-1",),
            location="witness.json",
            reason="the EnvironmentServer must receive a failed response retaining the model error and the "
            "outcome ladder's ng_agent_outcome: model_request_rejected for a 4xx other than 408/425/429 before any "
            "model reply, else model_request_error with the witnessed HTTP status",
            predicate=lambda: (
                _terminal_error_reported(attempts, agent_responses, http_errors[-1]) if terminal_error else True
            ),
            available=available,
            applies=terminal_error,
        )
    )
    results.run(
        BehavioralCheck(
            id="model.finished",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason="the harness did not finish the scripted model exchange",
            predicate=lambda: bool((witness or {}).get("finished")),
            available=available,
            applies=not terminal_error and final_answer,
        )
    )
    results.run(
        BehavioralCheck(
            id="tools.executed",
            tier="P0",
            evidence=("TE-4", "TE-5"),
            location="witness.json",
            reason="prescribed tool executions and their returned results were not all witnessed",
            predicate=lambda: (
                len(tools) == tool_steps and all((t.get("executed") and t.get("result_seen") for t in tools))
            ),
            available=available,
            applies=tool_steps > 0 and tool_results,
        )
    )
    verifications = (witness or {}).get("verifications", [])
    results.run(
        BehavioralCheck(
            id="verifier.outcome",
            tier="P0",
            evidence=("TE-6",),
            location="witness.json",
            reason="expected verifier outcome was not observed",
            predicate=lambda: len(verifications) == 1 and verifications[0]["reward"] == expected_reward,
            available=available,
            applies=graded,
        )
    )
    results.run(
        BehavioralCheck(
            id="verifier.answer",
            tier="P0",
            evidence=("TE-6",),
            location="witness.json",
            reason="the verifier did not receive the scripted final answer",
            predicate=lambda: len(verifications) == 1 and verifications[0].get("answer_seen"),
            available=available,
            applies=graded and final_answer,
        )
    )
    results.run(
        BehavioralCheck(
            id="verifier.saved_reward",
            tier="P0",
            evidence=("TE-6",),
            location="$.reward",
            reason="rollout reward differs from the verifier witness",
            predicate=lambda: record.get("reward") == expected_reward,
            available=record is not None and available,
            applies=graded,
        )
    )
    results.run(
        BehavioralCheck(
            id="verifier.saved_outcome",
            tier="P0",
            evidence=("TE-6",),
            location="$",
            reason="completed verification must retain evaluation_completed=true and mask_sample=false",
            predicate=lambda: record.get("evaluation_completed") is True and record.get("mask_sample") is False,
            available=record is not None and available,
            applies=graded,
        )
    )
    judges = (witness or {}).get("judge_calls", [])
    judge_id = (judges[0].get("response") or {}).get("id") if len(judges) == 1 else None
    trajectory = _mapping((record or {}).get("ng_trajectory"))
    judge_calls = [
        call
        for call in _objects(trajectory.get("model_calls"))
        if judge_id is not None and _mapping(call.get("response_metadata")).get("response_id") == judge_id
    ]
    verifier_refs = {
        ref.get("model_call_id")
        for inv in _objects(trajectory.get("verifier_invocations"))
        for ref in _objects(inv.get("model_calls"))
    }
    results.run(
        BehavioralCheck(
            id="verifier.judge_reference",
            tier="P1",
            evidence=("TE-6", "TE-8"),
            location="$.judgement_id",
            reason="the grade must reference the witnessed judge response with explicit verifier ownership",
            predicate=lambda: (
                judge_id is not None
                and len(judge_calls) == 1
                and (record or {}).get("judgement_id") == judge_id
                and judge_calls[0].get("model_call_id") in verifier_refs
                and "CONFORMANCE_DONE" in json.dumps(judges[0].get("request"))
            ),
            available=available and record is not None,
            applies=judge_verifier,
        )
    )
    return (
        results.dump()
        + outcome_checks(
            witness,
            record,
            outcomes=outcomes,
            target_outcome=target_outcome,
            retry_budget_seconds=retry_budget_seconds,
        )
        + model_checks(record, attempts + judges, fingerprint=fingerprint, available=available)
        + tool_checks(
            record,
            tools + _objects((witness or {}).get("native_tool_calls")),
            finish_id=((witness or {}).get("finish_call") or {}).get("id"),
            applies=tool_steps > 0 and tool_results,
            available=available,
        )
    )


def outcome_checks(
    witness: dict | None,
    record: dict | None,
    *,
    outcomes: Sequence[ExpectedOutcome],
    target_outcome: ExpectedOutcome | None = None,
    retry_budget_seconds: float | None = None,
) -> list[dict]:
    """Compare the agent's reply and the Environment Server's action with the outcome ladder's expectation.

    The witness records the agent's HTTP replies to the Environment Server (``agent_replies``) and the responses it
    accepted (``agent_responses``); the record is the graded rollout or the failure row collection kept.
    """
    results = Results()
    available = witness is not None
    replies = _objects((witness or {}).get("agent_replies"))
    responses = _objects((witness or {}).get("agent_responses"))
    response = responses[0] if len(responses) == 1 else None
    matched = next((o for o in outcomes if response is not None and o.matches(response)), None)
    action = environment_action(record)
    expected = ", ".join(o.describe() for o in outcomes)
    results.run(
        BehavioralCheck(
            id="outcome.http_ok",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason="the agent must reply to the Environment Server with HTTP 200 and report the run's outcome in the "
            "response, never with a 5xx",
            predicate=lambda: bool(replies) and all(reply.get("http_status") == 200 for reply in replies),
            available=available,
            applies=bool(outcomes),
        )
    )
    results.run(
        BehavioralCheck(
            id="outcome.ladder",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason=f"the agent's response must be one the outcome ladder reports for this run: {expected}",
            predicate=lambda: matched is not None,
            available=available,
            applies=bool(outcomes),
        )
    )
    results.run(
        BehavioralCheck(
            id="outcome.disposition",
            tier="P0",
            evidence=(),
            location="$",
            reason="the Environment Server must take the reported outcome's action: verify (a graded rollout), retry "
            "(a failure row with _ng_failure_terminal false) or fail (_ng_failure_terminal true)",
            predicate=lambda: action == matched.action,
            available=available and record is not None,
            applies=bool(outcomes),
            depends_on=("outcome.ladder",),
        )
    )
    results.run(
        BehavioralCheck(
            id="outcome.target",
            tier="P1",
            evidence=(),
            location="witness.json",
            reason=f"the outcome ladder's target for this run is {target_outcome.describe() if target_outcome else ''}",
            predicate=lambda: response is not None and target_outcome.matches(response),
            available=available,
            applies=target_outcome is not None,
        )
    )
    attempts = _objects((witness or {}).get("attempts"))
    started = [a["received_at"] for a in attempts if isinstance(a.get("received_at"), (int, float))]
    replied = [r["replied_at"] for r in replies if isinstance(r.get("replied_at"), (int, float))]
    results.run(
        BehavioralCheck(
            id="model.retries_bounded",
            tier="P0",
            evidence=(),
            location="witness.json",
            reason=f"the harness must stop retrying the failing model and reply within {retry_budget_seconds or 0:g}s "
            "of its first model request",
            predicate=lambda: bool(started) and bool(replied) and replied[0] - min(started) <= retry_budget_seconds,
            available=available,
            applies=retry_budget_seconds is not None,
        )
    )
    return results.dump()


def model_checks(
    record: dict | None, attempts: list[dict], *, fingerprint: Callable, available: bool = True
) -> list[dict]:
    """Compare canonical fields to the independently scripted endpoint exchanges."""
    calls = _objects(_mapping((record or {}).get("ng_trajectory")).get("model_calls"))
    expected_exchanges, observed_exchanges = Counter(), Counter()
    expected_metadata, observed_metadata = Counter(), Counter()
    expected_tokens, observed_tokens = Counter(), Counter()
    token_fields = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens")
    for attempt in attempts:
        response = attempt["response"] or {}
        exchange = fingerprint(attempt["request"], attempt["status_code"], response)
        expected_exchanges[exchange] += 1
        chat = "messages" in attempt["request"]
        choices = response.get("choices") or []
        # An incomplete Responses reply names its stop in incomplete_details; a completed one names none.
        finish = (
            (choices[0].get("finish_reason") if choices else None)
            if chat
            else _mapping(response.get("incomplete_details")).get("reason")
        )
        expected_metadata[
            (exchange, json.dumps(response.get("id")), json.dumps(response.get("status")), json.dumps(finish))
        ] += 1
        usage = response.get("usage") or {}
        counts = (
            usage.get("prompt_tokens" if chat else "input_tokens"),
            usage.get("completion_tokens" if chat else "output_tokens"),
            (usage.get("completion_tokens_details" if chat else "output_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            usage.get("total_tokens"),
            (usage.get("prompt_tokens_details" if chat else "input_tokens_details") or {}).get("cached_tokens"),
        )
        expected_tokens[(exchange, json.dumps(counts))] += 1
    for call in calls:
        metadata = _mapping(call.get("response_metadata"))
        exchange = fingerprint(call.get("request"), metadata.get("status_code"), call.get("response"))
        observed_exchanges[exchange] += 1
        observed_metadata[
            (
                exchange,
                json.dumps(metadata.get("response_id")),
                json.dumps(metadata.get("response_status")),
                json.dumps(metadata.get("finish_reason")),
            )
        ] += 1
        stats = _mapping(call.get("token_stats"))
        observed_tokens[(exchange, json.dumps([stats.get(field) for field in token_fields]))] += 1
    result = Results()
    result.run(
        BehavioralCheck(
            id="model.saved_exchanges",
            tier="P0",
            evidence=("TE-1", "TE-4", "TE-7"),
            location="$.ng_trajectory.model_calls",
            reason="canonical exchanges differ from the independent endpoint witness",
            predicate=lambda: expected_exchanges == observed_exchanges,
            available=record is not None and available,
        )
    )
    result.run(
        BehavioralCheck(
            id="model.response_metadata",
            tier="P0",
            evidence=("TE-1",),
            location="$.ng_trajectory.model_calls",
            reason="canonical response metadata differs from the independent endpoint witness",
            predicate=lambda: expected_metadata == observed_metadata,
            available=record is not None and available,
        )
    )
    result.run(
        BehavioralCheck(
            id="model.token_counts",
            tier="P0",
            evidence=("TE-2",),
            location="$.ng_trajectory.model_calls",
            reason="canonical token counts differ from the independent endpoint witness",
            predicate=lambda: expected_tokens == observed_tokens,
            available=record is not None and available,
        )
    )
    return result.dump()
