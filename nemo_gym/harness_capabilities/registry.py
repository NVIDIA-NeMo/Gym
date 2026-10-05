# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory and execution order of the existing conformance checks.

This catalogs the shipped gym-p0/v1 behavior, not the stricter RFC baseline.
Locations describe the input to each check after reader hydration. Capture paths
may contain payloads projected from ng_trajectory; existing fallbacks are unchanged.
Artifact IDs retain the report's evidence/assertion pair. Native-model errors keep
Pydantic's detailed codes; their catalog entry identifies the validated collection.
Behavioral prerequisites with no TE association affect the whole selected scenario.
"""

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal


CheckKind = Literal["schema", "semantic", "behavioral"]


@dataclass(frozen=True)
class CheckDefinition:
    """One named requirement; implementations may evaluate several related checks."""

    id: str
    kind: CheckKind
    evidence: tuple[str, ...]
    locations: tuple[str, ...]
    requirement: str
    implementation: str
    failure_message: str | None = None


@dataclass(frozen=True)
class ArtifactCheckGroup:
    """Related assertions evaluated together, in dependency order."""

    implementation: str
    checks: tuple[CheckDefinition, ...]


_Definition = tuple[str, CheckKind, tuple[str, ...], str]


def _definitions(implementation: str, rows: tuple[_Definition, ...]) -> tuple[CheckDefinition, ...]:
    return tuple(
        CheckDefinition(check_id, kind, (check_id.split(".", 1)[0],), locations, requirement, implementation)
        for check_id, kind, locations, requirement in rows
    )


def _group(implementation: str, rows: tuple[_Definition, ...]) -> ArtifactCheckGroup:
    return ArtifactCheckGroup(implementation, _definitions("checker." + implementation, rows))


CALL = "$.ng_model_call_capture.calls[*]"
TURN = "$.ng_trajectory.turns[*]"
INVOCATIONS = ("$.ng_agent_observations.records[*]", "$.ng_trajectory.invocations[*]")
TOOLS = ("$.ng_trajectory.tool_calls[*]", "$.ng_agent_observations.records[*]")
GAPS = ("$.ng_model_call_capture.gaps[*]", "$.ng_agent_observations.gaps[*]", "$.ng_trajectory.gaps[*]")


NATIVE_CHECKS = _definitions(
    "contracts.model_errors",
    (
        (
            "record.model.ng_model_call_capture.calls",
            "schema",
            ("$.ng_model_call_capture.calls",),
            "If supplied, the collection strictly validates as a list of ModelCallRecord objects.",
        ),
        (
            "record.model.ng_trajectory.model_calls",
            "schema",
            ("$.ng_trajectory.model_calls",),
            "If supplied, the collection strictly validates as a list of TrajectoryModelCall objects.",
        ),
        (
            "record.model.ng_trajectory.turns",
            "schema",
            ("$.ng_trajectory.turns",),
            "If supplied, the collection strictly validates as a list of TrajectoryTurn objects.",
        ),
        (
            "record.model.ng_trajectory.invocations",
            "schema",
            ("$.ng_trajectory.invocations",),
            "If supplied, the collection strictly validates as a list of AgentInvocation objects.",
        ),
        (
            "record.model.ng_trajectory.tool_calls",
            "schema",
            ("$.ng_trajectory.tool_calls",),
            "If supplied, the collection strictly validates as a list of TrajectoryToolCall objects.",
        ),
        (
            "record.model.ng_agent_observations.records",
            "schema",
            ("$.ng_agent_observations.records",),
            "If supplied, the collection strictly validates as a list of AgentObservationRecord objects.",
        ),
    ),
)


ARTIFACT_CHECK_GROUPS = (
    _group(
        "check_identity",
        (
            (
                "record.identity.rollout",
                "schema",
                ("$._ng_rollout_id", "$.ng_model_call_capture.rollout_id", "$.ng_trajectory.rollout_id"),
                "A nonempty rollout identity exists on a supported surface.",
            ),
            (
                "record.identity.task",
                "schema",
                ("$._ng_task_index", "$.ng_trajectory.task_id"),
                "A task index or truthy trajectory task identity exists.",
            ),
            (
                "record.schema.version",
                "schema",
                ("$.ng_trajectory.schema_version",),
                "A populated trajectory uses schema version 1.0.",
            ),
            (
                "record.identity.conflict",
                "semantic",
                ("$._ng_rollout_id", "$.ng_model_call_capture.rollout_id", "$.ng_trajectory.rollout_id"),
                "Populated trajectory and selected rollout identities agree.",
            ),
            (
                "record.reader.integrity",
                "semantic",
                ("$._capability_reader_issues",),
                "The reader reports no conflicting or unreadable retained evidence.",
            ),
        ),
    ),
    _group(
        "check_capture_presence",
        (
            (
                "TE-1.calls.required",
                "schema",
                ("$.ng_model_call_capture.calls",),
                "At least one effective captured call exists when the capability applies.",
            ),
            (
                "TE-2.calls.required",
                "schema",
                ("$.ng_model_call_capture.calls",),
                "At least one effective captured call exists when the capability applies.",
            ),
            (
                "TE-4.calls.required",
                "schema",
                ("$.ng_model_call_capture.calls",),
                "At least one effective captured call exists when the capability applies.",
            ),
            (
                "TE-7.calls.required",
                "schema",
                ("$.ng_model_call_capture.calls",),
                "At least one effective captured call exists when the capability applies.",
            ),
            (
                "TE-8.calls.required",
                "schema",
                ("$.ng_model_call_capture.calls",),
                "At least one effective captured call exists when the capability applies.",
            ),
            (
                "TE-9.calls.required",
                "schema",
                ("$.ng_model_call_capture.calls",),
                "At least one effective captured call exists when the capability applies.",
            ),
        ),
    ),
    _group(
        "check_model_calls",
        (
            ("TE-1.identity.unique", "semantic", (CALL + ".model_call_id",), "Call identities are unique."),
            (
                "TE-1.identity.required",
                "schema",
                (CALL + ".model_call_id", CALL + ".dialect"),
                "Call identity and dialect are nonempty.",
            ),
            (
                "TE-1.model_ref.required",
                "schema",
                (CALL + ".model_ref",),
                "The model reference has a nonempty server name.",
            ),
            ("TE-1.dialect.supported", "schema", (CALL + ".dialect",), "Dialect is responses, chat or messages."),
            (
                "TE-1.status.range",
                "schema",
                (CALL + ".status_code",),
                "A supplied HTTP status is between 100 and 599.",
            ),
            (
                "TE-1.outcome.required",
                "schema",
                (CALL + ".status_code", CALL + ".error_category"),
                "An HTTP status or explicit transport error exists.",
            ),
            (
                "TE-1.outcome.terminal",
                "schema",
                (CALL + ".status_code", CALL + ".response", CALL + ".error_category"),
                "The exchange has terminal outcome evidence; incomplete Responses include a reason.",
            ),
            (
                "TE-1.response.identity",
                "semantic",
                (CALL + ".response_id", CALL + ".response.id"),
                "Successful responses have an ID, and saved metadata matches a supplied body ID.",
            ),
            (
                "TE-1.outcome.preservation",
                "semantic",
                (CALL + ".response_status", CALL + ".finish_reason", CALL + ".response"),
                "Terminal metadata agrees with the retained response.",
            ),
            ("TE-1.outcome.error", "semantic", (CALL + ".response",), "A body error is not classified as success."),
            (
                "TE-1.timing.order",
                "semantic",
                (CALL + ".started_at", CALL + ".completed_at"),
                "When both clocks exist, completion does not precede start.",
            ),
        ),
    ),
    _group(
        "check_token_counts",
        (
            (
                "TE-2.usage.count",
                "schema",
                (
                    CALL + ".tokens_in",
                    CALL + ".tokens_out",
                    CALL + ".tokens_reasoning",
                    CALL + ".tokens_total",
                    CALL + ".cached_tokens",
                    CALL + ".response.usage",
                ),
                "Supplied token counts are nonnegative integers, not booleans.",
            ),
            (
                "TE-2.usage.preservation",
                "semantic",
                (CALL, CALL + ".response.usage"),
                "Saved counts equal normalized provider usage, including missing values and derived totals.",
            ),
            (
                "TE-2.usage.cache_subset",
                "semantic",
                (CALL + ".cached_tokens", CALL + ".tokens_in"),
                "Cached tokens do not exceed prompt tokens.",
            ),
            (
                "TE-2.usage.reasoning_subset",
                "semantic",
                (CALL + ".tokens_reasoning", CALL + ".tokens_out"),
                "Chat/Responses reasoning tokens do not exceed completion tokens.",
            ),
            (
                "TE-2.usage.total",
                "semantic",
                (CALL + ".tokens_total", CALL + ".tokens_in", CALL + ".tokens_out"),
                "Available total equals prompt plus completion.",
            ),
        ),
    ),
    _group(
        "check_content_references",
        (
            (
                "TE-4.payload.content_reference",
                "semantic",
                (CALL + ".request", CALL + ".response"),
                "Media and opaque references are resolvable by the reader; inline image data is valid.",
            ),
            (
                "TE-7.payload.content_reference",
                "semantic",
                (CALL + ".request", CALL + ".response"),
                "Media and opaque references are resolvable by the reader; inline image data is valid.",
            ),
        ),
    ),
    _group(
        "check_history",
        (
            (
                "TE-4.request.history",
                "schema",
                (CALL + ".request",),
                "Input/messages history exists as an array or string in a request object.",
            ),
            (
                "TE-4.request.previous_response",
                "semantic",
                (CALL + ".request.previous_response_id", CALL + ".response_id"),
                "Referenced previous response history is retained earlier in the call array.",
            ),
            (
                "TE-4.response.history",
                "schema",
                (CALL + ".response",),
                "Successful response objects contain a protocol-specific output array.",
            ),
        ),
    ),
    _group(
        "check_payloads",
        (
            (
                "TE-7.payload.request",
                "schema",
                (CALL + ".request", CALL + ".request_raw"),
                "A request object or raw request string is retained.",
            ),
            (
                "TE-7.payload.response",
                "schema",
                (CALL + ".response", CALL + ".response_raw", CALL + ".status_code", CALL + ".error_category"),
                "A response body is retained unless the exchange explicitly failed without a response.",
            ),
        ),
    ),
    _group(
        "check_ownership",
        (
            (
                "TE-8.invocations.required_unique",
                "semantic",
                INVOCATIONS,
                "At least one invocation exists and invocation IDs are unique.",
            ),
            (
                "TE-8.identity.required",
                "schema",
                ("$.ng_agent_observations.records[*].invocation_id", "$.ng_trajectory.invocations[*].invocation_id"),
                "Invocation identity is nonempty.",
            ),
            (
                "TE-8.references.required",
                "schema",
                ("$.ng_agent_observations.records[*].model_calls", "$.ng_trajectory.invocations[*].model_calls"),
                "Invocation call-reference arrays are explicitly present.",
            ),
            ("TE-8.invocation.parent", "semantic", INVOCATIONS, "Invocation parent references exist and are acyclic."),
            (
                "TE-8.ownership.reference",
                "semantic",
                ("$.ng_agent_observations.records[*]", "$.ng_trajectory.invocations[*]", CALL),
                "Each invocation call reference resolves uniquely.",
            ),
            (
                "TE-8.ownership.exactly_once",
                "semantic",
                ("$.ng_agent_observations.records[*]", "$.ng_trajectory.invocations[*]", CALL),
                "Every captured attempt has exactly one invocation owner.",
            ),
            (
                "TE-8.ownership.session",
                "semantic",
                ("$.ng_agent_observations.records[*]", "$.ng_trajectory.invocations[*]", CALL + ".client_session_id"),
                "Invocation ownership agrees with the captured client session when supplied.",
            ),
        ),
    ),
    _group(
        "check_turns",
        (
            (
                "TE-3.scope.contradiction",
                "semantic",
                ("$.ng_trajectory.turns",),
                "A declared step-free pair contains no turns.",
            ),
            (
                "TE-9.scope.contradiction",
                "semantic",
                ("$.ng_trajectory.turns",),
                "A declared step-free pair contains no turns.",
            ),
            ("TE-3.identity.repeat", "schema", ("$._ng_rollout_index",), "An explicit repeat identity exists."),
            (
                "TE-3.records.required",
                "schema",
                ("$.ng_trajectory.turns",),
                "Applicable step evidence contains at least one turn.",
            ),
            (
                "TE-3.turn.unique",
                "semantic",
                (TURN + ".invocation_id", TURN + ".turn_no"),
                "Invocation and turn-number pairs are unique.",
            ),
            (
                "TE-3.identity.required",
                "schema",
                (TURN + ".invocation_id", TURN + ".task_id", TURN + ".rollout_id"),
                "Turn invocation, task and rollout identities are nonempty.",
            ),
            ("TE-3.turn.question", "schema", (TURN + ".question",), "A non-null model-visible prompt is retained."),
            (
                "TE-3.turn.resolved",
                "schema",
                (TURN + ".resolved",),
                "Resolution is explicitly present, including unknown/null.",
            ),
            (
                "TE-3.turn.identity",
                "semantic",
                (TURN + ".task_id", TURN + ".rollout_id", "$.ng_trajectory.task_id", "$.ng_trajectory.rollout_id"),
                "Turn task and rollout identity agree with the containing trajectory.",
            ),
            (
                "TE-3.turn.content",
                "schema",
                (TURN + ".answer", TURN + ".reasoning_content"),
                "An answer/tool request or reasoning value exists.",
            ),
        ),
    ),
    _group(
        "check_step_join",
        (
            (
                "TE-9.ownership.conflict",
                "semantic",
                ("$.ng_agent_observations.records[*]", "$.ng_trajectory.invocations[*]", TURN + ".model_calls"),
                "A step join does not bypass conflicting invocation ownership.",
            ),
            (
                "TE-9.auxiliary.reference",
                "semantic",
                ("$.ng_agent_observations.records[*].model_calls", CALL),
                "Compaction helper references resolve uniquely.",
            ),
            (
                "TE-9.step.identity",
                "semantic",
                (TURN + ".invocation_id", TURN + ".turn_no"),
                "Explicit step identities are nonempty, positive and unique.",
            ),
            (
                "TE-9.turn.references",
                "schema",
                (TURN + ".model_calls",),
                "Turn call references are an explicit list of objects.",
            ),
            (
                "TE-9.turn.reference",
                "semantic",
                (TURN + ".model_calls", CALL),
                "Each step call reference resolves uniquely.",
            ),
            (
                "TE-9.turn.owner",
                "semantic",
                (
                    "$.ng_agent_observations.records[*]",
                    "$.ng_trajectory.invocations[*]",
                    TURN + ".invocation_id",
                    CALL + ".client_session_id",
                ),
                "Step owners agree with explicit invocation ownership and client sessions.",
            ),
            (
                "TE-9.accounting.exactly_once",
                "semantic",
                (TURN + ".model_calls", CALL, "$.ng_agent_observations.records[*]"),
                "Each retained policy attempt belongs to exactly one step and compaction helpers belong to none.",
            ),
        ),
    ),
    _group(
        "check_tools",
        (
            (
                "TE-5.scope.contradiction",
                "semantic",
                (
                    "$.ng_trajectory.tool_calls[*]",
                    "$.ng_agent_observations.records[*]",
                    "$.ng_trajectory.invocations[*]",
                    "$.response.output",
                ),
                "A declared tool-free pair contains no tool evidence.",
            ),
            ("TE-5.records.required", "schema", TOOLS, "Applicable tool evidence contains at least one execution."),
            ("TE-5.tool.unique", "semantic", TOOLS, "Tool invocation/execution identity pairs are unique."),
            (
                "TE-5.tool.execution",
                "semantic",
                (
                    "$.ng_trajectory.tool_calls[*]",
                    "$.ng_agent_observations.records[*]",
                    "$.ng_trajectory.invocations[*]",
                ),
                "Every conversation tool result has an execution record.",
            ),
            ("TE-5.identity.required", "schema", TOOLS, "Tool invocation ID, call ID and name are nonempty."),
            (
                "TE-5.tool.terminal",
                "schema",
                ("$.ng_trajectory.tool_calls[*].status", "$.ng_agent_observations.records[*].status"),
                "Tool outcome is completed, failed, timeout or cancelled.",
            ),
            (
                "TE-5.tool.request",
                "semantic",
                (
                    "$.ng_trajectory.tool_calls[*]",
                    "$.ng_agent_observations.records[*]",
                    "$.ng_trajectory.invocations[*]",
                ),
                "Executed tool joins to a matching name and supplied arguments in the conversation.",
            ),
            (
                "TE-5.tool.output",
                "semantic",
                (
                    "$.ng_trajectory.tool_calls[*]",
                    "$.ng_agent_observations.records[*]",
                    "$.ng_trajectory.invocations[*]",
                ),
                "Supplied execution output agrees with the conversation result.",
            ),
            (
                "TE-5.tool.outcome",
                "schema",
                (
                    "$.ng_trajectory.tool_calls[*]",
                    "$.ng_agent_observations.records[*]",
                    "$.ng_trajectory.invocations[*]",
                ),
                "An execution has model-visible output or explicit error evidence.",
            ),
        ),
    ),
    _group(
        "check_verifier",
        (
            (
                "TE-6.scope.contradiction",
                "semantic",
                ("$.reward", "$.mask_sample", "$.evaluation_completed", "$.verification_error", TURN + ".resolved"),
                "A declared verifier-free pair contains no verification evidence.",
            ),
            (
                "TE-6.verifier.boolean",
                "schema",
                ("$.mask_sample", "$.evaluation_completed"),
                "Verification flags are booleans when present.",
            ),
            (
                "TE-6.reward.required",
                "schema",
                ("$.reward",),
                "A finite numeric reward exists even for masked results.",
            ),
            (
                "TE-6.verifier.error",
                "schema",
                (
                    "$.failure_kind",
                    "$.failure_reason",
                    "$.mask_sample",
                    "$.evaluation_completed",
                    "$.verification_error",
                ),
                "Masked or incomplete verification has nonblank failure kind and reason.",
            ),
            (
                "TE-6.reward.scope",
                "semantic",
                (TURN + ".resolved",),
                "Episode resolution is not copied to intermediate steps.",
            ),
        ),
    ),
    _group(
        "check_gaps",
        (
            (
                "TE-1.capture.gap",
                "semantic",
                GAPS,
                "No producer gap declares missing captures or failed observation joins.",
            ),
            (
                "TE-4.capture.gap",
                "semantic",
                GAPS,
                "No producer gap declares missing captures or failed observation joins.",
            ),
            (
                "TE-7.capture.gap",
                "semantic",
                GAPS,
                "No producer gap declares missing captures or failed observation joins.",
            ),
            (
                "TE-8.capture.gap",
                "semantic",
                GAPS,
                "No producer gap declares missing captures or failed observation joins.",
            ),
            (
                "TE-9.capture.gap",
                "semantic",
                GAPS,
                "No producer gap declares missing captures or failed observation joins.",
            ),
            ("TE-8.ownership.gap", "semantic", GAPS, "No producer gap declares unresolved model-call ownership."),
            ("TE-3.turn.gap", "semantic", GAPS, "No producer gap declares unavailable steps when steps apply."),
            (
                "TE-9.accounting.gap",
                "semantic",
                GAPS,
                "No producer gap declares incomplete step-call membership when steps apply.",
            ),
        ),
    ),
)


REPORT_CHECKS = _definitions(
    "checker.result",
    (
        ("TE-1.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
        ("TE-2.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
        ("TE-3.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
        ("TE-4.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
        ("TE-5.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
        ("TE-6.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
        ("TE-7.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
        ("TE-8.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
        ("TE-9.record.integrity", "semantic", ("$",), "Record or reader integrity must pass before this TE can pass."),
    ),
)


BEHAVIORAL_CHECKS = (
    CheckDefinition(
        "probe.tools.identities",
        "behavioral",
        ("TE-5",),
        ("$.ng_trajectory.tool_calls[*]", "$.ng_agent_observations.records[*]"),
        "Witnessed and retained tool IDs match exactly.",
        "runner._tool_witness_issues",
        "retained tool identities differ from the independent tool witness",
    ),
    CheckDefinition(
        "probe.tools.join",
        "behavioral",
        ("TE-5",),
        (
            "$.ng_trajectory.tool_calls[*]",
            "$.ng_agent_observations.records[*]",
            "$.ng_trajectory.invocations[*]",
        ),
        "Each witnessed tool joins to one retained request and result.",
        "runner._tool_witness_issues",
        "retained tool request/result does not join uniquely to the witnessed execution",
    ),
    CheckDefinition(
        "probe.tools.request",
        "behavioral",
        ("TE-5",),
        (
            "$.ng_trajectory.tool_calls[*]",
            "$.ng_agent_observations.records[*]",
            "$.ng_trajectory.invocations[*]",
        ),
        "Tool name and arguments match the independent witness.",
        "runner._tool_witness_issues",
        "retained tool name or arguments differ from the independent tool witness",
    ),
    CheckDefinition(
        "probe.tools.status",
        "behavioral",
        ("TE-5",),
        ("$.ng_trajectory.tool_calls[*]", "$.ng_agent_observations.records[*]"),
        "Tool status matches the prescribed exit outcome.",
        "runner._tool_witness_issues",
        "retained tool status differs from the independent tool witness",
    ),
    CheckDefinition(
        "probe.tools.output",
        "behavioral",
        ("TE-5",),
        (
            "$.ng_trajectory.tool_calls[*]",
            "$.ng_agent_observations.records[*]",
            "$.ng_trajectory.invocations[*]",
        ),
        "Retained output matches the independently observed tool result.",
        "runner._tool_witness_issues",
        "retained tool output differs from the independent tool witness",
    ),
    CheckDefinition(
        "probe.execution.timeout",
        "behavioral",
        (),
        ("execution.timed_out",),
        "The episode completes within its timeout.",
        "runner.inspect_episode",
        "episode exceeded its timeout",
    ),
    CheckDefinition(
        "probe.execution.returncode",
        "behavioral",
        (),
        ("execution.returncode",),
        "The episode process exits successfully.",
        "runner.inspect_episode",
        "episode process failed; see episode.log",
    ),
    CheckDefinition(
        "probe.episode.seeded",
        "behavioral",
        (),
        ("witness.json:$.seeded",),
        "Exactly one fresh episode initialization is witnessed.",
        "runner.inspect_episode",
        "expected one fresh episode initialization",
    ),
    CheckDefinition(
        "probe.model.reached",
        "behavioral",
        ("TE-1",),
        ("witness.json:$.attempts",),
        "The harness reaches the controlled model endpoint.",
        "runner.inspect_episode",
        "the harness never reached the controlled model endpoint",
    ),
    CheckDefinition(
        "probe.model.failures",
        "behavioral",
        ("TE-1",),
        ("witness.json:$.attempts[*].status_code",),
        "The prescribed model-failure sequence occurs.",
        "runner.inspect_episode",
        "not all prescribed model failures were observed",
    ),
    CheckDefinition(
        "probe.model.retry_request",
        "behavioral",
        ("TE-1",),
        ("witness.json:$.attempts[*].request",),
        "Initial failure attempts repeat the same request.",
        "runner.inspect_episode",
        "retry scenario did not repeat the same request body",
    ),
    CheckDefinition(
        "probe.model.terminal_error",
        "behavioral",
        ("TE-1",),
        ("witness.json:$.attempts[*].status_code",),
        "The terminal model failure persists across attempts.",
        "runner.inspect_episode",
        "terminal model failure was not observed",
    ),
    CheckDefinition(
        "probe.model.finished",
        "behavioral",
        (),
        ("witness.json:$.finished",),
        "The harness finishes the scripted model exchange.",
        "runner.inspect_episode",
        "the harness did not finish the scripted model exchange",
    ),
    CheckDefinition(
        "probe.tools.executed",
        "behavioral",
        ("TE-4", "TE-5"),
        ("witness.json:$.tool_calls",),
        "All prescribed tool executions and returned results are witnessed.",
        "runner.inspect_episode",
        "prescribed tool executions and their returned results were not all witnessed",
    ),
    CheckDefinition(
        "probe.verifier.outcome",
        "behavioral",
        ("TE-6",),
        ("witness.json:$.verifications",),
        "One verification returns the prescribed grade.",
        "runner.inspect_episode",
        "expected verifier outcome was not observed",
    ),
    CheckDefinition(
        "probe.verifier.answer",
        "behavioral",
        ("TE-6",),
        ("witness.json:$.verifications[*].answer_seen",),
        "The verifier receives the scripted final answer.",
        "runner.inspect_episode",
        "the verifier did not receive the scripted final answer",
    ),
    CheckDefinition(
        "probe.episode.rollout",
        "behavioral",
        (),
        ("rollouts.jsonl",),
        "Exactly one rollout is collected.",
        "runner.inspect_episode",
        "expected exactly one collected rollout",
    ),
    CheckDefinition(
        "probe.model.exchanges",
        "behavioral",
        ("TE-1", "TE-2", "TE-4", "TE-7"),
        ("$.ng_model_call_capture.calls[*]", "witness.json:$.attempts"),
        "Retained exchanges match the independent endpoint witness.",
        "runner.inspect_episode",
        "retained model attempts differ from the independent endpoint witness",
    ),
    CheckDefinition(
        "probe.verifier.reward",
        "behavioral",
        ("TE-6",),
        ("$.reward", "witness.json:$.verifications"),
        "Saved reward equals the witnessed grade.",
        "runner.inspect_episode",
        "rollout reward differs from the verifier witness",
    ),
    CheckDefinition(
        "probe.model.protocol",
        "behavioral",
        (),
        ("witness.json:$.violations",),
        "The scripted provider reports no protocol violations.",
        "runner.inspect_episode",
    ),
)


_ALL_CHECKS = (
    *NATIVE_CHECKS,
    *(check for group in ARTIFACT_CHECK_GROUPS for check in group.checks),
    *REPORT_CHECKS,
    *BEHAVIORAL_CHECKS,
)
CHECKS = MappingProxyType({check.id: check for check in _ALL_CHECKS})
if len(CHECKS) != len(_ALL_CHECKS):
    raise ValueError("conformance check IDs must be unique")


def behavioral_issue(check_id: str) -> str:
    """Return an existing probe diagnostic through its registered assertion."""
    check = CHECKS[check_id]
    if check.kind != "behavioral" or check.failure_message is None:
        raise ValueError(f"{check_id} has no behavioral failure message")
    return check.failure_message


def check_catalog() -> list[dict[str, object]]:
    """Return a serializable inventory without importing a harness or launching work."""
    return [asdict(check) for check in CHECKS.values()]
