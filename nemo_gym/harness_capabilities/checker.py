# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic model and semantic checks over native Gym JSON."""

import base64
import binascii
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, TypeGuard

from pydantic import ValidationError

from nemo_gym.rollout_observability import ModelCallRef

from .contracts import TOKEN_FIELDS, model_errors


NAMES = {
    "TE-1": "model_call_status",
    "TE-2": "token_counts",
    "TE-3": "steps",
    "TE-4": "history",
    "TE-5": "tool_record",
    "TE-6": "verifier_outcome",
    "TE-7": "payloads",
    "TE-8": "run_join",
    "TE-9": "step_join",
}
PROFILE = "gym-p0/v1"
P0 = ("TE-1", "TE-2", "TE-3", "TE-4", "TE-5", "TE-6", "TE-7")


@dataclass(frozen=True)
class EvidenceScope:
    """Applicability of a harness/benchmark pair; absence never implies N/A."""

    tools: bool = True
    verifier: bool = True
    steps: bool = True


def gate_passes(evidence: dict) -> bool:
    """All applicable P0 rows and at least one exact call join must pass."""
    return all(evidence[key]["verdict"] in {"fulfilled", "not_applicable"} for key in P0) and any(
        evidence[key]["verdict"] == "fulfilled" for key in ("TE-8", "TE-9")
    )


@dataclass(frozen=True)
class Finding:
    evidence: str
    assertion: str
    location: str
    reason: str


def _objects(value: object) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _successful(call: dict) -> bool:
    code = call.get("status_code")
    return (
        type(code) is int
        and 200 <= code < 300
        and not call.get("error_category")
        and call.get("response_status") not in {"failed", "error", "cancelled"}
    )


def _missing_content(value: object) -> bool:
    """Reject unresolved media/opaque references; validate inline image data."""
    if isinstance(value, list):
        return any(_missing_content(item) for item in value)
    if isinstance(value, dict):
        unavailable = any(value.get(key) for key in ("file_id", "file_url", "encrypted_content"))
        if value.get("image_url"):
            image = value["image_url"]
            image = image.get("url") if isinstance(image, dict) else image
            if not isinstance(image, str) or ";base64," not in image or not image.startswith("data:"):
                unavailable = True
            else:
                try:
                    if not base64.b64decode(image.split(";base64,", 1)[1], validate=True):
                        unavailable = True
                except (ValueError, binascii.Error):
                    unavailable = True
        return bool(unavailable) or any(_missing_content(item) for item in value.values())
    return False


def _resolve(reference: dict, calls: list[dict]) -> list[int]:
    try:
        ModelCallRef.model_validate(reference, strict=True)
    except ValidationError:
        return []
    # TE joins require nonempty identifiers even where the model allows empty strings.
    if any(reference.get(key) == "" for key in ("model_call_id", "response_id")):
        return []
    if reference.get("model_ref") is not None and not reference["model_ref"].get("name"):
        return []
    return [
        index
        for index, call in enumerate(calls)
        if all(
            call.get(key) == reference[key]
            for key in ("model_call_id", "model_ref", "response_id")
            if reference.get(key) is not None
        )
    ]


def _provider_usage(usage: dict) -> dict:
    def first(*values):
        return next((v for v in values if v is not None), None)

    prompt = first(usage.get("input_tokens"), usage.get("prompt_tokens"))
    completion = first(usage.get("output_tokens"), usage.get("completion_tokens"))
    cache_read, cache_write = usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens")
    if all(v is None or _number(v) for v in (prompt, cache_read, cache_write)) and (
        cache_read is not None or cache_write is not None
    ):
        if prompt is not None or (cache_read or 0) + (cache_write or 0) > 0:
            prompt = (prompt or 0) + (cache_read or 0) + (cache_write or 0)
    total = usage.get("total_tokens")
    if total is None and _number(prompt) and _number(completion):
        total = prompt + completion
    return {
        "tokens_in": prompt,
        "tokens_out": completion,
        "tokens_total": total,
        "tokens_reasoning": first(
            _mapping(usage.get("output_tokens_details")).get("reasoning_tokens"),
            _mapping(usage.get("completion_tokens_details")).get("reasoning_tokens"),
            usage.get("reasoning_output_tokens"),
        ),
        "cached_tokens": first(
            _mapping(usage.get("input_tokens_details")).get("cached_tokens"),
            _mapping(usage.get("prompt_tokens_details")).get("cached_tokens"),
            _mapping(usage.get("input_tokens_details")).get("cached_input_tokens"),
            _mapping(usage.get("prompt_tokens_details")).get("cached_input_tokens"),
            cache_read,
            usage.get("cached_input_tokens"),
        ),
    }


class _RecordInspector:
    """Shared evidence and findings for capability-specific artifact checks."""

    def __init__(self, record: dict, source: str, scope: EvidenceScope) -> None:
        self.record = record
        self.scope = scope
        self.not_applicable: set[str] = set()
        self.source = source
        self.findings: list[Finding] = []
        self.trajectory = _mapping(self.record.get("ng_trajectory"))
        self.capture = _mapping(self.record.get("ng_model_call_capture"))
        self.bundle = _mapping(self.record.get("ng_agent_observations"))
        self.calls = _objects(self.capture.get("calls"))
        self.observations = _objects(self.bundle.get("records"))
        self.invocation_locations = [
            f"/ng_agent_observations/records/{i}"
            for i, r in enumerate(self.observations)
            if r.get("kind") == "agent_invocation"
        ]
        self.invocations = [r for r in self.observations if r.get("kind") == "agent_invocation"]
        if not self.invocations:
            self.invocations = _objects(self.trajectory.get("invocations"))
            self.invocation_locations = [f"/ng_trajectory/invocations/{i}" for i in range(len(self.invocations))]
        self.turns = _objects(self.trajectory.get("turns"))
        self.tools = _objects(self.trajectory.get("tool_calls")) or [
            r for r in self.observations if r.get("kind") == "tool_call"
        ]
        self.tool_locations = (
            [f"/ng_trajectory/tool_calls/{i}" for i in range(len(self.tools))]
            if _objects(self.trajectory.get("tool_calls"))
            else [
                f"/ng_agent_observations/records/{i}"
                for i, r in enumerate(self.observations)
                if r.get("kind") == "tool_call"
            ]
        )
        self.owners: dict[int, list[str]] = {i: [] for i in range(len(self.calls))}
        self.invocation_ids = [i.get("invocation_id") for i in self.invocations]

    def _fail(self, capability: str, assertion: str, location: str, reason: str) -> None:
        self.findings.append(Finding(capability, assertion, self.source + location, reason))

    def check_identity(self) -> None:
        """Record integrity: readable records, supported schema and consistent identity."""
        rollout_id = (
            self.record.get("_ng_rollout_id") or self.capture.get("rollout_id") or self.trajectory.get("rollout_id")
        )
        if not isinstance(rollout_id, str) or not rollout_id:
            self._fail(
                "record",
                "identity.rollout",
                "",
                "explicit rollout identity is required",
            )
        if self.record.get("_ng_task_index") is None and not self.trajectory.get("task_id"):
            self._fail("record", "identity.task", "", "explicit task identity is required")
        if self.trajectory and self.trajectory.get("schema_version") != "1.0":
            self._fail(
                "record",
                "schema.version",
                "/ng_trajectory/schema_version",
                "unsupported trajectory version",
            )
        if self.trajectory.get("rollout_id") and self.trajectory["rollout_id"] != rollout_id:
            self._fail(
                "record",
                "identity.conflict",
                "/ng_trajectory/rollout_id",
                "trajectory and capture identities differ",
            )
        for issue in self.record.get("_capability_reader_issues", []):
            self._fail("record", "reader.integrity", "", issue)

    def check_capture_presence(self) -> None:
        """Missing captures invalidate every capability that depends on calls."""
        for capability in ("TE-1", "TE-2", "TE-4", "TE-7", "TE-8", "TE-9"):
            if not self.calls and (capability != "TE-9" or self.scope.steps):
                self._fail(
                    capability,
                    "calls.required",
                    "/ng_model_call_capture/calls",
                    "no captured calls",
                )

    def check_model_calls(self) -> None:
        """TE-1: faithful terminal outcomes, including failure and cancellation."""
        call_ids = [c.get("model_call_id") for c in self.calls]
        if len(set(call_ids)) != len(call_ids):
            self._fail("TE-1", "identity.unique", "/ng_model_call_capture/calls", "duplicate call identity")
        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            self._require_text("TE-1", call, ("model_call_id", "dialect"), location)
            if not call.get("model_ref") or not call["model_ref"].get("name"):
                self._fail("TE-1", "model_ref.required", location, "explicit model server identity is required")
            if call.get("dialect") not in {"responses", "chat", "messages"}:
                self._fail("TE-1", "dialect.supported", location + "/dialect", "unsupported capture dialect")
            if call.get("status_code") is not None and not 100 <= call["status_code"] <= 599:
                self._fail("TE-1", "status.range", location + "/status_code", "invalid HTTP status")
            response = _mapping(call.get("response"))
            status = call.get("status_code")
            http_success = type(status) is int and 200 <= status < 300
            if status is None and not call.get("error_category"):
                self._fail("TE-1", "outcome.required", location, "HTTP outcome or transport error is required")
            if type(status) is int and status < 200:
                self._fail("TE-1", "outcome.terminal", location, "informational HTTP status is not a terminal outcome")
            if _successful(call) and not call.get("response_id"):
                self._fail("TE-1", "response.identity", location, "a returned successful response needs identity")
            if response.get("id") is not None and call.get("response_id") != response["id"]:
                self._fail("TE-1", "response.identity", location, "response identity differs from retained body")
            expected_status = response.get("status")
            choices = _objects(response.get("choices"))
            finish = next(
                (
                    v
                    for v in (
                        response.get("stop_reason"),
                        choices[0].get("finish_reason") if choices else None,
                        _mapping(response.get("incomplete_details")).get("reason"),
                    )
                    if isinstance(v, str)
                ),
                None,
            )
            for field, expected in (("response_status", expected_status), ("finish_reason", finish)):
                if response and call.get(field) != expected:
                    self._fail(
                        "TE-1",
                        "outcome.preservation",
                        location + "/" + field,
                        "terminal metadata differs from the retained response",
                    )
            if http_success and not call.get("error_category"):
                terminal = (
                    expected_status in {"completed", "incomplete", "failed", "error", "cancelled"}
                    if call.get("dialect") == "responses"
                    else bool(finish)
                )
                if not terminal or (expected_status == "incomplete" and not finish):
                    self._fail("TE-1", "outcome.terminal", location, "response termination evidence is missing")
                if response.get("error") and expected_status not in {"failed", "error", "cancelled"}:
                    self._fail("TE-1", "outcome.error", location, "response error is recorded as success")
            # Clocks are optional for equivalent capture paths. Validate them when retained.
            start, end = call.get("started_at"), call.get("completed_at")
            if _number(start) and _number(end) and end < start:
                self._fail("TE-1", "timing.order", location, "completion precedes start")

    def check_token_counts(self) -> None:
        """TE-2: preserve supplied usage on all attempts; report availability separately."""
        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            self._check_token_values(call, location)
            usage = _mapping(_mapping(call.get("response")).get("usage"))
            expected = _provider_usage(usage)
            self._check_token_values(expected, location + "/response/usage")
            for key, value in expected.items():
                if call.get(key) != value:
                    self._fail(
                        "TE-2",
                        "usage.preservation",
                        location + "/" + key,
                        "count differs from provider usage or documented derived total, including absence",
                    )
            prompt, completion, reasoning, total, cached = (call.get(k) for k in TOKEN_FIELDS)
            if _number(prompt) and _number(cached) and cached > prompt:
                self._fail("TE-2", "usage.cache_subset", location, "cached count exceeds prompt count")
            if (
                call.get("dialect") in {"responses", "chat"}
                and _number(reasoning)
                and _number(completion)
                and reasoning > completion
            ):
                self._fail("TE-2", "usage.reasoning_subset", location, "reasoning count exceeds completion count")
            if all(_number(v) for v in (prompt, completion, total)) and total != prompt + completion:
                self._fail("TE-2", "usage.total", location, "total differs from prompt plus completion")

    def check_content_references(self) -> None:
        """TE-4/TE-7: reject unavailable media and opaque payload dependencies."""
        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            request, response = call.get("request"), call.get("response")
            if _missing_content(request) or _missing_content(response):
                for capability in ("TE-4", "TE-7"):
                    self._fail(
                        capability,
                        "payload.content_reference",
                        location,
                        "external, encrypted or invalid content is not resolved by this reader",
                    )

    def check_history(self) -> None:
        """TE-4: ordered model request and response history."""
        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            request, response = call.get("request"), call.get("response")
            request_key = "input" if call.get("dialect") == "responses" else "messages"
            if not isinstance(request, dict) or not isinstance(request.get(request_key), (list, str)):
                self._fail(
                    "TE-4",
                    "request.history",
                    location + "/request",
                    "ordered model input is unavailable",
                )
            previous = _mapping(request).get("previous_response_id")
            if previous and not any(
                c.get("response_id") == previous and c.get("request") is not None and c.get("response") is not None
                for c in self.calls[:index]
            ):
                self._fail(
                    "TE-4", "request.previous_response", location, "referenced prior response history is not retained"
                )
            if _successful(call):
                response_key = {
                    "responses": "output",
                    "chat": "choices",
                    "messages": "content",
                }.get(str(call.get("dialect")), "")
                if not isinstance(response, dict) or not isinstance(response.get(response_key), list):
                    self._fail(
                        "TE-4",
                        "response.history",
                        location + "/response",
                        "model output is unavailable",
                    )

    def check_payloads(self) -> None:
        """TE-7: retained payloads or explicit no-response transport failures."""
        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            request, response = call.get("request"), call.get("response")
            request_present = isinstance(request, dict) or isinstance(call.get("request_raw"), str)
            response_present = isinstance(response, dict) or isinstance(call.get("response_raw"), str)
            if not request_present:
                self._fail(
                    "TE-7",
                    "payload.request",
                    location,
                    "request payload unavailable",
                )
            if not response_present and (call.get("status_code") is not None or not call.get("error_category")):
                self._fail(
                    "TE-7",
                    "payload.response",
                    location,
                    "response body unavailable; not an explicit no-response failure",
                )

    def check_ownership(self) -> None:
        """TE-8: each captured attempt resolves to one explicit invocation owner."""
        if not self.invocations or len(set(self.invocation_ids)) != len(self.invocation_ids):
            self._fail(
                "TE-8",
                "invocations.required_unique",
                "/ng_agent_observations",
                "unique invocations are required",
            )
        parents = {i.get("invocation_id"): i.get("parent_invocation_id") for i in self.invocations}
        for index, invocation in enumerate(self.invocations):
            location = self.invocation_locations[index]
            self._require_text("TE-8", invocation, ("invocation_id",), location)
            if "model_calls" not in invocation:
                self._fail("TE-8", "references.required", location + "/model_calls", "explicit call refs are required")
            parent = invocation.get("parent_invocation_id")
            seen = {invocation.get("invocation_id")}
            while parent is not None and parent not in seen and parent in parents:
                seen.add(parent)
                parent = parents[parent]
            if parent is not None:
                self._fail(
                    "TE-8",
                    "invocation.parent",
                    location,
                    "parent is missing or cyclic",
                )
            for reference in _objects(invocation.get("model_calls")):
                matches = _resolve(reference, self.calls)
                if len(matches) != 1:
                    self._fail(
                        "TE-8",
                        "ownership.reference",
                        location,
                        "call reference does not resolve uniquely",
                    )
                elif isinstance(invocation.get("invocation_id"), str):
                    self.owners[matches[0]].append(invocation["invocation_id"])
        for index, assigned in self.owners.items():
            if len(assigned) != 1:
                self._fail(
                    "TE-8",
                    "ownership.exactly_once",
                    f"/ng_model_call_capture/calls/{index}",
                    "call must have exactly one explicit owner",
                )
            elif self.calls[index].get("client_session_id") and self.calls[index]["client_session_id"] != assigned[0]:
                self._fail(
                    "TE-8",
                    "ownership.session",
                    f"/ng_model_call_capture/calls/{index}",
                    "owner conflicts with captured client session",
                )

    def _require_text(self, capability: str, item: dict, fields: tuple[str, ...], location: str) -> None:
        for field in fields:
            if not item.get(field):
                self._fail(capability, "identity.required", location + "/" + field, "nonempty value is required")

    def _check_token_values(self, values: dict, location: str) -> None:
        for field in TOKEN_FIELDS:
            value = values.get(field)
            if value is not None and (type(value) is not int or value < 0):
                self._fail(
                    "TE-2", "usage.count", location + "/" + field, "count must be a nonnegative integer or null"
                )

    def _check_records(self, capability: str, records: list[dict], location: str) -> None:
        if not records:
            self._fail(capability, "records.required", location, "required records are absent")

    def check_turns(self) -> None:
        """TE-3 contains step fields; TE-9 independently checks call linkage."""
        if not self.scope.steps:
            if self.turns:
                self._fail(
                    "TE-3", "scope.contradiction", "/ng_trajectory/turns", "steps exist in a declared step-free pair"
                )
                self._fail(
                    "TE-9", "scope.contradiction", "/ng_trajectory/turns", "steps exist in a declared step-free pair"
                )
            else:
                self.not_applicable.update(("TE-3", "TE-9"))
            return
        if self.record.get("_ng_rollout_index") is None:
            self._fail("TE-3", "identity.repeat", "", "explicit repeat identity is required")
        self._check_records("TE-3", self.turns, "/ng_trajectory/turns")
        keys = [(t.get("invocation_id"), t.get("turn_no")) for t in self.turns]
        if len(set(keys)) != len(keys):
            self._fail("TE-3", "turn.unique", "/ng_trajectory/turns", "duplicate step identity")
        for index, turn in enumerate(self.turns):
            location = f"/ng_trajectory/turns/{index}"
            self._require_text("TE-3", turn, ("invocation_id", "task_id", "rollout_id"), location)
            if turn.get("question") is None:
                self._fail(
                    "TE-3", "turn.question", location + "/question", "non-null model-visible prompt is required"
                )
            if "resolved" not in turn:
                self._fail(
                    "TE-3", "turn.resolved", location + "/resolved", "resolution or explicit unknown is required"
                )
            if any(turn.get(k) != self.trajectory.get(k) for k in ("task_id", "rollout_id")):
                self._fail("TE-3", "turn.identity", location, "step identity differs from rollout")
            if turn.get("answer") is None and turn.get("reasoning_content") is None:
                self._fail("TE-3", "turn.content", location, "answer/tool or reasoning field is required")

    def check_step_join(self) -> None:
        """TE-9: exact policy-attempt membership, independent of invocation refs."""
        if not self.scope.steps:
            return
        if any(
            f.evidence == "TE-8" and f.assertion in {"ownership.reference", "invocation.parent", "ownership.session"}
            for f in self.findings
        ):
            self._fail(
                "TE-9",
                "ownership.conflict",
                "/ng_agent_observations",
                "contradictory run refs cannot be bypassed by a step join",
            )
        auxiliary: set[int] = set()
        for observation in self.observations:
            if observation.get("kind") == "context_compaction":
                for reference in _objects(observation.get("model_calls")):
                    matches = _resolve(reference, self.calls)
                    if len(matches) != 1:
                        self._fail(
                            "TE-9",
                            "auxiliary.reference",
                            "/ng_agent_observations",
                            "helper reference does not resolve uniquely",
                        )
                    else:
                        auxiliary.add(matches[0])
        keys = [(t.get("invocation_id"), t.get("turn_no")) for t in self.turns]
        if len(set(keys)) != len(keys) or any(
            not isinstance(k[0], str) or not k[0] or type(k[1]) is not int or k[1] < 1 for k in keys
        ):
            self._fail("TE-9", "step.identity", "/ng_trajectory/turns", "unique explicit step identities are required")
        refs: Counter = Counter()
        for turn in self.turns:
            references = turn.get("model_calls")
            if not isinstance(references, list) or len(_objects(references)) != len(references):
                self._fail("TE-9", "turn.references", "/ng_trajectory/turns", "step references must be objects")
            for reference in _objects(references):
                matches = _resolve(reference, self.calls)
                if len(matches) != 1:
                    self._fail(
                        "TE-9",
                        "turn.reference",
                        "/ng_trajectory/turns",
                        "step call reference does not resolve uniquely",
                    )
                else:
                    index = matches[0]
                    owner = turn.get("invocation_id")
                    session = self.calls[index].get("client_session_id")
                    if (self.owners[index] and self.owners[index] != [owner]) or (session and session != owner):
                        self._fail(
                            "TE-9",
                            "turn.owner",
                            "/ng_trajectory/turns",
                            "step owner conflicts with explicit call ownership",
                        )
                    refs[index] += 1
        policy = set(range(len(self.calls))) - auxiliary
        if not self.turns or not policy or any(refs[i] != 1 for i in policy) or any(refs[i] for i in auxiliary):
            self._fail(
                "TE-9",
                "accounting.exactly_once",
                "/ng_trajectory/turns",
                "each retained policy attempt must belong to exactly one step; helpers must not be policy steps",
            )

    def check_tools(self) -> None:
        """TE-5: unique executed tools, request joins and terminal outcomes."""
        if not self.scope.tools:
            has_tool_items = any(
                item.get("type") in {"function_call", "function_call_output"}
                for inv in self.invocations
                for item in _objects(inv.get("conversation"))
            )
            has_tool_items = has_tool_items or any(
                item.get("type") == "function_call_output"
                for item in _objects(_mapping(self.record.get("response")).get("output"))
            )
            if self.tools or has_tool_items:
                self._fail(
                    "TE-5", "scope.contradiction", "/tool_calls", "tool evidence exists in a declared tool-free pair"
                )
            else:
                self.not_applicable.add("TE-5")
            return
        self._check_records("TE-5", self.tools, "/ng_trajectory/tool_calls")
        conversations = {
            (
                invocation.get("invocation_id"),
                item.get("call_id"),
                item.get("type"),
            ): item
            for invocation in self.invocations
            for item in _objects(invocation.get("conversation"))
            if item.get("call_id")
        }
        tool_keys = [(t.get("invocation_id"), t.get("tool_call_id")) for t in self.tools]
        if len(set(tool_keys)) != len(tool_keys):
            self._fail(
                "TE-5",
                "tool.unique",
                "/ng_trajectory/tool_calls",
                "duplicate execution identity",
            )
        for invocation_id, call_id, kind in conversations:
            if kind == "function_call_output" and (invocation_id, call_id) not in tool_keys:
                self._fail("TE-5", "tool.execution", "/tool_calls", "a tool result has no execution record")
        for index, tool in enumerate(self.tools):
            location = self.tool_locations[index]
            self._require_text("TE-5", tool, ("invocation_id", "tool_call_id", "tool_name"), location)
            if tool.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
                self._fail("TE-5", "tool.terminal", location + "/status", "terminal tool outcome is required")
            tool_key = (tool.get("invocation_id"), tool.get("tool_call_id"))
            request = conversations.get((*tool_key, "function_call"), {})
            output = conversations.get((*tool_key, "function_call_output"), {}).get("output", tool.get("output"))
            if request.get("arguments") is None or request.get("name") != tool.get("tool_name"):
                self._fail(
                    "TE-5",
                    "tool.request",
                    location,
                    "execution cannot be joined to tool name and arguments",
                )
            if tool.get("output") is not None and output != tool["output"]:
                self._fail("TE-5", "tool.output", location, "execution output differs from the model-visible result")
            if output is None and not tool.get("error_type"):
                self._fail(
                    "TE-5",
                    "tool.outcome",
                    location,
                    "execution has neither output nor error evidence",
                )

    def check_verifier(self) -> None:
        """TE-6's shipped Gym surface: rollout verification reward and terminal resolution."""
        reward = self.record.get("reward")
        failed = (
            self.record.get("mask_sample") is True
            or self.record.get("evaluation_completed") is False
            or self.record.get("verification_error") is not None
        )
        if not self.scope.verifier:
            if reward is not None or failed or any(t.get("resolved") is not None for t in self.turns):
                self._fail(
                    "TE-6",
                    "scope.contradiction",
                    "/reward",
                    "verifier evidence exists in a declared verifier-free pair",
                )
            else:
                self.not_applicable.add("TE-6")
            return
        for field in ("mask_sample", "evaluation_completed"):
            if field in self.record and type(self.record[field]) is not bool:
                self._fail("TE-6", "verifier.boolean", "/" + field, "verification flags must be booleans when present")
        # Gym requires a numeric reward even when the mask excludes it from scoring.
        if not _number(reward):
            self._fail(
                "TE-6",
                "reward.required",
                "/reward",
                "finite verifier reward is required; binary resolution is not a floating grade",
            )
        if failed:
            for field in ("failure_kind", "failure_reason"):
                value = self.record.get(field)
                if not isinstance(value, str) or not value.strip():
                    self._fail(
                        "TE-6",
                        "verifier.error",
                        "/" + field,
                        "masked or incomplete verification needs nonblank failure metadata",
                    )
        for invocation in {t.get("invocation_id") for t in self.turns}:
            turns = sorted(
                (t for t in self.turns if t.get("invocation_id") == invocation), key=lambda t: t.get("turn_no", 0)
            )
            for turn in turns[:-1]:
                if turn.get("resolved") is not None:
                    self._fail(
                        "TE-6",
                        "reward.scope",
                        "/ng_trajectory/turns",
                        "episode resolution must not be copied to intermediate steps",
                    )

    def check_gaps(self) -> None:
        """Propagate explicit producer gaps to dependent capabilities."""
        gaps: Iterable[dict] = (
            *_objects(self.capture.get("gaps")),
            *_objects(self.bundle.get("gaps")),
            *_objects(self.trajectory.get("gaps")),
        )
        for gap in gaps:
            code = str(gap.get("code", ""))
            if code.startswith("model_call_capture") or code == "agent_observation_join_failed":
                for capability in ("TE-1", "TE-4", "TE-7", "TE-8", "TE-9"):
                    self._fail(
                        capability,
                        "capture.gap",
                        "/gaps",
                        "producer declares missing or unreadable capture evidence",
                    )
            if code.startswith("model_call_reference") or code == "model_call_ownership_unavailable":
                self._fail(
                    "TE-8",
                    "ownership.gap",
                    "/gaps",
                    "producer declares unresolved ownership",
                )

            if self.scope.steps and code in {
                "turns_unavailable",
                "turn_evidence_incomplete",
                "trajectory_projection_failed",
            }:
                self._fail("TE-3", "turn.gap", "/gaps", "producer declares unavailable step evidence")
            if self.scope.steps and code == "turn_model_call_scope_incomplete":
                self._fail("TE-9", "accounting.gap", "/gaps", "producer declares incomplete step-call membership")

    def result(self) -> dict:
        """Report evidence correctness, metric availability and delivery independently."""
        failed = {f.evidence for f in self.findings}
        if "record" in failed:
            for key in NAMES:
                self._fail(key, "record.integrity", "", "record identity or reader integrity failed")
            failed.update(NAMES)
        evidence = {
            key: {
                "name": name,
                "verdict": "not_fulfilled"
                if key in failed
                else "not_applicable"
                if key in self.not_applicable
                else "fulfilled",
                "basis": "retained_artifacts",
            }
            for key, name in NAMES.items()
        }
        availability = {
            key: {"available": sum(c.get(key) is not None for c in self.calls), "calls": len(self.calls)}
            for key in TOKEN_FIELDS
        }
        return {
            "source": self.source,
            "verdict": "fulfilled" if gate_passes(evidence) else "not_fulfilled",
            "evidence": evidence,
            "token_availability": availability,
            "is_behavioral_qualification": False,
            "scope_closure": "not_independently_witnessed",
            "findings": [asdict(f) for f in self.findings],
        }


def inspect_record(record: dict, *, source: str = "record", scope: EvidenceScope = EvidenceScope()) -> dict:
    """Check TE-1–TE-9 on retained JSONL; do not infer missing evidence or applicability."""
    checks = _RecordInspector(record, source, scope)
    checks.findings.extend(
        Finding("record", code, source + path, "object at the evidence path does not validate its Gym model")
        for path, code in model_errors(record)
    )
    if checks.findings:
        # Semantic checks require typed identities and containers. An invalid native
        # object is a record integrity failure, never partially valid evidence.
        return checks.result()
    checks.check_identity()
    checks.check_capture_presence()
    checks.check_model_calls()
    checks.check_token_counts()
    checks.check_content_references()
    checks.check_history()
    checks.check_payloads()
    checks.check_ownership()
    checks.check_turns()
    checks.check_step_join()
    checks.check_tools()
    checks.check_verifier()
    checks.check_gaps()
    return checks.result()
