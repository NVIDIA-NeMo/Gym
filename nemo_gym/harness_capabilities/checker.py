# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic checks over native Gym JSON; no producer runtime imports."""

import base64
import binascii
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, TypeGuard

from jsonschema import Draft202012Validator

from .schemas import CALL_REF, SCHEMAS


NAMES = {
    "C0": "readable records and identity",
    "C1": "model calls and statuses",
    "C2": "five token counts",
    "C3": "canonical semantic turns",
    "C4": "model-visible history",
    "C5": "tool execution evidence",
    "C6": "independent parallel timing",
    "C7": "resource lifecycle",
    "C8": "verifier resolution and rewards",
    "C9": "model-call payloads",
    "C10": "invocation ownership",
    "C11": "complete turn-call accounting",
}
P0 = ("C0", "C1", "C2", "C4", "C9", "C10")
PENDING_VALIDATORS = frozenset({"C6", "C7", "C8", "C11"})
PROFILES = {
    "gym-artifacts-p0/v1": P0,
    "gym-artifacts-p1/v1": (*P0, "C5", "C6"),
    "gym-artifacts-all/v1": tuple(NAMES),
}


@dataclass(frozen=True)
class Finding:
    capability: str
    assertion: str
    location: str
    reason: str


def _schema_issues(capability: str, value: object, location: str) -> list[Finding]:
    # Never put error.message in reports: jsonschema embeds source payloads.
    return [
        Finding(
            capability,
            "schema." + str(error.validator),
            location + "/" + "/".join(map(str, error.absolute_path)),
            "required artifact contract is not satisfied",
        )
        for error in Draft202012Validator(SCHEMAS[capability]).iter_errors(value)
    ]


def _objects(value: object) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _successful(call: dict) -> bool:
    code = call.get("status_code")
    return isinstance(code, int) and 200 <= code < 300 and not call.get("error_category")


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
    if not Draft202012Validator(CALL_REF).is_valid({k: v for k, v in reference.items() if v is not None}):
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


class _RecordInspector:
    """Shared evidence and findings for capability-specific artifact checks."""

    def __init__(self, record: dict, source: str) -> None:
        self.record = record
        self.source = source
        self.findings: list[Finding] = []
        self.trajectory = _mapping(self.record.get("ng_trajectory"))
        self.capture = _mapping(self.record.get("ng_model_call_capture"))
        self.bundle = _mapping(self.record.get("ng_agent_observations"))
        self.calls = _objects(self.capture.get("calls"))
        self.observations = _objects(self.bundle.get("records"))
        self.invocations = [r for r in self.observations if r.get("kind") == "agent_invocation"]
        if not self.invocations:
            self.invocations = _objects(self.trajectory.get("invocations"))
        self.turns = _objects(self.trajectory.get("turns"))
        self.tools = _objects(self.trajectory.get("tool_calls")) or [
            r for r in self.observations if r.get("kind") == "tool_call"
        ]
        self.owners: dict[int, list[str]] = {i: [] for i in range(len(self.calls))}
        self.invocation_ids = [i.get("invocation_id") for i in self.invocations]

    def _fail(self, capability: str, assertion: str, location: str, reason: str) -> None:
        self.findings.append(Finding(capability, assertion, self.source + location, reason))

    def check_identity(self) -> None:
        """C0: readable records, supported schema and consistent identity."""
        rollout_id = self.record.get("_ng_rollout_id") or self.capture.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            self._fail(
                "C0",
                "identity.rollout",
                "",
                "explicit rollout identity is required",
            )
        if self.record.get("_ng_task_index") is None and not self.trajectory.get("task_id"):
            self._fail("C0", "identity.task", "", "explicit task identity is required")
        if self.trajectory and self.trajectory.get("schema_version") != "1.0":
            self._fail(
                "C0",
                "schema.version",
                "/ng_trajectory/schema_version",
                "unsupported trajectory version",
            )
        if self.trajectory.get("rollout_id") and self.trajectory["rollout_id"] != rollout_id:
            self._fail(
                "C0",
                "identity.conflict",
                "/ng_trajectory/rollout_id",
                "trajectory and capture identities differ",
            )
        if not isinstance(self.capture.get("calls"), list) or len(self.calls) != len(self.capture["calls"]):
            self._fail(
                "C0",
                "capture.shape",
                "/ng_model_call_capture/calls",
                "a list of call objects is required",
            )
        for issue in self.record.get("_capability_reader_issues", []):
            self._fail("C0", "reader.integrity", "", issue)

    def check_capture_presence(self) -> None:
        """Missing captures invalidate every capability that depends on calls."""
        for capability in ("C1", "C2", "C4", "C9", "C10", "C11"):
            if not self.calls:
                self._fail(
                    capability,
                    "calls.required",
                    "/ng_model_call_capture/calls",
                    "no captured calls",
                )

    def check_model_calls(self) -> None:
        """C1: unique call identities, model identity, status and timing."""
        call_ids = [c.get("model_call_id") for c in self.calls]
        if len(set(call_ids)) != len(call_ids):
            self._fail(
                "C1",
                "identity.unique",
                "/ng_model_call_capture/calls",
                "duplicate call identity",
            )

        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            self.findings.extend(_schema_issues("C1", call, self.source + location))
            start, end = call.get("started_at"), call.get("completed_at")
            if _number(start) and _number(end) and end < start:
                self._fail("C1", "timing.order", location, "completion precedes start")
            if not call.get("model") and not _mapping(call.get("model_ref")).get("name"):
                self._fail(
                    "C1",
                    "model.identity",
                    location,
                    "model identity unavailable",
                )

    def check_token_counts(self) -> None:
        """C2: complete token counts consistent with retained provider usage."""
        if not any(_successful(c) for c in self.calls):
            self._fail(
                "C2",
                "usage.eligible",
                "/ng_model_call_capture/calls",
                "no successful calls with token evidence",
            )

        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            response = call.get("response")
            if _successful(call):
                self.findings.extend(_schema_issues("C2", call, self.source + location))
                prompt, cached = call.get("tokens_in"), call.get("cached_tokens")
                if _number(prompt) and _number(cached) and cached > prompt:
                    self._fail(
                        "C2",
                        "usage.cache_subset",
                        location,
                        "cached count exceeds prompt count",
                    )
                # Responses/chat count reasoning inside completion. Anthropic has
                # different cache semantics and does not supply reasoning counts.
                if call.get("dialect") in ("responses", "chat"):
                    completion, reasoning, total = (
                        call.get(k)
                        for k in (
                            "tokens_out",
                            "tokens_reasoning",
                            "tokens_total",
                        )
                    )
                    if _number(reasoning) and _number(completion) and reasoning > completion:
                        self._fail(
                            "C2",
                            "usage.reasoning_subset",
                            location,
                            "reasoning count exceeds completion count",
                        )
                    if _number(prompt) and _number(completion) and _number(total) and total != prompt + completion:
                        self._fail(
                            "C2",
                            "usage.total",
                            location,
                            "total differs from prompt plus completion",
                        )
                usage = _mapping(_mapping(response).get("usage"))
                if call.get("dialect") in ("responses", "chat"):
                    is_responses = call.get("dialect") == "responses"
                    input_key, output_key = (
                        ("input_tokens", "output_tokens") if is_responses else ("prompt_tokens", "completion_tokens")
                    )
                    expected = {
                        "tokens_in": usage.get(input_key),
                        "tokens_out": usage.get(output_key),
                        "tokens_total": usage.get("total_tokens"),
                        "tokens_reasoning": _mapping(usage.get(output_key + "_details")).get("reasoning_tokens"),
                        "cached_tokens": _mapping(usage.get(input_key + "_details")).get("cached_tokens"),
                    }
                    for key, value in expected.items():
                        if call.get(key) != value:
                            self._fail(
                                "C2",
                                "usage.preservation",
                                location + "/" + key,
                                "count differs from captured provider usage, including absence",
                            )

    def check_content_references(self) -> None:
        """C4/C9: reject unavailable media and opaque payload dependencies."""
        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            request, response = call.get("request"), call.get("response")
            if _missing_content(request) or _missing_content(response):
                for capability in ("C4", "C9"):
                    self._fail(
                        capability,
                        "payload.content_reference",
                        location,
                        "external, encrypted or invalid content is not resolved by this reader",
                    )

    def check_history(self) -> None:
        """C4: ordered model request and response history."""
        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            request, response = call.get("request"), call.get("response")
            request_key = "input" if call.get("dialect") == "responses" else "messages"
            if not isinstance(request, dict) or not isinstance(request.get(request_key), (list, str)):
                self._fail(
                    "C4",
                    "request.history",
                    location + "/request",
                    "ordered model input is unavailable",
                )
            if _successful(call):
                response_key = {
                    "responses": "output",
                    "chat": "choices",
                    "messages": "content",
                }.get(str(call.get("dialect")), "")
                if not isinstance(response, dict) or not isinstance(response.get(response_key), list):
                    self._fail(
                        "C4",
                        "response.history",
                        location + "/response",
                        "model output is unavailable",
                    )

    def check_payloads(self) -> None:
        """C9: retained payloads or explicit no-response transport failures."""
        for index, call in enumerate(self.calls):
            location = f"/ng_model_call_capture/calls/{index}"
            request, response = call.get("request"), call.get("response")
            request_present = isinstance(request, dict) or isinstance(call.get("request_raw"), str)
            response_present = isinstance(response, dict) or isinstance(call.get("response_raw"), str)
            if not request_present:
                self._fail(
                    "C9",
                    "payload.request",
                    location,
                    "request payload unavailable",
                )
            if not response_present and (call.get("status_code") is not None or not call.get("error_category")):
                self._fail(
                    "C9",
                    "payload.response",
                    location,
                    "response body unavailable; not an explicit no-response failure",
                )

    def check_ownership(self) -> None:
        """C10: each captured attempt resolves to one explicit invocation owner."""
        if not self.invocations or len(set(self.invocation_ids)) != len(self.invocation_ids):
            self._fail(
                "C10",
                "invocations.required_unique",
                "/ng_agent_observations",
                "unique invocations are required",
            )
        parents = {i.get("invocation_id"): i.get("parent_invocation_id") for i in self.invocations}
        for index, invocation in enumerate(self.invocations):
            location = f"/invocations/{index}"
            self.findings.extend(_schema_issues("C10", invocation, self.source + location))
            parent = invocation.get("parent_invocation_id")
            seen = {invocation.get("invocation_id")}
            while parent is not None and parent not in seen and parent in parents:
                seen.add(parent)
                parent = parents[parent]
            if parent is not None:
                self._fail(
                    "C10",
                    "invocation.parent",
                    location,
                    "parent is missing or cyclic",
                )
            for reference in _objects(invocation.get("model_calls")):
                matches = _resolve(reference, self.calls)
                if len(matches) != 1:
                    self._fail(
                        "C10",
                        "ownership.reference",
                        location,
                        "call reference does not resolve uniquely",
                    )
                elif isinstance(invocation.get("invocation_id"), str):
                    self.owners[matches[0]].append(invocation["invocation_id"])
        for index, assigned in self.owners.items():
            if len(assigned) != 1:
                self._fail(
                    "C10",
                    "ownership.exactly_once",
                    f"/ng_model_call_capture/calls/{index}",
                    "call must have exactly one explicit owner",
                )
            elif self.calls[index].get("client_session_id") and self.calls[index]["client_session_id"] != assigned[0]:
                self._fail(
                    "C10",
                    "ownership.session",
                    f"/ng_model_call_capture/calls/{index}",
                    "owner conflicts with captured client session",
                )

    def _check_records(self, capability: str, records: list[dict]) -> None:
        if not records:
            self._fail(
                capability,
                "records.required",
                "/ng_trajectory",
                "required records are absent",
            )
        for index, item in enumerate(records):
            self.findings.extend(_schema_issues(capability, item, f"{self.source}/{capability}/{index}"))

    def check_turns(self) -> Counter:
        """C3: canonical turns; also report C11 reference defects and count joins."""
        self._check_records("C3", self.turns)
        turn_refs: Counter = Counter()
        turn_keys = [(t.get("invocation_id"), t.get("turn_no")) for t in self.turns]
        if len(set(turn_keys)) != len(turn_keys):
            self._fail(
                "C3",
                "turn.unique",
                "/ng_trajectory/turns",
                "duplicate semantic turn",
            )
        for turn in self.turns:
            if turn.get("invocation_id") not in self.invocation_ids or any(
                turn.get(k) != self.trajectory.get(k) for k in ("task_id", "rollout_id")
            ):
                self._fail(
                    "C3",
                    "turn.identity",
                    "/ng_trajectory/turns",
                    "turn identity does not match trajectory/invocation",
                )
            for reference in _objects(turn.get("model_calls")):
                matches = _resolve(reference, self.calls)
                if len(matches) != 1 or self.owners[matches[0]] != [turn.get("invocation_id")]:
                    self._fail(
                        "C3",
                        "turn.reference",
                        "/ng_trajectory/turns",
                        "turn call is unmatched or owned by another invocation",
                    )
                    self._fail(
                        "C11",
                        "turn.reference",
                        "/ng_trajectory/turns",
                        "turn call is unmatched or owned by another invocation",
                    )
                else:
                    turn_refs[matches[0]] += 1
        return turn_refs

    def check_accounting(self, turn_refs: Counter) -> None:
        """C11: turn-call accounting and the missing independent closure evidence."""
        if not self.turns or any(turn_refs[i] != 1 for i in range(len(self.calls))):
            self._fail(
                "C11",
                "accounting.exactly_once",
                "/ng_trajectory/turns",
                "calls are not assigned to turns exactly once",
            )
        # Gym 1.0 has neither closure/witness records nor auxiliary-purpose buckets.
        self._fail(
            "C11",
            "accounting.closure",
            "/ng_trajectory",
            "independent closed-scope ledger and auxiliary accounting required",
        )

    def check_tools(self) -> None:
        """C5: unique executed tools, request joins, outcomes and intervals."""
        self._check_records("C5", self.tools)
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
                "C5",
                "tool.unique",
                "/ng_trajectory/tool_calls",
                "duplicate execution identity",
            )
        for index, tool in enumerate(self.tools):
            location = f"/tool_calls/{index}"
            tool_key = (tool.get("invocation_id"), tool.get("tool_call_id"))
            request = conversations.get((*tool_key, "function_call"), {})
            output = conversations.get((*tool_key, "function_call_output"), {}).get("output", tool.get("output"))
            if not request.get("arguments") or request.get("name") != tool.get("tool_name"):
                self._fail(
                    "C5",
                    "tool.request",
                    location,
                    "execution cannot be joined to tool name and arguments",
                )
            if output is None and not tool.get("error_type"):
                self._fail(
                    "C5",
                    "tool.outcome",
                    location,
                    "execution has neither output nor error evidence",
                )
            if (
                _number(tool.get("started_at"))
                and _number(tool.get("completed_at"))
                and tool["completed_at"] < tool["started_at"]
            ):
                self._fail(
                    "C5",
                    "tool.interval",
                    location,
                    "completion precedes start",
                )

    def check_parallel_timing(self) -> None:
        """C6: independent timing witnesses remain unavailable."""
        self._fail(
            "C6",
            "timing.witness",
            "/tool_calls",
            "overlap needs independent monotonic-clock tool witnesses",
        )

    def check_lifecycle(self) -> None:
        """C7: full resource lifecycle evidence remains unavailable."""
        self._fail(
            "C7",
            "sandbox.lifecycle",
            "/ng_agent_observations",
            "session identity alone does not provide setup, verification and cleanup evidence",
        )

    def check_verifier(self) -> None:
        """C8: authoritative verifier provenance remains unavailable."""
        self._fail(
            "C8",
            "reward.provenance",
            "/reward",
            "reward alone does not provide authoritative verifier scope and version",
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
                for capability in ("C1", "C4", "C9", "C10", "C11"):
                    self._fail(
                        capability,
                        "capture.gap",
                        "/gaps",
                        "producer declares missing or unreadable capture evidence",
                    )
            if code.startswith("model_call_reference") or code == "model_call_ownership_unavailable":
                self._fail(
                    "C10",
                    "ownership.gap",
                    "/gaps",
                    "producer declares unresolved ownership",
                )

    def result(self) -> dict:
        """Produce payload-free verdicts and propagate C0 integrity failures."""
        failed = {f.capability for f in self.findings}
        if "C0" in failed:
            for capability in NAMES:
                if capability != "C0":
                    self._fail(
                        capability,
                        "record.integrity",
                        "",
                        "record identity or reader integrity failed",
                    )
            failed = set(NAMES)
        return {
            "source": self.source,
            "capabilities": {
                capability: {
                    "name": name,
                    "verdict": ("not_fulfilled" if capability in failed else "fulfilled"),
                    "basis": "retained_artifacts",
                    "validator_status": ("incomplete" if capability in PENDING_VALIDATORS else "artifact_subset"),
                    "behavioral_qualification": False,
                }
                for capability, name in NAMES.items()
            },
            "findings": [asdict(f) for f in self.findings],
        }


def inspect_record(record: dict, *, source: str = "record") -> dict:
    """Check artifact availability/correctness; never qualify unseen scenarios.

    Full per-call payloads may be hydrated by the reader. Missing artifacts fail
    their capabilities rather than being silently excluded from the denominator.
    Reports contain field locations and codes, never prompts or response bodies.
    """
    checks = _RecordInspector(record, source)
    checks.check_identity()
    checks.check_capture_presence()
    checks.check_model_calls()
    checks.check_token_counts()
    checks.check_content_references()
    checks.check_history()
    checks.check_payloads()
    checks.check_ownership()
    turn_refs = checks.check_turns()
    checks.check_accounting(turn_refs)
    checks.check_tools()
    checks.check_parallel_timing()
    checks.check_lifecycle()
    checks.check_verifier()
    checks.check_gaps()
    return checks.result()
