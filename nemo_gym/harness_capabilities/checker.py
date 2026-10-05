# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused evidence checks. TE labels are reporting metadata, not dispatch keys."""

import base64
import binascii
from collections import Counter
from dataclasses import dataclass

from jsonschema import Draft202012Validator

from . import schemas as s
from .results import Results


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
PROFILE = "gym-p0/v3"
P0 = tuple(f"TE-{i}" for i in range(1, 8))
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens")


@dataclass(frozen=True)
class EvidenceScope:
    """Declared applicability; absent artifacts never imply an exemption."""

    tools: bool = True
    verifier: bool = True
    steps: bool = True
    require_sandbox: bool = False


def gate_passes(evidence: dict) -> bool:
    return all(evidence[k]["verdict"] in {"fulfilled", "not_applicable"} for k in P0) and any(
        evidence[k]["verdict"] == "fulfilled" for k in ("TE-8", "TE-9")
    )


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


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
    if not Draft202012Validator(s.MODEL_CALL_REF).is_valid(reference):
        return []
    return [
        i
        for i, call in enumerate(calls)
        if all(
            (call.get(key) if key == "model_call_id" else _mapping(call.get("response_metadata")).get(key)) == value
            for key, value in reference.items()
            if key in {"model_call_id", "model_ref", "response_id"} and value is not None
        )
    ]


class Inspector:
    """Run independent predicates with focused prerequisites and explicit outcomes."""

    def __init__(self, record: dict | None, scope: EvidenceScope) -> None:
        self.available = record is not None
        self.record = record or {}
        self.scope = scope
        self.trajectory = _mapping(self.record.get("ng_trajectory"))
        self.results = Results()
        self.calls = _objects(self.trajectory.get("model_calls"))
        self.invocations = _objects(self.trajectory.get("invocations"))
        self.turns = _objects(self.trajectory.get("turns"))
        self.tools = _objects(self.trajectory.get("tool_calls"))
        self.observations = _objects(_mapping(self.record.get("ng_agent_observations")).get("records"))

    def schema(
        self,
        key: str,
        value: object,
        schema: dict,
        path: str,
        evidence: tuple[str, ...],
        *,
        depends_on: tuple[str, ...] = (),
        applies: bool = True,
    ) -> None:
        # Error messages can contain payloads. Retain only rule names and JSON paths.
        validator = Draft202012Validator(schema)
        errors = list(validator.iter_errors(value))
        self.results.check(
            key,
            "schema",
            not errors,
            evidence=evidence,
            location=path,
            reason="required evidence does not match its schema",
            depends_on=depends_on,
            available=self.available,
            applies=applies,
        )
        row = self.results.rows[key]
        if errors and row.status == "fail":
            for error in errors:
                location = path + "".join(
                    f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path
                )
                if location not in row.locations:
                    row.locations.append(location)

    def semantic(
        self,
        key: str,
        condition: bool,
        path: str,
        evidence: tuple[str, ...],
        *,
        depends_on: tuple[str, ...] = (),
        applies: bool = True,
        reason: str,
    ) -> None:
        self.results.check(
            key,
            "semantic",
            condition,
            location=path,
            evidence=evidence,
            reason=reason,
            available=self.available,
            depends_on=depends_on,
            applies=applies,
        )

    def collection(self, name: str, evidence: tuple[str, ...], *, applies: bool = True) -> None:
        self.schema(
            name + ".present",
            self.trajectory.get(name),
            {**s.NONEMPTY, "items": {"type": "object"}},
            "$.ng_trajectory." + name,
            evidence,
            applies=applies,
        )

    def fields(self, collection: str, rules: tuple, *, applies: bool = True) -> None:
        # Each field rule evaluates the collection once; nested errors retain item paths.
        for key, schema, evidence in rules:
            self.schema(
                key,
                self.trajectory.get(collection),
                {"type": "array", "items": schema},
                "$.ng_trajectory." + collection,
                evidence,
                depends_on=(collection + ".present",),
                applies=applies,
            )

    def model_calls(self) -> None:
        self.collection("model_calls", ("TE-1", "TE-2", "TE-4", "TE-7", "TE-8", "TE-9"))
        self.fields(
            "model_calls",
            (
                ("calls.identity", s.required_object(model_call_id=s.NONBLANK), ("TE-1",)),
                (
                    "calls.model",
                    s.required_object(response_metadata=s.required_object(model_ref=s.MODEL_REF)),
                    ("TE-1",),
                ),
                (
                    "calls.protocol",
                    s.required_object(response_metadata=s.required_object(dialect=s.DIALECT)),
                    ("TE-1",),
                ),
                ("calls.timing", s.required_object(started_at=s.TIMESTAMP, completed_at=s.TIMESTAMP), ("TE-1",)),
                ("calls.outcome", s.required_object(response_metadata=s.OUTCOME), ("TE-1",)),
                ("calls.response_id", s.required_object(response_metadata=s.RESPONSE_ID), ("TE-1",)),
                ("calls.request", s.REQUEST, ("TE-4", "TE-7")),
                ("calls.response", s.RESPONSE, ("TE-4", "TE-7")),
            ),
        )
        for field in TOKEN_FIELDS:
            # Optional metrics retain missing/null as unavailable; no normalization recheck.
            schema = s.required_object(
                token_stats={"type": "object", "properties": {field: {"type": ["integer", "null"], "minimum": 0}}}
            )
            self.fields("model_calls", (("tokens." + field, schema, ("TE-2",)),))
        self.semantic(
            "content.media",
            not any(_missing_content(c.get(k)) for c in self.calls for k in ("request", "response")),
            "$.ng_trajectory.model_calls",
            ("TE-4", "TE-7"),
            depends_on=("model_calls.present",),
            reason="external, encrypted or invalid media is unavailable to this reader",
        )
        valid = True
        for index, call in enumerate(self.calls):
            previous = _mapping(call.get("request")).get("previous_response_id")
            if previous:
                matches = [
                    c
                    for c in self.calls[:index]
                    if _mapping(c.get("response_metadata")).get("response_id") == previous
                ]
                valid &= (
                    len(matches) == 1
                    and matches[0].get("request") is not None
                    and matches[0].get("response") is not None
                )
        self.semantic(
            "content.previous_response",
            valid,
            "$.ng_trajectory.model_calls",
            ("TE-4",),
            depends_on=("model_calls.present", "calls.request"),
            reason="previous response has no unique retained history",
        )

    def structure(self) -> None:
        for field in ("task_id", "rollout_id"):
            self.schema(
                "trajectory." + field,
                self.trajectory,
                s.required_object(**{field: s.NONBLANK}),
                "$.ng_trajectory",
                ("TE-3", "TE-8"),
            )
        self.collection("invocations", ("TE-8",))
        self.fields(
            "invocations",
            (
                ("invocations.identity", s.required_object(invocation_id=s.NONBLANK), ("TE-8",)),
                (
                    "invocations.references",
                    s.required_object(model_calls={"type": "array", "items": s.MODEL_CALL_REF}),
                    ("TE-8",),
                ),
            ),
        )
        self.collection("turns", ("TE-3", "TE-9"), applies=self.scope.steps)
        self.fields(
            "turns",
            (
                ("steps.invocation", s.required_object(invocation_id=s.NONBLANK), ("TE-3", "TE-9")),
                ("steps.number", s.required_object(turn_no={"type": "integer", "minimum": 1}), ("TE-3", "TE-9")),
                ("steps.timestamp", s.required_object(timestamp=s.TIMESTAMP), ("TE-3",)),
                ("steps.resolution", s.required_object(resolved={"type": ["boolean", "null"]}), ("TE-3",)),
                ("steps.references", s.TURN_CALLS, ("TE-9",)),
            ),
            applies=self.scope.steps,
        )
        self.collection("tool_calls", ("TE-5",), applies=self.scope.tools)
        self.fields(
            "tool_calls",
            (
                ("tools.identity", s.required_object(tool_call_id=s.NONBLANK), ("TE-5",)),
                ("tools.name", s.required_object(tool_name=s.NONBLANK), ("TE-5",)),
                ("tools.invocation", s.required_object(invocation_id=s.NONBLANK), ("TE-5",)),
                (
                    "tools.status",
                    s.required_object(status={"enum": ["completed", "failed", "timeout", "cancelled"]}),
                    ("TE-5",),
                ),
                ("tools.output", s.TOOL_OUTPUT, ("TE-5",)),
            ),
            applies=self.scope.tools,
        )
        for collection, records, prefix, evidence, applies in (
            ("turns", self.turns, "steps", ("TE-3", "TE-9"), self.scope.steps),
            ("tool_calls", self.tools, "tools", ("TE-5",), self.scope.tools),
        ):
            self.semantic(
                prefix + ".invocation_target",
                all(
                    sum(i.get("invocation_id") == r.get("invocation_id") for i in self.invocations) == 1
                    for r in records
                ),
                "$.ng_trajectory." + collection,
                evidence,
                applies=applies,
                depends_on=(
                    collection + ".present",
                    prefix + ".invocation",
                    "invocations.present",
                    "invocations.identity",
                ),
                reason="invocation reference does not resolve to exactly one saved invocation",
            )
        # Arguments currently live only in the invocation conversation. Tool output has its own authority above.
        requests = [
            [
                item
                for inv in self.invocations
                if inv.get("invocation_id") == tool.get("invocation_id")
                for item in _objects(inv.get("conversation"))
                if item.get("type") == "function_call" and item.get("call_id") == tool.get("tool_call_id")
            ]
            for tool in self.tools
        ]
        self.semantic(
            "tools.request",
            all(
                len(items) == 1
                and isinstance(items[0].get("arguments"), str)
                and items[0].get("name") == tool.get("tool_name")
                for items, tool in zip(requests, self.tools)
            ),
            "$.ng_trajectory.invocations[*].conversation",
            ("TE-5",),
            applies=self.scope.tools,
            depends_on=("tool_calls.present", "tools.identity", "tools.name", "tools.invocation_target"),
            reason="tool execution lacks a unique saved request with name and arguments",
        )

    def ownership(self) -> None:
        parent_valid = True
        for invocation in self.invocations:
            current = invocation
            seen = {id(current)}
            while current.get("parent_invocation_id") is not None:
                matches = [i for i in self.invocations if i.get("invocation_id") == current["parent_invocation_id"]]
                if len(matches) != 1 or id(matches[0]) in seen:
                    parent_valid = False
                    break
                current = matches[0]
                seen.add(id(current))
        self.semantic(
            "invocations.parent",
            parent_valid,
            "$.ng_trajectory.invocations[*].parent_invocation_id",
            ("TE-8",),
            depends_on=("invocations.present", "invocations.identity"),
            reason="parent invocation is missing, ambiguous or cyclic",
        )
        owners: dict[int, list[str]] = {i: [] for i in range(len(self.calls))}
        valid = True
        for inv in self.invocations:
            for reference in _objects(inv.get("model_calls")):
                matches = _resolve(reference, self.calls)
                valid &= len(matches) == 1
                if len(matches) == 1:
                    owners[matches[0]].append(inv.get("invocation_id"))
        path = "$.ng_trajectory.invocations[*].model_calls"
        dependencies = (
            "model_calls.present",
            "calls.identity",
            "invocations.present",
            "invocations.references",
            "invocations.identity",
        )
        self.semantic(
            "ownership.call_target",
            valid,
            path,
            ("TE-8",),
            depends_on=dependencies,
            reason="call reference does not resolve uniquely with all supplied identifiers",
        )
        self.semantic(
            "ownership.call_owner",
            all(len(v) == 1 for v in owners.values()),
            path,
            ("TE-8",),
            depends_on=("ownership.call_target",),
            reason="each saved call must have exactly one invocation owner",
        )
        helpers = set()
        auxiliary_valid = True
        for observation in self.observations:
            if observation.get("kind") == "context_compaction":
                refs = observation.get("model_calls")
                auxiliary_valid &= Draft202012Validator({"type": "array", "items": s.MODEL_CALL_REF}).is_valid(refs)
                for reference in _objects(refs):
                    matches = _resolve(reference, self.calls)
                    auxiliary_valid &= len(matches) == 1
                    helpers.update(matches)
        self.semantic(
            "steps.compaction_target",
            auxiliary_valid,
            "$.ng_agent_observations.records",
            ("TE-9",),
            applies=self.scope.steps,
            depends_on=("model_calls.present",),
            reason="compaction helper reference is invalid or unresolved",
        )
        refs = Counter()
        turn_valid, owner_valid = True, True
        for turn in self.turns:
            for reference in _objects(turn.get("model_calls")):
                matches = _resolve(reference, self.calls)
                turn_valid &= len(matches) == 1
                if len(matches) == 1:
                    index = matches[0]
                    refs[index] += 1
                    owner_valid &= not owners[index] or owners[index] == [turn.get("invocation_id")]
        dependencies = (
            "model_calls.present",
            "calls.identity",
            "turns.present",
            "steps.references",
            "steps.invocation_target",
        )
        self.semantic(
            "steps.call_target",
            turn_valid,
            "$.ng_trajectory.turns[*].model_calls",
            ("TE-9",),
            applies=self.scope.steps,
            depends_on=dependencies,
            reason="step call reference does not resolve uniquely with all supplied identifiers",
        )
        self.semantic(
            "steps.call_owner",
            owner_valid,
            "$.ng_trajectory.turns[*].model_calls",
            ("TE-9",),
            applies=self.scope.steps,
            depends_on=("steps.call_target", "ownership.call_target"),
            reason="call ownership contradicts its step invocation",
        )
        policy = set(range(len(self.calls))) - helpers
        self.semantic(
            "steps.attempt_accounting",
            bool(policy) and all(refs[i] == 1 for i in policy) and not any(refs[i] for i in helpers),
            "$.ng_trajectory.turns[*].model_calls",
            ("TE-9",),
            applies=self.scope.steps,
            depends_on=("steps.call_target", "steps.compaction_target"),
            reason="every policy attempt needs one step; compaction helper calls must remain separate",
        )

    def evaluation(self) -> None:
        for field, schema in (
            ("reward", {"type": "number"}),
            ("evaluation_completed", {"type": "boolean"}),
            ("mask_sample", {"type": "boolean"}),
        ):
            self.schema(
                "evaluation." + field,
                self.record,
                s.required_object(**{field: schema}),
                "$",
                ("TE-6",),
                applies=self.scope.verifier,
            )
        failed = self.record.get("mask_sample") is True or self.record.get("evaluation_completed") is False
        for field in ("failure_kind", "failure_reason"):
            self.schema(
                "evaluation." + field,
                self.record,
                s.required_object(**{field: s.NONBLANK}),
                "$",
                ("TE-6",),
                applies=self.scope.verifier and failed,
            )
        sandbox = [r for r in self.observations if r.get("kind") == "sandbox"]
        self.schema(
            "sandbox.present",
            sandbox,
            s.NONEMPTY,
            "$.ng_agent_observations.records",
            ("TE-6",),
            applies=self.scope.require_sandbox,
        )
        for key, schema in (
            ("identity", s.required_object(sandbox_id=s.NONBLANK)),
            ("outcome", s.SANDBOX_OUTCOME),
            ("error", s.SANDBOX_ERROR),
        ):
            self.schema(
                "sandbox." + key,
                sandbox,
                {"type": "array", "items": schema},
                "$.ng_agent_observations.records",
                ("TE-6",),
                applies=bool(sandbox) or self.scope.require_sandbox,
                depends_on=("sandbox.present",) if self.scope.require_sandbox else (),
            )

    def gaps(self) -> None:
        # Canonical projection retains producer gaps. Duplicate attachment gaps are not compared.
        codes = [str(g.get("code", "")) for g in _objects(self.trajectory.get("gaps"))]
        for key, bad, evidence, applies in (
            (
                "calls.capture_gap",
                any(c.startswith("model_call_capture") or c == "agent_observation_join_failed" for c in codes),
                ("TE-1", "TE-4", "TE-7", "TE-8", "TE-9"),
                True,
            ),
            (
                "ownership.gap",
                any(c.startswith("model_call_reference") or c == "model_call_ownership_unavailable" for c in codes),
                ("TE-8",),
                True,
            ),
            (
                "steps.gap",
                any(
                    c in {"turns_unavailable", "turn_evidence_incomplete", "trajectory_projection_failed"}
                    for c in codes
                ),
                ("TE-3",),
                self.scope.steps,
            ),
            ("steps.accounting_gap", "turn_model_call_scope_incomplete" in codes, ("TE-9",), self.scope.steps),
        ):
            self.semantic(
                key,
                not bad,
                "$.ng_trajectory.gaps",
                evidence,
                applies=applies,
                depends_on=("model_calls.present",),
                reason="producer explicitly reports unavailable evidence",
            )

    def result(self, source: str) -> dict:
        checks = self.results.dump()
        evidence = {}
        for key, name in NAMES.items():
            statuses = {c["status"] for c in checks if key in c["evidence"]}
            if (
                (key in {"TE-3", "TE-9"} and not self.scope.steps)
                or (key == "TE-5" and not self.scope.tools)
                or (key == "TE-6" and not self.scope.verifier)
            ):
                statuses = {"not_applicable"}
            verdict = (
                "not_fulfilled"
                if "fail" in statuses
                else "not_assessed"
                if "not_assessed" in statuses
                else "fulfilled"
                if "pass" in statuses
                else "not_applicable"
            )
            evidence[key] = {"name": name, "verdict": verdict, "basis": "retained_artifacts"}
        return {
            "source": source,
            "checks": checks,
            "evidence": evidence,
            "verdict": "fulfilled" if gate_passes(evidence) else "not_fulfilled",
            "token_availability": {
                key: {
                    "available": sum(_mapping(c.get("token_stats")).get(key) is not None for c in self.calls),
                    "calls": len(self.calls),
                }
                for key in TOKEN_FIELDS
            },
            "is_behavioral_qualification": False,
            "scope_closure": "not_independently_witnessed",
            "findings": [
                {"evidence": te, "assertion": c["id"], "location": source + ":" + path, "reason": reason}
                for c in checks
                if c["status"] == "fail"
                for te in c["evidence"]
                for path in c["locations"]
                for reason in c["reasons"]
            ],
        }


def inspect_record(record: dict | None, *, source: str = "record", scope: EvidenceScope = EvidenceScope()) -> dict:
    """Validate designated fields; absent input produces unassessed checks."""
    inspector = Inspector(record, scope)
    inspector.model_calls()
    inspector.structure()
    inspector.ownership()
    inspector.evaluation()
    inspector.gaps()
    return inspector.result(source)
