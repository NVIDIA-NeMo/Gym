# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify immutable replay prefixes and request another generic agent activation."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.partner_if.native_evidence import LiveAgentCapture
from resources_servers.partner_if.receipt import masked_fields
from responses_api_agents.ngc_recorded_context.prefix import ordering_outcome
from responses_api_agents.ngc_recorded_context.runtime_contract import OrderingContract
from responses_api_agents.ngc_recorded_context.saved_prefix import prefix_digest, verification_digest


if TYPE_CHECKING:
    from resources_servers.partner_if.app import VerifyRequest, VerifyResponse


_CONTROLS = (
    "ngc_replay_parent_turn_limit",
    "ngc_replay_saved_prefix",
    "ngc_replay_saved_prefix_sha256",
)
_SAFE_TERMINALS = frozenset({"turn_limit", "policy_call_limit", "first_answer_draft", "final_response"})
_REPLAY_FIELDS = frozenset({"replay_ordering_contract", "replay_artifacts_dir", "replay_diagnostics"})


def is_ordering_response(response: NeMoGymResponse) -> bool:
    """Only replay envelopes opt into ordering; ordinary live-agent captures retain their behavior."""
    return bool(_REPLAY_FIELDS & (response.model_extra or {}).keys())


@dataclass(frozen=True)
class OrderingPrefix:
    directory: Path
    capture: LiveAgentCapture
    events: list[dict]
    diagnostics: dict
    ordering: OrderingContract
    parent_turn_limit: int
    saved_prefix_sha256: str | None

    @classmethod
    def read(cls, body: "VerifyRequest", artifacts_dir: Path) -> "OrderingPrefix":
        """Bind body-carried evidence to hash-checked receipts beneath an explicitly configured root."""
        extra = body.response.model_extra or {}
        root = artifacts_dir.expanduser().resolve()
        raw_directory = extra.get("replay_artifacts_dir")
        if not isinstance(raw_directory, str):
            raise ValueError("Replay response omits its artifact directory")
        directory = Path(raw_directory).resolve()
        if directory == root or not directory.is_relative_to(root):
            raise ValueError("Replay artifacts must be below ordering_artifacts_dir")

        def path(name: str) -> Path:
            value = (directory / name).resolve()
            if not value.is_relative_to(directory) or not value.is_relative_to(root) or not value.is_file():
                raise ValueError("Replay receipt is missing or escapes its artifact root: " + name)
            return value

        def read(name: str):
            return json.loads(path(name).read_text())

        # Check each fixed path before the digest helper opens it, including symlink targets.
        for name in (
            "identity.json",
            "events.json",
            "drafts.json",
            "terminal_reply.json",
            "result.json",
            "visible-output.json",
            "ordering-verification-input.json",
            "native-step.json",
        ):
            path(name)
        expected = extra.get("replay_verification_sha256")
        if not isinstance(expected, str) or verification_digest(directory) != expected:
            raise ValueError("Replay verification receipt checksum mismatch")

        diagnostics, step = read("result.json"), read("native-step.json")
        if not isinstance(diagnostics, dict) or diagnostics != extra.get("replay_diagnostics"):
            raise ValueError("Replay diagnostics differ from their immutable receipt")
        calls = diagnostics.get("model_calls")
        if not isinstance(calls, int) or isinstance(calls, bool) or calls < 0:
            raise ValueError("Replay policy-call count must be a nonnegative integer")
        limit = step.get("parent_turn_limit") if isinstance(step, dict) else None
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 32:
            raise ValueError("Replay parent-turn limit is invalid")
        identity = read("identity.json")
        if (
            not isinstance(identity, dict)
            or identity.get("model") != body.response.model
            or not isinstance(identity.get("settings"), dict)
            or identity["settings"].get("max_new_turns") != limit
        ):
            raise ValueError("Replay model or turn identity differs from its immutable receipt")
        ordering = OrderingContract.model_validate(step.get("ordering_contract"))
        if ordering.model_dump() != extra.get("replay_ordering_contract"):
            raise ValueError("Replay ordering declaration differs from its immutable receipt")
        contract = body.verifier_metadata.partner_if
        if len(contract.rubrics) != 1 or contract.prerequisites:
            raise ValueError("Ordering evaluation requires one rubric and no scored prerequisites")
        if (
            contract.task_id != ordering.task_id
            or getattr(body, "task_name", None) not in ordering.task_names
            or contract.rubric_version != ordering.rubric_version
            or contract.rubrics[0].id != ordering.criterion_id
        ):
            raise ValueError("Task and rubric identity differ from the frozen replay contract")

        captured = LiveAgentCapture.model_validate(extra.get("verification_input"))
        if body.verification_input is not None and body.verification_input != captured:
            raise ValueError("Top-level and replay capture evidence disagree")
        if LiveAgentCapture.model_validate(read("ordering-verification-input.json")) != captured:
            raise ValueError("Replay capture differs from its immutable receipt")
        params = body.responses_create_params.model_dump(exclude_none=True)
        if captured.input != params["input"]:
            raise ValueError("The actor's input differs from the frozen row")
        raw_output = read("visible-output.json")
        expected_response = NeMoGymResponse.model_validate(body.response.model_dump() | {"output": raw_output})
        if [v.model_dump(exclude_none=True) for v in expected_response.output] != [
            v.model_dump(exclude_none=True) for v in body.response.output
        ]:
            raise ValueError("Replay output differs from its immutable receipt")
        events = read("events.json")
        receipt = captured.artifacts.get("replay_receipts", {})
        if receipt.get("exists") is not True or not isinstance(receipt.get("text"), str):
            raise ValueError("Replay capture omits its ordering receipts")
        receipts = json.loads(receipt["text"])
        if (
            not isinstance(receipts, dict)
            or not isinstance(events, list)
            or any(not isinstance(event, dict) for event in events)
            or receipts.get("new_events") != events
            or receipts.get("evaluation_boundary") != "ordering_prefix_v1"
            or receipts.get("supplied_history_is_context") is not True
            or receipts.get("same_response_results_are_not_prior_evidence") is not True
        ):
            raise ValueError("Replay event receipts are inconsistent")

        saved_sha = extra.get("saved_prefix_sha256")
        if diagnostics.get("termination") in _SAFE_TERMINALS:
            # api-order names every parent/delegate request. Validate all targets before prefix_digest reads them.
            order_path = path("api-order.jsonl")
            entries = [json.loads(line) for line in order_path.read_text().splitlines()]
            for entry in entries:
                request = entry.get("request") if isinstance(entry, dict) else None
                if not isinstance(request, str) or not request.endswith(".request.json"):
                    raise ValueError("Replay request receipt path is invalid")
                path(request)
                path(request.replace(".request.json", ".response.json"))
                path(request.replace(".request.json", ".receipt.json"))
            if len(entries) != calls or not isinstance(saved_sha, str) or prefix_digest(directory) != saved_sha:
                raise ValueError("Replay saved-prefix identity or call count is inconsistent")
        elif saved_sha is not None:
            raise ValueError("A technically incomplete replay cannot advertise a resumable prefix")
        return cls(directory, captured, events, diagnostics, ordering, limit, saved_sha)

    def next_request(self, params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponseCreateParamsNonStreaming:
        """Keep frozen actor parameters intact; only the agent adapter consumes these transport controls."""
        value = params.model_dump(exclude_none=True, exclude_unset=True)
        metadata = dict(value.pop("metadata", None) or {})
        for name in _CONTROLS:
            metadata.pop(name, None)
        metadata.update(
            ngc_replay_parent_turn_limit=str(self.parent_turn_limit + 1),
            ngc_replay_saved_prefix=str(self.directory),
            ngc_replay_saved_prefix_sha256=self.saved_prefix_sha256,
        )
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(value | {"metadata": metadata})


async def verify_ordering(
    body: "VerifyRequest",
    *,
    artifacts_dir: Path,
    max_policy_calls: int,
    verify: Callable[["VerifyRequest"], Awaitable["VerifyResponse"]],
) -> dict:
    """Use the unchanged shared judge, then apply the existing decisive-prefix boundary."""
    contract = body.verifier_metadata.partner_if
    prefix = None
    grade = {}

    def masked(terminal: str, reason: str, *, next_request=None) -> dict:
        diagnostics = dict(prefix.diagnostics) if prefix is not None else {}
        diagnostics.update(termination=terminal, evaluation_completed=False, evaluation_boundary="ordering_prefix_v1")
        directory = (body.response.model_extra or {}).get("replay_artifacts_dir")
        return (
            grade
            | masked_fields(contract, "partner_if:replay_" + terminal, reason)
            | {
                "evaluation_completed": False,
                "grading_result": None,
                "response": body.response.model_copy(update={"status": "incomplete"}).model_dump(mode="json"),
                "replay_diagnostics": diagnostics,
                "replay_artifacts_dir": directory if isinstance(directory, str) else None,
                "next_agent_request": next_request,
            }
        )

    try:
        prefix = OrderingPrefix.read(body, artifacts_dir)
    except (OSError, ValueError, TypeError) as error:
        return masked("invalid_capture", str(error))
    terminal = prefix.diagnostics.get("termination")
    if terminal not in _SAFE_TERMINALS:
        return masked(str(terminal or "invalid_capture"), "Technical replay exclusion; see immutable capture receipts")
    if prefix.diagnostics["model_calls"] > max_policy_calls:
        return masked("policy_call_limit", "Replay exceeded its configured policy-call ceiling")

    grade = (await verify(body.model_copy(update={"verification_input": prefix.capture}))).model_dump(mode="json")
    try:
        outcome = ordering_outcome(
            grade,
            prefix.events,
            probe_namespace=prefix.ordering.probe_namespace,
            expected_criterion_id=prefix.ordering.criterion_id,
        )
    except ValueError:
        return masked("ordering_judge_error", "The shared verifier did not return a usable ordering verdict")
    if outcome is not None:
        diagnostics = prefix.diagnostics | {
            "termination": outcome,
            "evaluation_completed": True,
            "evaluation_boundary": "ordering_prefix_v1",
        }
        return grade | {
            "evaluation_completed": True,
            "replay_diagnostics": diagnostics,
            "replay_artifacts_dir": str(prefix.directory),
            "next_agent_request": None,
        }
    if terminal in {"first_answer_draft", "final_response"}:
        return masked("ordering_undecided", "No decisive ordering event before the captured final response")
    if terminal == "policy_call_limit" or prefix.diagnostics["model_calls"] >= max_policy_calls:
        return masked("policy_call_limit", "No decisive ordering event before the policy-call ceiling")
    if prefix.parent_turn_limit >= 32:
        return masked("turn_limit", "No decisive ordering event before the replay parent-turn ceiling")
    return masked(
        "ordering_pending",
        "The captured prefix is safe but has no decisive ordering event",
        next_request=prefix.next_request(body.responses_create_params).model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        ),
    )
