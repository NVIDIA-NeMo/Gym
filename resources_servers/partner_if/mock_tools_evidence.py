# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A mock-tool episode as evidence: the actor's visible trajectory, joined with the backend's call receipts.

mock_tools_app.py records a receipt for every call a backend ran and hands the receipts here at verify; the
shared grader scores the result. Importable without starting the server.
"""

import json

from pydantic import JsonValue

from nemo_gym.core import NeMoGymResponse
from resources_servers.partner_if.app import is_complete, output_cap_cutoff
from resources_servers.partner_if.evidence import Artifact, EvidenceBundle
from resources_servers.partner_if.grading_contract import StrictModel
from resources_servers.partner_if.native_evidence import visible_events


class RecordedToolCall(StrictModel):
    name: str
    arguments: dict[str, JsonValue]
    output: JsonValue


def _same_output(recorded: JsonValue, text: str) -> bool:
    """Plain-text tools return their text verbatim; JSON tools are compared as parsed values."""
    if isinstance(recorded, str):
        return recorded == text
    try:
        return json.loads(text) == recorded
    except ValueError:
        return False


def to_evidence(
    initial_params: dict[str, JsonValue],
    response: NeMoGymResponse,
    calls: list[RecordedToolCall],
    *,
    artifacts: dict[str, str] | None = None,
) -> EvidenceBundle:
    """Join captured calls/results with authoritative receipts, retaining only visible events.

    Function call IDs come from the trusted harness, not the HTTP tool arguments. Results
    match by call ID and receipts by name/arguments/output, which permits parallel result
    ordering without guessing. Calls the harness rejects before any tool runs (unadvertised
    names, non-object arguments) are actor errors without receipts, not capture gaps.

    A response cut off by the output-token cap ends the episode like the step cap: its calls
    never ran, so the completed tool exchanges before it are the evidence. Receipts still have
    to match, so a cutoff cannot hide an executed call. Whether the episode lacks a required
    final answer is for the caller to decide.
    """
    bundle = EvidenceBundle(
        context=initial_params["input"],
        instructions=initial_params.get("instructions"),
        tools=initial_params.get("tools") or [],
        artifacts={name: Artifact(status="present", text=text) for name, text in (artifacts or {}).items()},
    )
    advertised = {tool["name"] for tool in bundle.tools}
    try:
        items = [item.model_dump(mode="json") for item in response.output]
        if output_cap_cutoff(response):
            last_output = max(
                (i for i, item in enumerate(items) if item["type"] == "function_call_output"), default=-1
            )
            items = items[: last_output + 1]
        elif not is_complete(response):
            raise ValueError("Actor trajectory is incomplete")
        bundle.trajectory = visible_events(items)
        pending, seen, remaining, last_result = {}, set(), list(calls), -1
        for event in bundle.trajectory:
            if event.kind == "tool_call":
                if not event.call_id or event.call_id in seen:
                    raise ValueError("Missing or duplicate tool call ID")
                pending[event.call_id] = event
                seen.add(event.call_id)
            elif event.kind == "tool_result":
                call = pending.pop(event.call_id, None)
                if call is None:
                    raise ValueError("Tool result has no preceding unmatched call")
                last_result = event.source_index
                match = next(
                    (
                        i
                        for i, receipt in enumerate(remaining)
                        if receipt.name == call.tool_name
                        and receipt.arguments == call.arguments
                        and _same_output(receipt.output, event.text)
                    ),
                    None,
                )
                if match is not None:
                    remaining.pop(match)
                    continue
                try:
                    local = json.loads(event.text)
                except ValueError:
                    local = None
                rejected_locally = call.tool_name not in advertised or not isinstance(call.arguments, dict)
                if (
                    not rejected_locally
                    or not isinstance(local, dict)
                    or not (local.get("error") or local.get("detail"))
                ):
                    raise ValueError("Tool output lacks a matching authoritative receipt")
        if pending or remaining:
            raise ValueError("Tool calls, outputs and receipts do not form a complete trajectory")
        bundle.answer = "\n".join(
            event.text for event in bundle.trajectory if event.kind == "assistant" and event.source_index > last_result
        )
    except (ValueError, TypeError) as error:
        bundle.trajectory_error = str(error)
    return bundle
