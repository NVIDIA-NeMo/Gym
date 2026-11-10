# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify OpenCode's context-management records without rewriting saved exports."""

import hashlib
import json
from collections.abc import Sequence

from nemo_gym.core import NeMoGymResponseOutputItem
from resources_servers.agentic_if.episode.opencode_export import export_output


def generated_content(output: Sequence[NeMoGymResponseOutputItem]) -> list[dict]:
    """Compare public output and reasoning, not randomly generated document IDs."""
    result = []
    for item in output:
        kind = getattr(item, "type", None)
        if kind == "message" and item.role == "assistant":
            result.append(
                {
                    "type": kind,
                    "content": [
                        {"type": part.type, "text": part.text if part.type == "output_text" else part.refusal}
                        for part in item.content
                    ],
                }
            )
        elif kind == "reasoning":
            result.append({"type": kind, "summary": [part.text for part in item.summary]})
        elif kind == "function_call":
            result.append({"type": kind, "call_id": item.call_id, "name": item.name, "arguments": item.arguments})
    return result


def output_digest(output: Sequence[NeMoGymResponseOutputItem]) -> str:
    """SHA-256 of the generated content of ``output``, independent of document IDs."""
    return hashlib.sha256(json.dumps(generated_content(output), sort_keys=True).encode()).hexdigest()


def _followup(
    messages: list[dict], index: int, parent_index: int, *, controlled_ids: frozenset[str]
) -> tuple[dict, bool]:
    """Recognize stock 1.17.11 followups, never matching instruction prose.

    Overflow copies have no OpenCode provenance marker. Only an exclusively owned
    invocation can attest that no competing user submission occurred; otherwise
    preserve the message and expose an evidence gap. Part times/metadata are
    copied by stock OpenCode and must match, not be normalized away.
    """
    parent = messages[parent_index]
    markers = [part for part in parent["parts"] if part.get("type") == "compaction"]
    if len(markers) != 1 or markers[0].get("auto") is not True or len(parent["parts"]) != 1:
        return {}, False
    candidate = messages[index + 1] if index + 1 < len(messages) else {}
    if not isinstance(candidate, dict):
        candidate = {}
    info, parts = candidate.get("info", {}), candidate.get("parts", [])
    summary = messages[index]["info"]
    candidate_ok = (
        info.get("role") == "user"
        and isinstance(info.get("id"), str)
        and info.get("sessionID") == summary.get("sessionID")
        and isinstance(parts, list)
        and bool(parts)
        and all(isinstance(part, dict) for part in parts)
    )
    if candidate_ok and all(
        part.get("type") == "text"
        and part.get("synthetic") is True
        and isinstance(part.get("metadata"), dict)
        and part["metadata"].get("compaction_continue") is True
        for part in parts
    ):
        return {"followup_id": info["id"]}, False
    if markers[0].get("overflow") is not True:
        return {}, False
    originals = [
        message
        for message in messages[:parent_index]
        if isinstance(message, dict)
        and message.get("info", {}).get("role") == "user"
        and not any(part.get("type") == "compaction" for part in message.get("parts", []))
    ]
    if (
        summary.get("id") not in controlled_ids
        or info.get("id") not in controlled_ids
        or not candidate_ok
        or len(originals) < 2
    ):
        return {}, True
    original = originals[-1]
    copied_fields = ("agent", "model", "format", "tools", "system")
    allowed_fields = {*copied_fields, "id", "role", "sessionID", "time"}
    if set(info) - allowed_fields or any(info.get(key) != original["info"].get(key) for key in copied_fields):
        return {}, True
    # A second adjacent user means ownership/order is ambiguous, not permission
    # to discard whichever matching candidate was encountered first.
    if (
        index + 2 < len(messages)
        and isinstance(messages[index + 2], dict)
        and messages[index + 2].get("info", {}).get("role") == "user"
    ):
        return {}, True
    ignored = {"id", "messageID", "sessionID"}
    expected_parts = [{k: v for k, v in part.items() if k not in ignored} for part in original["parts"]]
    actual_parts = [{k: v for k, v in part.items() if k not in ignored} for part in parts]
    # Media replay is transformed by OpenCode's code. Do not guess unsupported
    # attachment semantics from display text; leave it visible and incomplete.
    if any(part.get("type") == "file" for part in expected_parts) or actual_parts != expected_parts:
        return {}, True
    return {"followup_id": info["id"], "replayed_user_id": original["info"]["id"]}, False


def compaction_context(
    document: dict, *, controlled_ids: frozenset[str] = frozenset()
) -> tuple[set[str], list[dict], list[str]]:
    """Recognize only completed OpenCode compaction summaries with retained parent/history.

    The content fingerprint binds each live summary to a captured inference
    receipt. Frozen replay summaries instead bind to their immutable prefix.
    Neither becomes task-solving prose or a new user instruction for grading.
    """
    session = document.get("info", {}).get("id")
    messages = document.get("messages", [])
    if not isinstance(messages, list):
        return set(), [], []
    by_id = {m.get("info", {}).get("id"): (index, m) for index, m in enumerate(messages) if isinstance(m, dict)}
    internal, records, issues = set(), [], []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        info = message.get("info", {})
        if info.get("summary") is not True:
            continue
        identity, parent_id = info.get("id"), info.get("parentID")
        parent_index, parent = by_id.get(parent_id, (-1, {}))
        parts = message.get("parts", [])
        parent_parts = parent.get("parts", [])
        valid = (
            isinstance(identity, str)
            and bool(identity)
            and info.get("role") == "assistant"
            and info.get("agent") == "compaction"
            and info.get("finish") == "stop"
            and not info.get("error")
            and 0 < parent_index < index
            and parent.get("info", {}).get("role") == "user"
            and any(p.get("type") == "compaction" for p in parent_parts if isinstance(p, dict))
            and isinstance(parts, list)
            and all(isinstance(p, dict) and p.get("type") != "tool" for p in parts)
            and any(p.get("type") == "text" and p.get("text") for p in parts)
        )
        if not valid:
            issues.append(f"invalid_opencode_compaction:{session}:{identity}")
            continue
        try:
            digest = output_digest(export_output({"messages": [message]}))
        except (KeyError, TypeError, ValueError, NotImplementedError):
            issues.append(f"invalid_opencode_compaction:{session}:{identity}")
            continue
        internal.add(identity)
        if all(p.get("type") == "compaction" for p in parent_parts):
            internal.add(parent_id)
        followup, ambiguous = _followup(messages, index, parent_index, controlled_ids=controlled_ids)
        if followup:
            internal.add(followup["followup_id"])
        if ambiguous:
            issues.append(f"ambiguous_opencode_compaction_followup:{session}:{identity}")
        records.append(
            {
                "session_id": session,
                "message_id": identity,
                "parent_id": parent_id,
                "output_sha256": digest,
                **followup,
            }
        )
    return internal, records, issues


def actor_export(document: dict, *, controlled_ids: frozenset[str] = frozenset()) -> dict:
    """Project task-solving messages; raw exports remain untouched on disk."""
    internal, _, _ = compaction_context(document, controlled_ids=controlled_ids)
    return document | {"messages": [m for m in document["messages"] if m["info"]["id"] not in internal]}
