# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental append/reset storage for captured model-call prompts.

This operates on a completed rollout, outside model adapters. It is deliberately
not enabled in the collector: readers must call decode_rollout before consuming
this format. The outer format marker prevents a stored row from looking like an
ordinary rollout. Each row is independent; references never cross JSONL lines.
References use call-list positions: decode before filtering or reordering calls.

Proposed collector write boundary, after trajectory/token finalization:
    serialized = orjson.dumps(encode_rollout(result))

Matching read boundary, before validation/health/export consumers:
    result = decode_rollout(orjson.loads(line))

Those call sites are examples, not enabled integrations. Resume, aggregation,
health and export readers would all need the decoder before adoption. Capture
files and model adapters do not change. This encodes message arrays, not token-ID
arrays, and makes no inference about turn ownership or compaction semantics.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


FORMAT = "gym-model-call-prompts-v1"
PROMPT_FIELDS = ("messages", "input")


def _json(value: Any) -> str:
    # Serialized comparison distinguishes true/1, 1/1.0 and -0.0/0.0. Requiring
    # identical object key order is conservative: it may miss a saving, never data.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _calls(row: dict[str, Any]) -> list[Any]:
    trajectory = row.get("ng_trajectory")
    calls = trajectory.get("model_calls") if isinstance(trajectory, dict) else None
    return calls if isinstance(calls, list) else []


def _request(call: Any) -> dict[str, Any]:
    request = call.get("request") if isinstance(call, dict) else None
    return request if isinstance(request, dict) else {}


def encode_rollout(row: dict[str, Any]) -> dict[str, Any]:
    """Return a separate storage document; do not mutate the original JSON row.

    Only request.messages and request.input are encoded. Every present prompt field
    is wrapped, including strings/nulls, so literal data cannot collide with an
    append marker. All other fields, including counts and responses, are unchanged.
    The base is the immediately preceding call in stored order, not a causal claim.
    """
    if not isinstance(row, dict):
        raise ValueError("expected a rollout object")
    if row.get("format") == FORMAT:
        raise ValueError("decode the storage document before encoding it again")
    stored = deepcopy(row)
    calls = _calls(row)
    for index, call in enumerate(_calls(stored)):
        request = _request(call)
        previous = _request(calls[index - 1]) if index else {}
        for field in PROMPT_FIELDS:
            if field not in request:
                continue
            prompt = request[field]
            representation = {"full": prompt}  # Reset, also used for the first call.
            base = previous.get(field)
            if isinstance(prompt, list) and isinstance(base, list) and len(prompt) >= len(base):
                if all(_json(old) == _json(new) for old, new in zip(base, prompt)):
                    append = {"append_to": index - 1, "items": prompt[len(base) :]}
                    if len(_json(append).encode("utf-8")) < len(_json(representation).encode("utf-8")):
                        representation = append
            request[field] = representation
    return {"format": FORMAT, "rollout": stored}


def decode_rollout(document: dict[str, Any]) -> dict[str, Any]:
    """Expand a storage document into an ordinary rollout without mutating it.

    Reject malformed/unknown formats rather than passing a partial prompt to a
    consumer. This explicitly decodes encoded documents; plain rows bypass it.
    """
    if not isinstance(document, dict) or document.keys() != {"format", "rollout"}:
        raise ValueError("expected a prompt-storage document with format and rollout")
    if document["format"] != FORMAT:
        raise ValueError(f"unsupported prompt-storage format: {document['format']!r}")
    if not isinstance(document["rollout"], dict):
        raise ValueError("rollout must be an object")
    row = deepcopy(document["rollout"])
    calls = _calls(row)
    for index, call in enumerate(calls):
        request = _request(call)
        for field in PROMPT_FIELDS:
            if field not in request:
                continue
            encoded = request[field]
            path = f"ng_trajectory.model_calls[{index}].request.{field}"
            if not isinstance(encoded, dict):
                raise ValueError(f"{path}: expected full or append representation")
            if encoded.keys() == {"full"}:
                request[field] = encoded["full"]
                continue
            if encoded.keys() != {"append_to", "items"}:
                raise ValueError(f"{path}: expected full or append representation")
            base_index = encoded["append_to"]
            if type(base_index) is not int or not 0 <= base_index < index:
                raise ValueError(f"{path}: append_to must name an earlier call in this row")
            base = _request(calls[base_index]).get(field)
            if not isinstance(base, list) or not isinstance(encoded["items"], list):
                raise ValueError(f"{path}: base prompt and appended items must be arrays")
            request[field] = deepcopy(base) + encoded["items"]
    return row
