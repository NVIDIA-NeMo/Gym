# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Represent explicit provider policy errors as typed refusal responses."""

import json
import time
from uuid import uuid4


def provider_refusal(status: int, body: object, allowed_codes: frozenset[str]) -> dict[str, str] | None:
    if status != 400 or not allowed_codes:
        return None
    pending = [body.decode(errors="replace") if isinstance(body, bytes) else body]
    for _ in range(32):
        if not pending:
            break
        value = pending.pop(0)
        if isinstance(value, dict):
            if (
                isinstance(value.get("code"), str)
                and value["code"] in allowed_codes
                and isinstance(value.get("message"), str)
            ):
                return {"code": value["code"], "message": value["message"]}
            if isinstance(value.get("error"), dict):
                pending.append(value["error"])
            if isinstance(value.get("message"), str):
                pending.append(value["message"])
        elif isinstance(value, str):
            # Decode enclosing JSON strings before looking inside their object text.
            starts = [index for char in ("{", '"') if (index := value.find(char)) >= 0]
            if not starts:
                continue
            fragment = value[min(starts) :]
            try:
                decoded, _ = json.JSONDecoder().raw_decode(fragment)
            except ValueError:
                # Some LiteLLM envelopes escape only quotes, not backslashes.
                # Undo that one layer without touching provider-value escapes.
                if not fragment.startswith("{"):
                    continue
                try:
                    decoded, _ = json.JSONDecoder().raw_decode(fragment.replace('\\"', '"'))
                except ValueError:
                    continue
            pending.append(decoded)
    return None


def refusal_response(model: str, refusal: dict, raw_body) -> dict:
    if isinstance(raw_body, bytes):
        raw_body = raw_body.decode(errors="replace")
    raw = raw_body if isinstance(raw_body, str) else json.dumps(raw_body, sort_keys=True)
    evidence = {
        "http_status": 400,
        "error_code": refusal["code"],
        "message": refusal["message"],
        "raw_error_body": raw,
    }
    return {
        "id": "resp_provider_refusal_" + uuid4().hex,
        "created_at": time.time(),
        "model": model,
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "msg_" + uuid4().hex,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": refusal["message"]}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        # The provider did not report usage; do not invent zero billable tokens.
        "usage": None,
        "metrics": {"provider_http_refusal": evidence},
    }
