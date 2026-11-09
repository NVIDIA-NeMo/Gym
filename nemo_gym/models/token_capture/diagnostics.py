# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional bounded structural evidence; never an input to capture decisions.

Set NEMO_GYM_CAPTURE_DIAGNOSTICS_DIR in the model-server environment before
startup. Each process appends JSONL beside (not inside) authoritative ledgers.
Only structural identifiers are retained; no content or token arrays are read.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nemo_gym.models.token_capture.lineage import LineageNode
    from nemo_gym.models.token_capture.sink import CaptureContext

logger = logging.getLogger(__name__)
MAX_ITEMS = 32
MAX_CANDIDATES = 32
MAX_STRING = 128
MAX_BYTES = 32768
_LOCK = threading.Lock()
_FIELDS = ("type", "id", "role", "status", "call_id", "name")


def _enabled() -> bool:
    return bool(os.environ.get("NEMO_GYM_CAPTURE_DIAGNOSTICS_DIR"))


def _scalar(value: object) -> object:
    return value[:MAX_STRING] if isinstance(value, str) else value if type(value) in (int, bool) else None


def _items(items: list[dict[str, object]]) -> dict[str, object]:
    result = []
    truncated = len(items) > MAX_ITEMS
    for item in items[:MAX_ITEMS]:
        if not isinstance(item, dict):
            result.append({})
            continue
        result.append({key: _scalar(item[key]) for key in _FIELDS if key in item})
        truncated |= any(isinstance(item.get(key), str) and len(item[key]) > MAX_STRING for key in _FIELDS)
    return {"items": result, "count": len(items), "truncated": truncated}


def remember_resolution(*, candidates: list[LineageNode], branch: str, fingerprint: str) -> None:
    """Retain only candidates inspected by the actual matcher, in its request context."""
    if not _enabled():
        return
    try:
        # Lazy import avoids making the matcher depend on sink initialization.
        from nemo_gym.models.token_capture.sink import current_capture_context

        context = current_capture_context()
        if context is None:
            return
        rows = []
        truncated = len(candidates) > MAX_CANDIDATES
        for node in candidates[:MAX_CANDIDATES]:
            row = {
                "model_call_id": node.call_id,
                "context_len": node.context_len,
                "context_digest": node.context_digest,
                "cum_len": node.cum_len,
                "digest": node.digest,
            }
            truncated |= any(isinstance(value, str) and len(value) > MAX_STRING for value in row.values())
            rows.append({key: _scalar(value) for key, value in row.items()})
        context.diagnostic_resolution = {
            "branch": branch,
            "fingerprint": _scalar(fingerprint),
            "candidates": rows,
            "candidate_count": len(candidates),
            "truncated": truncated,
        }
    except Exception:
        logger.warning("Capture structural resolution diagnostic failed")


def remember_response(context: CaptureContext, payload: dict[str, object], items: list[dict[str, object]]) -> None:
    """Keep bounded served-response metadata until the ledger commit succeeds."""
    if not _enabled():
        return
    try:
        details = payload.get("incomplete_details")
        fields = {
            "response_id": payload.get("id"),
            "status": payload.get("status"),
            "incomplete_reason": details.get("reason") if isinstance(details, dict) else None,
        }
        context.diagnostic_response = {
            **{key: _scalar(value) for key, value in fields.items()},
            "output": _items(items),
            "truncated": any(isinstance(v, str) and len(v) > MAX_STRING for v in fields.values()),
        }
    except Exception:
        logger.warning("Capture structural response diagnostic failed")


def emit(context: CaptureContext, *, event: str) -> None:
    """Append best-effort evidence after the authoritative commit or rejection."""
    directory = os.environ.get("NEMO_GYM_CAPTURE_DIAGNOSTICS_DIR")
    if not directory:
        return
    try:
        record = {
            "version": 1,
            "event": event,
            "capture_key": _scalar(context.capture_key),
            "model_call_id": _scalar(context.model_call_id),
            "truncated": False,
        }
        record["truncated"] = any(len(v) > MAX_STRING for v in (context.capture_key, context.model_call_id))
        if event == "commit":
            record["response"] = context.diagnostic_response
        else:
            record.update(
                request=_items(context.request_items or []),
                previous_response_id=_scalar(context.diagnostic_previous_response_id),
                resolution=context.diagnostic_resolution,
                reason=_scalar(context.parent_resolution.reason) if context.parent_resolution else None,
            )
            record["truncated"] |= len(context.diagnostic_previous_response_id or "") > MAX_STRING
        encoded = json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_BYTES:
            # Preserve identity and explicitly disclose omitted evidence. Never emit an oversized row.
            record = {key: record[key] for key in ("version", "event", "capture_key", "model_call_id")}
            record.update(truncated=True, truncation_reason="record_byte_limit")
            encoded = json.dumps(record, separators=(",", ":")).encode() + b"\n"
        with _LOCK:
            Path(directory).mkdir(parents=True, exist_ok=True)
            with (Path(directory) / f"capture-structure-{os.getpid()}.jsonl").open("ab") as handle:
                handle.write(encoded)
    except Exception:
        # Do not log exception text: filesystem errors may contain private configured paths.
        logger.warning("Capture structural diagnostic write failed; capture decision unchanged")
