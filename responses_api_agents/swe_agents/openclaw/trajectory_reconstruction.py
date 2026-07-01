# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side reconstruction of Responses-API output items from the StreamShim
JSONL log. Token IDs flow through verbatim — vllm_model attached them to each
turn's last output item; we preserve them by reference.

NOTE: will be replaced in the future with a native support in NeMo-Gym for trajectory reconstruction.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable


def _dedupe_superseded_turns(proxy_log: list[dict]) -> list[dict]:
    """Drop superseded same-`turn` proxy entries, keeping the last per turn.

    The shim only advances the turn number on a 2xx, so two entries share a turn
    ONLY when an earlier generation was abandoned/errored and re-issued (e.g. an
    idle-timeout retry). Keeping both would emit two trainable turns with an
    identical prompt and break NeMo-RL's on-policy contiguity assert. The
    last-logged entry is the one whose token IDs the shim spliced forward, so it
    matches what the model saw. No-op when turns are unique; entries without an
    integer turn are never deduped.
    """
    last_idx_for_turn: dict = {}
    for i, entry in enumerate(proxy_log):
        if isinstance(entry, dict) and entry.get("turn") is not None:
            last_idx_for_turn[entry["turn"]] = i
    kept_last = set(last_idx_for_turn.values())
    return [
        entry
        for i, entry in enumerate(proxy_log)
        if not (isinstance(entry, dict) and entry.get("turn") is not None) or i in kept_last
    ]


def reconstruct_responses_items(proxy_log: list[dict]) -> tuple[list, list, list]:
    """Return (input_items, output_items, tools).

    Algorithm:
      • Drop superseded same-turn retries (idle-timeout / errored re-issues)
        so abandoned generations never become trainable turns.
      • Initial input items = turn 0's request input.
      • For each turn N>0: new function_call_output items appended since
        turn N-1's request go between turn N-1's response and turn N's response.
      • Each turn's response output items appended verbatim (token IDs on
        last item preserved by reference).
    """
    if not proxy_log:
        return [], [], []
    proxy_log = _dedupe_superseded_turns(proxy_log)
    first = proxy_log[0]
    first_req = first.get("request") if isinstance(first, dict) else None
    if not isinstance(first_req, dict):
        return [], [], []
    initial_input = list(first_req.get("input", []))
    tools = list(first_req.get("tools", []))
    output_items: list = []
    prev_request_input = initial_input
    for turn in proxy_log:
        req = turn.get("request") if isinstance(turn, dict) else None
        resp = turn.get("response") if isinstance(turn, dict) else None
        # Refused / errored turn — `response` may be a raw error string (aiohttp
        # up.json() can return a non-dict body), or `request` may be missing.
        # Treat any non-dict req/resp as an errored turn: nothing to append.
        if not isinstance(req, dict) or not isinstance(resp, dict):
            continue
        new_inputs = req.get("input", [])[len(prev_request_input) :]
        for item in new_inputs:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                output_items.append(item)
        prev_request_input = req.get("input", [])
        output_items.extend(resp.get("output", []))
    return initial_input, output_items, tools


def classify_openclaw_agent_error(
    *,
    proxy_log: list[dict],
    trajectory_events: Iterable[dict],
    subprocess_timed_out: bool,
    subprocess_exit_code: int,
) -> str | None:
    """Return the agent_error_kind bucket, matching the existing rule in app.py."""
    if subprocess_timed_out:
        return None  # caller sets agent_timed_out=True separately

    # Scan proxy log entries in order; first match wins.
    for entry in proxy_log:
        if entry.get("error") == "max_iteration":
            return "max_iteration"
        status = entry.get("upstream_status")
        if status is not None and status >= 400:
            body_str = _stringify(entry.get("response"))
            lo = body_str.lower()
            if "context length" in lo or "maximum context" in lo:
                return "context_window"
            return "other"

    # Fall back to OpenClaw's trajectory events for clean-exit failures.
    for ev in trajectory_events:
        if ev.get("type") == "session.ended":
            reason = (ev.get("data") or {}).get("reason") or ""
            lo = reason.lower()
            if "max" in lo and "iter" in lo:
                return "max_iteration"
            if "context" in lo:
                return "context_window"

    if subprocess_exit_code != 0:
        return "other"
    return None


def _stringify(o) -> str:
    if isinstance(o, (dict, list)):
        return json.dumps(o)
    if o is None:
        return ""
    return str(o)


_STRIP_KEYS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")


def _strip_items(items) -> None:
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                for k in _STRIP_KEYS:
                    item.pop(k, None)


def strip_token_ids_in_proxy_log(jsonl_path: str) -> None:
    """In-place rewrite that drops token-id fields from every response output item AND every
    request input item.

    The shim injects the prior turn's token IDs onto the request's last assistant item (for
    NeMo-RL's on-policy splice), so both sides carry these large arrays and both must be
    stripped to keep logs small. Tolerant of refused/errored turns with `request`/`response: null`."""
    dirname = os.path.dirname(jsonl_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".strip_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as out, open(jsonl_path, "r") as inp:
            for line in inp:
                line = line.rstrip("\n")
                if not line:
                    continue
                entry = json.loads(line)
                response = entry.get("response")
                if isinstance(response, dict):
                    _strip_items(response.get("output"))
                request = entry.get("request")
                if isinstance(request, dict):
                    _strip_items(request.get("input"))
                out.write(json.dumps(entry, separators=(",", ":")))
                out.write("\n")
        os.replace(tmp_path, jsonl_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
