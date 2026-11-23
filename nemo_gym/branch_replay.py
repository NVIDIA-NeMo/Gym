# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Branch replay on the model server: record a rollout's chat completions, and replay a recorded prefix.

Both sides are off unless ``NEMO_GYM_BRANCH_REPLAY_DIR`` names a directory shared by the model server and the
training controller.

Recording.  After every chat completion served under a rollout path prefix, the response returned to the client
(without its token-id fields) is appended to ``<dir>/<rollout_id>.calls.jsonl`` together with the number of
assistant messages the request carried and a digest of the request's last message.  A rollout that never
compacted and never spawned a subagent issues its calls in order, so call ``j`` is the one whose request carried
``j`` assistant messages.

Replay.  The controller writes ``<dir>/<rollout_id>.plan.json`` for a new rollout before dispatching it::

    {"source_rollout_id": "...", "branch_turn": k}

For that rollout, a request carrying ``j < k`` assistant messages is answered with the source rollout's recorded
call ``j`` instead of a generation.  The harness executes the recorded tool calls itself, so its session and the
sandbox reach the state at the start of turn ``k``.  A replayed call is served before the capture resolves a
parent or admits the call, so the token capture never sees it: the first generated call (turn ``k``) finds no
captured row for the rollout and is admitted as a chain root whose prompt is the whole replayed history.  The
replayed turns are therefore prompt tokens of the training row, and only turns ``k`` onward are generated tokens.

Every replayed call appends ``{"call": j, "context_match": bool}`` to ``<dir>/<rollout_id>.replay.jsonl``:
``context_match`` says whether the last message of the request (normally the previous tool result) equals what
the source rollout saw at the same call, i.e. whether the re-executed tools reproduced the recorded state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

BRANCH_REPLAY_DIR_ENV = "NEMO_GYM_BRANCH_REPLAY_DIR"
FORCED_ID_PREFIX = "ngbr-"

LOG = logging.getLogger(__name__)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOKEN_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs", "routed_experts")
_lock = threading.Lock()
# rollout id -> plan (None: the rollout has no plan).  A plan is written before its rollout is dispatched, so a
# miss on the rollout's first call is final.
_plans: dict[str, Optional[dict[str, Any]]] = {}
# source rollout id -> {call index: record}, loaded when a plan first needs it.
_sources: dict[str, dict[int, dict[str, Any]]] = {}
_MAX_CACHED = 4096


def replay_dir() -> Optional[Path]:
    value = os.environ.get(BRANCH_REPLAY_DIR_ENV, "").strip()
    return Path(value) if value else None


def calls_path(root: Path, rollout_id: str) -> Path:
    return root / f"{rollout_id}.calls.jsonl"


def plan_path(root: Path, rollout_id: str) -> Path:
    return root / f"{rollout_id}.plan.json"


def replay_log_path(root: Path, rollout_id: str) -> Path:
    return root / f"{rollout_id}.replay.jsonl"


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _messages(params: Any) -> list[Any]:
    messages = _field(params, "messages")
    return list(messages) if isinstance(messages, list) else []


def assistant_count(params: Any) -> int:
    return sum(1 for message in _messages(params) if _field(message, "role") == "assistant")


def last_message_digest(params: Any) -> str:
    messages = _messages(params)
    if not messages:
        return ""
    last = messages[-1]
    payload = {"role": _field(last, "role"), "content": _field(last, "content")}
    if hasattr(payload["content"], "model_dump"):
        payload["content"] = payload["content"].model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def is_forced_response(response: Any) -> bool:
    response_id = _field(response, "id")
    return isinstance(response_id, str) and response_id.startswith(FORCED_ID_PREFIX)


def _without_token_fields(completion: Any) -> dict[str, Any]:
    data = completion.model_dump(mode="json") if hasattr(completion, "model_dump") else json.loads(json.dumps(completion))
    for choice in data.get("choices") or []:
        message = choice.get("message")
        if isinstance(message, dict):
            for name in _TOKEN_FIELDS:
                message.pop(name, None)
        choice.pop("logprobs", None)
        choice.pop("token_ids", None)
    data.pop("prompt_token_ids", None)
    return data


def record_completion(rollout_id: Optional[str], params: Any, completion: Any) -> None:
    """Append one served chat completion of ``rollout_id``.  Best effort: a failure never fails the call."""
    root = replay_dir()
    if root is None or not rollout_id or not _ID_RE.match(rollout_id) or is_forced_response(completion):
        return
    try:
        record = {
            "call": assistant_count(params),
            "context_digest": last_message_digest(params),
            "response": _without_token_fields(completion),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        root.mkdir(parents=True, exist_ok=True)
        with open(calls_path(root, rollout_id), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        LOG.warning("Branch replay could not record a call of rollout %s.", rollout_id, exc_info=True)


def _load_plan(root: Path, rollout_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        if rollout_id in _plans:
            return _plans[rollout_id]
    plan: Optional[dict[str, Any]] = None
    path = plan_path(root, rollout_id)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        source = loaded.get("source_rollout_id")
        turn = loaded.get("branch_turn")
        if not (isinstance(source, str) and _ID_RE.match(source) and isinstance(turn, int) and turn > 0):
            raise ValueError(f"Malformed branch replay plan {path}: {loaded!r}")
        plan = {"source_rollout_id": source, "branch_turn": turn}
    with _lock:
        if len(_plans) >= _MAX_CACHED:
            _plans.clear()
        _plans[rollout_id] = plan
    return plan


def _load_source(root: Path, source_rollout_id: str) -> dict[int, dict[str, Any]]:
    with _lock:
        cached = _sources.get(source_rollout_id)
    if cached is not None:
        return cached
    calls: dict[int, dict[str, Any]] = {}
    with open(calls_path(root, source_rollout_id), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # A retried call repeats its index; the last answer is the one the harness kept.
                calls[int(record["call"])] = record
    with _lock:
        if len(_sources) >= 256:
            _sources.clear()
        _sources[source_rollout_id] = calls
    return calls


def forced_completion(rollout_id: Optional[str], params: Any) -> Optional[dict[str, Any]]:
    """The recorded response to serve instead of generating, or None to generate normally.

    Raises when the rollout has a plan but the source recording cannot serve the call: generating there would
    silently turn a replay into a different rollout, so the call fails and the rollout is retried or masked.
    """
    root = replay_dir()
    if root is None or not rollout_id or not _ID_RE.match(rollout_id):
        return None
    plan = _load_plan(root, rollout_id)
    if plan is None:
        return None
    call = assistant_count(params)
    if call >= plan["branch_turn"]:
        return None
    source = _load_source(root, plan["source_rollout_id"])
    record = source.get(call)
    if record is None:
        raise RuntimeError(
            f"Branch replay of rollout {rollout_id}: source {plan['source_rollout_id']} has no recorded call {call} "
            f"(branch turn {plan['branch_turn']}, recorded calls {sorted(source)[:3]}..{len(source)})"
        )
    response = json.loads(json.dumps(record["response"]))
    response["id"] = FORCED_ID_PREFIX + str(response.get("id") or call)
    try:
        entry = {"call": call, "context_match": record.get("context_digest") == last_message_digest(params)}
        with open(replay_log_path(root, rollout_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        LOG.warning("Branch replay could not log a replayed call of rollout %s.", rollout_id, exc_info=True)
    return response
