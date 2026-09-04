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
"""Grade one recorded episode against the instruction-following constraints carried in its row metadata.

This is a port of the grading path of the offline scorer `score_if.py` (functions `segment`,
`to_tv_turns`, `prefix_turn_count`, `score_item`) into the gym, so that a rollout's verify response
can carry the per-constraint grades directly. Semantics, in the order they apply:

1. Segmentation. The Responses-API output items are cut into assistant turns: `message` and
   `function_call` items accumulate into the current turn until a `function_call_output` item
   closes it; `reasoning` items are ignored (the reasoning channel is never graded).
2. Turn conversion. Every turn becomes a verifier `Turn` (visible text = the assistant message
   texts joined by newlines; tool calls = name plus parsed JSON arguments). For the opencode
   persona the final turn is the last turn WITHOUT a tool call; for the codeact persona it is a
   turn that calls the `finish` tool (kept for parity with the offline scorer; the swe_agents rows
   graded here use the opencode persona).
3. Prefix handling. For items of type `last_step` or `interject` the row's input carries a
   replayed prefix conversation. If the output reproduces that prefix at its head (same tool-call
   sequence) those turns are skipped; otherwise the output is taken to be the continuation only
   and every turn is graded. Graded turns are then re-indexed from zero, so the `turn` index of a
   step is always relative to the CONTINUATION, never to the whole conversation.
4. Grading. `verifier.grade_ext` (the constraint verifier package) runs once per constraint with the verifier's default
   resolver updated with the row's tool binding (`metadata.tool_name_overrides`).

The public entry point is `grade_row`; it never raises.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from . import verifier as tv  # the constraint verifier: registries of matchers, triggers and templates


GRADING_ERROR_ID = "<grading_error>"

# Item types whose row input carries a replayed prefix conversation (data types 2 and 3).
PREFIX_TYPES = ("last_step", "interject")

# Rows built before the 2.0 row contract carry `data_type` (1, 2, 3) instead of `type`.
_LEGACY_DATA_TYPE = {1: "fresh", 2: "last_step", 3: "interject"}


# ------------------------------------------------------------------ Responses-API output -> verifier turns
def segment(items: List[dict], finish: str = "finish") -> List[dict]:
    """Assistant turns from Responses-API items: message/function_call items accumulate until a function_call_output.
    Returns a list of dicts {texts, calls:[(name,args)], call_ids}."""
    turns, cur = [], {"texts": [], "calls": [], "call_ids": []}
    for o in items:
        t = o.get("type")
        if t == "reasoning":
            continue
        if t == "function_call_output":
            if cur["texts"] or cur["calls"]:
                turns.append(cur)
            cur = {"texts": [], "calls": [], "call_ids": []}
        elif t == "message":
            if o.get("role") == "assistant":
                c = o.get("content")
                if isinstance(c, str):
                    cur["texts"].append(c)
                else:
                    for p in c or []:
                        if p.get("type") in ("output_text", "text") and p.get("text"):
                            cur["texts"].append(p["text"])
        elif t == "function_call":
            try:
                a = json.loads(o.get("arguments") or "{}")
            except Exception:
                a = {}
            cur["calls"].append((o.get("name") or "", a if isinstance(a, dict) else {}))
            cur["call_ids"].append(o.get("call_id"))
    if cur["texts"] or cur["calls"]:
        turns.append(cur)
    return turns


def to_tv_turns(segs: List[dict], persona: str = "opencode", finish: str = "finish") -> List[tv.Turn]:
    turns = []
    for i, s in enumerate(segs):
        visible = "\n".join(s["texts"])
        calls = [tv.ToolCall(name=n, args=a) for n, a in s["calls"]]
        if persona == "codeact":
            fin = [str(c.args["message"]) for c in calls if c.name == finish and "message" in c.args]
            if fin:
                visible = "\n".join([t for t in [visible] + fin if t])
        turns.append(tv.Turn(index=i, visible_text=visible, tool_calls=calls, is_final=False, preceding_messages=[]))
    if turns:
        last = turns[-1]
        genuine = (
            (len(last.tool_calls) == 0) if persona == "opencode" else any(c.name == finish for c in last.tool_calls)
        )
        last.is_final = bool(genuine)
    return turns


def prefix_turn_count(input_items: List[dict], output_items: List[dict]) -> Tuple[int, str, bool]:
    """How many leading assistant turns of the output are the replayed prefix (0 if the prefix is not present).

    Returns (turns_to_skip, note, prefix_present). `prefix_present` is True when the row's input carries a prefix
    conversation at all (assistant messages or tool calls after the system and user messages); the offline scorer
    conveys the same fact only through the wording of `note`."""
    pre = segment(
        [
            o
            for o in input_items
            if o.get("type") in ("message", "function_call", "function_call_output", "reasoning")
            and not (o.get("type") == "message" and o.get("role") in ("system", "user"))
        ]
    )
    if not pre:
        return 0, "no prefix in input", False
    out = segment(output_items)
    sig = lambda s: [n for n, _ in s["calls"]]  # noqa: E731
    n = len(pre)
    if len(out) >= n and all(sig(out[i]) == sig(pre[i]) for i in range(n)):
        return n, f"prefix of {n} assistant turns found at the head of the output (tool-call sequence matches)", True
    # partial match: count how many leading turns agree
    k = 0
    while k < min(n, len(out)) and sig(out[k]) == sig(pre[k]):
        k += 1
    return (
        0,
        f"row output is the continuation only (the harness replays the {n}-turn prefix in-episode and records only "
        f"LLM-generated turns; {k} leading turns matched); grading all {len(out)} continuation turns",
        True,
    )


# ------------------------------------------------------------------ row metadata helpers
def _parse_json_field(value: Any, name: str) -> Any:
    """Metadata values are strings (JSON-encoded when structured); accept an already-decoded object as well."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, (dict, list)):
        return value
    raise TypeError(f"metadata.{name} must be a JSON string or an object, got {type(value).__name__}")


def _item_type(sdg_item: dict) -> str:
    t = sdg_item.get("type")
    if isinstance(t, str) and t:
        return t
    dt = sdg_item.get("data_type")
    if dt in _LEGACY_DATA_TYPE:
        return _LEGACY_DATA_TYPE[dt]
    # Neither key present: an item with a stored prefix is a prefix item, anything else is fresh.
    return "last_step" if sdg_item.get("prefix") else "fresh"


# ------------------------------------------------------------------ grading
def _grade(
    metadata: Dict[str, Any], input_items: Optional[List[dict]], output_items: List[dict]
) -> Optional[List[dict]]:
    sdg_item = _parse_json_field(metadata.get("sdg_item"), "sdg_item")
    if not isinstance(sdg_item, dict):
        return None
    constraints = sdg_item.get("constraints")
    if not constraints:
        return None
    binding = _parse_json_field(metadata.get("tool_name_overrides"), "tool_name_overrides") or {}
    if not isinstance(binding, dict):
        raise TypeError("metadata.tool_name_overrides must decode to an object mapping identifier -> tool name")
    resolver = dict(tv.DEFAULT_RESOLVER)
    resolver.update(binding)
    persona = sdg_item.get("persona") or "opencode"

    segs = segment(output_items or [])
    turns = to_tv_turns(segs, persona)
    skip, continuation_only = 0, False
    if _item_type(sdg_item) in PREFIX_TYPES and input_items is not None:
        skip, _note, continuation_only = prefix_turn_count(input_items, output_items or [])
    graded = [t for t in turns if t.index >= skip]
    # re-index so the verifier's first_turn/final semantics apply to the continuation
    for j, t in enumerate(graded):
        t.index = j

    records = []
    for c in constraints:
        vp = c["verifier_parameter"]
        try:
            steps, q = tv.grade_ext(
                graded, vp, resolver=resolver
            )  # q = silent in-scope turns (no-answer count; owner ruling 2026-09-03)
            kind = tv.no_answer_policy(vp)
            err = None
        except ValueError as e:  # a retired or unknown matcher (e.g. `empty`, removed 2026-09-03): this constraint is not applicable, the row is not lost
            steps, q, kind, err = [], 0, None, "%s: %s" % (type(e).__name__, e)
        n = len(steps)
        p = sum(1 for s in steps if s.reward >= 1)
        records.append(
            {
                "id": c.get("id"),
                "trigger": vp.get("trigger"),
                "match": (vp.get("obligation") or {}).get("match"),
                "instruction": c.get("reference_instruction") or c.get("instruction"),
                "no_answer": kind,
                "n_steps": n,
                "n_pass": p,
                "n_silent": q,
                "step_avg": (p / n) if n else None,
                "all_pass": bool(n and p == n),
                "graded_turns": len(graded),
                "continuation_only": continuation_only,
                "steps": [{"turn": s.turn, "reward": s.reward, "detail": s.detail} for s in steps],
                **({"error": err} if err else {}),
            }
        )
    return records


def grade_row(
    metadata: Optional[Dict[str, Any]], input_items: Optional[List[dict]], output_items: List[dict]
) -> Optional[List[dict]]:
    """Return the `if_constraints` records for one episode, or None when the row carries no constraints.

    metadata     : the row's request metadata (`responses_create_params.metadata`); values are strings, structured
                   ones JSON-encoded (`sdg_item`, `tool_name_overrides`).
    input_items  : the row's request input items (system/user messages plus, for prefix items, the replayed prefix).
    output_items : the recorded Responses-API output items of the episode, as plain dicts.

    One record per constraint:
      {id, trigger, match, no_answer, instruction, n_steps, n_pass, n_silent, step_avg, all_pass, graded_turns, continuation_only,
       steps: [{turn, reward, detail}]}
    `n_steps` is the number of gradable steps (turns where the trigger fired), `n_pass` the number with reward 1,
    `step_avg` = n_pass / n_steps (None when the trigger never fired), `all_pass` = n_steps > 0 and n_pass == n_steps,
    `graded_turns` the number of assistant turns graded, `continuation_only` True when the row is a prefix item whose
    graded turns are the continuation of a replayed prefix (turn indices are then relative to the continuation).

    Never raises: any unexpected exception is returned as a single record {id: '<grading_error>', error: <message>}
    so that the rollout itself is never lost to a grading failure."""
    try:
        return _grade(metadata or {}, input_items, output_items)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: the rollout must survive any grading failure
        return [{"id": GRADING_ERROR_ID, "error": f"{type(exc).__name__}: {exc}"}]
