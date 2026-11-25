# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Prove that the judge's model calls are OpenCode's model calls, one to one (stdlib only).

The judge counts a rollout's model calls on ``response.output``: a call starts at item 0 and at every reasoning
item. The trainer maps judge call k to the k-th generated-token run of the training row, which is OpenCode's k-th
model call (*step* in ``export.json``, Gym's ``opencode_results_fpath``; ``response.output`` is built from the
export's parts). The two numberings agree only if every OpenCode step yields exactly one call start. They do not
when a step has no reasoning (merged into the previous call), when a step yields two reasoning items, or when the
session ends with steps that left no item at all (an empty model reply).

``call_alignment`` checks the mapping item by item:
- each judge call k starts in OpenCode step k: its reasoning item's text is the ``<think>`` content of step k (call 0
  may instead start with step 0's first visible item);
- each tool call item of judge call c is a tool part of step c (matched by call id), and every tool part appears;
- the steps that left no item (no reasoning, no visible text, no tool) all come after the last step that did.

It returns ``(aligned, trailing_empty, reason)``. ``aligned`` with ``trailing_empty = k`` means calls 0..n-1 are steps
0..n-1 exactly and the session has k more, empty, steps at its end: the training row may have up to k extra
generated-token runs after run n-1, and they carry no judge call. Anything not proven returns ``aligned = False``: a
label on a wrong turn is worse than no label.
"""

from typing import Any, Optional, Sequence


def _call_starts(output: Sequence[dict]) -> list[int]:
    # = merge.call_starts (duplicated so this module stays stdlib-only and self-contained)
    return [i for i, it in enumerate(output) if i == 0 or it.get("type") == "reasoning"]


def _call_of(starts: Sequence[int], idx: int) -> int:
    return sum(1 for s in starts if s <= idx) - 1


def _split_think(text: str) -> tuple[str, str]:
    """(think content, visible text) of an assistant text part, as the export stores it."""
    if "</think>" in text:
        think, visible = text.split("</think>", 1)
        think = think.split("<think>", 1)[1] if "<think>" in think else think
        return think.strip(), visible.strip()
    return "", text.strip()


def _reasoning_text(item: dict) -> str:
    parts = item.get("summary") or item.get("content") or []
    if isinstance(parts, list):
        return "".join(str((p or {}).get("text") or "") for p in parts if isinstance(p, dict)).strip()
    return str(parts).strip()


def export_steps(export: dict) -> list[dict]:
    """OpenCode's model calls in order: per step its think contents, visible texts and tool call ids.

    Raises ValueError when an assistant part sits outside a step or has an unknown type.
    """
    steps: list[dict] = []
    for message in export.get("messages") or []:
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        current: Optional[dict] = None
        for part in message.get("parts") or []:
            ptype = part.get("type")
            if ptype == "step-start":
                current = {"thinks": [], "visible": [], "call_ids": []}
                steps.append(current)
                continue
            if ptype in ("step-finish", "patch"):
                continue
            if current is None:
                raise ValueError(f"assistant part {ptype!r} outside a step")
            if ptype == "tool":
                current["call_ids"].append(part.get("callID"))
            elif ptype == "text":
                think, visible = _split_think(str(part.get("text") or ""))
                if think:
                    current["thinks"].append(think)
                if visible:
                    current["visible"].append(visible)
            elif ptype == "reasoning":
                think = str(part.get("text") or "").strip()
                if think:
                    current["thinks"].append(think)
            else:
                raise ValueError(f"unknown assistant part {ptype!r}")
    for s in steps:
        s["nonempty"] = bool(s["thinks"] or s["visible"] or s["call_ids"])
    return steps


def _count(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def step_output_tokens(export: Any) -> Optional[list[Optional[int]]]:
    """Generated tokens of each OpenCode model call (step), in order, or None when not attributable.

    OpenCode writes one assistant message per step, with the step's usage in ``info.tokens``: ``output`` plus
    ``reasoning`` (some steps report reasoning tokens apart) is the step's generated tokens. The trainer compares
    this list with the training row's generated-token runs: equal lengths, run by run, prove the row is the
    trajectory the judge read (a row re-dispatched under the same rollout id can train another session's tokens).
    An entry is None when the step never finished (no ``step-finish``: the session ended during its tool call, and
    its usage was never recorded). The whole list is None (the trainer then applies no label) when an assistant
    message with parts holds other than one step, or a finished step has no integer counts. Validated on 1,140
    labeled rows (LOG 095): every healthy row matches, every known mix-up fails at its first run.
    """
    if not isinstance(export, dict):
        return None
    out: list[Optional[int]] = []
    for message in export.get("messages") or []:
        info = message.get("info") or {}
        if info.get("role") != "assistant":
            continue
        parts = message.get("parts") or []
        if not parts:
            continue  # an empty assistant message: no step, no model call
        if sum(1 for part in parts if part.get("type") == "step-start") != 1:
            return None
        if not any(part.get("type") == "step-finish" for part in parts):
            out.append(None)
            continue
        tokens = info.get("tokens") or {}
        output, reasoning = _count(tokens.get("output")), _count(tokens.get("reasoning", 0))
        if output is None or reasoning is None:
            return None
        out.append(output + reasoning)
    return out


def call_alignment(export: Any, output: Sequence[dict]) -> tuple[bool, int, str]:
    """``(aligned, trailing_empty, reason)``; see the module docstring."""
    if not isinstance(export, dict):
        return False, 0, "no_export"
    try:
        steps = export_steps(export)
    except ValueError as error:
        return False, 0, f"structure: {error}"
    if not output:
        return False, 0, "no_output"
    nonempty = [i for i, s in enumerate(steps) if s["nonempty"]]
    if not nonempty:
        return False, 0, "no_nonempty_step"
    n = nonempty[-1] + 1
    if len(nonempty) != n:
        return False, 0, "inner_empty_step"
    starts = _call_starts(output)
    if len(starts) != n:
        return False, 0, f"calls {len(starts)} != steps {n}"
    # Each call k starts in step k.
    for k, idx in enumerate(starts):
        item = output[idx]
        step = steps[k]
        if item.get("type") == "reasoning":
            if len(step["thinks"]) != 1 or _reasoning_text(item) != step["thinks"][0]:
                return False, 0, f"call {k}: reasoning is not step {k}'s single think"
        elif k == 0:
            if step["thinks"]:
                return False, 0, "call 0 starts without reasoning but step 0 thinks"
        else:
            return False, 0, f"call {k} starts at a non-reasoning item"
    # Steps without thinking (beyond call 0) would have been merged into the previous call: rejected above, since
    # the call count matches only if each step contributes exactly one start. Tool calls must sit in their step.
    step_of_call_id: dict[str, int] = {}
    for s, step in enumerate(steps[:n]):
        for call_id in step["call_ids"]:
            if not call_id or call_id in step_of_call_id:
                return False, 0, "tool call id missing or repeated"
            step_of_call_id[call_id] = s
    seen = set()
    for idx, item in enumerate(output):
        if item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        if call_id not in step_of_call_id:
            return False, 0, "function_call not in the export"
        if _call_of(starts, idx) != step_of_call_id[call_id]:
            return False, 0, f"tool call {call_id} in call {_call_of(starts, idx)}, step {step_of_call_id[call_id]}"
        seen.add(call_id)
    if seen != set(step_of_call_id):
        return False, 0, "export tool call missing from output"
    return True, len(steps) - n, "ok"
