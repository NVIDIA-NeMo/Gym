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
"""Deterministic partial-credit reward: ACTION x DB x COMMUNICATE; NL_ASSERTION comes from judge.py.

strict = product of the components in the task's reward_basis; reward = strict + SHAPE * dense * (1 - strict),
so partial credit never reaches a full pass. Component decomposition follows tau2-bench; ``action_compare`` and
``state_normalize`` are derived from it (MIT, see their headers).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from resources_servers.indian_banking.core import engine
from resources_servers.indian_banking.core.action_compare import args_match
from resources_servers.indian_banking.core.state_normalize import normalize_state


SHAPE = 0.4
# Dense-mix weights; renormalised over whichever components a task actually has.
W_ACTION = 0.40
W_DB = 0.25
W_COMMUNICATE = 0.15
W_JUDGE = 0.20

# anti-inaction shaping: without these, a silent agent can outscore a partial attempt.
A_NAME = 0.35  # tool-name-only credit share
PURITY_PEN = 0.5  # cost of bad writes, task block only
CONV_FLOOR = 0.3  # min COMMUNICATE/JUDGE share if no tools attempted

# efficiency tie-breaker; applied ONLY when strict==1 (see score_trajectory). Set EFF=0.0 to disable.
EFF = 0.15
EFF_W_JUDGE = 0.5
EFF_W_ORDER = 0.2

# Order-strict action credit: in-order gold matches earn A_SEQ of the argument credit.
# Not comparable with set-based tau2-bench pass rates.
SEQ_STRICT = True
A_SEQ = 0.5  # in-order share of arg credit when SEQ_STRICT
# Anti-degenerate-loop penalty, off by default.
REPEAT_PENALTY_MAX = 0.0
REPEAT_PENALTY_MIN_STREAK = 3

# Keys of the per-episode session-state dict that score_trajectory reads.
WORLD_KEY = "tau_world"
TASK_KEY = "tau_task"  # {task_id, customer, evaluation_criteria, user_scenario}

# ---- internal-leak detection ------------------------------------------------
# Tokens that must never appear in a customer-facing message: tool names,
# backend category codes, and API field names.
_INTERNAL_CATEGORY_CODES = frozenset(
    {
        "failed_transaction",
        "unauthorized_debit",
        "service_quality",
        "app_issue",
        "charges_dispute",
        "general_query",
    }
)
_INTERNAL_API_FIELD_NAMES = frozenset({"related_transaction_id", "compare_args"})
_BANKING_TOOL_NAMES = frozenset(engine.MCP_TOOL_NAMES) | {"transfer_to_human_agents"}


def find_internal_leaks(text: str) -> list[str]:
    lower = text.lower()
    found: set[str] = set()
    for token in _BANKING_TOOL_NAMES | _INTERNAL_CATEGORY_CODES | _INTERNAL_API_FIELD_NAMES:
        if re.search(rf"\b{re.escape(token)}\b", lower):
            found.add(token)
    return sorted(found)


def _dict_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ---- ACTION -----------------------------------------------------------------
def _replay_gold(customer, gold_actions: list[dict]) -> tuple[dict, list[bool]]:
    """Replay gold actions on a fresh seed; returns (gold_world, per-action error flags).

    The DB hash comparison reads the replayed world; the error flags exist for data
    integrity checks (every flagged action must carry an explicit ``expect_error``)."""
    gold_world = engine.seed_world(customer)
    errors = []
    for a in gold_actions:
        engine.apply_tool(gold_world, a["name"], dict(a.get("arguments") or {}))
        errors.append(bool(gold_world["calls"][-1].get("error")))
    return gold_world, errors


def _action_reward(calls: list[dict], gold_actions: list[dict]) -> dict:
    """Score ACTION. Returns strict, action_frac, name_frac, seq_frac, purity_ok, bad_writes, n_writes."""
    if not gold_actions:
        return {
            "strict": 1.0,
            "action_frac": 1.0,
            "name_frac": 1.0,
            "seq_frac": 1.0,
            "purity_ok": True,
            "bad_writes": 0,
            "n_writes": 0,
        }

    def matches(gold: dict, call: dict) -> bool:
        if gold["name"] != call["name"]:
            return False
        # An errored call satisfies a gold action only if the task explicitly marks
        # that action with ``expect_error: true`` (tasks that deliberately exercise
        # an error path). Never inferred from the gold replay.
        if call.get("error") and not gold.get("expect_error"):
            return False
        return args_match(
            gold.get("arguments") or {},
            call.get("arguments") or {},
            gold.get("compare_args"),
        )

    matched = 0
    named = 0
    called_names = {c["name"] for c in calls}
    for gold in gold_actions:
        if any(matches(gold, c) for c in calls):
            matched += 1
        if gold["name"] in called_names:
            named += 1
    action_frac = matched / len(gold_actions)
    name_frac = named / len(gold_actions)
    all_gold_found = matched == len(gold_actions)

    # LCS(gold, calls)/n_gold: in-order match fraction.
    m = len(calls)
    prev_row = [0] * (m + 1)
    for g in gold_actions:
        row = [0] * (m + 1)
        for ci in range(1, m + 1):
            if matches(g, calls[ci - 1]):
                row[ci] = prev_row[ci - 1] + 1
            else:
                row[ci] = max(prev_row[ci], row[ci - 1])
        prev_row = row
    seq_frac = prev_row[m] / len(gold_actions)

    # Write purity, counted rather than short-circuited.
    gold_writes = [g for g in gold_actions if g["name"] in engine.WRITE_TOOLS]
    n_writes = 0
    bad_writes = 0
    for c in calls:
        if c["name"] not in engine.WRITE_TOOLS:
            continue
        if c.get("error"):
            continue  # an errored write mutated nothing; retry-after-error is not a wrong write
        n_writes += 1
        if not any(matches(g, c) for g in gold_writes):
            bad_writes += 1

    strict_ok = all_gold_found and bad_writes == 0
    if SEQ_STRICT:
        strict_ok = strict_ok and seq_frac >= 1.0
    return {
        "strict": 1.0 if strict_ok else 0.0,
        "action_frac": action_frac,
        "name_frac": name_frac,
        "seq_frac": seq_frac,
        "purity_ok": bad_writes == 0,
        "bad_writes": bad_writes,
        "n_writes": n_writes,
    }


def _gold_order_violated(calls: list[dict], gold_actions: list[dict]) -> bool:
    """True if matched gold actions were called out of canonical order (False if <2 matched)."""
    if len(gold_actions) < 2:
        return False
    positions = []
    for gold in gold_actions:
        for i, c in enumerate(calls):
            if c["name"] == gold["name"] and args_match(
                gold.get("arguments") or {}, c.get("arguments") or {}, gold.get("compare_args")
            ):
                positions.append(i)
                break
    if len(positions) < 2:
        return False
    return positions != sorted(positions)


def _efficiency_cost(
    calls: list[dict],
    gold_actions: list[dict],
    nl_history: list[dict],
    judge_score: float | None = None,
) -> tuple[float, dict]:
    """Cost in [0, 1] for a successful trajectory (duplicate calls, excess writes, out-of-order gold, judge shortfall).

    Never applied to failures; extra lookups and prose length are deliberately not charged.
    """
    n_gold = max(1, len(gold_actions))

    seen: set[str] = set()
    duplicates = 0
    writes = 0
    for c in calls:
        key = f"{c['name']}|{json.dumps(c.get('arguments') or {}, sort_keys=True, default=str)}"
        if key in seen:
            duplicates += 1
        seen.add(key)
        if c["name"] in engine.WRITE_TOOLS:
            writes += 1
    gold_writes = sum(1 for g in gold_actions if g["name"] in engine.WRITE_TOOLS)
    excess_writes = max(0, writes - gold_writes)
    waste = min(1.0, (duplicates + excess_writes) / n_gold)

    out_of_order = 1.0 if _gold_order_violated(calls, gold_actions) else 0.0
    judge_short = 0.0 if judge_score is None else max(0.0, 1.0 - float(judge_score))

    cost = min(
        1.0,
        EFF_W_JUDGE * judge_short + EFF_W_ORDER * out_of_order + max(0.0, 1.0 - EFF_W_JUDGE - EFF_W_ORDER) * waste,
    )
    return cost, {
        "eff_waste": waste,
        "eff_duplicates": float(duplicates),
        "eff_excess_writes": float(excess_writes),
        "eff_order": out_of_order,
        "eff_judge_short": judge_short,
        "prose_tokens": sum(len(str(m.get("content") or "")) for m in nl_history) / 4.0,
    }


# ---- DB ---------------------------------------------------------------------
def _db_reward(world: dict, gold_world: dict) -> float:
    """Hash the replayed gold world's DB and compare to the agent's live world DB.
    Both run under the same frozen clock and id seed (engine.SIM_CLOCK / SIM_SEED)
    so generated ids and timestamps match."""
    gold_hash = _dict_hash(normalize_state(gold_world["db"]))
    pred_hash = _dict_hash(normalize_state(world["db"]))
    return 1.0 if gold_hash == pred_hash else 0.0


# ---- COMMUNICATE ------------------------------------------------------------
def _communicate_reward(nl_history: list[dict], communicate_info: list[str]) -> tuple[float, float]:
    """Returns (strict, communicate_frac). A leak zeroes both — hard failure, not partial."""
    assistant_msgs = [m.get("content", "") for m in nl_history if m.get("role") == "assistant"]
    # Banking customer-facing gate: no internal identifier leaks.
    for msg in assistant_msgs:
        if find_internal_leaks(msg):
            return 0.0, 0.0

    required = list(communicate_info or [])
    if not required:
        return 1.0, 1.0
    # Required info substrings (case-insensitive, comma-stripped): tau2 semantics.
    hits = sum(1 for info in required if any(info.lower() in msg.lower().replace(",", "") for msg in assistant_msgs))
    frac = hits / len(required)
    return (1.0 if hits == len(required) else 0.0), frac


# ---- optional: anti-degenerate-loop penalty ----------------------------------
def _repeat_penalty(calls: list[dict]) -> tuple[float, int]:
    """Penalize 3+ back-to-back identical (name, args) calls. Non-consecutive repeats
    (e.g. check_balance before AND after a transfer) don't count. Returns
    (penalty, longest_streak); penalty is 0 unless REPEAT_PENALTY_MAX > 0."""
    if REPEAT_PENALTY_MAX <= 0 or len(calls) < REPEAT_PENALTY_MIN_STREAK:
        return 0.0, 0

    def key(c: dict) -> str:
        return f"{c.get('name')}::{json.dumps(c.get('arguments') or {}, sort_keys=True, default=str)}"

    longest = streak = 1
    prev = key(calls[0])
    for c in calls[1:]:
        k = key(c)
        streak = streak + 1 if k == prev else 1
        longest = max(longest, streak)
        prev = k

    if longest < REPEAT_PENALTY_MIN_STREAK:
        return 0.0, longest
    frac = min(1.0, (longest - REPEAT_PENALTY_MIN_STREAK + 1) / REPEAT_PENALTY_MIN_STREAK)
    return REPEAT_PENALTY_MAX * frac, longest


# ---- top-level --------------------------------------------------------------
def _plain(value: Any) -> Any:
    """Recursively convert numpy-like containers/scalars (``.tolist()`` / ``.item()``)
    back to plain Python. Duck-typed so numpy is not a dependency."""
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict, str, bytes)):
        return _plain(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict, str, bytes)):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def normalize_gold_actions(gold_actions: Any) -> list[dict]:
    """Undo parquet round-trip damage to ``evaluation_criteria.actions``.

    Drops ``None`` filler arguments introduced by schema unification and converts ndarray values back to lists.
    """
    if hasattr(gold_actions, "tolist"):
        gold_actions = gold_actions.tolist()
    if not gold_actions:
        return []

    out: list[dict] = []
    for action in gold_actions:
        action = dict(_plain(action))
        action["arguments"] = {k: v for k, v in (action.get("arguments") or {}).items() if v is not None}
        if "compare_args" in action and action["compare_args"] is not None:
            action["compare_args"] = list(action["compare_args"])
        out.append(action)
    return out


def score_trajectory(store: dict, judge_score: float | None = None) -> dict:
    """Compute the reward from an episode's session-state store.

    judge_score is passed in rather than computed here because judging is
    async/network-bound while this function is sync; None skips the judge term.
    """
    task = store.get(TASK_KEY) or {}
    ec = task.get("evaluation_criteria") or {}
    # Tolerates the parquet round-trip; see normalize_gold_actions.
    basis = set(_plain(ec.get("reward_basis")) or [])
    gold_actions = normalize_gold_actions(ec.get("actions"))
    communicate_info = _plain(ec.get("communicate_info")) or []
    customer = task.get("customer")

    world = store.get(WORLD_KEY)
    if world is None:  # policy never called a tool: empty predicted world
        world = engine.seed_world(customer)
    calls = world.get("calls", [])
    nl_history = store.get("nl_history", [])

    components: dict[str, float] = {}
    # task block (ACTION/DB) and conv block (COMMUNICATE/JUDGE) are gated
    # differently — see A_NAME/PURITY_PEN/CONV_FLOOR above.
    task_parts: list[tuple[float, float]] = []
    conv_parts: list[tuple[float, float]] = []
    action_frac = 1.0
    name_frac = 1.0
    seq_frac = 1.0
    purity_ok = True
    bad_writes = n_writes = 0
    gold_world = None
    if gold_actions and ("ACTION" in basis or "DB" in basis):
        gold_world, _ = _replay_gold(customer, gold_actions)
    if "ACTION" in basis:
        a = _action_reward(calls, gold_actions)
        action_frac, name_frac = a["action_frac"], a["name_frac"]
        seq_frac = a["seq_frac"]
        purity_ok, bad_writes, n_writes = a["purity_ok"], a["bad_writes"], a["n_writes"]
        components["ACTION"] = a["strict"]
        arg_frac = action_frac
        if SEQ_STRICT:
            arg_frac = (1.0 - A_SEQ) * action_frac + A_SEQ * seq_frac
        task_parts.append((W_ACTION, A_NAME * name_frac + (1.0 - A_NAME) * arg_frac))
    if "DB" in basis:
        components["DB"] = _db_reward(world, gold_world if gold_world is not None else engine.seed_world(customer))
        # Read-only-gold tasks: gate dense DB credit on engagement (a do-nothing agent trivially leaves the DB unchanged).
        gold_has_write = any(g["name"] in engine.WRITE_TOOLS for g in gold_actions)
        if gold_actions and not gold_has_write:
            db_gate = CONV_FLOOR + (1.0 - CONV_FLOOR) * name_frac
        else:
            db_gate = 1.0
        task_parts.append((W_DB, components["DB"] * db_gate))
    communicate_frac = 1.0
    if "COMMUNICATE" in basis:
        comm, communicate_frac = _communicate_reward(nl_history, communicate_info)
        components["COMMUNICATE"] = comm
        conv_parts.append((W_COMMUNICATE, communicate_frac))
    # NL_ASSERTION enters only via dense, never strict — see judge.py for why it's bounded.
    if judge_score is not None:
        conv_parts.append((W_JUDGE, judge_score))

    strict = 1.0
    for v in components.values():
        strict *= v
    if not components:  # NL-only basis (no shipped task uses one) -> neutral
        strict = 1.0

    # ---- deterministic floors (never inferred, always checkable offline) ----
    # A silent episode never passes: strict requires at least one non-empty
    # customer-facing assistant message, whatever the reward basis. This is what
    # separates a correct refusal-with-explanation from a mute agent on tasks
    # whose DB check passes trivially.
    if not any((m.get("content") or "").strip() for m in nl_history if m.get("role") == "assistant"):
        strict = 0.0
    # Optional per-task criteria in evaluation_criteria:
    #   max_tool_calls: N  -> more than N tool calls fails strict (0 = conversational-only task)
    #   require_transfer: true -> the episode must end in transfer_to_human_agents
    max_tool_calls = ec.get("max_tool_calls")
    if max_tool_calls is not None and len(calls) > int(max_tool_calls):
        strict = 0.0
    if ec.get("require_transfer") and not world.get("transferred"):
        strict = 0.0

    # Any partial credit stays below a true pass; purity only discounts the task block.
    purity_scale = 1.0 - PURITY_PEN * (bad_writes / n_writes) if n_writes else 1.0
    conv_gate = CONV_FLOOR + (1.0 - CONV_FLOOR) * name_frac
    task_w = sum(w for w, _ in task_parts)
    conv_w = sum(w for w, _ in conv_parts)
    total_w = task_w + conv_w
    if total_w:
        task_sum = sum(w * v for w, v in task_parts) * purity_scale
        conv_sum = sum(w * v for w, v in conv_parts) * conv_gate
        dense = (task_sum + conv_sum) / total_w
    else:
        dense = action_frac

    # Efficiency credit only for passing trajectories, so quitting early is never optimal.
    eff_cost, eff_detail = _efficiency_cost(calls, gold_actions, nl_history, judge_score)
    if strict >= 1.0 and EFF > 0.0:
        reward = 1.0 - EFF * eff_cost
    else:
        reward = strict + SHAPE * dense * (1.0 - strict)
    repeat_penalty, longest_streak = _repeat_penalty(calls)
    if repeat_penalty > 0.0:
        reward = max(0.0, reward - repeat_penalty)
    return {
        "score": float(reward),
        "strict": float(strict),
        "action_frac": float(action_frac),
        "name_frac": float(name_frac),
        "seq_frac": float(seq_frac),
        "communicate_frac": float(communicate_frac),
        "dense": float(dense),
        "judge": float(judge_score) if judge_score is not None else 0.0,
        "write_purity": 1.0 if purity_ok else 0.0,
        "bad_writes": float(bad_writes),
        "eff_cost": float(eff_cost),
        **{k: float(v) for k, v in eff_detail.items()},
        "n_calls": len(calls),
        "n_gold": len(gold_actions),
        "repeat_penalty": float(repeat_penalty),
        "longest_call_streak": longest_streak,
        **{k.lower(): float(v) for k, v in components.items()},
    }
