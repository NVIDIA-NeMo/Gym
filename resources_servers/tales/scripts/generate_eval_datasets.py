# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Deterministically generate the TALES tiered eval datasets.

Emits three JSONL files under ``resources_servers/tales/data/``:

* ``smoke.jsonl``     — 5 rows (one per family), wiring-only, small step budget.
* ``eval_v0.jsonl``   — PROVISIONAL discriminative subset (~64 rows), re-selectable.
* ``eval_full.jsonl`` — every test-eligible task_no (95 rows at seed=0).

Design constraints (see resources_servers/tales/README.md "Eval protocol"):

* Split / test-membership is computed PROGRAMMATICALLY from the upstream
  tale-suite split machinery — never from hand-written task_no lists.
  - jericho:      game NOT in ``JERICHO_TRAIN_GAMES`` -> test  (27 test / 27 train).
  - scienceworld: split is env-enforced (app.py:87-88) -> rows MUST set split="test".
  - textworld / textworld_express / alfworld: the server ignores the row's split
    (silent no-op, app.py:71-74) and always draws from the upstream TEST pool
    (textworld TWCookingEnv defaults to the test cooking pool; twx hardcodes
    gameFold="test"; alfworld seed=0 lands on the upstream test game file). We
    still stamp split="test" for honest provenance.

* seed=0 for every eval row. seed selects CONTENT for textworld / twx / alfworld /
  scienceworld and is stochasticity-only for jericho. seed=0 is MANDATORY for
  alfworld: ALFWorldTask.reset picks gamefiles[seed % len]; only index 0 is the
  upstream TALES *test* file — any other seed crosses into the TALES *train* split.

* max_episode_steps=100 (paper protocol, arXiv:2504.14128), NOT the server's
  25 default. The agent-side cap (tales.yaml `max_steps`) must ALSO be raised to
  100 or the agent loop stops at 25 regardless of this field.

This script must run inside the server venv (scienceworld's task list is derived
from the installed library at runtime). It performs no network-free assumptions
about the scienceworld task set and no randomness (fully deterministic).

Usage:
    python resources_servers/tales/scripts/generate_eval_datasets.py \
        [--out-dir resources_servers/tales/data] [--n-seeds 1]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Any, Dict, List


# --- Protocol constants -----------------------------------------------------

# Byte-identical to the paper-parity system prompt in data/example.jsonl.
SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score. "
    "Upon reading the text observation, provide a *single* short phrase to interact with the game, "
    "e.g. `get lamp` (without the backticks). When stuck, try using the `help` command to see what "
    "commands are available."
)

AGENT_REF = {"type": "responses_api_agents", "name": "tales_gymnasium_agent"}

EVAL_SEED = 0  # test-safe, deterministic content selection
EVAL_MAX_STEPS = 100  # paper protocol
SMOKE_MAX_STEPS = 10  # wiring-only; keep episodes cheap

FAMILIES = ["textworld", "textworld_express", "alfworld", "scienceworld", "jericho"]

# --- Capability + difficulty metadata (STRUCTURAL / PRIOR) ------------------
# All difficulty tiers are PLACEHOLDER (structural) until the k>=5 sweep runs;
# eval_v0 membership is PROVISIONAL and re-selectable from eval_full on these
# same slicing fields without a schema change.

# textworld_express: env_name (without the "TWX" prefix) -> (capability, tier)
TWX_META = {
    "CookingWorld": ("situated_object_manipulation", "easy"),
    "TextWorldCommonsense": ("commonsense_object_manipulation", "easy"),
    "CoinCollector": ("spatial_navigation", "easy"),
    "Arithmetic": ("arithmetic", "easy"),
    "MapReader": ("spatial_navigation_mapping", "medium"),
    "Sorting": ("ordering_reasoning", "medium"),
    "SimonSays10": ("instruction_following", "easy"),
    "SimonSays50": ("instruction_following", "medium"),
    "SimonSays100": ("instruction_following_long_horizon", "hard"),
    "SimonSaysWithMemory10": ("memory", "easy"),
    "SimonSaysWithMemory50": ("memory_long_horizon", "medium"),
    # Paper inclusion gate: capable models >=90% on 100-step SimonSaysWithMemory.
    "SimonSaysWithMemory100": ("memory_long_horizon", "anchor_saturation"),
    "SimonSaysWithMemory10Verbose": ("memory", "easy"),
    "SimonSaysWithMemory50Verbose": ("memory_long_horizon", "medium"),
    "SimonSaysWithMemory100Verbose": ("memory_long_horizon", "hard"),
    "PeckingOrder": ("ordering_commonsense", "medium"),
}

# alfworld: task_type -> capability
ALFWORLD_CAP = {
    "pick_and_place_simple": "embodied_object_manipulation",
    "look_at_obj_in_light": "embodied_tool_use",
    "pick_clean_then_place_in_recep": "multi_step_embodied_planning",
    "pick_heat_then_place_in_recep": "multi_step_embodied_planning",
    "pick_cool_then_place_in_recep": "multi_step_embodied_planning",
    "pick_two_obj_and_place": "multi_object_embodied_planning",
}
ALFWORLD_TASK_TYPES = list(ALFWORLD_CAP.keys())  # canonical order (alfworld_data.TASK_TYPES)

# jericho games whose walkthroughs are heavily republished (Infocom/Adventure
# canon). Flagged as a distinct contamination slice, never silently pooled.
JERICHO_CANON = {
    "advent",
    "zork1",
    "zork2",
    "zork3",
    "ztuu",
    "planetfall",
    "trinity",
    "sorcerer",
    "spellbrkr",
    "enchanter",
    "seastalker",
    "cutthroat",
    "infidel",
    "hhgg",
    "sherlock",
    "wishbringer",
    "ballyhoo",
    "hollywood",
}


def _row(
    framework: str,
    task_no: int,
    *,
    split: str,
    seed: int,
    max_steps: int,
    capability: str,
    difficulty_tier: str,
    split_provenance: str,
    contamination_flag: str,
    raw_id: str,
    tier_membership: str,
) -> Dict[str, Any]:
    """Assemble one Responses-API dataset row.

    SLICING SURVIVAL — two mechanisms (belt + suspenders):
      (a) PRIMARY: slicing fields are mirrored into
          ``responses_create_params.metadata`` (str->str). ``metadata`` is a
          typed Dict[str,str] Responses-API field and the agent echoes the input
          ``responses_create_params`` verbatim into every rollout row
          (gymnasium_agent/app.py:88 base_body = ...model_copy(deep=True) -> :164
          returned as responses_create_params), so these keys ARE self-describing
          in the rollout output. Kept <=16 keys / short values to respect the
          OpenAI metadata limit some providers enforce.
      (b) FALLBACK: the same fields are ALSO top-level for a deterministic
          ``_ng_task_index`` (0-based input line number) -> input-row join. The
          top-level custom fields themselves are NOT echoed to rollout rows
          (MEASURED on example_rollouts.jsonl), but the row order is
          deterministic so the join always recovers them.
    The server's reset() reads only the top-level selector keys via model_extra;
    the metadata mirror is inert to the server.
    """
    # str->str metadata mirror (<=16 keys) for self-describing rollouts.
    meta = {
        "family": framework,
        "framework": framework,
        "task_no": str(task_no),
        "split": split,
        "seed": str(seed),
        "raw_id": raw_id,
        "capability": capability,
        "difficulty_tier": difficulty_tier,
        "contamination_flag": contamination_flag,
        "tier_membership": tier_membership,
    }
    return {
        # --- server selector fields (arrive as reset() metadata) ---
        "framework": framework,
        "task_no": task_no,
        "split": split,
        "seed": seed,
        "max_episode_steps": max_steps,
        # --- slicing metadata (top-level fallback; recover via _ng_task_index join) ---
        "family": framework,
        "raw_id": raw_id,
        "capability": capability,
        "difficulty_tier": difficulty_tier,
        "split_provenance": split_provenance,
        "contamination_flag": contamination_flag,
        "tier_membership": tier_membership,
        # --- model call (metadata mirror survives into rollout output) ---
        "responses_create_params": {
            "input": [{"role": "system", "content": SYSTEM_PROMPT}],
            "metadata": meta,
        },
        "agent_ref": dict(AGENT_REF),
    }


def build_tables() -> Dict[str, List[Dict[str, Any]]]:
    """Return, per family, the ordered list of task descriptors (test-eligible only).

    Every task_no is resolved from the live ``tales.<framework>.environments``
    list so indices match exactly what the server will load.
    """
    tables: Dict[str, List[Dict[str, Any]]] = {}

    # --- textworld: 10 TWCooking difficulties (all test pool) ---
    tw = importlib.import_module("tales.textworld")
    tw_rows = []
    for task_no, (env_name, _v) in enumerate(tw.environments):
        difficulty = task_no + 1  # TWCookingLevel{difficulty}
        if difficulty <= 3:
            tier = "easy"
        elif difficulty <= 7:
            tier = "medium"
        else:
            tier = "hard"
        cap = "situated_object_manipulation"
        if difficulty >= 5:
            cap = "situated_object_manipulation_navigation"
        tw_rows.append(
            {
                "task_no": task_no,
                "raw_id": env_name,
                "split": "test",
                "capability": cap,
                "difficulty_tier": tier,
                "split_provenance": "server_test_pool (textworld_data.py:42; split field is a no-op)",
                "contamination_flag": "none_procedural",
            }
        )
    tables["textworld"] = tw_rows

    # --- textworld_express: 16 tasks, gameFold hardcoded "test" ---
    twx = importlib.import_module("tales.textworld_express")
    twx_rows = []
    for task_no, (env_name, _v) in enumerate(twx.environments):
        key = env_name[len("TWX") :] if env_name.startswith("TWX") else env_name
        cap, tier = TWX_META.get(key, ("instruction_following", "medium"))
        twx_rows.append(
            {
                "task_no": task_no,
                "raw_id": env_name,
                "split": "test",
                "capability": cap,
                "difficulty_tier": tier,
                "split_provenance": "server_test_fold (twx_env.py:34 gameFold=test; split field is a no-op)",
                "contamination_flag": "none_procedural",
            }
        )
    tables["textworld_express"] = twx_rows

    # --- alfworld: 12 tasks (seen 0-5, unseen 6-11); seed=0 -> upstream test file ---
    alf = importlib.import_module("tales.alfworld")
    alf_rows = []
    for task_no, (env_name, _v) in enumerate(alf.environments):
        split_half = "seen" if task_no < len(ALFWORLD_TASK_TYPES) else "unseen"
        task_type = ALFWORLD_TASK_TYPES[task_no % len(ALFWORLD_TASK_TYPES)]
        cap = ALFWORLD_CAP[task_type]
        base_tier = "easy" if task_type in ("pick_and_place_simple", "look_at_obj_in_light") else "medium"
        # unseen == generalization axis; bump one notch harder (PRIOR: alfworld
        # is the hardest family for small models, many floor at 0.0).
        tier = base_tier if split_half == "seen" else ("medium" if base_tier == "easy" else "hard")
        alf_rows.append(
            {
                "task_no": task_no,
                "raw_id": f"{env_name} [{task_type}/{split_half}]",
                "split": "test",
                "capability": f"{cap}|generalization_{split_half}",
                "difficulty_tier": tier,
                "split_provenance": (
                    "upstream_test_file@seed0 (get_alfworld_env_splits first sorted file; "
                    "seed!=0 crosses into TALES train split)"
                ),
                "contamination_flag": "none_procedural",
            }
        )
    tables["alfworld"] = alf_rows

    # --- scienceworld: 30 tasks, split env-enforced -> test ---
    sw = importlib.import_module("tales.scienceworld")
    sw_rows = []
    for task_no, (env_name, _v) in enumerate(sw.environments):
        # raw task name recovered from the parallel TASK_NAMES list (same order
        # the package __init__ builds `environments`).
        raw = sw.TASK_NAMES[task_no]
        category = raw.split("-")[0]
        sw_rows.append(
            {
                "task_no": task_no,
                "raw_id": raw,
                "split": "test",
                "capability": f"scientific_procedure|{category}",
                "difficulty_tier": "medium",
                "split_provenance": "env_enforced_test (app.py:87-88 make_kwargs['split']='test')",
                "contamination_flag": "none_procedural",
            }
        )
    tables["scienceworld"] = sw_rows

    # --- jericho: 54 games; test = game NOT in JERICHO_TRAIN_GAMES ---
    jericho = importlib.import_module("tales.jericho")
    from tales.jericho.jericho_data import GAMES_INFOS, JERICHO_TRAIN_GAMES

    train_games = set(JERICHO_TRAIN_GAMES)
    game_names = list(GAMES_INFOS.keys())  # same order the server builds `environments`
    assert len(game_names) == len(jericho.environments), "jericho env/index misalignment"
    jer_rows = []  # test-eligible only (for smoke/eval_v0/eval_full)
    jer_all_rows = []  # all 54 (for paper_parity harness-parity tier)
    for task_no, game in enumerate(game_names):
        contam = "published_walkthrough_canonical" if game in JERICHO_CANON else "published_walkthrough"
        is_train = game in train_games
        jer_all_rows.append(
            {
                "task_no": task_no,
                "raw_id": game,
                "split": "test",  # split inert for jericho (app.py:71-74 no-op)
                "capability": "long_horizon_puzzle|exploration_mapping",
                "difficulty_tier": "hard_tail",
                "split_provenance": (
                    f"paper_parity_POOLED ({'train' if is_train else 'test'} game; "
                    "harness parity only — NOT for training-adjacent eval; jericho_data.py:11-39)"
                ),
                "contamination_flag": contam,
            }
        )
        if is_train:
            continue  # excluded from eval; reported separately as jericho_train slice
        jer_rows.append(
            {
                "task_no": task_no,
                "raw_id": game,
                "split": "test",
                "capability": "long_horizon_puzzle|exploration_mapping",
                "difficulty_tier": "hard_tail",
                "split_provenance": "upstream_split (game not in JERICHO_TRAIN_GAMES; jericho_data.py:11-39)",
                "contamination_flag": contam,
            }
        )
    tables["jericho"] = jer_rows
    tables["_jericho_all"] = jer_all_rows
    return tables


def _select_v0(tables: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[int]]:
    """Deterministic PROVISIONAL v0 selection. Returns family -> list of task_no.

    Weighted toward the built-in gradients and capability variety; anchors kept
    at both extremes. Re-run to re-select; no hand-written index lists.
    """
    sel: Dict[str, List[int]] = {}
    # Full gradients: keep every textworld difficulty, every twx task, every alfworld task.
    sel["textworld"] = [r["task_no"] for r in tables["textworld"]]  # 10
    sel["textworld_express"] = [r["task_no"] for r in tables["textworld_express"]]  # 16
    sel["alfworld"] = [r["task_no"] for r in tables["alfworld"]]  # 12

    # scienceworld: category-balanced 12 via round-robin over leading-token groups.
    sw = tables["scienceworld"]
    groups: Dict[str, List[int]] = {}
    for r in sorted(sw, key=lambda x: x["raw_id"]):
        groups.setdefault(r["raw_id"].split("-")[0], []).append(r["task_no"])
    picked: List[int] = []
    ordered_group_keys = sorted(groups.keys())
    i = 0
    while len(picked) < min(12, len(sw)):
        progressed = False
        for k in ordered_group_keys:
            if i < len(groups[k]):
                picked.append(groups[k][i])
                progressed = True
                if len(picked) >= min(12, len(sw)):
                    break
        if not progressed:
            break
        i += 1
    sel["scienceworld"] = sorted(picked)

    # jericho: 14 hard-tail games, evenly spaced across the sorted test set for
    # canonical/indie variety, with the two most-published anchors (advent, zork1)
    # force-included if present.
    jer = sorted(tables["jericho"], key=lambda x: x["raw_id"])
    target = min(14, len(jer))
    names = {r["raw_id"]: r["task_no"] for r in jer}
    anchors = [a for a in ("advent", "zork1") if a in names]
    non_anchor = [r for r in jer if r["raw_id"] not in anchors]
    m = target - len(anchors)
    if m <= 1:
        idxs = [0] if m == 1 else []
    else:
        idxs = sorted({round(i * (len(non_anchor) - 1) / (m - 1)) for i in range(m)})
    chosen = {names[a] for a in anchors} | {non_anchor[i]["task_no"] for i in idxs}
    sel["jericho"] = sorted(chosen)
    return sel


def _emit(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=os.path.join(here, "data"))
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="content-seed expansion for textworld/twx/scienceworld in eval_full "
        "(alfworld is always pinned to seed=0). Default 1 (seed=0 only).",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tables = build_tables()

    # ---- eval_full: every test-eligible task_no ----
    full_rows: List[Dict[str, Any]] = []
    for fam in FAMILIES:
        content_seeded = fam in ("textworld", "textworld_express", "scienceworld")
        seeds = list(range(args.n_seeds)) if content_seeded else [EVAL_SEED]
        for t in tables[fam]:
            for seed in seeds:
                full_rows.append(
                    _row(
                        fam,
                        t["task_no"],
                        split=t["split"],
                        seed=seed,
                        max_steps=EVAL_MAX_STEPS,
                        capability=t["capability"],
                        difficulty_tier=t["difficulty_tier"],
                        split_provenance=t["split_provenance"],
                        contamination_flag=t["contamination_flag"],
                        raw_id=t["raw_id"],
                        tier_membership="eval_full",
                    )
                )
    _emit(os.path.join(args.out_dir, "eval_full.jsonl"), full_rows)

    # ---- eval_v0: PROVISIONAL discriminative subset (seed=0 only) ----
    sel = _select_v0(tables)
    v0_rows: List[Dict[str, Any]] = []
    for fam in FAMILIES:
        by_no = {t["task_no"]: t for t in tables[fam]}
        for task_no in sel[fam]:
            t = by_no[task_no]
            v0_rows.append(
                _row(
                    fam,
                    task_no,
                    split=t["split"],
                    seed=EVAL_SEED,
                    max_steps=EVAL_MAX_STEPS,
                    capability=t["capability"],
                    difficulty_tier=t["difficulty_tier"],
                    split_provenance=t["split_provenance"],
                    contamination_flag=t["contamination_flag"],
                    raw_id=t["raw_id"],
                    tier_membership="eval_v0_provisional",
                )
            )
    _emit(os.path.join(args.out_dir, "eval_v0.jsonl"), v0_rows)

    # ---- smoke: one cheap row per family, small step budget ----
    smoke_rows: List[Dict[str, Any]] = []
    for fam in FAMILIES:
        t = tables[fam][0]  # first test-eligible task of each family
        smoke_rows.append(
            _row(
                fam,
                t["task_no"],
                split=t["split"],
                seed=EVAL_SEED,
                max_steps=SMOKE_MAX_STEPS,
                capability=t["capability"],
                difficulty_tier=t["difficulty_tier"],
                split_provenance=t["split_provenance"],
                contamination_flag=t["contamination_flag"],
                raw_id=t["raw_id"],
                tier_membership="smoke",
            )
        )
    _emit(os.path.join(args.out_dir, "smoke.jsonl"), smoke_rows)

    # ---- paper_parity: FULL per-family populations for arXiv:2504.14128 harness
    # parity (tw 10 + twx 16 + alfworld 12 + sw 30 + jericho ALL 54 = 122). The
    # paper has no train/test split and scores all 54 jericho games; this tier
    # POOLS jericho train+test and exists ONLY to reproduce the paper's numbers —
    # never for training-adjacent eval (use eval_full/eval_v0 for that). seed=0,
    # cap 100, same schema. ----
    parity_rows: List[Dict[str, Any]] = []
    parity_tables = {fam: tables[fam] for fam in ("textworld", "textworld_express", "alfworld", "scienceworld")}
    parity_tables["jericho"] = tables["_jericho_all"]  # all 54, pooled
    for fam in FAMILIES:
        for t in parity_tables[fam]:
            parity_rows.append(
                _row(
                    fam,
                    t["task_no"],
                    split=t["split"],
                    seed=EVAL_SEED,
                    max_steps=EVAL_MAX_STEPS,
                    capability=t["capability"],
                    difficulty_tier=t["difficulty_tier"],
                    split_provenance=t["split_provenance"],
                    contamination_flag=t["contamination_flag"],
                    raw_id=t["raw_id"],
                    tier_membership="paper_parity",
                )
            )
    _emit(os.path.join(args.out_dir, "paper_parity.jsonl"), parity_rows)

    # ---- summary (no silent caps: report every count + exclusions) ----
    jer_all = list(importlib.import_module("tales.jericho").environments)
    jer_train_excluded = len(jer_all) - len(tables["jericho"])
    print("=== TALES eval dataset generation ===")
    for fam in FAMILIES:
        print(f"  {fam:20s} test-eligible={len(tables[fam]):3d}  v0={len(sel[fam]):3d}")
    print(f"  eval_full rows   : {len(full_rows)} (n_seeds={args.n_seeds})")
    print(f"  eval_v0 rows     : {len(v0_rows)}")
    print(f"  smoke rows       : {len(smoke_rows)}")
    print(f"  paper_parity rows: {len(parity_rows)} (POOLS jericho train+test; harness parity only)")
    print(
        f"  EXCLUDED from eval : jericho_train={jer_train_excluded} games "
        f"(reported slice; pooled ONLY in paper_parity)"
    )


if __name__ == "__main__":
    main()
