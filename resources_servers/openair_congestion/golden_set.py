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

"""Derived-oracle golden set: an objective per-state action benchmark.

No public benchmark exists for this telco control task, so we derive one from
the environment itself. The replay backend is deterministic, so for any state
the best single intervention is *computable*: reconstruct the state, try every
action in a finite grid, and keep the one whose value (below) is highest. The
label is derived from the dynamics, not opined by a model -- any reviewer can
recompute it, which is the whole point.

Each golden row is a decision point drawn from the do-nothing (noop)
trajectory of a fixed evaluation seed -- "the network has been left alone for k steps
and is now congested; what is the best single action?" -- scored by
*episode-level value*: apply the candidate now, then coast (noop) to the end,
and sum the reward from the action onward. Coasting after the action, not just
its one-step reward, is what makes the margin meaningful in this environment,
where a single control move has a small immediate delta but a compounding
effect over the horizon. Each row carries:

    golden action + value, the noop (all-coast) value, and their margin.

Margin is the opportunity: how much the best single action beats inaction
here. Near-zero margin means the state is not a real decision point, so
evaluation focuses on the positive-margin subset. The set spans all five
congestion regimes, which also surfaces *where* actions matter. Treat the
regime breakdown from each regenerated benchmark as the evidence for that run.

Scoring a policy is one intervention: at each golden state, apply the policy's
action, coast to the end, and measure how much of the golden margin it
recovers, `(policy_value - noop_value) / (golden_value - noop_value)`, clipped
to [0, 1]. The validity checks are intentionally minimal: an oracle that
replays the golden action recovers ~all the margin and noop recovers none.
The hand-written relief rule is reported descriptively, not used as a gate.

This is a v0: best-single-intervention value over a finite action grid, not
multi-step optimal control, on a fixed evaluation seed band. This contribution
does not establish that those seeds are disjoint from an external training run.

Usage:
    python resources_servers/openair_congestion/golden_set.py
    python resources_servers/openair_congestion/golden_set.py --out golden_set_v0.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from openair_congestion import render
from openair_congestion.schemas import Observation, ToolCall

from resources_servers.openair_congestion.backends import ReplayBackend


# Fixed evaluation seed band. Difficulty is high so the sampled states are
# congested and the decisions actually matter. External training workflows
# must enforce their own disjoint train/evaluation split.
_GOLDEN_SEEDS = (33001, 33002, 33003, 33004, 33005, 33006, 33007, 33008)
_GOLDEN_DIFFICULTY = 0.9
# The set spans every congestion regime, so the benchmark covers the scenario
# space and reveals where control actions matter.
_GOLDEN_REGIMES = ("prb_exhaustion", "bursty", "interference", "prach_storm", "qos_competition")
# Steps of inaction before we ask "what now?"; deeper prefixes are more
# congested. One row per (regime, seed, prefix).
_DECISION_STEPS = (3, 6, 9)
_MAX_STEPS = 16
# Actions barely better than coasting are not real decision points.
_OPPORTUNITY_MARGIN = 0.05

_NOOP = ToolCall(name="noop", arguments={})
_SCHEDULERS = ("PF", "RR", "MaxCI")


def _candidate_grid(obs: Observation) -> list[ToolCall]:
    """A finite, deterministic set of valid tool calls for this state.

    Small on purpose -- a representative discretization of each tool family
    over the observed topology, not the full continuous space -- so the
    brute-force argmax is exhaustive and reproducible. noop is included, so
    the golden action can be "do nothing" when acting does not help.
    """
    grid: list[ToolCall] = [_NOOP]
    for cell in obs.cells:
        cid = cell.cell_id
        for policy in _SCHEDULERS:
            grid.append(ToolCall(name="set_scheduler_policy", arguments={"cell_id": cid, "policy": policy}))
        for p0 in (-95, -90, -85, -80):
            grid.append(ToolCall(name="set_ul_power_control", arguments={"cell_id": cid, "p0_dbm": p0, "alpha": 0.8}))
        for mcs_max in (14, 28):
            grid.append(
                ToolCall(
                    name="set_mcs_bounds",
                    arguments={"cell_id": cid, "mcs_min": 0, "mcs_max": mcs_max, "target_bler": 0.1},
                )
            )
        for accept in (50, 100):
            grid.append(
                ToolCall(
                    name="set_admission_policy",
                    arguments={"cell_id": cid, "accept_threshold_pct": accept, "slice_reservation": {}},
                )
            )
        for ue in cell.ues:
            for max_prb in (50, 137, 273):
                grid.append(
                    ToolCall(
                        name="set_prb_cap",
                        arguments={"cell_id": cid, "target": "ue", "target_id": ue.ue_id, "max_prb": max_prb},
                    )
                )
    return grid


def _state_hash(obs: Observation) -> str:
    # Identity of the decision-point state: the rendered text the policy sees,
    # minus the per-episode id/time line that render prepends.
    text = render.to_user_text(obs)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _task_params(regime: str, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "difficulty": _GOLDEN_DIFFICULTY,
        "max_steps": _MAX_STEPS,
        "regime_mix": {regime: 1.0},
        "scenario_id": regime,
    }


def _reconstruct(backend: ReplayBackend, regime: str, seed: int, prefix_len: int) -> tuple[str, Observation]:
    """Reset seed and replay `prefix_len` noops; return the live episode + obs.

    Deterministic, so this always reproduces the same state for a given
    (regime, seed, prefix_len) -- that is what makes the golden label
    recomputable.
    """
    obs, meta = backend.reset(_task_params(regime, seed))
    for _ in range(prefix_len):
        obs, _reward, _done, _info = backend.step(meta.episode_id, _NOOP)
    return meta.episode_id, obs


def _value_of(backend: ReplayBackend, regime: str, seed: int, prefix_len: int, action: ToolCall) -> float:
    # Episode-level value of one intervention: reconstruct the exact
    # decision-point state, apply `action`, then coast (noop) to the episode
    # end, summing reward from the action onward. A fresh episode per call
    # keeps the canonical state untouched, so candidates are scored from the
    # identical state.
    episode_id, _obs = _reconstruct(backend, regime, seed, prefix_len)
    try:
        _next, reward, done, _info = backend.step(episode_id, action)
        total, step_idx = float(reward), prefix_len + 1
        while not done and step_idx < _MAX_STEPS:
            _obs, reward, done, _info = backend.step(episode_id, _NOOP)
            total += float(reward)
            step_idx += 1
        return total
    finally:
        backend.close(episode_id)


@dataclass
class GoldenRow:
    regime: str
    seed: int
    prefix_len: int
    state_hash: str
    golden_action: dict[str, Any]
    golden_value: float
    noop_value: float
    margin: float
    n_candidates: int
    ranked: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "seed": self.seed,
            "prefix_len": self.prefix_len,
            "state_hash": self.state_hash,
            "golden_action": self.golden_action,
            "golden_value": round(self.golden_value, 5),
            "noop_value": round(self.noop_value, 5),
            "margin": round(self.margin, 5),
            "is_opportunity": self.margin > _OPPORTUNITY_MARGIN,
            "n_candidates": self.n_candidates,
            "top_actions": self.ranked,
        }


def generate_golden_set(
    seeds: tuple[int, ...] = _GOLDEN_SEEDS,
    regimes: tuple[str, ...] = _GOLDEN_REGIMES,
    decision_steps: tuple[int, ...] = _DECISION_STEPS,
) -> list[GoldenRow]:
    backend = ReplayBackend(pool_size=8, max_steps_default=_MAX_STEPS)
    rows: list[GoldenRow] = []
    for regime in regimes:
        for seed in seeds:
            for prefix_len in decision_steps:
                episode_id, state = _reconstruct(backend, regime, seed, prefix_len)
                try:
                    candidates = _candidate_grid(state)
                    scored = [(action, _value_of(backend, regime, seed, prefix_len, action)) for action in candidates]
                    scored.sort(key=lambda ar: ar[1], reverse=True)
                    golden_action, golden_value = scored[0]
                    noop_value = next(v for a, v in scored if a.name == "noop")
                    rows.append(
                        GoldenRow(
                            regime=regime,
                            seed=seed,
                            prefix_len=prefix_len,
                            state_hash=_state_hash(state),
                            golden_action={"name": golden_action.name, "arguments": golden_action.arguments},
                            golden_value=golden_value,
                            noop_value=noop_value,
                            margin=golden_value - noop_value,
                            n_candidates=len(candidates),
                            ranked=[
                                {"name": a.name, "arguments": a.arguments, "value": round(v, 5)} for a, v in scored[:3]
                            ],
                        )
                    )
                finally:
                    backend.close(episode_id)
    return rows


# --- Scoring a policy against the golden set ---------------------------------
# A policy maps a rendered observation to a tool call dict. Scored one-step:
# apply the policy's action at each golden state and measure margin recovered.

PolicyFn = Callable[[str], dict[str, Any]]


def _recovery(policy_value: float, noop_value: float, golden_value: float) -> float:
    span = golden_value - noop_value
    if span <= 0:
        return 1.0  # nothing to recover: any action ties the (already optimal) noop
    return max(0.0, min(1.0, (policy_value - noop_value) / span))


def score_policy_against_golden(rows: list[GoldenRow], policy: PolicyFn, *, opportunities_only: bool = True) -> dict:
    backend = ReplayBackend(pool_size=8, max_steps_default=_MAX_STEPS)
    exact_hits, recoveries, scored_rows = 0, [], 0
    for row in rows:
        if opportunities_only and row.margin <= _OPPORTUNITY_MARGIN:
            continue
        episode_id, state = _reconstruct(backend, row.regime, row.seed, row.prefix_len)
        try:
            chosen = policy(render.to_user_text(state))
            value = _value_of(
                backend,
                row.regime,
                row.seed,
                row.prefix_len,
                ToolCall(name=chosen["name"], arguments=chosen["arguments"]),
            )
            scored_rows += 1
            if chosen["name"] == row.golden_action["name"] and chosen["arguments"] == row.golden_action["arguments"]:
                exact_hits += 1
            recoveries.append(_recovery(value, row.noop_value, row.golden_value))
        finally:
            backend.close(episode_id)
    return {
        "scored_rows": scored_rows,
        "exact_match_rate": round(exact_hits / scored_rows, 4) if scored_rows else 0.0,
        "mean_margin_recovered": round(sum(recoveries) / len(recoveries), 4) if recoveries else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="write the golden set as JSONL to this path")
    parser.add_argument("--no-validate", action="store_true", help="skip the anchor-policy validation pass")
    args = parser.parse_args()

    rows = generate_golden_set()
    opportunities = [r for r in rows if r.margin > _OPPORTUNITY_MARGIN]
    print(f"\ngolden set v0: {len(rows)} states, {len(opportunities)} opportunities (margin > {_OPPORTUNITY_MARGIN})")
    print(f"mean margin over opportunities: {sum(r.margin for r in opportunities) / max(1, len(opportunities)):.4f}")
    print(f"\n{'regime':<18} {'states':>7} {'opportunities':>14} {'max margin':>12}")
    print("-" * 53)
    for regime in _GOLDEN_REGIMES:
        rr = [r for r in rows if r.regime == regime]
        opp = [r for r in rr if r.margin > _OPPORTUNITY_MARGIN]
        top = max((r.margin for r in rr), default=0.0)
        print(f"{regime:<18} {len(rr):>7} {len(opp):>14} {top:>12.3f}")
    print("\n(opportunities are states where one action beats coasting)\n")

    if args.out:
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row.to_json()) + "\n")
        print(f"golden set written to {args.out}\n")

    if not args.no_validate:
        # Reuse the small offline client heuristic; hosted-model evaluation is
        # intentionally outside this deterministic benchmark.
        from resources_servers.openair_congestion.client import choose_action

        # The benchmark's validity claims are only that the oracle recovers
        # the margin and noop does not. The scripted relief rule is reported
        # as a regime-specific finding rather than an ordering gate.
        anchors: dict[str, PolicyFn] = {
            "oracle": _oracle_policy(rows),
            "scripted": lambda obs, _c=choose_action: _c(obs, 0),
            "noop": lambda obs: {"name": "noop", "arguments": {}},
        }
        print(f"{'policy':<16} {'exact-match':>12} {'margin-recovered':>18}")
        print("-" * 48)
        recovered = {}
        for label, policy in anchors.items():
            result = score_policy_against_golden(rows, policy)
            recovered[label] = result["mean_margin_recovered"]
            print(f"{label:<16} {result['exact_match_rate']:>12.3f} {result['mean_margin_recovered']:>18.3f}")

        # Per-regime relief recovery: the finding that the hand-written relief
        # rule is regime-specific -- an argument for learning the policy.
        print("\nscripted recovery by regime (the rule is tuned for PRB exhaustion):")
        for regime in _GOLDEN_REGIMES:
            regime_rows = [r for r in rows if r.regime == regime and r.margin > _OPPORTUNITY_MARGIN]
            if not regime_rows:
                continue
            r = score_policy_against_golden(regime_rows, anchors["scripted"])
            print(f"  {regime:<18} {r['mean_margin_recovered']:>6.3f}  ({r['scored_rows']} opportunities)")

        ok = recovered["oracle"] > 0.95 and recovered["noop"] < 0.01
        print(f"\nbenchmark validity (oracle recovers ~1, noop ~0): {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(1)


def _oracle_policy(rows: list[GoldenRow]) -> PolicyFn:
    # Looks up the golden action by state hash: a policy that always plays it
    # must recover the full margin, which validates the scoring harness.
    by_hash = {row.state_hash: row.golden_action for row in rows}

    def _fn(obs: str) -> dict[str, Any]:
        return by_hash.get(hashlib.sha256(obs.encode()).hexdigest()[:16], {"name": "noop", "arguments": {}})

    return _fn


if __name__ == "__main__":
    main()
