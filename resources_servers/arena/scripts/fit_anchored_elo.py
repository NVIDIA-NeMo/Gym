#!/usr/bin/env python3
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
"""Fit one model's Elo while holding its opponents at fixed Elo values.

The dataset must identify each battle, its opponent, and its human outcome.
Both the analysis schema (``uid``, ``opponent_model``, ``human_winner``) and
validation schema (``question_id``, ``baseline_model``, ``winner``) are accepted.

Outcome mapping
---------------
- Human labels per battle (1 observation/battle):
    policy/other      -> 1
    opponent/baseline -> 0
    tie      -> 0.5
    both_bad -> excluded in lmarena_v2, 0.5 in lmarena_v3
- Judge verdicts per game (2 observations/battle from position swap):
    [[A>>B]], [[A>B]] (A side better) -> 1 if policy was A else 0
    [[A=B]]                            -> 0.5
    [[B>A]], [[B>>A]]                  -> 1 if policy was B else 0
    [[BB]]                             -> excluded in lmarena_v2, 0.5 in lmarena_v3

Only battles whose opponent has a fixed Elo value are used. Bootstrap samples
individual observations.
"""

import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar


VERDICT_TO_A_SCORE = {
    "A>>B": 1.0,
    "A>B": 1.0,
    "A=B": 0.5,
    "tie": 0.5,
    "B>A": 0.0,
    "B>>A": 0.0,
}


def load_fixed_elos(path: str) -> dict[str, float]:
    """Load `{model: Elo}`, `{"overall": {model: Elo}}`, or model/rating rows from JSON."""
    with open(path) as f:
        data = json.load(f)
    if "overall" in data:
        data = data["overall"]
    if isinstance(data, dict):
        return {model: float(elo) for model, elo in data.items()}
    return {row["model"]: float(row["rating"]) for row in data}


def load_dataset(path: str) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r.get("uid") or r["question_id"]] = r
    return out


def load_verdicts(path: str) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r.get("uid") or r["question_id"]] = r
    return out


def neg_loglik(theta_policy: float, opp_ratings: np.ndarray, scores: np.ndarray) -> float:
    # p = 1 / (1 + 10**((opp - policy)/400))
    diff = (opp_ratings - theta_policy) / 400.0
    log10_inv = np.log(10.0) * diff  # log(1 + 10^x) requires safe form
    # log p = -log(1 + 10^diff); log(1-p) = diff*ln10 - log(1 + 10^diff)
    # Use logsumexp-style stable form for log(1+10^x) = log(1+e^(x*ln10))
    x = log10_inv
    log1p_ex = np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    log_p = -log1p_ex
    log_1mp = x - log1p_ex
    ll = scores * log_p + (1.0 - scores) * log_1mp
    return -ll.sum()


def fit_policy_elo(opp_ratings: np.ndarray, scores: np.ndarray) -> float:
    # Fit only the evaluated model; opponent Elo values remain fixed.
    res = minimize_scalar(
        neg_loglik,
        args=(opp_ratings, scores),
        bounds=(0.0, 3000.0),
        method="bounded",
        options={"xatol": 1e-3},
    )
    return float(res.x)


def bootstrap_ci(
    opp_ratings: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
):
    # Resample individual observations, as in proxy scoring.
    rng = np.random.default_rng(seed)
    n = len(scores)
    fits = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fits.append(fit_policy_elo(opp_ratings[idx], scores[idx]))
    fits = np.array(fits)
    return float(np.percentile(fits, 2.5)), float(np.percentile(fits, 97.5))


def build_observations(
    dataset: dict,
    verdicts: dict,
    fixed_elos: dict[str, float],
    label_source: str,
    benchmark: str,
):
    """Build policy scores against opponents with fixed Elo values."""
    opp_ratings = []
    scores = []
    n_no_fixed_elo = 0
    n_no_verdict = 0
    n_no_human = 0
    n_used = 0

    for uid, row in dataset.items():
        opp = row.get("opponent_model") or row["baseline_model"]
        if opp not in fixed_elos:
            n_no_fixed_elo += 1
            continue
        opp_r = fixed_elos[opp]

        # Human battles contribute one policy score per opponent.
        if label_source == "human":
            hw = row.get("human_winner") or row.get("winner")
            if hw in ("policy", "other"):
                s = 1.0
            elif hw in ("opponent", "baseline"):
                s = 0.0
            elif hw == "tie" or (hw == "both_bad" and benchmark == "lmarena_v3"):
                s = 0.5
            else:
                n_no_human += 1
                continue
            opp_ratings.append(opp_r)
            scores.append(s)
            n_used += 1
        else:
            v = verdicts.get(uid)
            if v is None:
                n_no_verdict += 1
                continue
            # Accept current NeMo Gym games or the older game0/game1 format.
            games = v.get("games") or [v.get("game0"), v.get("game1")]
            if len(games) != 2:
                n_no_verdict += 1
                continue
            battle_scores = []
            for game_index, g in enumerate(games):
                if not g or g.get("verdict") is None:
                    break
                verdict = g["verdict"].strip("[]")
                a_score = 0.5 if verdict == "BB" and benchmark == "lmarena_v3" else VERDICT_TO_A_SCORE.get(verdict)
                if a_score is None:
                    break
                # The second game swaps the policy and opponent positions.
                policy_is_a = game_index == 0
                battle_scores.append(a_score if policy_is_a else (1.0 - a_score))
            if len(battle_scores) != 2:
                n_no_verdict += 1
                continue
            opp_ratings.extend([opp_r] * len(battle_scores))
            scores.extend(battle_scores)
            n_used += 1

    return (
        np.array(opp_ratings),
        np.array(scores),
        n_used,
        n_no_fixed_elo,
        n_no_verdict,
        n_no_human,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--elo-file", required=True, help="JSON containing fixed Elo values")
    p.add_argument("--dataset", required=True, help="Battle dataset JSONL")
    p.add_argument("--verdicts", help="judge verdicts JSONL (omit for human-only fit)")
    p.add_argument("--policy-name", required=True)
    p.add_argument("--label", choices=["human", "judge"], required=True)
    p.add_argument("--version", choices=["lmarena_v2", "lmarena_v3"], required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    fixed_elos = load_fixed_elos(args.elo_file)
    ds = load_dataset(args.dataset)
    vd = load_verdicts(args.verdicts) if args.verdicts else {}

    opp_r, scores, n_used, n_no_fixed_elo, n_no_v, n_no_h = build_observations(
        ds, vd, fixed_elos, args.label, args.version
    )
    n_obs = len(scores)
    if n_obs == 0:
        print("no observations after filtering — aborting")
        return

    elo = fit_policy_elo(opp_r, scores)
    ci_lo, ci_hi = bootstrap_ci(opp_r, scores, n_boot=args.bootstrap, seed=args.seed)

    print(f"policy:          {args.policy_name}")
    print(f"label source:    {args.label}")
    print(f"battles used:    {n_used}")
    print(f"observations:    {n_obs}")
    print(f"  dropped (opponent has no fixed Elo): {n_no_fixed_elo}")
    if args.label == "judge":
        print(f"  dropped (no verdict for battle):  {n_no_v}")
    else:
        print(f"  dropped (no human label):         {n_no_h}")
    print(f"mean score:      {scores.mean():.4f}")
    print(f"fitted Elo:      {elo:.1f}  [{ci_lo:.1f}, {ci_hi:.1f}]  (95% boot, n_boot={args.bootstrap})")

    # JSON dump for easy collection
    print("---JSON---")
    print(
        json.dumps(
            {
                "policy": args.policy_name,
                "label": args.label,
                "n_battles": n_used,
                "n_obs": n_obs,
                "n_dropped_no_fixed_elo": n_no_fixed_elo,
                "n_dropped_no_verdict": n_no_v,
                "n_dropped_no_human": n_no_h,
                "mean_score": float(scores.mean()),
                "elo": elo,
                "elo_ci_lo": ci_lo,
                "elo_ci_hi": ci_hi,
            }
        )
    )


if __name__ == "__main__":
    main()
