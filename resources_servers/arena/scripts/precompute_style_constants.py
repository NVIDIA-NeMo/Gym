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
"""Precompute style normalization constants and BT coefficients for arena-hard-auto datasets.

These constants are computed from a multi-model dataset and stored in the dataset's YAML
config file (e.g. configs/lmarena_260311.yaml).

Usage (from the nemo-gym root):

    python resources_servers/arena/scripts/precompute_style_constants.py \
        --arena_data_dir /path/to/arena-hard-auto/data/lmarena-260311 \
        --judge gemini-3.1-pro-preview

Outputs per-category YAML snippets ready to paste into the dataset config under the
`style_norm_mean`, `style_norm_std`, and `style_coefs` keys.

Design:
  - Normalization (mean/std) at judgment level with ddof=1: each question contributes once
    regardless of verdict strength, so constants are independent of judge/model quality.
  - Verdict weighting applied at battle level for BT coefficient estimation: strong verdicts
    (>>) contribute weight=3 rows, matching arena-hard-auto's methodology.
  - Contrast coding (+1/-1): matching arena-hard-auto's one_hot_encode, avoids ill-conditioned
    Hessian when one model wins near-100% of battles.
  - Bootstrap (--n_rounds, default 100): style coefficients are averaged over resampled fits.
    BFGS may report 'precision loss' with extreme model coefficients even when grad_norm~1e-7;
    this is a false alarm counted as failure only when grad_norm > 1e-3.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from resources_servers.arena.arena import (
    _AH_AUTO_LABEL_TO_VERDICT,
    _DEFAULT_VERDICT_WEIGHT,
    _bt_neg_ll_from_logits,
    _bt_neg_ll_grad,
    _raw_style_feature_from_counts,
    _weighted_scores_as_a,
    _weighted_scores_as_b,
)


def _metadata_counts(meta: dict) -> tuple[int, int, int, int]:
    """Flatten arena-hard-auto nested metadata into (token_len, headers, lists, bold)."""
    return (
        int(meta["token_len"]),
        int(sum(meta["header_count"].values())),
        int(sum(meta["list_count"].values())),
        int(sum(meta["bold_count"].values())),
    )


def _bfgs(X: np.ndarray, y: np.ndarray):
    """Fit a BT model via BFGS; return the scipy OptimizeResult (.x for coefficients, .jac for gradient norm)."""
    result = minimize(
        lambda beta: _bt_neg_ll_from_logits(X @ beta, y),
        x0=np.zeros(X.shape[1]),
        jac=lambda beta: _bt_neg_ll_grad(X, beta, y),
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-7},
    )
    return result


def main(
    arena_data_dir: str,
    judge: str | None,
    include_models: list[str] | None,
    baseline_models: list[str] | None,
    n_rounds: int,
    seed: int,
) -> None:
    data_dir = Path(arena_data_dir)
    weight = _DEFAULT_VERDICT_WEIGHT

    # ── Load model answer metadata ────────────────────────────────────────────
    all_model_meta: dict[str, dict[str, dict]] = {}
    for fpath in sorted((data_dir / "model_answer").glob("*.jsonl")):
        is_evaluated = include_models is None or fpath.stem in include_models
        is_baseline = baseline_models is None or fpath.stem in baseline_models
        if not (is_evaluated or is_baseline):
            continue
        with open(fpath) as f:
            rows = [json.loads(line) for line in f]
        all_model_meta[fpath.stem] = {r["uid"]: r["metadata"] for r in rows}

    loaded_baselines = [s for s in all_model_meta if baseline_models is None or s in baseline_models]
    loaded_evaluated = [s for s in all_model_meta if include_models is None or s in include_models]
    print(f"Baseline models loaded ({len(loaded_baselines)}): {sorted(loaded_baselines)}")
    print(f"Evaluated models loaded ({len(loaded_evaluated)}): {sorted(loaded_evaluated)}")
    for label, requested, loaded in [
        ("baseline", baseline_models, loaded_baselines),
        ("evaluated", include_models, loaded_evaluated),
    ]:
        if requested is not None:
            missing = set(requested) - set(loaded)
            if missing:
                print(f"WARNING: {label} models not found in model_answer/: {sorted(missing)}")

    # ── Select judgment files ─────────────────────────────────────────────────
    if judge is not None:
        judgment_dir = data_dir / "model_judgment" / judge
        if not judgment_dir.is_dir():
            available = sorted(p.name for p in (data_dir / "model_judgment").iterdir() if p.is_dir())
            raise FileNotFoundError(f"Judge not found: {judgment_dir}\nAvailable: {available}")
        judgment_files = sorted(judgment_dir.glob("*.jsonl"))
        print(f"Judge: {judge}  |  Judgment files: {len(judgment_files)}")
    else:
        judgment_files = sorted((data_dir / "model_judgment").rglob("*.jsonl"))
        print(f"No --judge filter — loading all {len(judgment_files)} judgment files")

    # ── Load judgments ────────────────────────────────────────────────────────
    raw_features: list[np.ndarray] = []
    scores_per_judgment: list[list[float]] = []
    model_per_judgment: list[str] = []
    baseline_per_judgment: list[str] = []
    categories_per_judgment: list[str | None] = []
    skipped = 0

    for jf in judgment_files:
        with open(jf) as f:
            for line in f:
                row = json.loads(line)
                model, uid = row["model"], row["uid"]
                if include_models is not None and model not in include_models:
                    continue
                games = row.get("games") or []
                if len(games) < 2 or games[0] is None or games[1] is None:
                    skipped += 1
                    continue
                s0, s1 = games[0].get("score"), games[1].get("score")
                if s0 not in _AH_AUTO_LABEL_TO_VERDICT or s1 not in _AH_AUTO_LABEL_TO_VERDICT:
                    skipped += 1
                    continue
                baseline_key = row.get("baseline", "baseline")
                if (
                    model not in all_model_meta
                    or uid not in all_model_meta[model]
                    or baseline_key not in all_model_meta
                    or uid not in all_model_meta[baseline_key]
                ):
                    skipped += 1
                    continue

                m = _metadata_counts(all_model_meta[model][uid])
                b = _metadata_counts(all_model_meta[baseline_key][uid])
                # Same scoring path as live evaluation: as_a for games[1], as_b for games[0].
                raw_features.append(_raw_style_feature_from_counts(*m, *b))
                scores_per_judgment.append(
                    _weighted_scores_as_a(_AH_AUTO_LABEL_TO_VERDICT[s1], weight)
                    + _weighted_scores_as_b(_AH_AUTO_LABEL_TO_VERDICT[s0], weight)
                )
                model_per_judgment.append(model)
                baseline_per_judgment.append(baseline_key)
                categories_per_judgment.append(row.get("category"))

    if include_models is not None:
        missing = set(include_models) - set(model_per_judgment)
        if missing:
            print(f"WARNING: no judgments found for: {sorted(missing)}")

    print(f"Valid judgments: {len(raw_features)}  (skipped: {skipped})")
    if not raw_features:
        raise RuntimeError("No valid judgments found. Check --arena_data_dir, --judge, and --include_models.")

    # ── Group by category ─────────────────────────────────────────────────────
    # Single-category datasets (e.g. lmarena-260311) are normalised to "default"
    # because NeMo-Gym input JSONL for such datasets has no category field, so
    # _get_style_constants(None) must find "default".
    raw_cats = {c or "default" for c in categories_per_judgment}
    if len(raw_cats) == 1:
        categories_per_judgment = ["default"] * len(categories_per_judgment)
        unique_cats = ["default"]
    else:
        unique_cats = sorted(raw_cats)
    print(f"Categories ({len(unique_cats)}): {unique_cats}")

    per_cat_norm_mean: dict[str, np.ndarray] = {}
    per_cat_norm_std: dict[str, np.ndarray] = {}
    per_cat_style_coefs: dict[str, np.ndarray] = {}

    for cat in unique_cats:
        cat_mask = [c == cat for c in categories_per_judgment]
        cat_raw_feats = np.array([f for f, m in zip(raw_features, cat_mask) if m])
        cat_scores = [s for s, m in zip(scores_per_judgment, cat_mask) if m]
        cat_models = [m for m, mask in zip(model_per_judgment, cat_mask) if mask]
        cat_baselines = [b for b, m in zip(baseline_per_judgment, cat_mask) if m]

        print(f"\n{'=' * 60}\nCategory: {cat!r}  |  Judgments: {len(cat_raw_feats)}")

        # Normalization at judgment level (ddof=1): each question contributes once
        # regardless of verdict strength.
        norm_mean = cat_raw_feats.mean(axis=0)
        norm_std = np.where((s := cat_raw_feats.std(axis=0, ddof=1)) < 1e-8, 1.0, s)
        per_cat_norm_mean[cat] = norm_mean
        per_cat_norm_std[cat] = norm_std
        print(f"Norm mean: {norm_mean}\nNorm std:  {norm_std}")

        # Explode to per-battle rows (verdict weighting applied here).
        z_feats = (cat_raw_feats - norm_mean) / norm_std
        battle_feats, battle_scores, battle_models, battle_baselines = [], [], [], []
        for i, (scores, model, baseline) in enumerate(zip(cat_scores, cat_models, cat_baselines)):
            for s in scores:
                battle_feats.append(z_feats[i])
                battle_scores.append(s)
                battle_models.append(model)
                battle_baselines.append(baseline)

        print(f"Battles (after verdict weighting): {len(battle_scores)}")

        # Contrast-coded design matrix (+1 evaluated model, -1 baseline).
        unique_models = sorted(set(battle_models) | set(battle_baselines))
        n_models = len(unique_models)
        midx = {m: i for i, m in enumerate(unique_models)}
        X_model = np.zeros((len(battle_models), n_models))
        for i, (m, b) in enumerate(zip(battle_models, battle_baselines)):
            X_model[i, midx[m]] = 1.0
            X_model[i, midx[b]] = -1.0
        X = np.concatenate([X_model, np.array(battle_feats)], axis=1)
        y = np.array(battle_scores)

        # Bootstrap BT fit — average style coefs across rounds.
        rng = np.random.default_rng(seed)
        N = len(y)
        all_coefs = np.zeros((n_rounds, 4))
        n_failures = 0
        print(f"Bootstrapping ({n_rounds} rounds, seed={seed})...")
        for r in range(n_rounds):
            idx = rng.integers(0, N, size=N)
            result = _bfgs(X[idx], y[idx])
            if np.linalg.norm(result.jac) > 1e-3:
                n_failures += 1
            all_coefs[r] = result.x[n_models:]

        if n_failures:
            print(f"WARNING: {n_failures}/{n_rounds} rounds did not converge (grad_norm > 1e-3).")

        coefs = all_coefs.mean(axis=0)
        coefs_std = all_coefs.std(axis=0)
        per_cat_style_coefs[cat] = coefs

        print(f"Style coefs (mean ± std over {n_rounds} rounds):")
        for dim, (c, s) in enumerate(zip(coefs, coefs_std)):
            print(f"  dim {dim}: {c:+.6f}  ±  {s:.6f}")

    # ── Output ────────────────────────────────────────────────────────────────
    def _fmt(d: dict[str, np.ndarray], name: str) -> str:
        lines = [f"{name}:"]
        for k, v in d.items():
            vals = ", ".join(f"{x:.6g}" for x in v)
            lines.append(f'  "{k}": [{vals}]')
        return "\n".join(lines)

    print("\n\n# ── Paste these into the dataset config YAML ────────────────────")
    print(_fmt(per_cat_norm_mean, "style_norm_mean"))
    print(_fmt(per_cat_norm_std, "style_norm_std"))
    print(_fmt(per_cat_style_coefs, "style_coefs"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arena_data_dir", required=True, help="Path to dataset directory (e.g. data/lmarena-260311.")
    parser.add_argument(
        "--judge",
        default=None,
        metavar="JUDGE",
        help="Judge subdirectory under model_judgment/. If omitted, all judges are used.",
    )
    parser.add_argument(
        "--include_models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Evaluated model names to include (stems of model_answer/*.jsonl). Default: all.",
    )
    parser.add_argument(
        "--baseline_models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Baseline model names to load. Rows whose baseline isn't loaded are skipped. Default: all.",
    )
    parser.add_argument("--n_rounds", type=int, default=100, metavar="N", help="Bootstrap rounds (default: 100).")
    parser.add_argument("--seed", type=int, default=42, metavar="SEED", help="Bootstrap random seed (default: 42).")
    args = parser.parse_args()
    main(args.arena_data_dir, args.judge, args.include_models, args.baseline_models, args.n_rounds, args.seed)
