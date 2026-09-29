# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Pure building blocks for multi-stage adaptive ELO estimation on GDPVal.

Instead of comparing the evaluated model against every reference model on all
tasks, multi-stage ELO runs a sequence of *stages*. Each stage:

1. Samples ``T`` tasks from a task-distribution JSON file (see
   ``responses_api_agents.stirrup_agent.task_distribution``); ``T`` is
   configurable per stage and **defaults to the full task set**,
2. Includes a set of ``M`` reference models and assigns **each task a single
   reference** drawn uniformly (equal weight) from that set — so a task is
   judged against one reference, not all ``M``,
3. Fits an anchored Bradley-Terry MLE ELO from that stage's win/loss/tie
   battles pooled per reference (reusing ``comparison.calculate_mle_elo``), and
4. Uses that estimate to choose the ``M`` references for the next stage.

Across stages, ``M`` typically shrinks (zooming in on references whose known
ELO is closest to the evaluated model's current estimate) while ``T`` grows (up
to the full task set) — spending the judge budget on a tighter final estimate.
Because each task's deliverable is reference-independent, a deliverable is
produced once and reused when its task recurs in a later stage — that stage only
re-judges it against its freshly assigned reference.

This module is intentionally **pure / server-agnostic** — reference selection,
per-task reference assignment, ELO fitting, vote pooling, and distribution
loading, with no server or rollout I/O. The orchestration that wires these into
Gym's standard rollout-collection flow (the single supported entry point) lives
in ``multistage_orchestrator`` and is enabled with ``++multistage.enabled=true``.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from resources_servers.gdpval.comparison import calculate_mle_elo


# A mapping ``ref_id -> {"wins": int, "losses": int, "ties": int,
# "reference_elo": float}`` as produced (per task, then pooled) by the GDPVal
# comparison verifier. This is the unit the ELO MLE is fit over.
PerReferenceTotals = Dict[str, Dict[str, float]]


@dataclass
class StageSpec:
    """Configuration for a single stage.

    ``num_tasks`` is ``T`` (the number of tasks sampled from the distribution and
    judged this stage). ``None`` (the default) means **the full task set** — every
    task in the distribution (e.g. all 220 GDPVal tasks). ``num_models`` is ``M``
    (the number of reference models included this stage); ``None`` means "all
    available references" (used for the first, broad stage). Each task is judged
    against **one** reference sampled uniformly from the included set. ``seed``
    makes this stage's task sampling and per-task reference assignment
    reproducible.
    """

    num_tasks: Optional[int] = None
    num_models: Optional[int] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Reference selection
# ---------------------------------------------------------------------------


def select_references(
    reference_elos: Mapping[str, float],
    eval_elo: Optional[float],
    num_models: Optional[int],
) -> List[str]:
    """Choose reference ids for a stage.

    Returns all references (sorted by id) when ``num_models`` is ``None`` or the
    estimate is not yet available (the first, broad stage). Otherwise returns the
    ``num_models`` references whose anchor ELO is closest to ``eval_elo``, ties
    broken by ``ref_id`` for determinism.
    """
    all_ids = sorted(reference_elos)
    if num_models is None or eval_elo is None or num_models >= len(all_ids):
        return all_ids
    if num_models <= 0:
        return []
    ranked = sorted(all_ids, key=lambda rid: (abs(reference_elos[rid] - eval_elo), rid))
    chosen = ranked[:num_models]
    # Return in stable id order rather than distance order for readable output.
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Per-task reference assignment
# ---------------------------------------------------------------------------


def all_task_ids(distribution: Mapping[str, Mapping[str, object]]) -> List[str]:
    """Every task id in the distribution, de-duplicated, in a stable order.

    Used to size the default (full) task set and, in nested sampling, as the
    total available count. Order is stable (distribution group order, then task
    order within each group).
    """
    ids: List[str] = []
    seen: set = set()
    for group in distribution.values():
        for tid in (group or {}).get("task_ids", []) or []:
            tid_str = str(tid)
            if tid_str not in seen:
                seen.add(tid_str)
                ids.append(tid_str)
    return ids


def stage_assignment_rng(seed: Optional[int], stage_seed: Optional[int], stage_index: int) -> random.Random:
    """Seed a reproducible RNG for a stage's per-task reference assignment.

    Derives the seed from the run seed, the stage's own ``seed``, and the stage
    index, so each stage draws an independent-but-reproducible assignment. Using
    a dedicated RNG (rather than a shared, state-advancing one) means a resumed
    stage recomputes exactly the same assignment it did originally.
    """
    payload = "|".join([repr(seed), repr(stage_seed), str(stage_index)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def assign_task_references(
    task_ids: Sequence[str],
    reference_ids: Sequence[str],
    *,
    rng: random.Random,
) -> Dict[str, str]:
    """Assign each task a single reference model, sampled with equal probability.

    Rather than judging every task against every included reference, each task is
    compared against **one** reference drawn uniformly (each included reference
    weighted equally) from ``reference_ids``. Returns a ``{task_id: ref_id}`` map;
    empty when no references are included.

    Deterministic given *rng*, so a stage's assignment replays identically on
    resume (it is also recorded in the stage journal).
    """
    refs = list(reference_ids)
    if not refs:
        return {}
    return {str(tid): rng.choice(refs) for tid in task_ids}


# ---------------------------------------------------------------------------
# Task planning
# ---------------------------------------------------------------------------


def plan_stage_task_ids(
    distribution: Mapping[str, Mapping[str, object]],
    stages: Sequence[StageSpec],
    *,
    rng: Optional[random.Random] = None,
    nested: bool = False,
) -> List[List[str]]:
    """Pre-sample the ``T`` tasks for every stage from a task distribution.

    Task selection is independent of any ELO estimate, so all stages' task sets
    can be planned up front. A stage's ``num_tasks`` (``T``) is the target count;
    ``None`` (the default) or a value ``>=`` the total means **the full task
    set**. ``T`` is always capped at the number of available tasks.

    ``nested=False`` (the default) samples each stage independently, honoring its
    own ``seed``.

    ``nested=True`` instead makes each stage's set a superset of the previous one.
    We get this for free in a single draw: ``sample_task_ids`` samples without
    replacement one task at a time, so a prefix of a large draw is identical to a
    smaller draw made with the same RNG. We therefore draw once, sized to the
    largest stage, and slice each stage's prefix from it — O(max T) work and
    exactly proportional per stage, with nesting guaranteed. A single shared RNG
    is used (per-stage ``seed`` only applies to independent sampling).
    """
    from responses_api_agents.stirrup_agent.task_distribution import sample_task_ids

    base_rng = rng or random.Random()
    total = len(all_task_ids(distribution))

    def _target(spec: StageSpec) -> int:
        # num_tasks=None (default) -> the full task set; always capped at total.
        return total if spec.num_tasks is None else min(spec.num_tasks, total)

    if not nested:
        return [
            sample_task_ids(
                distribution,
                _target(s),
                rng=random.Random(s.seed) if s.seed is not None else base_rng,
            )
            for s in stages
        ]

    max_target = max((_target(s) for s in stages), default=0)
    ordered = sample_task_ids(distribution, max_target, rng=base_rng)
    return [list(ordered[: _target(s)]) for s in stages]


# ---------------------------------------------------------------------------
# ELO fitting
# ---------------------------------------------------------------------------


def fit_stage_elo(
    per_reference: Mapping[str, Mapping[str, float]],
    reference_elos: Mapping[str, float],
) -> tuple[Optional[float], Optional[float], int]:
    """Fit the eval model's ELO for a stage from per-reference battle totals.

    A reference is included in the fit only if it has a known anchor ELO (from
    ``reference_elos`` or a ``reference_elo`` recorded on its counts) and at
    least one judged game (win + loss + tie > 0).

    Returns ``(elo, normalized_elo, num_references)``:
    - ``num_references`` is how many references met both criteria above and were
      passed to the MLE.
    - ``elo`` / ``normalized_elo`` are ``None`` when no reference qualified
      (``num_references == 0``) or when the MLE itself could not produce a rating;
      in the latter case ``num_references`` is still > 0.
    """
    battles: List[tuple[float, float, float, float]] = []
    for ref_id, counts in per_reference.items():
        ref_elo = reference_elos.get(ref_id, counts.get("reference_elo"))
        if ref_elo is None:
            continue
        wins = float(counts.get("wins", 0) or 0)
        losses = float(counts.get("losses", 0) or 0)
        ties = float(counts.get("ties", 0) or 0)
        if wins + losses + ties <= 0:
            continue
        battles.append((float(ref_elo), wins, losses, ties))

    if not battles:
        return None, None, 0

    mle = calculate_mle_elo(battles)
    if mle is None:
        return None, None, len(battles)
    elo, normalized = mle
    return elo, normalized, len(battles)


# ---------------------------------------------------------------------------
# Vote pooling
# ---------------------------------------------------------------------------


def pool_per_reference(verify_responses: Sequence[Mapping[str, Any]]) -> PerReferenceTotals:
    """Sum ``per_reference`` win/loss/tie counts across many verify responses."""
    totals: PerReferenceTotals = {}
    for vr in verify_responses:
        per_ref = vr.get("per_reference") or {}
        for ref_id, counts in per_ref.items():
            entry = totals.setdefault(ref_id, {"wins": 0, "losses": 0, "ties": 0, "reference_elo": None})
            entry["wins"] += int(counts.get("wins", 0) or 0)
            entry["losses"] += int(counts.get("losses", 0) or 0)
            entry["ties"] += int(counts.get("ties", 0) or 0)
            if entry["reference_elo"] is None:
                entry["reference_elo"] = counts.get("reference_elo")

    return totals


# ---------------------------------------------------------------------------
# Distribution loading
# ---------------------------------------------------------------------------


# Default location for distributions built on demand. Lives under the resources
# server's data dir so it is easy to inspect/reuse across runs.
DEFAULT_DISTRIBUTION_CACHE_DIR = Path(__file__).resolve().parent / "data" / "distributions"


def load_distribution(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load a task-distribution JSON file produced by ``task_distribution.py``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Distribution file {path} must be a JSON object.")
    return data


def ensure_distribution(
    distribution_path: Optional[str | Path] = None,
    *,
    dataset_path: Optional[str | Path] = None,
    columns: Optional[Sequence[str]] = None,
    cache_dir: Optional[str | Path] = None,
) -> tuple[Dict[str, Dict[str, Any]], Path]:
    """Return ``(distribution, path)``, building the distribution if needed.

    If ``distribution_path`` exists it is loaded as-is. Otherwise a distribution
    is built from ``dataset_path`` (or the default GDPVal dataset) grouped by
    ``columns`` (default ``["occupation"]``) via ``task_distribution``, then saved
    so subsequent runs reuse it. It is written to ``distribution_path`` when
    given, else to ``<cache_dir>/<columns>_distribution.json`` (cache_dir
    defaults to ``DEFAULT_DISTRIBUTION_CACHE_DIR``).
    """
    column_list = list(columns) if columns else ["occupation"]

    if distribution_path is not None and Path(distribution_path).is_file():
        return load_distribution(distribution_path), Path(distribution_path)

    from responses_api_agents.stirrup_agent.task_distribution import (
        build_distribution_from_dataset,
        resolve_default_dataset,
    )

    resolved_dataset = Path(dataset_path) if dataset_path is not None else resolve_default_dataset()
    if resolved_dataset is None:
        raise FileNotFoundError(
            "No distribution file was provided and no default GDPVal dataset could be found to "
            "build one from. Provide distribution_path, pass dataset_path, or prepare the GDPVal "
            "dataset (gym eval prepare --benchmark gdpval)."
        )

    distribution = build_distribution_from_dataset(resolved_dataset, column_list)

    if distribution_path is not None:
        out_path = Path(distribution_path)
    else:
        base = Path(cache_dir) if cache_dir is not None else DEFAULT_DISTRIBUTION_CACHE_DIR
        out_path = base / f"{'_'.join(column_list)}_distribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(distribution, handle, indent=2, ensure_ascii=False)
    print(
        f"[multistage-elo] built task distribution over {column_list} from {resolved_dataset} -> {out_path}",
        flush=True,
    )
    return distribution, out_path
