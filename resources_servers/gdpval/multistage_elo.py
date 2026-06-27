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
"""Multi-stage adaptive ELO estimation for GDPVal pairwise comparison.

Instead of comparing the evaluated model against every reference model on all
tasks, this runs a sequence of *stages*. Each stage:

1. fixes a set of ``T`` tasks sampled from a task-distribution JSON file (see
   ``responses_api_agents.stirrup_agent.task_distribution``),
2. judges the evaluated model against a set of ``M`` reference models on those
   tasks (delegated to an injected ``judge_stage`` callable),
3. fits an anchored Bradley-Terry MLE ELO from that stage's win/loss/tie
   battles (reusing ``comparison.calculate_mle_elo``), and
4. uses that estimate to choose the ``M`` references for the next stage.

Across stages, ``M`` typically shrinks (zooming in on references whose known
ELO is closest to the evaluated model's current estimate) while ``T`` grows
(spending the saved judge budget on a tighter final estimate).

This module is intentionally **pure / server-agnostic**: the actual judging
(running rollouts, calling ``/verify``, reading cached deliverables) is supplied
by the caller as a ``judge_stage`` callable, so the staging/selection/ELO logic
is unit-testable without any servers. The orchestration that wires this to the
GDPVal servers lives in the driver (see the module docstring there).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from resources_servers.gdpval.comparison import calculate_mle_elo


# A mapping ``ref_id -> {"wins": int, "losses": int, "ties": int,
# "reference_elo": float}`` as produced (per task, then pooled) by the GDPVal
# comparison verifier. This is the unit the ELO MLE is fit over.
PerReferenceTotals = Dict[str, Dict[str, float]]

# Signature of the injected judging step. Given the stage's fixed task ids and
# the selected reference ids, return pooled per-reference win/loss/tie totals
# for the evaluated model across those tasks.
JudgeStageFn = Callable[[Sequence[str], Sequence[str]], PerReferenceTotals]


@dataclass
class StageSpec:
    """Configuration for a single stage.

    ``num_tasks`` is ``T`` (the number of tasks judged this stage). ``num_models``
    is ``M`` (the number of reference models compared against); ``None`` means
    "all available references" (used for the first, broad stage). ``seed`` makes
    task sampling for this stage reproducible.
    """

    num_tasks: int
    num_models: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class StageResult:
    """Outcome of one stage."""

    stage_index: int
    task_ids: List[str]
    reference_ids: List[str]
    per_reference: PerReferenceTotals
    eval_elo: Optional[float]
    normalized_elo: Optional[float]
    # Number of reference models included in this stage's ELO fit.
    num_references: int


@dataclass
class MultiStageEloConfig:
    """End-to-end configuration for a multi-stage ELO run."""

    stages: List[StageSpec]
    # ref_id -> known/anchor ELO. Both the MLE (anchors) and reference selection
    # ("closest to the eval estimate") require these.
    reference_elos: Dict[str, float]

    # Task distribution source. When ``distribution_path`` is unset (or missing),
    # the driver builds a distribution from ``dataset_path`` (or the default
    # GDPVal dataset) grouped by ``column`` and caches it. See
    # ``multistage_elo_driver.ensure_distribution``.
    distribution_path: Optional[str] = None
    dataset_path: Optional[str] = None

    # Eval deliverables source. When set, pre-existing cached deliverables under
    # this directory (``task_<id>/repeat_<n>/``) are reused instead of producing
    # fresh rollouts. ``produce_missing`` controls whether tasks absent from the
    # cache are produced on demand (True) or dropped from the stage (False).
    eval_deliverables_dir: Optional[str] = None
    produce_missing: bool = True

    # Sampling behaviour across stages. ``nested=True`` makes each stage's task set
    # a superset of the previous stage's, which is cheaper (reuses produced
    # deliverables and judgments) but couples the stages' samples. The default
    # (False) samples each stage independently: later stages draw fresh tasks, so
    # the stages contribute more independent information to the ELO estimate.
    nested_tasks: bool = False

    selection: str = "closest"
    column: List[str] = field(default_factory=lambda: ["occupation"])

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("At least one stage is required.")
        if self.selection != "closest":
            raise ValueError(f"Unknown selection strategy: {self.selection!r}")


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
# Task planning
# ---------------------------------------------------------------------------


def plan_stage_task_ids(
    distribution: Mapping[str, Mapping[str, object]],
    stages: Sequence[StageSpec],
    *,
    rng: Optional[random.Random] = None,
    nested: bool = True,
) -> List[List[str]]:
    """Pre-sample the task set for every stage from a task distribution.

    Task selection is independent of any ELO estimate, so all stages' task sets
    can be planned up front.

    ``nested=True`` makes each stage's set a superset of the previous one. We get
    this for free in a single draw: ``sample_task_ids`` samples without
    replacement one task at a time, so a prefix of a large draw is identical to a
    smaller draw made with the same RNG. We therefore draw once, sized to the
    largest stage, and slice each stage's prefix from it — O(max T) work and
    exactly proportional per stage, with nesting guaranteed. A single shared RNG
    is used (per-stage ``seed`` only applies to independent sampling).

    ``nested=False`` samples each stage independently, honoring its own ``seed``.
    """
    from responses_api_agents.stirrup_agent.task_distribution import sample_task_ids

    base_rng = rng or random.Random()

    if not nested:
        return [
            sample_task_ids(
                distribution,
                s.num_tasks,
                rng=random.Random(s.seed) if s.seed is not None else base_rng,
            )
            for s in stages
        ]

    max_target = max(s.num_tasks for s in stages)
    ordered = sample_task_ids(distribution, max_target, rng=base_rng)
    return [list(ordered[: s.num_tasks]) for s in stages]


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
# Runner
# ---------------------------------------------------------------------------


class MultiStageEloRunner:
    """Drive the multi-stage ELO procedure.

    ``run`` first plans every stage's task set up front (task selection does not
    depend on any ELO estimate), then walks the stages sequentially: for each
    stage it selects the references (closest known ELO to the running estimate),
    judges the stage, fits the stage ELO, and threads that estimate into the next
    stage's reference selection. Matchup judging is not the runner's concern; it
    is supplied as ``judge_stage(task_ids, reference_ids) -> per_reference_totals``.

    ``run`` returns one ``StageResult`` per stage; the last stage's ``eval_elo``
    is the headline estimate.
    """

    def __init__(
        self,
        config: MultiStageEloConfig,
        distribution: Mapping[str, Mapping[str, object]],
        judge_stage: JudgeStageFn,
        *,
        rng: Optional[random.Random] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.config = config
        self.distribution = distribution
        self.judge_stage = judge_stage
        self.rng = rng or random.Random()
        # Optional progress hook. Called as ``on_event(name, data)`` for the
        # events "planned", "stage_start", and "stage_end". Kept as a callback so
        # this module performs no I/O itself; the driver/CLI does the printing.
        self.on_event = on_event

    def _emit(self, name: str, **data: object) -> None:
        if self.on_event is not None:
            self.on_event(name, data)

    def run(self) -> List[StageResult]:
        stage_task_sets = plan_stage_task_ids(
            self.distribution,
            self.config.stages,
            rng=self.rng,
            nested=self.config.nested_tasks,
        )
        total_stages = len(self.config.stages)
        self._emit("planned", stage_task_counts=[len(s) for s in stage_task_sets], total_stages=total_stages)

        results: List[StageResult] = []
        eval_elo: Optional[float] = None
        for index, stage in enumerate(self.config.stages):
            reference_ids = select_references(self.config.reference_elos, eval_elo, stage.num_models)
            task_ids = stage_task_sets[index]
            self._emit(
                "stage_start",
                index=index,
                total_stages=total_stages,
                reference_ids=list(reference_ids),
                num_tasks=len(task_ids),
                prior_elo=eval_elo,
            )
            per_reference = self.judge_stage(task_ids, reference_ids)
            stage_elo, normalized, num_references = fit_stage_elo(per_reference, self.config.reference_elos)
            if stage_elo is not None:
                eval_elo = stage_elo
            self._emit(
                "stage_end",
                index=index,
                total_stages=total_stages,
                eval_elo=stage_elo,
                normalized_elo=normalized,
                num_references=num_references,
                per_reference=dict(per_reference),
            )
            results.append(
                StageResult(
                    stage_index=index,
                    task_ids=list(task_ids),
                    reference_ids=list(reference_ids),
                    per_reference=dict(per_reference),
                    eval_elo=stage_elo,
                    normalized_elo=normalized,
                    num_references=num_references,
                )
            )
        return results
