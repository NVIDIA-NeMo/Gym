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
"""Secondary-structure folding tools."""

from __future__ import annotations

from dataclasses import dataclass

from ..registry import tool
from ..spec import Cost, field


def _validate(sequences: list[str]) -> None:
    if not sequences:
        raise ValueError("no sequences given")
    for seq in sequences:
        bad = set(seq) - set("ACGU")
        if bad:
            raise ValueError(f"non-ACGU characters {sorted(bad)} in {seq[:20]}...")


@dataclass
class EternaFoldIn:
    sequences: list[str] = field("RNA sequences (A/C/G/U), or a selector: @all, @unscored, @top:N.")
    max_free_energy: float = field("Gate: report pass when dG is at or above this, in kcal/mol.", default=-19.0)


def _ensure_arnie_tmp() -> None:
    """Create the scratch directory named in $ARNIEFILE if it is missing. arnie writes there but never creates it."""
    import os
    from pathlib import Path

    path = os.environ.get("ARNIEFILE")
    if not path or not Path(path).is_file():
        return
    for line in Path(path).read_text().splitlines():
        if line.startswith("TMP:"):
            Path(line.split(":", 1)[1].strip()).mkdir(parents=True, exist_ok=True)
            return


@tool(
    name="fold.eternafold",
    summary=(
        "Fold RNA sequences with EternaFold and return the structure, the free "
        "energy, and whether it clears a dG gate. EternaFold is a learned "
        "model trained on chemical-mapping and affinity data, so it disagrees "
        "with a thermodynamic engine most where a structure is weak."
    ),
    cost=Cost.CHEAP,
    model=EternaFoldIn,
)
def fold_eternafold(inp: EternaFoldIn) -> dict:
    # lazy: both need ARNIEFILE and the built binary
    from arnie.free_energy import free_energy
    from arnie.mfe import mfe

    _ensure_arnie_tmp()
    _validate(inp.sequences)
    results = []
    for seq in inp.sequences:
        dg = free_energy(seq, package="eternafold")
        results.append(
            {
                "sequence": seq,
                "structure": mfe(seq, package="eternafold"),
                "dg": round(dg, 2),
                "passed": dg >= inp.max_free_energy,
            }
        )
    return {
        "tool": "fold.eternafold",
        "n": len(results),
        "n_passed": sum(r["passed"] for r in results),
        "max_free_energy": inp.max_free_energy,
        "results": results,
    }


@dataclass
class FoldMatchIn:
    sequences: list[str] = field("RNA sequences to fold and compare.")
    target: str = field("Campaign target name whose structure is the reference.")
    engine: str = field(
        "Folding engine. EternaFold is the only one here.",
        default="eternafold",
    )


@tool(
    name="score.fold_match",
    summary=(
        "Fold each design with a thermodynamic engine and score the result "
        "against the campaign's target structure as pair-set F1. This is how "
        "the published inverse-folding benchmarks judge a design — recover the "
        "target fold under ViennaRNA — so it is the faithful metric for those "
        "suites, and it runs on a CPU. It cannot express pseudoknots: on a "
        "crossing target it measures only the nested part, which is why the "
        "pseudoknot suites rank on a predictor that can."
    ),
    cost=Cost.CHEAP,
    model=FoldMatchIn,
    needs=("secondary_structure",),
)
def score_fold_match(inp: FoldMatchIn) -> dict:
    from ..structure import compare
    from ..targets import TARGETS

    _validate(inp.sequences)
    if inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    reference = TARGETS[inp.target].target_structure
    # The engine is named, not spelled: refusing "EternaFold" costs a round and teaches nothing.
    if inp.engine.strip().lower() != "eternafold":
        raise ValueError(
            f"unknown engine {inp.engine!r}; this harness folds with EternaFold only, because ViennaRNA's "
            "licence restricts redistributing the package and one engine everywhere keeps the arms comparable"
        )

    def folder(seqs):
        return fold_eternafold(EternaFoldIn(sequences=seqs))

    results = []
    for row in folder(inp.sequences)["results"]:
        match = compare(row["structure"], reference)
        results.append(
            {"sequence": row["sequence"], "structure": row["structure"], "dg": row.get("dg"), "match": match}
        )
    best = max((r["match"]["f1"] for r in results), default=0.0)
    return {
        "tool": "score.fold_match",
        "target": inp.target,
        "engine": inp.engine,
        "n": len(results),
        "best_f1": best,
        "results": results,
    }


@dataclass
class SwitchMatchIn:
    sequences: list[str] = field("RNA sequences to fold in every state.")
    target: str = field("Campaign target whose states give the shapes to reach", default="")
    engine: str = field("Folding engine; EternaFold is the only one here.", default="eternafold")


def _fold_state(sequence: str, state, engine: str) -> str:
    """The shape this sequence takes under one state's condition.

    A ligand is not in the thermodynamic model, so its bound positions are forced to pair as the state asks.
    Where the engine cannot take a constraint, the state folds free.
    """
    from arnie.mfe import mfe

    _ensure_arnie_tmp()
    from ..structure import to_pairs

    if not state.bound:
        return mfe(sequence, package=engine)
    # Constrain both ends of a held pair or neither. The engine refuses an unclosed bracket.
    partner = {}
    for left, right in to_pairs(state.structure):
        partner[left] = right
        partner[right] = left
    constraint = ["." for _ in sequence]
    for position in state.bound:
        index = position - 1  # the challenge counts from one
        other = partner.get(index)
        if other is None or not (0 <= index < len(sequence)) or not (0 <= other < len(sequence)):
            continue
        constraint[min(index, other)] = "("
        constraint[max(index, other)] = ")"
    if not any(c != "." for c in constraint):
        return mfe(sequence, package=engine)
    try:
        return mfe(sequence, package=engine, constraint="".join(constraint))
    except TypeError:
        return mfe(sequence, package=engine)


@tool(
    name="score.switch_match",
    summary=(
        "Fold each design in every state the target asks for and score it "
        "against that state's shape. An Eterna lab round is usually a switch: "
        "the same molecule folds one way alone and another way with a ligand "
        "bound, and a design that reaches one state and not the other has not "
        "solved it. Reports the match per state, the worst state — which is "
        "what the design is worth — and how far the two shapes actually move "
        "apart, since a sequence that ignores the ligand scores well on both "
        "states and switches nothing."
    ),
    cost=Cost.CHEAP,
    model=SwitchMatchIn,
    needs=("secondary_structure", "states"),
)
def score_switch_match(inp: SwitchMatchIn) -> dict:
    from ..structure import compare, to_pairs
    from ..targets import TARGETS

    _validate(inp.sequences)
    if inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    target = TARGETS[inp.target]
    states = target.all_states()
    if len(states) < 2:
        raise ValueError(f"target {inp.target!r} has one state; score.fold_match is the tool for a single shape")

    results = []
    for sequence in inp.sequences:
        per_state, folded = {}, {}
        for state in states:
            structure = _fold_state(sequence, state, inp.engine)
            folded[state.condition or f"state{len(folded)}"] = structure
            per_state[state.condition or f"state{len(per_state)}"] = compare(structure, state.structure)["f1"]
        shapes = list(folded.values())
        moved = len(set(map(frozenset, (to_pairs(s) for s in shapes))))
        results.append(
            {
                "sequence": sequence,
                "per_state": per_state,
                "structures": folded,
                # The design is worth its weakest state: an Eterna lab round pays
                # nothing for a switch that only works one way round.
                "worst_state_f1": min(per_state.values()) if per_state else 0.0,
                "switches": moved > 1,
            }
        )
    best = max((r["worst_state_f1"] for r in results), default=0.0)
    return {
        "tool": "score.switch_match",
        "target": inp.target,
        "engine": inp.engine,
        "states": [s.condition for s in states],
        "n": len(results),
        "best_worst_state_f1": best,
        "n_switching": sum(r["switches"] for r in results),
        "results": results,
    }


@dataclass
class EnsembleDefectIn:
    sequences: list[str] = field("RNA sequences to measure.")
    target: str = field("Campaign target whose structure the ensemble is measured against", default="")
    state: str = field(
        "Which state's shape to measure against, when the target has several; empty uses the first.", default=""
    )


@tool(
    name="score.ensemble_defect",
    summary=(
        "How much of the ensemble is in the wrong shape, per nucleotide: the "
        "expected number of positions paired differently from the target, "
        "divided by length. Zero is a sequence that folds to the target and "
        "nowhere else. The minimum-free-energy structure can match a target "
        "exactly while most of the ensemble sits elsewhere, and a molecule "
        "that has to work in a tube is the ensemble, not its best member — "
        "which is why Eterna shows players more than the one structure."
    ),
    cost=Cost.CHEAP,
    model=EnsembleDefectIn,
    needs=("secondary_structure",),
)
def score_ensemble_defect(inp: EnsembleDefectIn) -> dict:
    from arnie.bpps import bpps

    from ..structure import to_pairs
    from ..targets import TARGETS

    _ensure_arnie_tmp()
    _validate(inp.sequences)
    if inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    target = TARGETS[inp.target]
    structure = target.target_structure
    for state in target.all_states():
        if inp.state and state.condition == inp.state:
            structure = state.structure
            break
    else:
        if inp.state:
            raise ValueError(
                f"target {inp.target!r} has no state {inp.state!r}; it has "
                f"{[s.condition for s in target.all_states()]}"
            )

    partner = {}
    for left, right in to_pairs(structure):
        partner[left] = right
        partner[right] = left

    results = []
    for sequence in inp.sequences:
        matrix = bpps(sequence, package="eternafold")
        defect = 0.0
        for position in range(len(sequence)):
            if position in partner:
                # Probability this position pairs with the one the target says.
                defect += 1.0 - float(matrix[position][partner[position]])
            else:
                # Probability it is paired at all, when the target leaves it open.
                defect += float(sum(matrix[position]))
        results.append({"sequence": sequence, "ensemble_defect": round(defect / len(sequence), 4)})
    best = min((r["ensemble_defect"] for r in results), default=None)
    return {
        "tool": "score.ensemble_defect",
        "target": inp.target,
        "state": inp.state or "apo",
        "n": len(results),
        "best_ensemble_defect": best,
        "results": results,
    }
