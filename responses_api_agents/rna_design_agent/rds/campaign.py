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
"""A campaign is a declared design task plus how its result is scored, so changing either is a file edit."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import registry


@dataclass(frozen=True)
class Scoring:
    """How a finished campaign becomes one number.

    `primary` is the ranked metric. `gates` are pass/fail scores a candidate must satisfy to count at all.
    """

    primary: str
    direction: str = "max"  # "max" or "min"
    gates: dict[str, Any] = field(default_factory=dict)
    top_k: int = 10
    #: The primary-metric value that counts as solved. 1.0 means recovering the target fold exactly.
    pass_threshold: float = 1.0

    def solved(self, best: float | None) -> bool:
        if best is None:
            return False
        return best >= self.pass_threshold if self.direction == "max" else best <= self.pass_threshold

    def passes_gates(self, scores: dict[str, Any]) -> bool:
        for key, rule in self.gates.items():
            value = scores.get(key)
            if value is None:
                return False
            if isinstance(rule, dict):
                if "min" in rule and value < rule["min"]:
                    return False
                if "max" in rule and value > rule["max"]:
                    return False
            elif value != rule:
                return False
        return True


@dataclass(frozen=True)
class Campaign:
    """One design task."""

    name: str
    target: str  # a key in rds.targets, or "inline"
    scoring: Scoring
    budget: dict[str, Any] = field(default_factory=dict)
    controller: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)  # empty means every tool
    #: Tools this campaign does not offer, whatever else admits them.
    exclude: list[str] = field(default_factory=list)
    notes: str = ""
    #: The target itself, for benchmarks that carry their own structures.
    inline_target: dict[str, Any] | None = None
    #: "atomic" offers one tool per step. "composite" offers the packaged pipelines instead.
    exposure: str = "atomic"
    #: "strict" offers a tool only when the task supplies what it needs. "leaky" offers every admissible tool.
    disclosure: str = "strict"
    #: "derived" builds the control from the objective. "handwritten" uses `control_cycle`.
    control: str = "derived"
    #: Steps for the hand-written control, as [tool, {args}] pairs.
    control_cycle: list[Any] = field(default_factory=list)
    #: Whether the agent may write and run its own code.
    allow_code: bool = False
    #: Rounds between scheduled reflections. 0 leaves them off. A reflection round offers only report.rules,
    #: charged to its own budget axis.
    reflect_every: int = 0
    #: What the agent is told about how it will be judged. "metric": the ranked metric, gates and top-k.
    #: "goal": the brief and hard constraints only, scored afterwards by the protocol. "hidden": the brief alone.
    objective_visibility: str = "metric"

    @property
    def target_name(self) -> str:
        """The name tools resolve. An inline target's own name, never the literal "inline"."""
        if self.target == "inline" and self.inline_target:
            return self.inline_target["name"]
        return self.target

    @staticmethod
    def load(path: str | Path) -> Campaign:
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        scoring = Scoring(**data.pop("scoring"))
        return Campaign(scoring=scoring, **data)


#: The axis the ground truth judges, keyed by the primary metric's family. A pseudoknotted target has
#: several readings of "a good design", so the axis is named while the engine and threshold stay hidden.
SUCCESS_NOTE = {
    ("score.fold_match", "f1"): (
        "Success is the target's base pairs actually forming in your design. "
        "Pairs you add that the target does not have count against you."
    ),
    ("struct.rnet_ss", "crossed_pair_f1"): (
        "This target is pseudoknotted: some of its pairs cross others. Success "
        "is those crossing pairs forming — a design that gets every nested "
        "pair right and no crossing pair counts for nothing here."
    ),
    ("struct.rnet_ss", "f1"): (
        "Success is the target's base pairs actually forming in your design, "
        "as a structure predictor reads them rather than as a folding free "
        "energy ranks them."
    ),
}


def _success_note(primary: str) -> str | None:
    """The axis a design is judged on, named without naming the scorer."""
    parts = primary.split(".")
    if len(parts) < 3:
        return None
    return SUCCESS_NOTE.get((".".join(parts[:2]), parts[-1]))


def objective_view(campaign: Campaign) -> dict:
    """What the controller is told about the job.

    `metric` shows the ranked metric and gates. `goal`, the benchmark setting, shows the brief and hard
    constraints only, and the ground-truth protocol scores the submission afterwards.
    """
    from .targets import TARGETS

    target = TARGETS.get(campaign.target_name)
    brief = {
        "campaign": campaign.name,
        "target": campaign.target_name,
        "target_length": target.length if target else None,
        "target_structure": getattr(target, "target_structure", "") or None,
        # Not "pass this to every tool": tools that take no target refuse it.
        "pass_this_name_to_tools_that_take_a_target": campaign.target_name,
    }
    constraints = {
        "fixed_prefix": getattr(target, "fixed_prefix", "") or None,
        "conserved_positions": getattr(target, "conserved", None) or None,
        "max_free_energy": getattr(target, "max_free_energy", None),
    }
    brief["sequence_constraints"] = {k: v for k, v in constraints.items() if v is not None} or None
    if campaign.objective_visibility == "metric":
        return {
            **brief,
            "rank_candidates_by": campaign.scoring.primary,
            "direction": campaign.scoring.direction,
            "gates": campaign.scoring.gates,
            "report_top_k": campaign.scoring.top_k,
            "note": (
                "Only candidates carrying the ranked metric and passing "
                "every gate count. A candidate that is generated but never "
                "scored contributes nothing to the campaign result."
            ),
        }
    if campaign.objective_visibility not in ("goal", "hidden"):
        raise ValueError(
            f"unknown objective_visibility {campaign.objective_visibility!r}; expected metric, goal or hidden"
        )

    task = {
        **brief,
        "task": (
            "Design RNA sequences that fold into this target's secondary "
            "structure. Submit the designs you would stand behind with "
            "report.submission before the budget ends."
        ),
        "how_you_will_be_judged": (
            "By an evaluation protocol you are not "
            "shown, applied to what you submit. Design "
            "for the structure, not for a number."
        ),
    }
    if campaign.objective_visibility == "goal":
        # Hard constraints are part of the task, not the scoring, so they are always shown.
        task["hard_constraints"] = {
            "must_pass": sorted(campaign.scoring.gates),
            "note": "These are requirements of the task, not the ranking.",
        }
        note = _success_note(campaign.scoring.primary)
        if note:
            task["what_counts_as_success"] = note
    return task


def _field_names(tool: str) -> set[str]:
    from . import registry

    return {f.name for f in dataclasses.fields(registry.get(tool).model)}


def _scorer_step(tool: str, target: str) -> dict[str, Any] | None:
    """The arguments this scorer needs, read off its declaration. None if it takes no sequences."""
    fields = _field_names(tool)
    if "sequences" not in fields:
        return None
    step: dict[str, Any] = {"sequences": "@unscored"}
    if "target" in fields:
        step["target"] = target
    return step


def applicable_tools(campaign: Campaign) -> list:
    """The tools this campaign may legitimately use, applied to every arm.

    A tool is offered when the challenge supplies everything it needs and the campaign's allowlist, if any,
    names it.
    """
    from . import registry
    from .targets import TARGETS

    target = TARGETS.get(campaign.target_name)
    supplied = target.provides if target else frozenset()
    if campaign.disclosure == "leaky":
        # Every capability token, whether the task supplied it or not.
        supplied = frozenset(
            {"secondary_structure", "backbone_3d", "ligand", "wild_type", "sequence_constraints", "states"}
        )
    elif campaign.disclosure != "strict":
        raise ValueError(f"unknown disclosure {campaign.disclosure!r}; expected strict or leaky")
    allowed = set(campaign.tools)
    excluded = set(campaign.exclude)
    if not campaign.allow_code:
        excluded.update({"code.run", "code.call", "code.list"})
    admissible = [
        spec
        for spec in registry.all_tools()
        if spec.needs <= supplied
        and (not allowed or spec.name in allowed)
        and spec.name not in excluded
        # A composite that calls an excluded tool is excluded too.
        and not (set(spec.steps) & excluded)
    ]

    composites = [spec for spec in admissible if spec.steps]
    if campaign.exposure == "composite":
        packaged = {step for spec in composites for step in spec.steps}
        return [spec for spec in admissible if spec.name not in packaged]
    return [spec for spec in admissible if not spec.steps]


def default_cycle(
    campaign: Campaign, generator: str = "generate.inverse_fold", n: int = 100, depth: str = "minimal"
) -> list[tuple[str, dict[str, Any]]]:
    """The `fixed` arm's cycle: the generator, then the tools the objective needs, in a fixed order.

    Built from atomic steps under both exposures, so the control does not change with the packaging.
    `depth="minimal"` runs the ranked metric and gates. `depth="full"` adds every other admissible scorer.
    """
    from .state import _producer_of
    from .targets import TARGETS

    target = campaign.target_name
    fields = {f.name for f in dataclasses.fields(registry.get(generator).model)}
    count = "n" if "n" in fields else ""
    if not count:
        raise ValueError(f"{generator} declares no count argument")
    step: dict[str, Any] = {count: n, "target": target}
    # gRNAde samples range(n // batch_size), so the batch must not exceed n.
    if "batch_size" in fields:
        # RibonanzaNet memory grows as batch x length squared, so shrink the batch for long targets.
        length = getattr(TARGETS.get(target), "length", 0) or 0
        batch = 64 if length <= 150 else max(4, int(64 * (150 / length) ** 2))
        step["batch_size"] = min(n, batch)
    # Turn off the generator's own filter, which can drop every sample. The campaign's gates decide.
    if "pass_threshold" in fields:
        step["pass_threshold"] = 0.0
    cycle: list[tuple[str, dict[str, Any]]] = [(generator, step)]
    needed = [campaign.scoring.primary]
    needed += [k for k in campaign.scoring.gates if k not in needed]
    for metric in needed:
        tool = _producer_of(metric)
        if tool == "unknown":
            raise ValueError(
                f"no registered tool produces {metric!r}; the fixed arm cannot "
                f"be derived for campaign {campaign.name!r}"
            )
        step = _scorer_step(tool, target)
        if step is None:
            raise ValueError(
                f"{tool} produces {metric!r} but takes no list of sequences "
                f"({', '.join(_field_names(tool))}); it reads a prior tool's "
                f"output, so the fixed arm cannot derive a cycle for this "
                f"campaign — rank on a metric a sequence-level scorer produces"
            )
        if (tool, step) not in cycle:
            cycle.append((tool, step))

    if depth == "full":
        scored_by = {name for name, _ in cycle}
        # Folding engines and structure predictors count as evaluation depth too.
        for spec in applicable_tools(campaign):
            if spec.group not in ("score", "struct", "fold") or spec.name in scored_by:
                continue
            fields = {f.name for f in dataclasses.fields(spec.model)}
            if "sequences" not in fields or "rules_file" in fields:
                continue  # it reads a prior tool's output, which this cycle never writes
            step = {"sequences": "@unscored"}
            if "target" in fields:
                step["target"] = target
            cycle.append((spec.name, step))
    elif depth != "minimal":
        raise ValueError(f"unknown depth {depth!r}; expected minimal or full")
    return cycle


def evaluation_depth(rows: list[dict]) -> dict:
    """How many distinct tools measured each candidate, counted from its score keys."""
    per_candidate = []
    for row in rows:
        producers = {key.split(".", 2)[0] + "." + key.split(".", 2)[1] for key in row["scores"] if key.count(".") >= 1}
        per_candidate.append(len(producers))
    if not per_candidate:
        return {"mean_tools_per_candidate": 0.0, "max": 0, "unmeasured": 0}
    return {
        "mean_tools_per_candidate": round(sum(per_candidate) / len(per_candidate), 3),
        "max": max(per_candidate),
        "unmeasured": sum(1 for n in per_candidate if n == 0),
    }


def score_campaign(state, campaign: Campaign) -> dict:
    """Turn a finished campaign's candidates into one comparable result, gated and ungated reported apart."""
    rows = state.candidates()
    metric = campaign.scoring.primary
    reverse = campaign.scoring.direction == "max"

    scored = [r for r in rows if metric in r["scores"]]
    gated = [r for r in scored if campaign.scoring.passes_gates(r["scores"])]
    gated.sort(key=lambda r: r["scores"][metric], reverse=reverse)
    top = gated[: campaign.scoring.top_k]

    def summarise(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        ordered = sorted(values, reverse=reverse)
        return {
            "n": len(values),
            "best": ordered[0],
            "median": ordered[len(ordered) // 2],
            "mean": sum(values) / len(values),
        }

    return {
        "campaign": campaign.name,
        "metric": metric,
        "direction": campaign.scoring.direction,
        "candidates": len(rows),
        "scored": len(scored),
        "depth": evaluation_depth(rows),
        "gated": len(gated),
        "gate_rate": round(len(gated) / len(scored), 4) if scored else 0.0,
        "all": summarise([r["scores"][metric] for r in scored]),
        "passing": summarise([r["scores"][metric] for r in gated]),
        "top": [{"sequence": r["sequence"], "source": r["source"], "scores": r["scores"]} for r in top],
    }


def write_result(directory: str | Path, result: dict) -> Path:
    path = Path(directory) / "result.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    return path
