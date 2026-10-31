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
"""Hard-constraint checking, independent of whatever produced the sequence."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

from ..registry import tool
from ..spec import Cost, field
from ..targets import TARGETS


@dataclass
class ConstraintsIn:
    sequences: list[str] = field("RNA sequences to check, or a selector: @all, @unscored, @top:N.")
    target: str = field("Campaign target name")


def _violations(seq: str, t) -> list[str]:
    out: list[str] = []
    if len(seq) != t.length:
        out.append(f"length {len(seq)} != {t.length}")
    if not seq.startswith(t.fixed_prefix):
        out.append(f"prefix {seq[: len(t.fixed_prefix)]!r} != {t.fixed_prefix!r}")
    bad = sorted(set(seq) - set("ACGU"))
    if bad:
        out.append(f"non-ACGU {bad}")
    # Everything below indexes the sequence, so only ask once the length is right.
    if len(seq) == t.length:
        for pos, base in sorted(t.conserved.items()):
            if seq[pos - 1] != base:
                out.append(f"conserved {pos}{base} is {seq[pos - 1]}")
        # A de novo puzzle has no wild type to differ from.
        if t.wild_type and t.min_changes_from_wt:
            changes = sum(1 for a, b in zip(seq, t.wild_type, strict=True) if a != b)
            if changes < t.min_changes_from_wt:
                out.append(f"{changes} changes from WT, need >= {t.min_changes_from_wt}")
        out += _puzzle_violations(seq, t)
        # Fold last, and only when the target states a limit.
        if t.max_free_energy is not None and not out:
            from arnie.free_energy import free_energy

            from .fold import _ensure_arnie_tmp

            _ensure_arnie_tmp()
            dg = free_energy(seq, package="eternafold")
            if dg < t.max_free_energy:
                out.append(f"free energy {dg:.2f} is below the limit {t.max_free_energy}")
    return out


def _puzzle_violations(seq: str, t) -> list[str]:
    """Eterna puzzle limits on pair-type counts and runs of one base, counted over the target's pairs."""
    from ..structure import to_pairs

    out: list[str] = []
    if t.max_consecutive:
        longest, run = 1, 1
        for before, after in itertools.pairwise(seq):
            run = run + 1 if before == after else 1
            longest = max(longest, run)
        if longest > t.max_consecutive:
            out.append(f"{longest} identical bases in a row, allowed {t.max_consecutive}")
    if t.pair_limits:
        counts = {"GC": 0, "AU": 0, "GU": 0}
        for left, right in to_pairs(t.target_structure):
            if left >= len(seq) or right >= len(seq):
                continue
            pair = "".join(sorted(seq[left] + seq[right]))
            kind = {"CG": "GC", "AU": "AU", "GU": "GU"}.get(pair)
            if kind:
                counts[kind] += 1
        for kind, (low, high) in t.pair_limits.items():
            kind = kind.upper()
            have = counts.get(kind, 0)
            if low is not None and have < low:
                out.append(f"{have} {kind} pairs, need >= {low}")
            if high is not None and have > high:
                out.append(f"{have} {kind} pairs, allowed <= {high}")
    return out


@tool(
    name="score.constraints",
    summary=(
        "Check RNA sequences against a campaign's hard constraints — length, "
        "fixed prefix, conserved residues, minimum changes from wild type, and "
        "the puzzle's own limits where a round states them: how many GC, AU or "
        "GU pairs the target's pairs may be, and how many identical bases may "
        "sit in a row. Returns a pass/fail verdict and the reason for every "
        "failure."
    ),
    cost=Cost.CHEAP,
    model=ConstraintsIn,
)
def score_constraints(inp: ConstraintsIn) -> dict:
    if inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    t = TARGETS[inp.target]
    results = []
    for seq in inp.sequences:
        seq = seq.strip().upper().replace("T", "U")
        violations = _violations(seq, t)
        results.append(
            {
                "sequence": seq,
                "passed": not violations,
                "violations": violations,
                # Only meaningful once the length matches. A short sequence would
                # otherwise report a count taken over the part that happens to line up.
                "changes_from_wt": (
                    sum(1 for a, b in zip(seq, t.wild_type, strict=True) if a != b)
                    if t.wild_type and len(seq) == t.length
                    else None
                ),
            }
        )
    return {
        "tool": "score.constraints",
        "target": t.name,
        "n": len(results),
        "n_passed": sum(r["passed"] for r in results),
        "results": results,
    }


@dataclass
class RulesIn:
    rules: list[dict] = field(
        "The design rules this campaign discovered. Each one needs a plain "
        "`statement` a person can read, a `context` (stem, loop, crossing, "
        "five_prime_end, three_prime_end, all), a `quantity` (gc_fraction, "
        "a_fraction, u_fraction, g_run_max, purine_fraction), and a `minimum` "
        "and/or `maximum`. A `rationale` is optional and read by people."
    )
    settings: dict = field('Generator arguments the rules imply, e.g. {"gc_bias": 0.65}.', default_factory=dict)
    notes: str = field("What the rules are for, in a sentence or two.", default="")
    discovered_on: str = field("The suite these were learned on", default="")
    out_dir: str = field("Where to write the ruleset", default="", operator=True)


@tool(
    name="report.rules",
    summary=(
        "Report the design rules this campaign found, as statements that hold "
        "for targets you have not seen. A rule prescribes a composition over a "
        "structural context — stems, loops, crossing pairs, the ends — or a "
        "generator setting. Rules naming a sequence fragment or a specific "
        "target are rejected: what is being asked for is what transfers, not "
        "what worked here. Reporting rules does not score candidates; it is "
        "how a run leaves behind something a later run can use."
    ),
    cost=Cost.CHEAP,
    model=RulesIn,
)
def report_rules(inp: RulesIn) -> dict:
    import tempfile
    from pathlib import Path

    from ..rules import SETTING_ALIASES, SETTINGS, Ruleset, as_rule

    # Split a comma-separated context into one rule per context. Settings that name a context
    # (stem_gc_bias) are converted to rules and reported. Anything else still fails validation.
    rules = []
    for rule in inp.rules or []:
        contexts = [c.strip() for c in str(rule.get("context", "all")).split(",")]
        for context in contexts or ["all"]:
            rules.append({**rule, "context": context})

    settings, converted = {}, []
    for key, value in (inp.settings or {}).items():
        if key in SETTINGS:
            settings[key] = value
            continue
        if key in SETTING_ALIASES:
            for real in SETTING_ALIASES[key]:
                settings[real] = value
            continue
        rule = as_rule(key, value)
        if rule is None:
            settings[key] = value  # unknown: let validate refuse it
        else:
            converted.append({"from": key, "to": rule})

    ruleset = Ruleset.from_dict(
        {
            "rules": rules + [r["to"] for r in converted],
            "settings": settings,
            "notes": inp.notes,
            "discovered_on": inp.discovered_on,
        }
    )
    ruleset.validate()
    out = Path(inp.out_dir or os.environ.get("RDS_RUN_DIR") or tempfile.mkdtemp(prefix="rules_"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "rules.json").write_text(ruleset.to_json() + "\n")
    (out / "rules.md").write_text(ruleset.to_markdown() + "\n")
    return {
        "tool": "report.rules",
        "n_rules": len(ruleset.rules),
        "converted_settings": converted,
        "rules_dir": str(out),
        "markdown": ruleset.to_markdown(),
    }


@dataclass
class RuleCheckIn:
    sequences: list[str] = field("Candidates to check against a ruleset.")
    target: str = field("Campaign target whose structure gives the contexts", default="")
    rules_file: str = field("Path to a rules.json from report.rules", default="")


@tool(
    name="score.rule_obedience",
    summary=(
        "Check candidates against a ruleset discovered elsewhere: does each one "
        "satisfy every rule, and which rule does it break. This is how a rule "
        "earns its keep — it is applied to targets it was not learned on, and "
        "what it buys there is the only evidence that it is a principle rather "
        "than a memory of one search."
    ),
    cost=Cost.CHEAP,
    model=RuleCheckIn,
    needs=("secondary_structure",),
)
def score_rule_obedience(inp: RuleCheckIn) -> dict:
    from ..rules import Ruleset, check
    from ..targets import TARGETS

    if inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    if not inp.rules_file:
        raise ValueError("no rules_file; run report.rules first, or pass one")
    ruleset = Ruleset.load(inp.rules_file)
    structure = TARGETS[inp.target].target_structure
    results = []
    for sequence in inp.sequences:
        verdict = check(sequence, structure, ruleset)
        results.append(
            {
                "sequence": sequence,
                "obeys": verdict["obeys"],
                "n_violations": len(verdict["violations"]),
                "violations": "; ".join(verdict["violations"]),
            }
        )
    obeying = sum(1 for r in results if r["obeys"])
    return {
        "tool": "score.rule_obedience",
        "target": inp.target,
        "n": len(results),
        "n_obeying": obeying,
        "obey_rate": round(obeying / len(results), 4) if results else 0.0,
        "results": results,
    }
