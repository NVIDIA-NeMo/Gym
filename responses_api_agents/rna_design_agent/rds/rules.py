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
"""Design rules an agent discovered, in a form a person reads and a machine runs.

A rule prescribes a composition over a structural context, or a generator setting, and nothing else.
The narrow schema keeps target-specific answers from passing as rules that transfer.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


#: Structural contexts a rule may speak about, resolved from the target's dot-bracket.
CONTEXTS = ("stem", "loop", "crossing", "closing", "isolated", "five_prime_end", "three_prime_end", "all")

#: Quantities a rule may prescribe over a context.
QUANTITIES = (
    "gc_fraction",
    "a_fraction",
    "u_fraction",
    "g_run_max",
    "purine_fraction",
    #: The fraction of target pairs inside a context that are G-U.
    "gu_pair_fraction",
)

#: Generator arguments a ruleset may set.
SETTINGS = ("gc_bias", "n_mutations", "weak_bias", "temperature_min", "temperature_max")


@dataclass(frozen=True)
class Rule:
    """One prescription, with the sentence a person would read."""

    statement: str  # "stems should be GC-rich"
    context: str = "all"
    quantity: str = "gc_fraction"
    minimum: float | None = None
    maximum: float | None = None
    rationale: str = ""

    def validate(self) -> None:
        if self.context not in CONTEXTS:
            raise ValueError(f"unknown context {self.context!r}; expected {CONTEXTS}")
        if self.quantity not in QUANTITIES:
            raise ValueError(f"unknown quantity {self.quantity!r}; expected {QUANTITIES}")
        if self.minimum is None and self.maximum is None:
            raise ValueError(f"rule {self.statement!r} prescribes nothing")
        if not self.statement.strip():
            raise ValueError("a rule without a statement cannot be read by anyone")


@dataclass
class Ruleset:
    """What one discovery run claims to have learned."""

    rules: list[Rule] = field(default_factory=list)
    settings: dict[str, float] = field(default_factory=dict)
    discovered_on: str = ""  # the suite the agent searched
    notes: str = ""

    def validate(self) -> None:
        for rule in self.rules:
            rule.validate()
        for key in self.settings:
            if key not in SETTINGS:
                # A per-context target written as a setting (stem_gc_bias): show the rule it meant.
                raise ValueError(
                    f"unknown setting {key!r}; settings are generator arguments "
                    f"{SETTINGS}. A target for one structural context is a "
                    f"rule, not a setting — send it in `rules` as "
                    f"{_as_rule(key, self.settings[key])}, with `minimum` or "
                    f"`maximum` depending on which way you meant it."
                )
        leak = target_specific(self)
        if leak:
            raise ValueError(
                f"a rule names something target-specific ({leak}); a rule has to "
                f"hold for targets nobody has searched yet"
            )

    def to_json(self) -> str:
        return json.dumps(
            {
                "rules": [asdict(r) for r in self.rules],
                "settings": self.settings,
                "discovered_on": self.discovered_on,
                "notes": self.notes,
            },
            indent=1,
        )

    def to_markdown(self) -> str:
        """The form a reviewer reads before deciding whether to believe it."""
        lines = [f"# Design rules discovered on {self.discovered_on or 'an unnamed suite'}"]
        if self.notes:
            lines += ["", self.notes]
        for rule in self.rules:
            bound = []
            if rule.minimum is not None:
                bound.append(f">= {rule.minimum}")
            if rule.maximum is not None:
                bound.append(f"<= {rule.maximum}")
            lines.append(f"\n- **{rule.statement}**  \n  `{rule.context}.{rule.quantity}` {' and '.join(bound)}")
            if rule.rationale:
                lines.append(f"  _{rule.rationale}_")
        if self.settings:
            lines.append("\nGenerator settings: " + ", ".join(f"`{k}={v}`" for k, v in sorted(self.settings.items())))
        return "\n".join(lines)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Ruleset:
        # Drop unknown keys rather than fail. validate() still checks what a rule prescribes.
        fields = {f.name for f in dataclasses.fields(Rule)}
        return Ruleset(
            rules=[Rule(**{k: v for k, v in r.items() if k in fields}) for r in data.get("rules") or []],
            settings=dict(data.get("settings") or {}),
            discovered_on=data.get("discovered_on", ""),
            notes=data.get("notes", ""),
        )

    @staticmethod
    def load(path: str | Path) -> Ruleset:
        return Ruleset.from_dict(json.loads(Path(path).read_text()))


#: Settings a model writes that mean a setting this schema does have.
SETTING_ALIASES = {
    "temperature": ("temperature_min", "temperature_max"),
    "temp": ("temperature_min", "temperature_max"),
    "mutations": ("n_mutations",),
    "n_mutation": ("n_mutations",),
    "gc": ("gc_bias",),
    "gc_content": ("gc_bias",),
    "gc_target": ("gc_bias",),
}


def as_rule(key: str, value: float) -> dict | None:
    """The rule a settings key such as `stem_gc_bias` was reaching for, or None if it names no known context."""
    # Words a designer uses for a context that this schema spells otherwise.
    for spoken, meant in (
        ("target_global_", ""),
        ("global_", ""),
        ("three_prime_tail_", "three_prime_end_"),
        ("five_prime_tail_", "five_prime_end_"),
        ("terminal_unpaired_", "five_prime_end_"),
        ("unpaired_end_", "five_prime_end_"),
        ("unpaired_terminal_", "five_prime_end_"),
        ("isolated_pair_force_", "isolated_"),
        ("isolated_pair_", "isolated_"),
        ("lone_pair_", "isolated_"),
        ("terminal_pair_", "closing_"),
        ("closing_pair_", "closing_"),
        ("unpaired_", "loop_"),
        ("paired_", "stem_"),
        ("helix_", "stem_"),
        ("hairpin_", "loop_"),
        ("junction_", "loop_"),
    ):
        if key.startswith(spoken):
            key = meant + key[len(spoken) :]
            break

    # A key with no context in it is about the whole molecule.
    whole = {
        "max_g_run": "g_run_max",
        "g_run_max": "g_run_max",
        "gc_fraction": "gc_fraction",
        "gc": "gc_fraction",
        "a_fraction": "a_fraction",
        "u_fraction": "u_fraction",
        "purine_fraction": "purine_fraction",
    }.get(key)
    if whole:
        return {
            "statement": f"all: {whole} near {value}",
            "context": "all",
            "quantity": whole,
            "minimum": float(value),
            "rationale": f"converted from the setting {key!r}",
        }

    head, _, tail = key.partition("_")
    # "The ends" is read as the 5' end.
    if head in ("end", "ends", "terminal", "terminus", "tail", "tails"):
        head = "five_prime_end"
        tail = key.split("_", 1)[1] if "_" in key else ""
    if head not in CONTEXTS and head not in ("five", "three"):
        return None
    if head in ("five", "three"):  # five_prime_end_gc_bias
        for context in CONTEXTS:
            if key.startswith(context + "_"):
                head, tail = context, key[len(context) + 1 :]
                break
        else:
            return None
    stem = tail
    for suffix in ("_bias", "_fraction", "_target", "_content", "_level", "_min", "_max", "_floor", "_cap"):
        stem = stem.replace(suffix, "")
    quantity = {
        "gc": "gc_fraction",
        "a": "a_fraction",
        "u": "u_fraction",
        "purine": "purine_fraction",
        "g_run": "g_run_max",
        "g_run_max": "g_run_max",
        "gu": "gu_pair_fraction",
        "gu_pair": "gu_pair_fraction",
        "wobble": "gu_pair_fraction",
    }.get(stem)
    if quantity is None:
        return None
    return {
        "statement": f"{head}: {quantity} near {value}",
        "context": head,
        "quantity": quantity,
        "minimum": float(value),
        "rationale": f"converted from the setting {key!r}",
    }


def _as_rule(key: str, value: float) -> str:
    """The same thing as JSON, for an error message."""
    rule = as_rule(key, value) or {
        "statement": f"all: gc_fraction near {value}",
        "context": "all",
        "quantity": "gc_fraction",
        "minimum": value,
    }
    return json.dumps(rule)


#: Seven or more nucleotides in a row is a target-specific answer. Shorter motifs (GAAA, UUCG) are vocabulary.
SEQUENCE_FRAGMENT = re.compile(r"\b[ACGU]{7,}\b")


def target_specific(ruleset: Ruleset) -> str:
    """What in this ruleset would not survive being moved to another target. The notes travel with it too."""
    for text in [ruleset.notes] + [t for rule in ruleset.rules for t in (rule.statement, rule.rationale)]:
        found = SEQUENCE_FRAGMENT.search(text.upper())
        if found:
            return f"sequence fragment {found.group()!r}"
        if "eterna" in text.lower() or "openknot" in text.lower():
            return "a target or suite name"
    return ""


def contexts_of(structure: str) -> dict[str, list[int]]:
    """Which positions each context covers, read off the target's own notation."""
    from .structure import is_crossing, to_pairs

    pairs = to_pairs(structure)
    crossing = {p for p in pairs if any(is_crossing(p, q) for q in pairs)}
    paired = {i for pair in pairs for i in pair}
    length = len(structure)
    flank = max(1, length // 10)
    # isolated: a pair with no pair stacked on either side. closing: a pair next to an unpaired base.
    pairset = set(pairs)
    isolated = {
        i for (a, b) in pairs if (a - 1, b + 1) not in pairset and (a + 1, b - 1) not in pairset for i in (a, b)
    }
    closing = {
        i
        for pair in pairs
        for i in pair
        if any(n not in paired and 0 <= n < length for n in (pair[0] + 1, pair[0] - 1, pair[1] + 1, pair[1] - 1))
    }
    return {
        "stem": sorted(paired),
        "loop": [i for i in range(length) if i not in paired],
        "closing": sorted(closing),
        "isolated": sorted(isolated),
        "crossing": sorted({i for pair in crossing for i in pair}),
        "five_prime_end": list(range(min(flank, length))),
        "three_prime_end": list(range(max(0, length - flank), length)),
        "all": list(range(length)),
    }


def measure(sequence: str, positions: list[int], quantity: str, structure: str = "") -> float | None:
    """The quantity a rule prescribes, over the positions its context covers. None when it is inapplicable."""
    if not positions:
        return None
    if quantity == "gu_pair_fraction":
        # A pair quantity: positions alone cannot say what pairs with what.
        from .structure import to_pairs

        if not structure:
            return None
        covered = set(positions)
        pairs = [
            (i, j)
            for i, j in to_pairs(structure)
            if i in covered and j in covered and i < len(sequence) and j < len(sequence)
        ]
        if not pairs:
            return None
        wobble = sum(1 for i, j in pairs if {sequence[i], sequence[j]} == {"G", "U"})
        return wobble / len(pairs)
    bases = [sequence[i] for i in positions if i < len(sequence)]
    if not bases:
        return None
    if quantity == "g_run_max":
        longest = run = 0
        for base in bases:
            run = run + 1 if base == "G" else 0
            longest = max(longest, run)
        return float(longest)
    counts = {b: bases.count(b) / len(bases) for b in "ACGU"}
    return {
        "gc_fraction": counts["G"] + counts["C"],
        "a_fraction": counts["A"],
        "u_fraction": counts["U"],
        "purine_fraction": counts["A"] + counts["G"],
    }[quantity]


def check(sequence: str, structure: str, ruleset: Ruleset) -> dict[str, Any]:
    """How far a candidate is from obeying each rule, and whether it does."""
    contexts = contexts_of(structure)
    violations = []
    inapplicable = 0
    for rule in ruleset.rules:
        value = measure(sequence, contexts[rule.context], rule.quantity, structure)
        if value is None:
            inapplicable += 1
            continue
        if rule.minimum is not None and value < rule.minimum:
            violations.append(f"{rule.context}.{rule.quantity}={value:.3f} < {rule.minimum}")
        if rule.maximum is not None and value > rule.maximum:
            violations.append(f"{rule.context}.{rule.quantity}={value:.3f} > {rule.maximum}")
    return {
        "obeys": not violations,
        "violations": violations,
        "n_rules": len(ruleset.rules),
        "inapplicable": inapplicable,
    }
