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
"""Compute per-skill deltas from rollout JSONLs.

Supports two input shapes, auto-detected from `verifier_metadata`:

  Two-arm (legacy): records have `with_skill` only. Renders with/without/Δ
  per skill on three axes (reward, tool calls, output tokens).

  2×2 (Phase-1+): records have `cell` in {blind, docs-only, skill-only,
  skill+docs}. Renders one table per skill with four cells, plus "effects":
    Δskill | refs=T   = skill+docs − docs-only (realistic reader)
    Δskill | refs=F   = skill-only − blind     (skill standalone)
    Δrefs  | skill=T  = skill+docs − skill-only
    Δrefs  | skill=F  = docs-only  − blind

Two-file mode: same-structure diff of v1 vs v2 with per-field provenance
attribution (which inputs changed: md / evals / fx / judge / harness).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path


_PROV_FIELDS = ("skill_md_sha", "evals_sha", "fixtures_sha", "judge_prompt_sha", "harness_version")
_PROV_ABBREV = {
    "skill_md_sha": "md",
    "evals_sha": "evals",
    "fixtures_sha": "fx",
    "judge_prompt_sha": "judge",
    "harness_version": "harness",
}

_CELL_ORDER = ("blind", "docs-only", "skill-only", "skill+docs")


@dataclass
class CellStats:
    rewards: list[float] = field(default_factory=list)
    tool_calls: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rewards)

    @property
    def mean_reward(self) -> float:
        return statistics.fmean(self.rewards) if self.rewards else float("nan")

    @property
    def mean_tools(self) -> float:
        return statistics.fmean(self.tool_calls) if self.tool_calls else float("nan")

    @property
    def mean_tokens(self) -> float:
        return statistics.fmean(self.output_tokens) if self.output_tokens else float("nan")


@dataclass
class SkillStats:
    cells: dict[str, CellStats] = field(default_factory=dict)
    prov: dict[str, str] = field(default_factory=dict)

    def add(self, cell_name: str, reward: float, n_tools: int, n_tokens: int) -> None:
        cs = self.cells.setdefault(cell_name, CellStats())
        cs.rewards.append(reward)
        cs.tool_calls.append(n_tools)
        cs.output_tokens.append(n_tokens)


def _sha12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _extract_prov(md: dict, with_skill: bool) -> dict[str, str]:
    prov = {f: str(md.get(f) or "") for f in _PROV_FIELDS}
    if not prov["skill_md_sha"] and with_skill:
        prov["skill_md_sha"] = _sha12(str(md.get("skill_md") or "").encode("utf-8"))
    return prov


def _cell_from_record(md: dict) -> str:
    """Return the cell name. Pre-Phase-1 records lack `cell` but still have
    `with_skill`; map those onto the two-arm legacy cells so downstream code
    can treat both uniformly."""
    if md.get("cell"):
        return str(md["cell"])
    return "legacy:with" if md.get("with_skill") else "legacy:without"


def load_scoreboard(path: Path) -> dict[str, SkillStats]:
    buckets: dict[str, SkillStats] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            md = r.get("verifier_metadata") or {}
            skill_name = md.get("skill_name")
            reward = r.get("reward")
            if not skill_name or reward is None:
                continue
            cell = _cell_from_record(md)
            n_tools = len(md.get("tool_calls") or [])
            usage = (r.get("response") or {}).get("usage") or {}
            n_tokens = int(usage.get("output_tokens") or 0)

            stats = buckets.setdefault(skill_name, SkillStats())
            stats.add(cell, float(reward), n_tools, n_tokens)

            record_prov = _extract_prov(md, with_skill=bool(md.get("with_skill")))
            for k, v in record_prov.items():
                if v and not stats.prov.get(k):
                    stats.prov[k] = v
    return buckets


def _is_2x2(board: dict[str, SkillStats]) -> bool:
    """True if at least one skill has any of the 2×2 cell labels."""
    return any(cell in _CELL_ORDER for stats in board.values() for cell in stats.cells)


def _fmt_signed(d: float, width: int = 7, prec: int = 3) -> str:
    if d != d:  # nan
        return f"{'—':>{width}s}"
    return f"{d:+{width}.{prec}f}"


def _fmt_cell(stats: CellStats, width_r: int = 6) -> str:
    if stats.n == 0:
        return f"{'—':>{width_r}s}  {'—':>5s}  {'—':>6s}"
    return f"{stats.mean_reward:{width_r}.3f}  {stats.mean_tools:5.1f}  {stats.mean_tokens:6.0f}"


def _effects_2x2(stats: SkillStats) -> dict[str, tuple[float, float, float]]:
    """Return the four marginal effects as (Δreward, Δtools, Δtokens) tuples.
    NaN when a needed cell is missing."""
    c = stats.cells
    nan = float("nan")

    def diff(a: str, b: str) -> tuple[float, float, float]:
        if a not in c or b not in c:
            return (nan, nan, nan)
        return (
            c[a].mean_reward - c[b].mean_reward,
            c[a].mean_tools - c[b].mean_tools,
            c[a].mean_tokens - c[b].mean_tokens,
        )

    return {
        "skill | refs=T": diff("skill+docs", "docs-only"),
        "skill | refs=F": diff("skill-only", "blind"),
        "refs | skill=T": diff("skill+docs", "skill-only"),
        "refs | skill=F": diff("docs-only", "blind"),
    }


def print_scoreboard_2x2(board: dict[str, SkillStats], label: str) -> None:
    print(f"\n=== {label} (2×2 mode) ===")
    for name in sorted(board):
        stats = board[name]
        present = [c for c in _CELL_ORDER if c in stats.cells]
        if not present:
            continue
        print(f"\n{name}")
        print(f"  {'cell':12s}  {'reward':>6s}  {'tools':>5s}  {'tokens':>6s}  n")
        for c in _CELL_ORDER:
            if c in stats.cells:
                cs = stats.cells[c]
                print(f"  {c:12s}  {_fmt_cell(cs)}  {cs.n}")
        print(f"  {'effect':14s}  {'Δreward':>8s}  {'Δtools':>7s}  {'Δtokens':>8s}")
        for effect_name, (dr, dt, dk) in _effects_2x2(stats).items():
            print(f"  {effect_name:14s}  {_fmt_signed(dr, 8, 3)}  {_fmt_signed(dt, 7, 2)}  {_fmt_signed(dk, 8, 0)}")
    _print_prov_footer(board)


def print_scoreboard_legacy(board: dict[str, SkillStats], label: str) -> None:
    """Pre-Phase-1 two-arm rendering. Kept for historical JSONLs."""
    print(f"\n=== {label} (legacy 2-arm mode) ===")
    print(f"{'skill':24s}  {'with':>8s}  {'without':>8s}  {'Δreward':>8s}  {'Δtools':>7s}  {'Δtokens':>8s}  {'n':>4s}")
    print("-" * 82)
    for name in sorted(board):
        stats = board[name]
        w = stats.cells.get("legacy:with") or CellStats()
        wo = stats.cells.get("legacy:without") or CellStats()
        if w.n == 0 or wo.n == 0:
            continue
        dr = w.mean_reward - wo.mean_reward
        dt = w.mean_tools - wo.mean_tools
        dk = w.mean_tokens - wo.mean_tokens
        print(
            f"{name:24s}  {w.mean_reward:8.3f}  {wo.mean_reward:8.3f}  "
            f"{_fmt_signed(dr, 8, 3)}  {_fmt_signed(dt, 7, 2)}  {_fmt_signed(dk, 8, 0)}  {w.n:4d}"
        )
    _print_prov_footer(board)


def _print_prov_footer(board: dict[str, SkillStats]) -> None:
    print("\n  provenance per skill:")
    for name in sorted(board):
        p = board[name].prov
        tags = [f"{_PROV_ABBREV[f]}={p.get(f) or '—'}" for f in _PROV_FIELDS]
        print(f"    {name:24s}  " + "  ".join(tags))


def _prov_change_tag(p1: dict[str, str], p2: dict[str, str]) -> tuple[str, str]:
    changed = []
    both_known = 0
    total_populated = 0
    for f in _PROV_FIELDS:
        a, b = p1.get(f, ""), p2.get(f, "")
        if a and b:
            both_known += 1
            if a != b:
                changed.append(_PROV_ABBREV[f])
        elif a or b:
            changed.append(f"{_PROV_ABBREV[f]}?")
        if a or b:
            total_populated += 1
    if total_populated == 0:
        return "—", "legacy"
    if changed:
        return "+".join(changed), ""
    if both_known == len(_PROV_FIELDS):
        return "—", "same-all"
    return "—", f"partial({both_known}/{len(_PROV_FIELDS)})"


def print_diff(v1: dict[str, SkillStats], v2: dict[str, SkillStats]) -> None:
    is_2x2 = _is_2x2(v1) and _is_2x2(v2)
    print(f"\n=== diff ({'2×2' if is_2x2 else 'legacy'}) ===")
    for name in sorted(set(v1) | set(v2)):
        s1, s2 = v1.get(name), v2.get(name)
        if s1 is None or s2 is None:
            print(f"\n{name}  ({'v2 only' if s1 is None else 'v1 only'})")
            continue
        prov_diff, note = _prov_change_tag(s1.prov, s2.prov)
        print(f"\n{name}  [prov: {prov_diff}  {note}]")

        cells = _CELL_ORDER if is_2x2 else ("legacy:with", "legacy:without")
        print(f"  {'cell':12s}  {'Δreward':>9s}  {'Δtools':>8s}  {'Δtokens':>9s}")
        for c in cells:
            c1 = s1.cells.get(c)
            c2 = s2.cells.get(c)
            if c1 is None or c2 is None or c1.n == 0 or c2.n == 0:
                continue
            dr = c2.mean_reward - c1.mean_reward
            dt = c2.mean_tools - c1.mean_tools
            dk = c2.mean_tokens - c1.mean_tokens
            print(f"  {c:12s}  {_fmt_signed(dr, 9, 3)}  {_fmt_signed(dt, 8, 2)}  {_fmt_signed(dk, 9, 0)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollouts", type=Path, help="Rollout JSONL (v1 if --v2 is given, else just scoreboard).")
    p.add_argument("--v2", type=Path, default=None, help="If provided, diff v1 vs v2 per skill.")
    p.add_argument("--v1-label", type=str, default="v1")
    p.add_argument("--v2-label", type=str, default="v2")
    return p


def _render(board: dict[str, SkillStats], label: str) -> None:
    (print_scoreboard_2x2 if _is_2x2(board) else print_scoreboard_legacy)(board, label)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv if argv is not None else sys.argv[1:])
    v1 = load_scoreboard(args.rollouts)
    if args.v2 is None:
        _render(v1, label=str(args.rollouts))
        return 0
    v2 = load_scoreboard(args.v2)
    _render(v1, label=f"{args.v1_label}: {args.rollouts}")
    _render(v2, label=f"{args.v2_label}: {args.v2}")
    print_diff(v1, v2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
