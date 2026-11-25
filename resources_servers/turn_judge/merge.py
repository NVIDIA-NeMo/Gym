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
"""Parse pair verdicts, check them mechanically against the trajectories, merge a group's pairs into one label file.

Checks (as turn_judge tools/check_labels.py): a decision label's idx is snapped to the first item of its model call
(a reasoning item starts a call); each quote `#N[ output| args]: "text"` (or 'text', as checks.py accepts) must occur in
item N or another item of its call; a decision label with no verifiable quote is dropped (precision first).

Merge (LOG 065): a rollout judged n times (pairs x samples) keeps a decision when >= vote_frac*n judgments give it
(both when n == 2): with a group decision list (variant B) one vote per judgment per (class, decision id), else per
±1-call cluster of the class; the label goes on the modal call (earliest on a tie). Group ill_posed: the stage-1
verdict, or >= ill_min_votes pairs (a pair votes by the majority of its samples) with a share >= ill_frac; then no
decision labels anywhere. In a labelable group, judgments that answered ill_posed count as missing."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Sequence

DECISION = ("good", "harmful")
QUOTE = re.compile(
    r"""#(\d+)(?:\s+(?:output|args))?\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')"""
)  # = checks.QUOTE


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    t = text.strip()
    for cand in (t, re.sub(r"^```(?:json)?\s*|\s*```$", "", t)):
        try:
            v = json.loads(cand, strict=False)
            return v if isinstance(v, dict) else None
        except Exception:
            pass
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0), strict=False) if m else None
    except Exception:
        return None


def call_starts(output: Sequence[dict]) -> list[int]:
    return [i for i, it in enumerate(output) if i == 0 or it.get("type") == "reasoning"]


def call_of(starts: Sequence[int], idx: int) -> int:
    return sum(1 for s in starts if s <= idx) - 1


def _as_text(v: Any) -> str:
    """A judge's free-text field as one string (a list of strings is joined; anything else str())."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_as_text(x) for x in v)
    return v if isinstance(v, str) else str(v)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def check_labels(entry: dict, output: Sequence[dict]) -> dict:
    """Snap idx to the call start, verify quotes; returns the entry with decision labels that pass, plus stats."""
    starts = call_starts(output)
    texts = [_norm(json.dumps(it, ensure_ascii=False)) for it in output]
    raw = [
        _norm(json.dumps(it, ensure_ascii=False).encode().decode("unicode_escape", errors="ignore")) for it in output
    ]
    kept, stats = [], Counter()
    seen_calls = set()
    for lab in entry.get("labels") or []:
        if lab.get("label") not in DECISION:
            kept.append(lab)
            continue
        stats["decision"] += 1
        idx = lab.get("idx")
        if not isinstance(idx, int) or not 0 <= idx < len(output):
            stats["bad_idx"] += 1
            continue
        c = call_of(starts, idx)
        first, last = starts[c], (starts[c + 1] - 1 if c + 1 < len(starts) else len(output) - 1)
        if idx != first:
            stats["snapped"] += 1
            lab = {**lab, "idx": first, "idx_original": idx}
        ev = lab.get("evidence") or ""
        quotes = QUOTE.findall(" ".join(map(str, ev)) if isinstance(ev, list) else str(ev))  # judges sometimes list
        ok = 0
        for n, q1, q2 in quotes:
            n, q = int(n), q1 or q2
            qn = _norm(q.encode().decode("unicode_escape", errors="ignore")) if "\\" in q else _norm(q)
            pool = range(first, last + 1) if first <= n <= last else [n] if 0 <= n < len(output) else []
            if qn and any(qn in texts[k] or qn in raw[k] for k in pool):
                ok += 1
        lab = {**lab, "mechanical": {"quotes": len(quotes), "verified": ok}}
        if ok == 0:
            stats["dropped_unverified"] += 1
            continue
        if (c, lab["label"]) in seen_calls or (c, "good" if lab["label"] == "harmful" else "harmful") in seen_calls:
            stats["dropped_second_in_call"] += 1  # one decision label per call (rule 6 / P7)
            continue
        seen_calls.add((c, lab["label"]))
        kept.append(lab)
    return {**entry, "labels": kept, "check": dict(stats)}


def side_entry(verdict: dict, side: str, line: int) -> dict | None:
    """The pair verdict's entry for one side: the requested {"fail": {...}, "pass": {...}} shape, or a LABELING §1
    style {"entries": [...]} list (matched by line), which judges sometimes write instead."""
    e = verdict.get(side)
    if isinstance(e, dict):
        return e
    for x in verdict.get("entries") or []:
        if isinstance(x, dict) and x.get("line") == line:
            return x
    return None


def _runs(calls) -> list[list[int]]:
    """Sorted distinct calls split into runs of consecutive (±1) calls."""
    out: list[list[int]] = []
    for c in sorted(set(calls)):
        if out and c - out[-1][-1] <= 1:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def _modal(calls) -> int:
    cc = Counter(calls).most_common()
    return min(c for c, n in cc if n == cc[0][1])


def vote_labels(
    judged: list[dict | None],
    starts: Sequence[int],
    vote_frac: float = 0.5,
    decision_ids: bool = True,
    good_earliest: bool = False,
) -> list[dict]:
    """judged = one checked entry per judgment of a rollout (None = no entry). Returns the kept decision labels
    (one per call; on a call wanted by both classes the better-voted wins, harmful on a tie, P7) and every note.
    good_earliest: a `good` with a list id is kept once per id when enough judgments give it, at the earliest call any
    of them proposed (samples agree on the decision but place goods late, LOG 067); harmful keeps the per-run vote
    (P8 / rule-15 second labels)."""
    n = len(judged)
    need = 2 if n == 2 else max(1, math.ceil(vote_frac * n - 1e-9))
    labs, notes = [], []  # (judgment, call, class, decision id, label)
    for i, e in enumerate(judged):
        for lab in (e or {}).get("labels") or []:
            if lab.get("label") in DECISION:
                labs.append(
                    (i, call_of(starts, lab["idx"]), lab["label"], str(lab.get("decision") or "").strip(), lab)
                )
            else:
                notes.append(lab)
    groups: dict[tuple, list] = {}
    for x in labs:
        if decision_ids and x[3]:
            groups.setdefault((x[2], "id", x[3]), []).append(x)
    for cls in DECISION:  # labels without a list id: ±1-call clusters
        rest = [x for x in labs if x[2] == cls and not (decision_ids and x[3])]
        for run in _runs(x[1] for x in rest):
            groups[(cls, "calls", str(run[0]))] = [x for x in rest if x[1] in run]
    kept: dict[int, tuple] = {}  # call -> (votes, class, label)
    for (cls, kind, _), mem in groups.items():
        picks = []
        if good_earliest and cls == "good" and kind == "id":
            if len({x[0] for x in mem}) >= need:
                picks.append((len({x[0] for x in mem}), min(x[1] for x in mem)))
            mem_picks = picks
        else:
            mem_picks = None
        for run in _runs(x[1] for x in mem) if mem_picks is None else []:  # P8: one label per ±1 run
            sub = [x for x in mem if x[1] in run]
            if len({x[0] for x in sub}) >= need:
                picks.append((len({x[0] for x in sub}), _modal(x[1] for x in sub)))
        if (
            mem_picks is None and not picks and kind == "id" and len({x[0] for x in mem}) >= need
        ):  # same id, other call
            picks.append((len({x[0] for x in mem}), _modal(x[1] for x in mem)))
        if kind == "id" and len(picks) > 1:
            # vote-split duplicates (LOG 072): real P8 multiples come from a judgment that labels both calls; runs of the
            # same (class, id) reached by disjoint voter sets are one decision placed differently -> keep the best-voted
            runs_v = [(v, c, {x[0] for x in mem if abs(x[1] - c) <= 1}) for v, c in picks]
            if not any(a[2] & b[2] for i, a in enumerate(runs_v) for b in runs_v[i + 1 :]):
                picks = [max(picks, key=lambda vc: (vc[0], -vc[1]))]
        for v, c in picks:
            lab = next(x[4] for x in mem if x[1] == c)
            prev = kept.get(c)
            if prev is None or (v, cls == "harmful") > (prev[0], prev[1] == "harmful"):
                kept[c] = (v, cls, {**lab, "votes": f"{v}/{n}"})
    # id-split fallback (LOG 072): no id of a class reached need, but >= need judgments put same-class labels on the same
    # +-1 call (under different ids) -> keep one label there
    for cls in DECISION:
        if any(k[1] == cls for k in kept.values()) or not decision_ids:
            continue
        mine = [x for x in labs if x[2] == cls]
        for run in _runs(x[1] for x in mine):
            sub = [x for x in mine if x[1] in run]
            if len({x[0] for x in sub}) >= need and n > 1:
                c = _modal(x[1] for x in sub)
                if c not in kept:
                    lab = next(x[4] for x in sub if x[1] == c)
                    kept[c] = (
                        len({x[0] for x in sub}),
                        cls,
                        {**lab, "votes": f"{len({x[0] for x in sub})}/{n}", "merge": "id-split fallback"},
                    )
    return [kept[c][2] for c in sorted(kept)] + notes


def merge_group(
    meta: dict,
    pairs: list[dict],
    outputs: dict[int, Sequence[dict]],
    ill_frac: float = 0.5,
    ill_min_votes: int = 1,
    vote_frac: float = 0.5,
    group_ill: bool | None = None,
    decision_ids: bool = True,
    good_earliest: bool = False,
) -> dict:
    """meta = group.json; pairs = [{fail, passing, larger_side, verdict (parsed pair.json or None)}], one item per
    judgment (a pair judged k times appears k times); outputs[k] = rollout k's response.output; decision_ids: the group
    had a stage-1 decision list, so labels vote by decision id. Returns the group label file (LABELING §1)."""
    judgments = [p for p in pairs if p.get("verdict")]
    by_pair: dict[tuple, list] = defaultdict(list)
    for p in judgments:
        by_pair[(p["fail"], p["passing"])].append(bool(p["verdict"].get("ill_posed")))
    # LABELING P6: one ill-posed failure mode makes the whole group ill_posed, and a pair only sees its own contrast.
    # group_ill: the stage-1 group analyst's verdict (variant B) decides when given, plus the pair vote: at least
    # ill_min_votes pairs AND a share >= ill_frac (round 2: single stray pair votes flipped labelable groups).
    votes = sum(1 for f in by_pair.values() if 2 * sum(f) >= len(f))
    ill = bool(group_ill) or (bool(by_pair) and votes >= max(1, ill_min_votes) and votes / len(by_pair) >= ill_frac)
    ev = next((p["verdict"].get("ill_posed_evidence", "") for p in judgments if p["verdict"].get("ill_posed")), "")
    lines = {r["R"]: r["line"] for r in meta["rollouts"]}
    per = defaultdict(list)
    for p in pairs:
        v = p.get("verdict") or {}
        if v.get("ill_posed") and not ill:  # a stray ill_posed judgment has no labels: missing, not a "no label" vote
            continue
        for side, k in (("fail", p["fail"]), ("pass", p["passing"])):
            e = side_entry(v, side, lines[k])
            per[k].append(check_labels(e, outputs[k]) if e else None)
    entries = []
    for r in meta["rollouts"]:
        k, verdict = r["R"], ("PASS" if r["reward"] > 0.5 else "FAIL")
        ent = {
            "file": r["file"],
            "line": r["line"],
            "verdict": verdict,
            "group_ill_posed": ill,
            "ill_posed_evidence": ev if ill else "",
            "cause": "",
            "summary": "",
            "labels": [],
        }
        got = [e for e in per.get(k, []) if e]
        ent["pairs"] = len(per.get(k, []))
        if ill:
            ent["cause"] = "ill_posed" if verdict == "FAIL" else ""
        elif got:
            ent["labels"] = vote_labels(per[k], call_starts(outputs[k]), vote_frac, decision_ids, good_earliest)
            # Judges sometimes write these free-text fields as lists (as `evidence`, LOG 067): flatten, never crash.
            causes = Counter(_as_text(e.get("cause")) for e in got if e.get("cause"))
            ent["cause"] = causes.most_common(1)[0][0] if verdict == "FAIL" and causes else ""
            ent["summary"] = " || ".join(_as_text(e.get("summary"))[:800] for e in got)
            ent["check"] = dict(sum((Counter(e.get("check", {})) for e in got), Counter()))
        entries.append(ent)
    return {
        "labeler": "turn-judge-pair (sandbox+opencode)",
        "rules": "LABELING.md (annotator edition)",
        "ill_votes": f"{votes}/{len(by_pair)} pairs ({sum(f for fs in by_pair.values() for f in fs)}/"
        f"{len(judgments)} judgments)",
        "entries": entries,
    }
