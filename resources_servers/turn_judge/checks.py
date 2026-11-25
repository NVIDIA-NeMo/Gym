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
"""Mechanical checks of a pair verdict against the rendered trajectories (stdlib only: the same file runs inside the
judge sandbox as /judge/bin/check_pair.py, which the judge must run before finishing, and in the merge).

    python3 /judge/bin/check_pair.py            # checks /judge/out/pair.json (and decisions.json if required)
    python3 /judge/bin/check_pair.py --decisions  # checks a group stage's /judge/out/decisions.json (variant C)

Checks: valid JSON with the required shape; every good/harmful label has its reasoning fields, an idx that is the first
item of its model call, and at least one verbatim quote `#N[ output| args]: "text"` (or 'text') found in item N or
another item of its call; no decision labels when ill_posed; at most one decision label per model call; with a group decision list (/judge/decisions.json, variant C) a
`decisions_addressed` item (label idx with that decision id, or `none` + reason) per list decision and rollout.
--decisions: the stage-1 list has the variant C shape (label_policy, membership of every rollout, no placement)."""

import json
import re
import sys
from pathlib import Path

DECISION = ("good", "harmful")
REQUIRED = ("note", "evidence", "info_state_test", "contrary_evidence_checked", "decision_turn_check")
HDR = re.compile(r"^===== #(-?\d+) \[call (-?\d+)\] (\S+) =====$")  # #-1 = end sentinel (flushes the last item)
QUOTE = re.compile(r"""#(\d+)(?:\s+(?:output|args))?\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')""")


def parse_trajectory(text: str) -> dict:
    """{"items": {idx: (call, text)}, "starts": {call: first idx}} from a trajectory.txt (wrapped lines rejoined)."""
    lines = []
    for ln in text.split("\n"):
        if ln.startswith("↪ ") and lines:
            lines[-1] += ln[2:]
        else:
            lines.append(ln)
    items, starts, types, cur, buf = {}, {}, {}, None, []
    for ln in lines + ["===== #-1 [call -1] end ====="]:
        m = HDR.match(ln)
        if m:
            if cur is not None:
                items[cur[0]] = (cur[1], "\n".join(buf))
                types[cur[0]] = cur[2]
            cur, buf = (int(m.group(1)), int(m.group(2)), m.group(3)), []
            if cur[0] >= 0:
                starts.setdefault(cur[1], cur[0])
        else:
            buf.append(ln)
    return {"items": items, "starts": starts, "types": types}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


REOPEN = re.compile(r"\b(wait|reconsider|let me (?:check|verify|confirm)|must check|decisive)\b|\?\s*$", re.I)
WARNINGS: list[str] = []


def check_entry(entry: dict, traj: dict, where: str, ill_posed: bool, need_but_for: bool = False) -> list[str]:
    errs = []
    if not isinstance(entry.get("labels"), list):
        return [f"{where}: 'labels' must be a list"]
    calls_used = set()
    for i, lab in enumerate(entry["labels"]):
        tag = f"{where} label {i} (idx {lab.get('idx')}, {lab.get('label')})"
        if lab.get("label") not in DECISION:
            continue
        if ill_posed:
            errs.append(f"{tag}: ill_posed is true, so no good/harmful labels are allowed")
            continue
        for f in REQUIRED:
            if not str(lab.get(f) or "").strip():
                errs.append(f"{tag}: field '{f}' is empty")
        if need_but_for and lab.get("label") == "harmful" and not str(lab.get("but_for") or "").strip():
            errs.append(
                f"{tag}: field 'but_for' is empty (failing assertion, its input, what this call decided about it)"
            )
        idx = lab.get("idx")
        if not isinstance(idx, int) or idx not in traj["items"]:
            errs.append(f"{tag}: idx must be an item number of this rollout's trajectory")
            continue
        call = traj["items"][idx][0]
        first = traj["starts"][call]
        if need_but_for:  # variant E: warn when the labeled call's reasoning ends open and the dtc ignores its ending
            tail = norm(traj["items"].get(first, (0, ""))[1])[-600:]
            dtc = str(lab.get("decision_turn_check") or "")
            if REOPEN.search(tail[-300:]) and not any(
                len(q) > 20 and norm(q) in tail for q in re.findall(r'"([^"]{20,})"', dtc)
            ):
                WARNINGS.append(
                    f'{tag}: call {call} ends open ("...{tail[-160:]}"); quote its ending in '
                    f"decision_turn_check and say whether it is a lean (precision rule 1)"
                )
        if idx != first:
            errs.append(f"{tag}: idx {idx} is inside model call {call}; the label idx must be its first item #{first}")
        if call in calls_used:
            errs.append(f"{tag}: a second decision label on model call {call} (one decision label per call)")
        calls_used.add(call)
        quotes = QUOTE.findall(str(lab.get("evidence") or ""))
        if not quotes:
            errs.append(f'{tag}: evidence has no quote of the form #N: "exact text"')
            continue
        found = False
        for n, q1, q2 in quotes:
            q = norm(q1 or q2)
            n = int(n)
            if n not in traj["items"] or not q:
                continue
            pool = [n] + [k for k, (c, _) in traj["items"].items() if c == traj["items"][n][0]]
            if any(q in norm(traj["items"][k][1]) for k in pool):
                found = True
                break
        if not found:
            errs.append(f"{tag}: none of its {len(quotes)} quotes occurs verbatim in the cited item or its call")
    return errs


POLICIES = ("good+harmful", "harmful_only", "none")
PLACEMENT = re.compile(
    r"\bcalls?\s+\d+|#\d+|\bitems?\s+\d+|\bturns?\s+\d+|decision turn|the write is|"
    r"silent(?:ly)? inside",
    re.I,
)
MEMBERSHIP = {"right", "wrong", "wrong-uncued", "n/a", "unknown"}


def check_addressed(pair: dict, listed: dict) -> list[str]:
    """Every decision of the group list is addressed for both rollouts: a label of that rollout with that decision id
    (label_idx) or `none` with a reason."""
    ids = [str(d.get("id")) for d in listed.get("decisions") or [] if isinstance(d, dict) and d.get("id")]
    got = pair.get("decisions_addressed")
    if not isinstance(got, list):
        return ["'decisions_addressed' must be a list with one item per decision of /judge/decisions.json and rollout"]
    errs, have = [], set()
    for x in got:
        if not isinstance(x, dict):
            errs.append("decisions_addressed: every item must be an object")
            continue
        did, rk = str(x.get("id")), str(x.get("rollout"))
        side = next(
            (
                pair.get(s)
                for s in ("fail", "pass")
                if isinstance(pair.get(s), dict) and str(pair[s].get("rollout")) == rk
            ),
            None,
        )
        if side is None:
            errs.append(f"decisions_addressed {did}/{rk}: rollout must be one of this pair's two rollouts")
            continue
        have.add((did, rk))
        li = x.get("label_idx")
        if li is not None:
            if not any(
                lab.get("idx") == li and lab.get("label") in DECISION and str(lab.get("decision")) == did
                for lab in side.get("labels") or []
            ):
                errs.append(
                    f"decisions_addressed {did}/{rk}: label_idx {li} is not a good/harmful label of {rk} with "
                    f"decision '{did}'"
                )
        elif not str(x.get("none") or "").strip():
            errs.append(f"decisions_addressed {did}/{rk}: give label_idx or a non-empty 'none' reason")
    for s_ in ("fail", "pass"):
        rk = str((pair.get(s_) or {}).get("rollout") or "") if isinstance(pair.get(s_), dict) else ""
        for did in ids:
            if rk and (did, rk) not in have:
                errs.append(f"decisions_addressed: no item for decision {did} and {rk}")
    return errs


SYMBOL = re.compile(r"`([^`]{3,80})`")


def output_text(traj: dict) -> str:
    """All tool outputs of a rollout (what the agent saw)."""
    return "\n".join(t for k, (c, t) in traj["items"].items() if traj["types"].get(k) == "function_call_output")


def check_decisions(
    dec: dict, rollouts: list[str], trajs: dict[str, dict] | None = None, strict: bool = False
) -> list[str]:
    """The variant C group-stage list (strict, variant E: mechanical membership checks)."""
    errs = []
    if not isinstance(dec.get("group_ill_posed"), bool):
        errs.append("'group_ill_posed' must be true or false")
    ds = dec.get("decisions")
    if not isinstance(ds, list):
        return errs + ["'decisions' must be a list"]
    for i, d in enumerate(ds):
        if not isinstance(d, dict):
            errs.append(f"decision {i}: must be an object")
            continue
        tag = f"decision {d.get('id') or i}"
        for f in ("id", "what", "correct", "wrong", "tests", "label_policy", "membership", "how_to_recognize"):
            if not d.get(f):
                errs.append(f"{tag}: field '{f}' is empty")
        if d.get("label_policy") and d["label_policy"] not in POLICIES:
            errs.append(f"{tag}: label_policy must be one of {POLICIES}")
        m = d.get("membership")
        if isinstance(m, dict):
            miss = [r for r in rollouts if r not in m]
            if miss:
                errs.append(f"{tag}: membership must cover every rollout; missing {miss}")
            bad = sorted({str(v) for v in m.values()} - MEMBERSHIP)
            if bad:
                errs.append(f"{tag}: membership values must be one of {sorted(MEMBERSHIP)}, not {bad}")
        elif m:
            errs.append(f'{tag}: membership must be an object {{"R0": "right", ...}}')
        if strict and isinstance(m, dict):
            notes = d.get("membership_notes") if isinstance(d.get("membership_notes"), dict) else {}
            for r, v in m.items():
                if str(v) in ("n/a", "unknown") and not str(notes.get(r) or "").strip():
                    errs.append(
                        f"{tag}: {r} is '{v}': give the reason in membership_notes[\"{r}\"] (a rollout whose "
                        f"final patch contains the decision's code is right or wrong, not unknown)"
                    )
            wrong = [r for r, v in m.items() if str(v) == "wrong"]
            if (
                d.get("label_policy") == "good+harmful"
                and len(wrong) == 1
                and not str(d.get("visible_alternative") or "").strip()
            ):
                errs.append(
                    f"{tag}: good+harmful with a single wrong rollout ({wrong[0]}): name the visible alternative "
                    f"the others weighed in visible_alternative, or use harmful_only"
                )
            syms = [x for x in SYMBOL.findall(str(d.get("how_to_recognize") or "")) if x.strip()]
            if trajs and syms:
                for r in wrong:
                    if r in trajs and not any(x in output_text(trajs[r]) for x in syms):
                        errs.append(
                            f"{tag}: {r} is 'wrong' but none of the recognition symbols {syms[:4]} appears in "
                            f"its tool outputs: if it never saw the deciding fact use 'wrong-uncued', else add "
                            f"the symbol it saw to how_to_recognize"
                        )
        for f in ("what", "correct", "wrong", "how_to_recognize", "policy_reason"):
            hit = PLACEMENT.search(str(d.get(f) or ""))
            if hit:
                errs.append(
                    f"{tag}: '{f}' names a call / item / turn ('{hit.group(0)}'): give recognition cues only "
                    f"(files, symbols, commands, phrases), no placement"
                )
    return errs


def check_pair(
    pair: dict,
    trajs: dict[str, dict],
    need_decisions: dict | None = None,
    listed: dict | None = None,
    need_but_for: bool = False,
) -> list[str]:
    errs = []
    ill = pair.get("ill_posed")
    if not isinstance(ill, bool):
        errs.append("'ill_posed' must be true or false")
    for side in ("fail", "pass"):
        e = pair.get(side)
        if not isinstance(e, dict):
            errs.append(f"'{side}' must be an object (the {side}ing rollout's entry)")
            continue
        rk = str(e.get("rollout") or "")
        if rk not in trajs:
            errs.append(f"'{side}.rollout' must be one of {sorted(trajs)}")
            continue
        if side == "fail" and not e.get("cause"):
            errs.append("'fail.cause' is required")
        errs += check_entry(e, trajs[rk], side, bool(ill), need_but_for)
    if need_decisions is not None and not isinstance(need_decisions.get("decisions"), list):
        errs.append("decisions.json must have a 'decisions' list")
    if listed is not None and ill is not True:
        errs += check_addressed(pair, listed)
    return errs


def main() -> int:
    root = Path("/judge")
    if "--decisions" in sys.argv:
        try:
            dec = json.loads((root / "out/decisions.json").read_text())
            meta = json.loads((root / "ws/group.json").read_text())
        except FileNotFoundError as e:
            print(f"FAIL: {e.filename} does not exist")
            return 1
        except json.JSONDecodeError as e:
            print(f"FAIL: /judge/out/decisions.json is not valid JSON: {e}")
            return 1
        strict = (root / "DECISIONS_STRICT").exists()
        trajs = (
            {
                p.parent.name: parse_trajectory(p.read_text(errors="replace"))
                for p in sorted((root / "ws").glob("R*/trajectory.txt"))
            }
            if strict
            else None
        )
        errs = check_decisions(dec, [f"R{r['R']}" for r in meta["rollouts"]], trajs, strict)
        if errs:
            print(f"FAIL: {len(errs)} problem(s); fix them in /judge/out/decisions.json and run this check again:")
            for e in errs:
                print(" -", e)
            return 1
        print("PASS: /judge/out/decisions.json has the required format")
        return 0
    try:
        pair = json.loads((root / "out/pair.json").read_text())
    except FileNotFoundError:
        print("FAIL: /judge/out/pair.json does not exist")
        return 1
    except json.JSONDecodeError as e:
        print(f"FAIL: /judge/out/pair.json is not valid JSON: {e}")
        return 1
    trajs = {
        p.parent.name: parse_trajectory(p.read_text(errors="replace"))
        for p in sorted((root / "ws").glob("R*/trajectory.txt"))
    }
    dec = None
    if (root / "DECISIONS_REQUIRED").exists():
        try:
            dec = json.loads((root / "out/decisions.json").read_text())
        except Exception as e:
            print(f"FAIL: /judge/out/decisions.json missing or invalid: {e}")
            return 1
    listed = None
    if (root / "DECISIONS_LIST").exists():
        try:
            listed = json.loads((root / "decisions.json").read_text())
        except Exception:
            listed = None
    errs = check_pair(pair, trajs, dec, listed, (root / "BUT_FOR_REQUIRED").exists())
    for w in WARNINGS:
        print("WARN:", w)
    if errs:
        print(f"FAIL: {len(errs)} problem(s); fix them in /judge/out/pair.json and run this check again:")
        for e in errs:
            print(" -", e)
        return 1
    print("PASS: /judge/out/pair.json has the required format and every quote was found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
