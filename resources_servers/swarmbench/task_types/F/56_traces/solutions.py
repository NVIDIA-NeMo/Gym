"""Type 56 solutions: tool-only oracle and the plausible-wrong readings of the rules."""
import json
import re

from . import type as track


def _fns(task):
    return {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}


def read_all(task):
    fns, cids, off = _fns(task), [], 0
    while True:
        page = fns["list_classes"]({"offset": off})
        cids += re.findall(r"^(C\d{4}) \|", page, re.M)
        if "[end]" in page:
            break
        off += 100
    out = {}
    for cid in cids:
        tids = re.findall(r"^([0-9a-f]{16}) \|", fns["list_traces"]({"class_id": cid}), re.M)
        ts = []
        for tid in tids:
            rows = fns["get_trace"]({"trace_id": tid}).splitlines()[2:]
            ts.append([[p[0], p[1], p[2], p[3], int(p[4]), int(p[5]), p[6]] for p in (r.split(" | ") for r in rows)])
        out[cid] = ts
    return out


def best(stats):
    return max(stats.items(), key=lambda x: x[1])[0]


def oracle(task):
    return json.dumps({c: best(track.class_stats(ts)) for c, ts in read_all(task).items()})


def _naive(ts, kind):
    tot, k = {}, 0
    for spans in ts:
        if kind != "duration_incl_errors" and any(s[1] == "-" and s[6] == "ERROR" for s in spans):
            continue
        k += 1
        child = {}
        for s in spans:
            child[s[1]] = child.get(s[1], 0) + s[5]
        for s in spans:
            if s[1] == "-" and kind.startswith("duration"):
                continue
            v = s[5] if kind.startswith("duration") else s[5] - child.get(s[0], 0)
            key = f"{s[2]} {s[3]}"
            tot[key] = tot.get(key, 0) + v
    return {key: v / max(k, 1) for key, v in tot.items()}


def minus_sum(task):
    # Diagnostic, not a wrong solution: duration minus the SUM of child durations gives the same answer in 90-100%
    # of classes (README), because the planted operation dominates its class.
    return json.dumps({c: best(_naive(ts, "minus_sum")) for c, ts in read_all(task).items()})


def wrong(task):
    data = read_all(task)
    return {
        "longest_non_root_span": json.dumps({c: best(_naive(ts, "duration")) for c, ts in data.items()}),
        "errors_included": json.dumps({c: best(track.class_stats(ts, True)) for c, ts in data.items()}),
    }


def units(task, k, rng):
    gold = json.loads(open(task["gold_path"]).read())["answers"]
    rules = "\n".join(f"- {x.lstrip('- ')}" for x in track.T["rules"])
    out = []
    for cid in rng.sample(sorted(gold), min(k, len(gold))):
        g = gold[cid]
        prompt = track.T["unit_prompt"].replace("{cid2}", cid).replace("{cid}", cid).replace("{rules}", rules)

        def check(text, g=g, cid=cid):
            m = re.findall(rf"UNIT\W*\|\W*{cid}\W*\|\s*(.+)", text or "")
            return None if not m else track._norm(m[-1]).strip("*` ") == g
        out.append({"id": f"{task['id']}:{cid}", "kind": "class", "prompt": prompt, "gold": g, "check": check})
    return out
