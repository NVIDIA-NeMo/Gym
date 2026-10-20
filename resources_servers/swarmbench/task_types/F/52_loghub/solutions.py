"""Type 52 solutions: tool-only oracle (reads every window, applies the shipped rulebook), plausible-wrong
heuristics, single units."""
import json
import re
from collections import Counter

from . import type as track
from .bgl_rules import categorize, split


def _fns(task):
    return {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}


def read_all(task):
    fns, ids, off = _fns(task), [], 0
    while True:
        page = fns["list_windows"]({"offset": off})
        ids += re.findall(r"^(W\d{4}) \|", page, re.M)
        if "[end]" in page:
            break
        off += 200
    out = {}
    for wid in ids:
        lines, start = [], 1
        while True:
            txt = fns["read_window"]({"window_id": wid, "start_line": start})
            rows = re.findall(r"^\d\d: (.*)$", txt, re.M)
            lines += rows
            m = re.search(r"continue with start_line=(\d+)", txt)
            if not m:
                break
            start = int(m.group(1))
        out[wid] = lines
    return out


def oracle(task):
    ans = {}
    for wid, lines in read_all(task).items():
        cats = sorted({c for c in map(categorize, lines) if c})
        if cats:
            ans[wid] = cats
    return json.dumps(ans)


def wrong(task):
    wins = read_all(task)
    sev, sev_cat, kw = {}, {}, {}
    common_cat = Counter()
    for lines in wins.values():
        common_cat.update(c for c in map(categorize, lines) if c)
    top = common_cat.most_common(1)[0][0] if common_cat else "KERNDTLB"
    for wid, lines in wins.items():
        lv = [split(x) for x in lines]
        if any(l in ("FATAL", "FAILURE") for _, l, _ in lv):
            sev[wid] = [top]
            # severity + component guess: KERNEL -> the kernel category seen most, APP -> APPSEV, ...
            comp = Counter(c for c, l, _ in lv if l in ("FATAL", "FAILURE")).most_common(1)[0][0]
            sev_cat[wid] = [{"KERNEL": "KERNDTLB", "APP": "APPSEV", "LINKCARD": "LINKIAP",
                             "BGLMASTER": "MASABNORM", "MONITOR": "MONPOW"}.get(comp, top)]
        if any(re.search(r"interrupt|failed|error", m, re.I) for _, l, m in lv if l in ("FATAL", "FAILURE")):
            kw[wid] = [top]
    return {"fatal_any": json.dumps(sev), "fatal_by_component": json.dumps(sev_cat), "keyword": json.dumps(kw),
            "all_windows_top": json.dumps({w: [top] for w in wins})}


def units(task, k, rng):
    gold = {}
    for w, c in json.loads(open(task["gold_path"]).read())["pairs"]:
        gold.setdefault(w, set()).add(c)
    wins = json.loads(open(task["world_dir"] + "/windows.json").read())
    anom = sorted(gold)
    decoy = sorted(w for w in wins if w not in gold and sum(split(x)[1] != "INFO" for x in wins[w]) >= 5)
    pick = rng.sample(anom, min(len(anom), (k + 1) // 2)) + rng.sample(decoy, min(len(decoy), k // 2))
    rules = track.rulebook()
    out = []
    for w in pick:
        g = gold.get(w, set())
        prompt = (track.T["unit_prompt"].replace("{window2}", w).replace("{window}", w).replace("{rules}", rules))

        def check(text, g=g, w=w):
            m = re.findall(rf"UNIT\W*\|\W*{w}\W*\|\s*([A-Z, ]+)", (text or "").upper())
            if not m:
                return None
            got = {x for x in re.split(r"[,\s]+", m[-1]) if x and x != "NONE"}
            return got == g
        out.append({"id": f"{task['id']}:{w}", "kind": "alert" if g else "decoy", "prompt": prompt,
                    "gold": sorted(g), "check": check})
    return out
