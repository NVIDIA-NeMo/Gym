"""Type 52 loghub: real-log incident triage over LogHub BGL. Label every 50-line window with the alert categories in
it. Unit = one window. Gold = the dataset's own per-line alert labels (bgl_rules reproduces them exactly)."""
import gzip
import json
import random
import re
import threading
from pathlib import Path

from .. import common
from .bgl_rules import RULES, categorize

TYPE_ID = "52_loghub"
TIERS = {"easy": (64, 1800), "medium": (256, 1200), "hard": (2048, 600)}
HERE = Path(__file__).parent
POOL = HERE / "data" / "bgl_pool.jsonl.gz"
T = common.load_text(HERE)
CATS = [r[0] for r in RULES]
ANOM_SHARE = 0.25
_pool, _lock = None, threading.Lock()


def pool():
    global _pool
    with _lock:
        if _pool is None:
            with gzip.open(POOL, "rt") as f:
                _pool = [json.loads(x) for x in f]
    return _pool


def rulebook():
    rows = [f"- {c} | component {comp} | level {lvl} | message signature /{rx}/ | {T['categories'][c]}"
            for c, comp, lvl, rx in RULES]
    return T["rules_intro"] + "\n" + "\n".join(rows)


def generate(n, seed, out_dir):
    rng = random.Random(f"52:{seed}:{n}")
    p = pool()
    anom = [r for r in p if r["cats"]]
    rich = [r for r in p if not r["cats"] and r["n_decoy"] >= 5]
    plain = [r for r in p if not r["cats"] and r["n_decoy"] < 5]
    k_anom = round(ANOM_SHARE * n)
    k_plain = (n - k_anom) // 5
    # Anomalous windows are drawn category-first (uniform over categories with enough windows in the pool), so no
    # single category dominates and guessing the most common one scores low.
    by_cat = {}
    for r in anom:
        for c in r["cats"]:
            by_cat.setdefault(c, []).append(r)
    cats = sorted(c for c, rs in by_cat.items() if len(rs) >= 8)
    picked = {}
    while len(picked) < k_anom:
        r = rng.choice(by_cat[rng.choice(cats)])
        picked.setdefault(r["start"], r)
    chosen = (sorted(picked.values(), key=lambda r: r["start"]) + rng.sample(rich, n - k_anom - k_plain)
              + rng.sample(plain, k_plain))
    rng.shuffle(chosen)
    ids = [f"W{i:04d}" for i in range(1, n + 1)]
    windows = {wid: r["lines"] for wid, r in zip(ids, chosen)}
    gold = {wid: r["cats"] for wid, r in zip(ids, chosen) if r["cats"]}
    for wid, lines in windows.items():  # the shipped rules must reproduce the dataset labels exactly
        assert sorted({c for c in map(categorize, lines) if c}) == gold.get(wid, []), wid
    example = json.dumps({ids[0]: [CATS[0]], ids[1]: [CATS[22], CATS[3]]})
    prompt = "\n\n".join([T["intro"].replace("{n}", str(n)).replace("{lines}", "50"), rulebook(),
                          "Tools:\n" + "\n".join(f"- {k}: {v}" for k, v in T["tools"].items()),
                          T["answer"].replace("{example}", example), T["score"]])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt,
                         {"pairs": sorted([w, c] for w, cs in gold.items() for c in cs), "n": n, "seed": seed,
                          "source_starts": {w: r["start"] for w, r in zip(ids, chosen)}},
                         TIERS, files={"windows.json": json.dumps(windows)})


_worlds, _wlock = {}, threading.Lock()


def _load(world_dir):
    with _wlock:
        if world_dir not in _worlds:
            _worlds[world_dir] = json.loads((Path(world_dir) / "windows.json").read_text())
        return _worlds[world_dir]


def _ts(line):
    return line.split(" ", 4)[3].split(".")[0]  # 2005-06-03-15.42.50 (microseconds dropped to keep pages short)


def tools(world_dir):
    def list_windows(a):
        w = _load(world_dir)
        ids = sorted(w)
        off, lim = common.page_args(a, 200, 200)
        rows = [f"{i} | {_ts(w[i][0])} .. {_ts(w[i][-1])}" for i in ids[off:off + lim]]
        tail = f"[next: offset={off + lim}]" if off + lim < len(ids) else "[end]"
        return f"{len(ids)} windows; showing {off}-{off + len(rows) - 1}\n" + "\n".join(rows) + "\n" + tail

    def read_window(a):
        w = _load(world_dir)
        wid = str(a.get("window_id", "")).strip().upper()
        if wid not in w:
            raise KeyError(f"unknown window_id {a.get('window_id')!r}; ids look like W0001")
        lines = w[wid]
        start = int(a.get("start_line") or 1)
        count = int(a.get("max_lines") or len(lines))
        if not 1 <= start <= len(lines) or count < 1:
            raise ValueError(f"start_line must be 1..{len(lines)} and max_lines >= 1")
        out, size = [], 0
        for i in range(start, min(len(lines), start + count - 1) + 1):
            row = f"{i:02d}: {lines[i - 1]}"
            if size + len(row) > common.MAX_OUT - 120:
                out.append(f"[cut at line {i - 1} of {len(lines)}; continue with start_line={i}]")
                break
            out.append(row)
            size += len(row) + 1
        return f"{wid} ({len(lines)} lines)\n" + "\n".join(out)

    S, I = {"type": "string"}, {"type": "integer"}
    return [common.tool("list_windows", T["tools"]["list_windows"], {"offset": I, "limit": I}, [], list_windows),
            common.tool("read_window", T["tools"]["read_window"],
                        {"window_id": S, "start_line": I, "max_lines": I}, ["window_id"], read_window),
            common.tool("get_rulebook", T["tools"]["get_rulebook"], {}, [], lambda a: rulebook())]


WID = re.compile(r"^W\d{4}$")


def parse_answer(answer):
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(WID.match(str(k).strip().upper()) for k in o))
    out = {}
    if obj is not None:
        for k, v in obj.items():
            k = str(k).strip().upper()
            if WID.match(k):
                vals = v if isinstance(v, list) else re.split(r"[,\s]+", str(v))
                out[k] = {str(x).strip().upper() for x in vals if str(x).strip()}
        return out
    for line in (answer or "").splitlines():  # fallback: "W0012 | KERNDTLB, APPSEV"
        m = re.match(r"^\W*(W\d{4})\W*[|:]\s*(.+)$", common.clean_line(line))
        if m:
            out[m.group(1)] = {x for x in re.split(r"[,\s]+", m.group(2).upper()) if x and x != "NONE"}
    return out


def score(task, answer):
    gold = {tuple(p) for p in common.gold_of(task)["pairs"]}
    pred = {(w, c) for w, cs in parse_answer(answer).items() for c in cs if c in CATS}
    f, p, r = common.f1(pred, gold)
    gw, pw = {w for w, _ in gold}, {w for w, _ in pred}
    return {"score": f, "score_partial": f, "score_exact": float(pred == gold), "precision": p, "recall": r,
            "window_f1": common.f1(pw, gw)[0], "n_gold_pairs": len(gold), "n_pred_pairs": len(pred)}
