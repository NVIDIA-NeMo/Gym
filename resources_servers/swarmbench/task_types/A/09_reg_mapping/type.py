"""Type A09 regulation mapping: trace many internal policy clauses to the regulation paragraph each implements.

Real source: ten parts of the US Code of Federal Regulations (eCFR, 2026-10-01; public domain): Regulations B, E, P,
V, X and DD, the COPPA, Safeguards and Identity Theft rules, and the Bank Secrecy Act rules for banks. Each clause was
written by self-hosted K3 from one first-level paragraph (e.g. 12 CFR 1024.35(e)) in a company's own voice, without
numbers or names of the regulation, and kept only if a blind K3 pass picked the right paragraph out of 8 similar ones
(data/build, README). One unit = one clause; it counts if the cited section and first-level paragraph are right (a
deeper citation inside that paragraph is accepted). Parallel provisions (error resolution vs information requests,
initial vs annual vs revised notices, furnisher vs reseller duties) make near misses common.
"""
import json
import math
import random
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

from .. import common

TYPE_ID = "09_reg_mapping"
TIERS = {"easy": (6, 1800), "medium": (60, 1200), "hard": (180, 1200)}
HERE = Path(__file__).parent
T = common.load_text(HERE)
_pool, _regs, _lock = [], {}, threading.Lock()


def pool():
    with _lock:  # loaded whole under a lock (instances are generated concurrently in threads)
        if not _pool:
            _pool.extend([json.loads(x) for x in open(HERE / "data" / "pool.jsonl")])
    return _pool


def regs():
    """sections by number, the flat paragraph list for search, and an inverted index (built once per process)."""
    with _lock:
        if not _regs:
            d = json.loads((HERE / "data" / "regs.json").read_text())
            _regs["parts"] = d["parts"]
            _regs["sections"] = {s["section"]: s for s in d["sections"]}
            paras = [(s["section"], p["des"], p["text"]) for s in d["sections"] for p in s["paras"]]
            post, lens = defaultdict(list), []
            for i, (sec, _, t) in enumerate(paras):
                tf = Counter(_toks(_regs["sections"][sec]["heading"] + " " + t))
                lens.append(sum(tf.values()))
                for w, c in tf.items():
                    post[w].append((i, c))
            _regs.update(paras=paras, post=post, lens=lens, avg=sum(lens) / len(lens))
    return _regs


def generate(n, seed, out_dir):
    order = list(range(len(pool())))
    random.Random(f"a09:{seed}").shuffle(order)
    units = [pool()[i] for i in order[:n]]
    gold, lines = {}, []
    for k, u in enumerate(units):
        kid = f"K{k + 1:04d}"
        gold[kid] = {"cite": u["cite"], "section": u["section"], "para": u["cite"].split(u["section"])[1]}
        lines.append(f"{kid}\t{u['clause']}")
    src = json.loads((HERE / "data" / "SOURCES.json").read_text())
    files = {"clauses.tsv": "clause_id\tclause\n" + "\n".join(lines) + "\n",
             "SNAPSHOT.txt": f"regs.json sha256 {src['regs_json_sha256']}\n"}
    prompt = common.fill(T["prompt"], n_clauses=n, n_parts=len(regs()["parts"]))
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"clauses": gold}, TIERS, files)


# ---------------------------------------------------------------- tools
_WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on at to for and or is be by with as from that this which any such shall may must its it "
           "under not are has have if".split())


def _toks(s):
    return [w for w in _WORD.findall(s.lower()) if w not in STOP]


def tools(world_dir):
    world = Path(world_dir)

    def list_clauses(a):
        ls = (world / "clauses.tsv").read_text().splitlines()
        off, lim = common.page_args(a, 40, 80)
        body = ls[1 + off:1 + off + lim]
        return f"{ls[0]}\n" + "\n".join(body) + f"\n[clauses {off + 1}-{off + len(body)} of {len(ls) - 1}]"

    def list_sections(a):
        r = regs()
        part = str(a.get("part") or "").strip()
        secs = [s for s in r["sections"].values() if not part or s["part"] == part]
        if not secs:
            raise KeyError(f"unknown part {part!r}; parts: " + ", ".join(r["parts"]))
        off, lim = common.page_args(a, 120, 250)
        body = [f"{s['cite']}\t{s['heading']}" for s in secs[off:off + lim]]
        head = "parts: " + "; ".join(f"{p} = {n}" for p, n in r["parts"].items())
        return head + "\n" + "\n".join(body) + f"\n[sections {off + 1}-{off + len(body)} of {len(secs)}]"

    def read_section(a):
        r = regs()
        m = re.search(r"(\d{3,4}\.\d+[a-z]?)", str(a.get("section") or ""))
        s = r["sections"].get(m.group(1)) if m else None
        if not s:
            raise KeyError(f"unknown section {a.get('section')!r}; use list_sections or search")
        start = int(a.get("start") or 0)
        if start < 0:
            raise ValueError("start must be >= 0")
        out, size = [f"{s['cite']} {s['heading']}  [{len(s['paras'])} paragraphs]"], 0
        k = start
        for k in range(start, len(s["paras"])):
            line = f"[{k}] {s['paras'][k]['text']}"
            if size + len(line) > 7000:
                out.append(f"... continue with start={k}")
                break
            out.append(line)
            size += len(line)
        return "\n".join(out)

    def search(a):
        r = regs()
        q = _toks(str(a.get("query") or ""))[:30]
        if not q:
            raise ValueError("query needs at least one content word")
        lim = min(int(a.get("limit") or 8), 15)
        sc = defaultdict(float)
        N = len(r["paras"])
        for w in set(q):
            ps = r["post"].get(w, [])
            idf = math.log(1 + (N - len(ps) + 0.5) / (len(ps) + 0.5))
            for i, c in ps:
                sc[i] += idf * c * 2.2 / (c + 1.2 * (0.25 + 0.75 * r["lens"][i] / r["avg"]))
        top = sorted(sc, key=lambda i: (-sc[i], i))[:lim]
        if not top:
            return "no matching paragraph"
        return "\n".join(f"{r['sections'][r['paras'][i][0]]['title']} CFR {r['paras'][i][0]}{r['paras'][i][1]}\t"
                         f"{r['paras'][i][2][:180]}" for i in top)

    page = {"offset": {"type": "integer"}, "limit": {"type": "integer"}}
    return [
        common.tool("list_clauses", T["tool_list_clauses"], page, [], list_clauses),
        common.tool("list_sections", T["tool_list_sections"], {"part": {"type": "string"}, **page}, [], list_sections),
        common.tool("read_section", T["tool_read_section"], {"section": {"type": "string"}, "start": {"type": "integer"}},
                    ["section"], read_section),
        common.tool("search", T["tool_search"], {"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"],
                    search),
    ]


# ---------------------------------------------------------------- scoring
_CITE = re.compile(r"(\d{3,4}\.\d+[a-z]?)\s*((?:\([A-Za-z0-9]{1,4}\))*)")


def norm_cite(v):
    """'12 CFR 1024.35(e)(1)' -> ('1024.35', '(e)'); None if there is no section with a first-level paragraph."""
    m = _CITE.search(str(v))
    if not m or not m.group(2):
        return None
    first = re.match(r"\([A-Za-z0-9]{1,4}\)", m.group(2)).group()
    return m.group(1), first


def parse_answer(answer):
    obj = common.last_json(answer if isinstance(answer, str) else json.dumps(answer),
                           lambda o: isinstance(o, dict) and any(re.fullmatch(r"K\d{4}", str(k)) for k in o))
    return {str(k): norm_cite(v) for k, v in (obj or {}).items() if re.fullmatch(r"K\d{4}", str(k))}


def score(task, answer):
    gold = common.gold_of(task)["clauses"]
    pred = parse_answer(answer)
    ok = [pred.get(k) == (g["section"], g["para"]) for k, g in gold.items()]
    sec = sum(1 for k, g in gold.items() if pred.get(k) and pred[k][0] == g["section"])
    return {"score": sum(ok) / len(ok), "score_exact": float(all(ok)), "score_partial": sec / len(gold),
            "n_correct": sum(ok), "n_units": len(ok), "sections_correct": sec, "answered": sum(k in pred for k in gold)}
