"""Type A06 citation check: audit every citation of a long literature report against the cited papers' abstracts.

Real source: arXiv cs.CL abstracts (March-April 2024; arXiv metadata is CC0). Each citing sentence was written by
self-hosted K3 from one abstract, either faithfully or with one concrete detail changed so the abstract contradicts
it, and kept only if a blind K3 judgement agreed (data/build, README). Three verdicts per citation:
  supported     - the cited abstract states it;
  unsupported   - the cited paper is the right one but the sentence misstates it (no paper in the bibliography
                  states it);
  wrong_source  - the sentence is a faithful statement of a DIFFERENT paper in the bibliography (a topically close
                  one); the agent must also give that paper's reference id.
One unit = one report paragraph of 5 citing sentences; it counts only if all 5 verdicts (and the correct reference for
every wrong_source) are right. Task-facing prose is K3's (text/k3.json); abstracts, titles and arXiv ids are real.
"""
import json
import math
import random
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

from .. import common

TYPE_ID = "06_citation_check"
TIERS = {"easy": (4, 1800), "medium": (40, 1200), "hard": (150, 1200)}
HERE = Path(__file__).parent
T = common.load_text(HERE)
PER_PARA = 5
P_KIND = (("supported", 0.45), ("unsupported", 0.4), ("wrong_source", 0.15))
DECOYS = 1  # extra uncited bibliography entry per paragraph
_refs, _pool, _index, _lock = {}, [], {}, threading.Lock()


def load():
    with _lock:  # loaded whole under a lock (instances are generated concurrently in threads)
        if not _refs:
            pool = [json.loads(x) for x in open(HERE / "data" / "pool.jsonl")]
            refs = {r["arxiv"]: r for r in (json.loads(x) for x in open(HERE / "data" / "refs.jsonl"))}
            _pool.extend(pool)
            _refs.update(refs)
    return _refs, _pool


def generate(n, seed, out_dir):
    refs, pool = load()
    order = list(range(len(pool)))
    random.Random(f"a06:{seed}").shuffle(order)
    used, slots = set(), []
    for i in order:
        e = pool[i]  # {"arxiv", "faithful", "distorted", "wrong_cite" (validated neighbour or null)}
        rng = random.Random(f"a06:{seed}:{e['arxiv']}")
        kind = rng.choices([k for k, _ in P_KIND], [w for _, w in P_KIND])[0]
        if kind == "wrong_source" and not e.get("wrong_cite"):
            kind = "supported"
        if kind == "unsupported" and not e.get("distorted"):
            kind = "supported"
        need = {e["arxiv"]} | ({e["wrong_cite"]} if kind == "wrong_source" else set())
        if need & used:
            continue
        used |= need
        slots.append((e, kind))
        if len(slots) == n * PER_PARA:
            break
    rest = [a for a in sorted(refs) if a not in used]
    random.Random(f"a06d:{seed}").shuffle(rest)
    bib = sorted(used | set(rest[:DECOYS * n]))
    random.Random(f"a06b:{seed}:{n}").shuffle(bib)
    rid = {a: f"R{k + 1:04d}" for k, a in enumerate(bib)}
    gold, paras = {}, []
    for p in range(n):
        lines = [f"[P{p + 1:03d}]"]
        for j in range(PER_PARA):
            e, kind = slots[p * PER_PARA + j]
            sid = f"S{p * PER_PARA + j + 1:04d}"
            cited = e["wrong_cite"] if kind == "wrong_source" else e["arxiv"]
            text = (e["distorted"] if kind == "unsupported" else e["faithful"]).replace("[CITE]", f"[{rid[cited]}]")
            gold[sid] = {"para": f"P{p + 1:03d}", "verdict": kind, "cited": rid[cited],
                         "correct_ref": rid[e["arxiv"]] if kind == "wrong_source" else None, "arxiv": e["arxiv"]}
            lines.append(f"{sid} {text}")
        paras.append("\n".join(lines))
    files = {"report.txt": "\n\n".join(paras) + "\n",
             "bibliography.jsonl": "".join(json.dumps({"id": rid[a], "arxiv": a, "title": refs[a]["title"],
                                                       "abstract": refs[a]["abstract"]}) + "\n"
                                           for a in sorted(bib, key=lambda a: rid[a]))}
    prompt = common.fill(T["prompt"], n_paragraphs=n, n_sentences=n * PER_PARA, n_refs=len(bib))
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"sentences": gold}, TIERS, files)


# ---------------------------------------------------------------- tools
_WORD = re.compile(r"[a-z0-9][a-z0-9-]+")
STOP = set("the a an of in on at to for and or is are be by with as from that this which we our their its it these "
           "than into using use used based".split())


def _toks(s):
    return [w for w in _WORD.findall(s.lower()) if w not in STOP]


def index(world):
    key = str(world)
    with _lock:
        if key not in _index:
            docs = [json.loads(x) for x in open(world / "bibliography.jsonl")]
            post, lens = defaultdict(list), []
            for i, d in enumerate(docs):
                tf = Counter(_toks(d["title"] + " " + d["title"] + " " + d["abstract"]))
                lens.append(sum(tf.values()))
                for w, c in tf.items():
                    post[w].append((i, c))
            _index[key] = (docs, {d["id"]: d for d in docs}, post, lens, sum(lens) / len(lens))
        return _index[key]


def tools(world_dir):
    world = Path(world_dir)

    def read_report(a):
        paras = (world / "report.txt").read_text().strip().split("\n\n")
        off, lim = common.page_args(a, 10, 20)
        body = paras[off:off + lim]
        return "\n\n".join(body) + f"\n[paragraphs {off + 1}-{off + len(body)} of {len(paras)}]"

    def list_refs(a):
        docs = index(world)[0]
        off, lim = common.page_args(a, 60, 120)
        body = [f"{d['id']}\t{d['title']}" for d in docs[off:off + lim]]
        return "\n".join(body) + f"\n[references {off + 1}-{off + len(body)} of {len(docs)}]"

    def read_refs(a):
        by_id = index(world)[1]
        ids = a.get("ids")
        ids = ids if isinstance(ids, list) else re.findall(r"R\d{4}", str(ids or ""))
        if not ids:
            raise ValueError("give a list of reference ids, e.g. [\"R0012\"]")
        out = []
        for x in ids[:6]:
            d = by_id.get(str(x).strip())
            out.append(f"{d['id']} | {d['title']}\n{d['abstract']}" if d else f"{x}: no such reference")
        return "\n\n".join(out)

    def search_refs(a):
        docs, _, post, lens, avg = index(world)
        q = _toks(str(a.get("query") or ""))[:30]
        if not q:
            raise ValueError("query needs at least one content word")
        lim = min(int(a.get("limit") or 8), 15)
        sc = defaultdict(float)
        for w in set(q):
            ps = post.get(w, [])
            idf = math.log(1 + (len(docs) - len(ps) + 0.5) / (len(ps) + 0.5))
            for i, c in ps:
                sc[i] += idf * c * 2.2 / (c + 1.2 * (0.25 + 0.75 * lens[i] / avg))
        top = sorted(sc, key=lambda i: (-sc[i], i))[:lim]
        if not top:
            return "no matching reference"
        return "\n".join(f"{docs[i]['id']}\t{docs[i]['title']}" for i in top)

    page = {"offset": {"type": "integer"}, "limit": {"type": "integer"}}
    return [
        common.tool("read_report", T["tool_read_report"], page, [], read_report),
        common.tool("list_refs", T["tool_list_refs"], page, [], list_refs),
        common.tool("read_refs", T["tool_read_refs"], {"ids": {"type": "array", "items": {"type": "string"}}},
                    ["ids"], read_refs),
        common.tool("search_refs", T["tool_search_refs"], {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    ["query"], search_refs),
    ]


# ---------------------------------------------------------------- scoring
def _verdict(v):
    s = re.sub(r"[\s-]+", "_", str(v).strip().lower())
    return {"supported": "supported", "unsupported": "unsupported", "wrong_source": "wrong_source",
            "wrongsource": "wrong_source", "misattributed": "wrong_source"}.get(s)


def parse_answer(answer):
    obj = common.last_json(answer if isinstance(answer, str) else json.dumps(answer),
                           lambda o: isinstance(o, dict) and any(re.fullmatch(r"S\d{4}", str(k)) for k in o))
    out = {}
    for k, v in (obj or {}).items():
        if not re.fullmatch(r"S\d{4}", str(k)):
            continue
        if isinstance(v, dict):
            ref = re.search(r"R\d{4}", str(v.get("correct_ref") or ""))
            out[str(k)] = (_verdict(v.get("verdict")), ref.group() if ref else None)
        else:
            ref = re.search(r"R\d{4}", str(v))
            out[str(k)] = (_verdict(str(v).split(":")[0]), ref.group() if ref else None)
    return out


def sentence_ok(p, g):
    if not p or p[0] != g["verdict"]:
        return False
    return g["verdict"] != "wrong_source" or p[1] == g["correct_ref"]


def score(task, answer):
    gold = common.gold_of(task)["sentences"]
    pred = parse_answer(answer)
    by_para = defaultdict(list)
    for s, g in gold.items():
        by_para[g["para"]].append(sentence_ok(pred.get(s), g))
    units = [all(v) for v in by_para.values()]
    sent = sum(sentence_ok(pred.get(s), g) for s, g in gold.items())
    return {"score": sum(units) / len(units), "score_exact": float(all(units)), "score_partial": sent / len(gold),
            "n_correct": sum(units), "n_units": len(units), "sentences_correct": sent, "n_sentences": len(gold),
            "answered": sum(s in pred for s in gold)}
