"""Type A01 evidence matrix: fill a literature matrix (intervention x comparator x outcome rows) for many randomized
controlled trial reports with the reported effect direction.

Real source: Evidence Inference 2.0 (MIT; PubMed Central open-access full texts). One unit = one article with 3-5
rows; an article counts only if every row is right. Gold = the label two annotators agreed on (and a verifier
accepted), kept only where self-hosted K3 reading the whole article agreed (data/build, README). Task-facing prose is
K3's (text/k3.json); article text, ICO strings and ids are real.
"""
import gzip
import json
import random
import re
import threading
from pathlib import Path

from .. import common

TYPE_ID = "01_evidence_matrix"
TIERS = {"easy": (5, 1800), "medium": (60, 1200), "hard": (120, 1200)}
HERE = Path(__file__).parent
T = common.load_text(HERE)
LABELS = ("increased", "decreased", "no_difference")
MAX_ROWS = 5
_pool, _plock = [], threading.Lock()


def pool():
    """Articles with at least one row whose annotators needed more than the abstract (v2; see README)."""
    # loaded whole under a lock: the runner generates several instances concurrently in threads, and a lazily
    # extended list let early threads see a partial pool (3 calibration rows were generated from 3 articles)
    with _plock:
        if not _pool:
            rows = [json.loads(x) for x in gzip.open(HERE / "data" / "pool.jsonl.gz", "rt")]
            _pool.extend([r for r in rows if any(not p["in_abstract"] for p in r["prompts"][:MAX_ROWS])])
    return _pool


def pick_rows(rec, rng):
    ps = list(rec["prompts"])
    if len(ps) <= MAX_ROWS:
        return ps
    for _ in range(50):
        sel = sorted(rng.sample(ps, MAX_ROWS), key=lambda p: p["pid"])
        if len({p["label"] for p in sel}) >= 2:
            return sel
    return ps[:MAX_ROWS]


def generate(n, seed, out_dir):
    order = list(range(len(pool())))
    random.Random(f"a01:{seed}").shuffle(order)
    arts = [pool()[i] for i in order[:n]]
    rows, gold, files = [], {}, {}
    for i, rec in enumerate(arts):
        sel = pick_rows(rec, random.Random(f"a01:{seed}:{rec['pmcid']}"))
        files[f"articles/{rec['pmcid']}.txt"] = rec["text"]
        for p in sel:
            rows.append((rec["pmcid"], p))
    # rows stay grouped by article, in a seeded article order (a literature matrix is read article by article)
    lines = []
    for k, (pmcid, p) in enumerate(rows):
        rid = f"R{k + 1:04d}"
        gold[rid] = {"article": pmcid, "label": p["label"], "pid": p["pid"], "in_abstract": p["in_abstract"]}
        lines.append("\t".join([rid, pmcid] + [" ".join(p[k].split()) for k in ("intervention", "comparator", "outcome")]))
    files["matrix.tsv"] = "row_id\tarticle\tintervention\tcomparator\toutcome\n" + "\n".join(lines) + "\n"
    files["articles.tsv"] = "article\ttitle\tchars\n" + "\n".join(
        f"{r['pmcid']}\t{r['title']}\t{len(r['text'])}" for r in arts) + "\n"
    prompt = common.fill(T["prompt"], n_rows=len(rows), n_articles=n)
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"rows": gold}, TIERS, files)


# ---------------------------------------------------------------- tools
def tools(world_dir):
    world = Path(world_dir)
    cache = {}

    def lines(name):
        return (world / name).read_text().splitlines()

    def article(aid):
        aid = str(aid).strip()
        if not re.fullmatch(r"PMC\d+", aid) or not (world / "articles" / f"{aid}.txt").is_file():
            raise KeyError(f"unknown article {aid!r}; see list_articles")
        if aid not in cache:
            cache[aid] = (world / "articles" / f"{aid}.txt").read_text()
        return cache[aid]

    def list_rows(a):
        ls = lines("matrix.tsv")
        off, lim = common.page_args(a, 50, 100)
        body = ls[1 + off:1 + off + lim]
        return f"{ls[0]}\n" + "\n".join(body) + f"\n[rows {off + 1}-{off + len(body)} of {len(ls) - 1}]"

    def list_articles(a):
        ls = lines("articles.tsv")
        off, lim = common.page_args(a, 50, 200)
        body = ls[1 + off:1 + off + lim]
        return f"{ls[0]}\n" + "\n".join(body) + f"\n[articles {off + 1}-{off + len(body)} of {len(ls) - 1}]"

    def read_article(a):
        text = article(a.get("article"))
        off = int(a.get("offset") or 0)
        lim = min(int(a.get("limit") or 6000), 7000)
        if off < 0 or lim < 1 or off >= len(text):
            raise ValueError(f"offset must be in [0, {len(text)}) and limit >= 1")
        return f"[chars {off}-{min(len(text), off + lim)} of {len(text)}]\n" + text[off:off + lim]

    art = {"article": {"type": "string", "description": "article id, e.g. PMC1234567"}}
    page = {"offset": {"type": "integer"}, "limit": {"type": "integer"}}
    return [
        common.tool("list_rows", T["tool_list_rows"], page, [], list_rows),
        common.tool("list_articles", T["tool_list_articles"], page, [], list_articles),
        common.tool("read_article", T["tool_read_article"], {**art, **page}, ["article"], read_article),
    ]


# ---------------------------------------------------------------- scoring
SYN = {"increased": "increased", "significantly increased": "increased", "increase": "increased",
       "higher": "increased", "decreased": "decreased", "significantly decreased": "decreased",
       "decrease": "decreased", "lower": "decreased", "no difference": "no_difference",
       "no significant difference": "no_difference", "nodifference": "no_difference"}


def norm_label(v):
    return SYN.get(re.sub(r"[\s_-]+", " ", str(v).strip().lower()))


def parse_answer(answer):
    obj = common.last_json(answer if isinstance(answer, str) else json.dumps(answer),
                           lambda o: isinstance(o, dict) and any(re.fullmatch(r"R\d{4}", str(k)) for k in o))
    out = {}
    for k, v in (obj or {}).items():
        lab = norm_label(v)
        if re.fullmatch(r"R\d{4}", str(k)) and lab:
            out[str(k)] = lab
    return out


def score(task, answer):
    gold = common.gold_of(task)["rows"]
    pred = parse_answer(answer)
    by_art = {}
    for rid, g in gold.items():
        by_art.setdefault(g["article"], []).append(pred.get(rid) == g["label"])
    units = [all(v) for v in by_art.values()]
    s = sum(units) / len(units) if units else 0.0
    rows_ok = sum(pred.get(r) == g["label"] for r, g in gold.items())
    return {"score": s, "score_exact": float(all(units)), "score_partial": rows_ok / len(gold),
            "n_correct": sum(units), "n_units": len(units), "rows_correct": rows_ok, "n_rows": len(gold),
            "rows_answered": sum(r in pred for r in gold)}
