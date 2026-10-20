"""A09 reference solution (the generation key), plausible-wrong strategies (tool-only) and unit cases."""
import json

from .. import common
from . import type as M


def oracle(task):
    return json.dumps({k: g["cite"] for k, g in common.gold_of(task)["clauses"].items()})


def _clauses(task):
    return [l.split("\t", 1) for l in open(task["world_dir"] + "/clauses.tsv").read().splitlines()[1:]]


def bm25_top(task, skip_first=False, rank=None):
    """Retrieval baseline: the clause text as a query; cite the first-level paragraph of the best hit."""
    s = {t["name"]: t["fn"] for t in M.tools(task["world_dir"])}["search"]
    out = {}
    for k, c in _clauses(task):
        hits = [h.split("\t")[0] for h in s({"query": c, "limit": 15}).splitlines() if "\t" in h]
        cites = []
        for h in hits:
            n = M.norm_cite(h)
            if n and n not in cites:
                cites.append(n)
        r = rank if rank is not None else (1 if skip_first else 0)
        pick = cites[r] if len(cites) > r else cites[0] if cites else None
        out[k] = f"{pick[0]}{pick[1]}" if pick else ""
    return json.dumps(out)


def wrong(task):
    g = common.gold_of(task)["clauses"]

    def neighbour(v):
        paras = sorted({p["des"].split(")")[0] + ")" for p in M.regs()["sections"][v["section"]]["paras"] if p["des"]})
        i = paras.index(v["para"]) if v["para"] in paras else 0
        return f"{v['section']}{paras[(i + 1) % len(paras)]}"
    return {"bm25_top_hit": bm25_top(task), "bm25_second_distinct_hit": bm25_top(task, skip_first=True),
            "bm25_fourth_distinct_hit": bm25_top(task, rank=3),
            "right_section_first_paragraph": json.dumps({k: f"{v['section']}(a)" if v["para"] != "(a)" else
                                                         f"{v['section']}(b)" for k, v in g.items()}),
            "right_section_next_paragraph": json.dumps({k: neighbour(v) for k, v in g.items()})}


def units(task, k, rng):
    g = common.gold_of(task)["clauses"]
    cl = dict(_clauses(task))
    out = []
    for kid in rng.sample(sorted(g), min(k, len(g))):
        prompt = common.CANARY + "\n\n" + common.fill(M.T["unit_prompt"], clause_id=kid) + f"\n\n{kid}\t{cl[kid]}"

        def check(text, kid=kid):
            p = M.parse_answer(text)
            if kid not in p:
                return None
            return p[kid] == (g[kid]["section"], g[kid]["para"])
        out.append({"id": kid, "kind": g[kid]["section"].split(".")[0], "prompt": prompt, "gold": g[kid]["cite"],
                    "check": check})
    return out
