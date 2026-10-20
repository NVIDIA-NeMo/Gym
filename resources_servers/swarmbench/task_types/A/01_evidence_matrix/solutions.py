"""A01 reference solution, plausible-wrong strategies and single-unit cases (check 5).

The gold is a real annotation key, so the oracle is the key itself (it checks the parser and scorer round-trip); the
wrong strategies read the world through the tools only."""
import json
import re

from .. import common
from . import type as M

NEG = re.compile(r"no (statistically )?significant|not (statistically )?significant|did not differ|no difference|"
                 r"similar|comparable|n\.?s\.?\b|p\s*[>=]\s*0?\.0[5-9]|p\s*>\s*0?\.[1-9]", re.I)
UP = re.compile(r"increas|higher|greater|more|improv|elevat|longer", re.I)
DOWN = re.compile(r"decreas|lower|reduc|less|fewer|shorter|declin", re.I)


def oracle(task):
    g = common.gold_of(task)["rows"]
    return json.dumps({r: v["label"] for r, v in g.items()})


def _rows(task):
    ls = (open(task["world_dir"] + "/matrix.tsv").read().splitlines())[1:]
    return [l.split("\t") for l in ls]


def abstract_keywords(task):
    """Find the abstract sentence that shares most words with the outcome; label it by cue words."""
    out = {}
    for rid, art, i, c, o in _rows(task):
        text = open(f"{task['world_dir']}/articles/{art}.txt").read()
        ab = text[:text.find("BODY")] if "BODY" in text else text[:4000]
        sents = re.split(r"(?<=[.;])\s+", ab)
        ow = set(re.findall(r"[a-z]{3,}", o.lower()))
        best = max(sents, key=lambda s: len(ow & set(re.findall(r"[a-z]{3,}", s.lower()))))
        out[rid] = "no_difference" if NEG.search(best) else "decreased" if DOWN.search(best) and not UP.search(best) \
            else "increased" if UP.search(best) else "no_difference"
    return json.dumps(out)


def flip(task):
    """Right significance, direction read the wrong way round (comparator vs intervention)."""
    g = common.gold_of(task)["rows"]
    sw = {"increased": "decreased", "decreased": "increased", "no_difference": "no_difference"}
    return json.dumps({r: sw[v["label"]] for r, v in g.items()})


def significance_only(task):
    """Knows which rows differ significantly but always says 'increased' for them."""
    g = common.gold_of(task)["rows"]
    return json.dumps({r: "no_difference" if v["label"] == "no_difference" else "increased" for r, v in g.items()})


def wrong(task):
    rows = _rows(task)
    return {"abstract_keywords": abstract_keywords(task), "all_no_difference":
            json.dumps({r[0]: "no_difference" for r in rows}), "flip_direction": flip(task),
            "significance_only": significance_only(task)}


def units(task, k, rng):
    g = common.gold_of(task)["rows"]
    rows = _rows(task)
    arts = sorted({r[1] for r in rows})
    out = []
    for art in rng.sample(arts, min(k, len(arts))):
        mine = [r for r in rows if r[1] == art]
        gold = {r[0]: g[r[0]]["label"] for r in mine}
        prompt = (common.CANARY + "\n\n" + common.fill(M.T["unit_prompt"], n_rows=len(mine), article=art) +
                  "\n\nrow_id\tarticle\tintervention\tcomparator\toutcome\n" + "\n".join("\t".join(r) for r in mine))

        def check(text, gold=gold):
            p = M.parse_answer(text)
            if not any(r in p for r in gold):
                return None
            return all(p.get(r) == v for r, v in gold.items())
        out.append({"id": art, "kind": "mixed" if len(set(gold.values())) > 1 else "single", "prompt": prompt,
                    "gold": gold, "check": check})
    return out
