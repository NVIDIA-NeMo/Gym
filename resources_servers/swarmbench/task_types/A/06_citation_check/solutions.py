"""A06 reference solution (the construction key), plausible-wrong strategies (tool-only) and unit cases."""
import json
import re
from collections import defaultdict

from .. import common
from . import type as M


def oracle(task):
    g = common.gold_of(task)["sentences"]
    return json.dumps({s: {"verdict": v["verdict"], **({"correct_ref": v["correct_ref"]} if v["correct_ref"] else {})}
                       for s, v in g.items()})


def _sentences(task):
    out = []
    for para in open(task["world_dir"] + "/report.txt").read().strip().split("\n\n"):
        for line in para.splitlines()[1:]:
            sid, text = line.split(" ", 1)
            out.append((sid, text, re.search(r"\[(R\d{4})\]", text).group(1)))
    return out


NUMS = re.compile(r"\d+(?:[.,]\d+)*%?")


def overlap(task, hi=0.5, margin=0.15, numbers=False):
    """Word-overlap judge: supported if most content words of the sentence occur in the cited abstract; otherwise
    wrong_source if another reference covers clearly more of them, else unsupported."""
    docs = M.index(__import__("pathlib").Path(task["world_dir"]))[0]
    words = {d["id"]: set(M._toks(d["title"] + " " + d["abstract"])) for d in docs}
    docs_by = {d["id"]: d["title"] + " " + d["abstract"] for d in docs}
    out = {}
    for sid, text, cited in _sentences(task):
        w = set(M._toks(re.sub(r"\[R\d{4}\]", "", text)))
        cov = {r: len(w & ws) / max(1, len(w)) for r, ws in words.items()}
        best = max(cov, key=cov.get)
        ab = docs_by[cited]
        bad_num = numbers and any(n not in ab for n in NUMS.findall(re.sub(r"\[R\d{4}\]", "", text)))
        if cov[cited] >= hi and not bad_num:
            out[sid] = {"verdict": "supported"}
        elif best != cited and cov[best] - cov[cited] >= margin:
            out[sid] = {"verdict": "wrong_source", "correct_ref": best}
        else:
            out[sid] = {"verdict": "unsupported"}
    return json.dumps(out)


def wrong(task):
    g = common.gold_of(task)["sentences"]
    return {"all_supported": json.dumps({s: {"verdict": "supported"} for s in g}),
            "word_overlap": overlap(task),
            "word_overlap_number_check": overlap(task, numbers=True),
            # finds the misattributed ones but calls every misstatement supported
            "no_detail_check": json.dumps({s: {"verdict": "wrong_source", "correct_ref": v["correct_ref"]}
                                           if v["verdict"] == "wrong_source" else {"verdict": "supported"}
                                           for s, v in g.items()}),
            # checks details but never looks for the right source
            "no_source_search": json.dumps({s: {"verdict": "supported" if v["verdict"] == "supported" else
                                                "unsupported"} for s, v in g.items()})}


def units(task, k, rng):
    g = common.gold_of(task)["sentences"]
    sents = _sentences(task)
    paras = defaultdict(list)
    for sid, text, _ in sents:
        paras[g[sid]["para"]].append((sid, text))
    out = []
    for p in rng.sample(sorted(paras), min(k, len(paras))):
        gold = {sid: g[sid] for sid, _ in paras[p]}
        prompt = (common.CANARY + "\n\n" + common.fill(M.T["unit_prompt"], paragraph_id=p, n_sentences=len(gold)) +
                  "\n\n[" + p + "]\n" + "\n".join(f"{sid} {t}" for sid, t in paras[p]))

        def check(text, gold=gold):
            pr = M.parse_answer(text)
            if not any(s in pr for s in gold):
                return None
            return all(M.sentence_ok(pr.get(s), v) for s, v in gold.items())
        kinds = sorted(v["verdict"] for v in gold.values())
        out.append({"id": p, "kind": "+".join(sorted(set(kinds))), "prompt": prompt,
                    "gold": {s: [v["verdict"], v["correct_ref"]] for s, v in gold.items()}, "check": check})
    return out
