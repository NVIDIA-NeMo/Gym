"""Caveman (telegraphic thinking) detector for search reasoning, vendored from
research/ppo/.cache/search_deep/caveman_detector (detector_search.py; data, labels and metrics in its search/metrics.md).

Turn level: one reasoning block (one model call) -> P(caveman), or None when it has fewer prose words than the
model's minimum after code, quotes, URLs, numbers and names are masked. Rollout level: the mean P over the scored turns.
Detector A: logistic regression on 15 prose-rate features. Detector B: logistic regression on masked unigrams, an
independent cross-check. Pure Python (re, math, json)."""
import json
import math
import os

from . import ngram
from .sfeatures import features

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS = {k: json.load(open(os.path.join(_DIR, f"model_{k}.json"))) for k in ("A", "B")}
THRESHOLDS = json.load(open(os.path.join(_DIR, "thresholds.json")))
# A reasoning block longer than this is scored on its first MAX_TURN_CHARS characters (features are rates).
MAX_TURN_CHARS = 100_000


def score_turn(text, detector="A"):
    """P(caveman) for one reasoning block, or None when it has too little prose to score."""
    text = (text or "")[:MAX_TURN_CHARS]
    m = _MODELS[detector]
    if detector == "A":
        f = features(text)
        if f["n_words"] < m["min_words"]:
            return None
        z = m["b"] + sum(w * (f[c] - mu) / sd for c, w, mu, sd in zip(m["feats"], m["w"], m["mu"], m["sd"]))
        return 1 / (1 + math.exp(-z))
    if features(text)["n_words"] < m["min_words"]:
        return None
    return ngram.score(m, text)[0]


def score_rollout(turns, detector="A"):
    """-> (mean P over scored turns or None, number of scored turns)."""
    ps = [p for p in (score_turn(t, detector) for t in turns) if p is not None]
    return (sum(ps) / len(ps) if ps else None), len(ps)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize_rollout(turns):
    """One pass over a rollout's reasoning turns (features computed once per turn). Returns a dict:
    score_a / score_b: mean P(caveman) over scored turns for detectors A and B (None when no turn is scored);
    n_scored: turns with enough prose to score (the same rule for A and B);
    article_rate / subjectless_share: mean over scored turns of the article share and of the share of sentences
    opening without a subject ("Need…", "Found…") - the two signals of telegraphic text that keeps its articles,
    which both detectors miss."""
    ma, mb = _MODELS["A"], _MODELS["B"]
    pa, pb, art, subj = [], [], [], []
    for text in turns:
        text = (text or "")[:MAX_TURN_CHARS]
        f = features(text)
        if f["n_words"] < ma["min_words"]:
            continue
        z = ma["b"] + sum(w * (f[c] - mu) / sd for c, w, mu, sd in zip(ma["feats"], ma["w"], ma["mu"], ma["sd"]))
        pa.append(1 / (1 + math.exp(-z)))
        if f["n_words"] >= mb["min_words"]:
            pb.append(ngram.score(mb, text)[0])
        art.append(f["art"])
        subj.append(f["subjless_start"])
    return dict(score_a=_mean(pa), score_b=_mean(pb), n_scored=len(pa), article_rate=_mean(art),
                subjectless_share=_mean(subj))
