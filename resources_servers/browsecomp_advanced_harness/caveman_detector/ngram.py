"""Vendored from research/ppo/.cache/search_deep/caveman_detector/search/ngram.py; the _URL substitution is replaced
by an equivalent linear-time version.

Detector B: logistic regression on masked word n-grams (pure Python at inference). Independent of detector A's
hand-built rates: it sees only token sequences, with names, numbers, quotes, URLs and code masked to placeholder
tokens and sentence starts marked, and learns which unigrams/bigrams signal telegraphic prose."""
import math, re
from collections import Counter

_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
_QUOTE = re.compile(r"\"[^\"\n]{0,200}\"|“[^”\n]{0,200}”")
_URL = re.compile(r"https?://\S+|\S+\.(?:com|org|net|gov|edu|pdf|html?)\S*|\S*[/\\_]\S*")
_TOK = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d[\d,.:%-]*|[.!?]+|\n")


_RUN = re.compile(r"\S+")
_TLD_AFTER_FIRST = re.compile(r"(?<=\S)\.(?:com|org|net|gov|edu|pdf|html?)")


def _url_sub(t):
    """Equal to _URL.sub(" <url> ", t) but linear-time: a whitespace-free run is replaced whole when it starts with
    http(s)://, contains / \\ or _, or has a .com/.org/.net/.gov/.edu/.pdf/.htm(l) after its first character."""
    def repl(m):
        run = m.group(0)
        if run.startswith(("http://", "https://")) or any(c in "/\\_" for c in run) or _TLD_AFTER_FIRST.search(run):
            return " <url> "
        return run
    return _RUN.sub(repl, t)


def tokens(text):
    t = _CODE.sub(" <code> ", text)
    t = _QUOTE.sub(" <q> ", t)
    t = _url_sub(t)
    out, start = ["<s>"], True
    for m in _TOK.finditer(t.replace("<code>", "CODEPH").replace("<q>", "QUOTEPH").replace("<url>", "URLPH")):
        w = m.group(0)
        if w in ("CODEPH", "QUOTEPH", "URLPH"):
            out.append({"CODEPH": "<code>", "QUOTEPH": "<q>", "URLPH": "<url>"}[w]); start = False
        elif w[0] in ".!?\n":
            if not start:
                out.append("<s>")
            start = True
        elif w[0].isdigit():
            out.append("<num>"); start = False
        elif not start and w[0].isupper() and w not in ("I", "I'm", "I'll", "I've"):
            out.append("<cap>")
        else:
            out.append(w.lower()); start = False
    return out


def grams(text, n=2):
    tk = tokens(text)
    c = Counter(tk)
    if n >= 2:
        c.update(a + " " + b for a, b in zip(tk, tk[1:]))
    if n >= 3:
        c.update(a + " " + b + " " + d for a, b, d in zip(tk, tk[1:], tk[2:]))
    return c


def vectorize(c, vocab):
    """Sublinear tf, L2-normalised; sparse dict index -> value."""
    v = {vocab[g]: 1 + math.log(k) for g, k in c.items() if g in vocab}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {i: x / norm for i, x in v.items()}


def score(model, text):
    c = grams(text, model["n"])
    v = vectorize(c, model["vocab"])
    z = model["b"] + sum(model["w"][i] * x for i, x in v.items())
    return 1 / (1 + math.exp(-z)), sum(1 for g in c if not g.startswith("<") and " " not in g)
