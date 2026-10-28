"""Search-reasoning features (vendored from research/ppo/.cache/search_deep/caveman_detector/search/sfeatures.py; the
TECH substitution is replaced by an equivalent linear-time version).

Search-reasoning features for telegraphic ('caveman') thinking. The text is cleaned first: code, backtick spans,
quoted strings, URLs, paths, numbers, footnote markers and markdown are removed, and capitalised words inside a
sentence (names, titles) are masked, so candidate lists, quotes and entity-dense notes are judged on the prose around
them. All features are rates, so they do not depend on text length."""
import re

CODEBLOCK = re.compile(r"```.*?```", re.S)
TICK = re.compile(r"`[^`\n]*`")
QUOTE = re.compile(r"\"[^\"\n]{0,200}\"|“[^”\n]{0,200}”|‘[^’\n]{0,120}’|(?<![A-Za-z])'[^'\n]{1,80}'(?![A-Za-z])")
TECH = re.compile(r"\[\^?\d+\^?\]|https?://\S+|\S*[/\\_=<>{}\[\]|$#@*^~+]\S*|\S*\.\S*[A-Za-z]\S*|\S*\d\S*")
TOKEN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|[.!?;:\n—–-]+|,")

ARTICLES = {"a", "an", "the"}
DET = ARTICLES | {"this", "that", "these", "those", "my", "our", "their", "his", "her", "its", "your", "some", "any",
                  "each", "every", "all", "both", "no", "another", "other", "such", "which", "what", "whose", "one",
                  "two", "three", "several", "many", "much", "more", "most", "few", "first", "second", "last", "next",
                  "same", "only", "either", "neither"}
SUBJ = {"i", "we", "it", "this", "that", "they", "he", "she", "you", "there", "these", "those", "which", "who",
        "what", "one"}
AUX = {"is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "has", "have", "had", "will",
       "would", "should", "can", "could", "may", "might", "must", "shall", "it's", "that's", "there's", "i'm", "i'll",
       "we'll", "let's", "isn't", "doesn't", "don't", "didn't", "won't", "can't", "wasn't", "aren't", "i've", "we've",
       "he's", "she's", "they're", "we're", "what's", "here's", "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't"}
MODAL = {"could", "can", "might", "may", "must", "should", "would", "will"}
PREP = {"in", "of", "for", "from", "at", "on", "with", "via", "by", "about", "into", "under", "between", "through",
        "within", "without", "after", "before", "per", "around", "across"}
PRON_OBJ = {"me", "us", "them", "him", "her", "it", "you", "itself", "themselves", "himself", "herself"}
FUNC = ARTICLES | AUX | SUBJ | PREP | {"to", "and", "or", "but", "if", "so", "as", "not", "me", "us", "them", "him",
                                       "her", "its", "their", "my", "our", "let"}
HEDGE = {"likely", "maybe", "perhaps", "possibly", "probably"}
SUBJLESS_START = {"need", "needs", "could", "can", "might", "may", "must", "should", "maybe", "perhaps", "likely",
                  "possibly", "probably", "seems", "looks", "found", "got", "fits", "matches", "says", "said", "shows",
                  "gives", "lists", "mentions", "confirms", "suggests", "indicates", "appears", "includes", "contains",
                  "has", "had", "was", "is", "are", "not", "no", "also", "still", "already", "now", "then"}
IMPER = {"search", "browse", "query", "use", "check", "verify", "inspect", "open", "read", "find", "look", "try",
         "ensure", "cite", "confirm", "grep", "fetch", "compute", "calculate", "identify", "locate", "examine",
         "investigate", "consider", "provide", "include", "mention", "answer", "update", "gather", "go", "visit",
         "get", "see", "note", "map", "list", "determine", "test", "explore", "research", "compare"}
NEED_BARE = re.compile(r"\b(?:need|needs|want|wants|try|tries)\s+(?!to\b|a\b|an\b|the\b|it\b|this\b|that\b|more\b|"
                       r"some\b|any\b|not\b|only\b|also\b|be\b|you\b|me\b|us\b|them\b|him\b|her\b|my\b|our\b|his\b|"
                       r"their\b|its\b|no\b|one\b|each\b|all\b|both\b|every\b|another\b|other\b|again\b|of\b|for\b|"
                       r"in\b|on\b|with\b|at\b|and\b|or\b|X\b|is\b|was\b|are\b)([a-z]+)", re.I)
DUE_BARE = re.compile(r"\bdue\s+(?!to\b)[a-z]", re.I)
END_NOT = re.compile(r"\b(?:not|no|yes|good|fine|enough|strong|okay)\s*[.?!;]", re.I)
LETME = re.compile(r"\blet me\b|\bi'll\b|\bi need to\b|\bi should\b|\bi think\b|\bi'm\b|\bi can\b|\bi will\b|"
                   r"\bi have\b|\bi want\b|\bwe need to\b|\bwe should\b|\bwe can\b|\bwe have\b|\bi've\b|\bi realize\b",
                   re.I)


_RUN = re.compile(r"\S+")
_FOOTNOTE_AT_START = re.compile(r"\[\^?\d+\^?\]")
_TECH_CHARS = set("/\\_=<>{}[]|$#@*^~+")
_TRAIL_PUNCT = re.compile(r"[.,;:!?]*$")


def _tech_run(run):
    """Replacement for one whitespace-free run, equal to what TECH.sub produces on it but in linear time (TECH's
    \\S*-prefixed alternatives backtrack quadratically on long runs)."""
    m = _FOOTNOTE_AT_START.match(run)
    if m:  # TECH's first alternative wins at the run start and matches only the footnote marker
        return " X " + _tech_run(run[m.end():]) if m.end() < len(run) else " X "
    if run.startswith(("http://", "https://")) or any(c in _TECH_CHARS or c.isdecimal() for c in run):
        hit = True
    else:
        dot = run.find(".")
        hit = dot >= 0 and any(c.isalpha() and c.isascii() for c in run[dot + 1:])
    return " X" + _TRAIL_PUNCT.search(run).group(0) + " " if hit else run


def _tech_sub(t):
    return _RUN.sub(lambda m: _tech_run(m.group(0)), t)


def clean(t):
    t = CODEBLOCK.sub(" ", t)
    t = TICK.sub(" ", t)
    t = QUOTE.sub(" X ", t)
    t = _tech_sub(t)
    return t.replace("**", " ").replace("__", " ")


def sentences(t):
    """Token lists per sentence; capitalised words after the first token are masked to 'X' (names, titles)."""
    out, cur = [], []
    for m in TOKEN.finditer(clean(t)):
        w = m.group(0)
        if w[0] in ".!?;:\n—–-":
            if cur:
                out.append(cur)
            cur = []
            continue
        if w == ",":
            cur.append(",")
            continue
        if cur and w[0].isupper() and w.lower() not in ("i", "i'm", "i'll", "i've"):
            cur.append("X")
        elif w == "X":
            cur.append("X")
        else:
            cur.append(w.lower())
    if cur:
        out.append(cur)
    return out


def features(text):
    S = sentences(text)
    words = [w for s in S for w in s if w not in ("X", ",")]
    n = max(1, len(words))
    prose = [s for s in S if sum(w not in ("X", ",") for w in s) >= 3]
    ns = max(1, len(prose))
    starts, seconds = [], []
    for s in prose:
        toks = [w for w in s if w != ","]
        i = next(j for j, w in enumerate(toks) if w != "X")
        starts.append(toks[i]); seconds.append(toks[i + 1] if i + 1 < len(toks) else "")
    # a modal or auxiliary opening is a question with a subject ("Could it be", "Is the date") unless the next word
    # is not a subject, determiner or name ("Could be X", "Could cite source")
    subjless = [w in SUBJLESS_START and not (w in AUX and (n2 in SUBJ or n2 in DET or n2 == "X"))
                for w, n2 in zip(starts, seconds)]
    modal = [w in MODAL and not (n2 in SUBJ or n2 in DET or n2 == "X") for w, n2 in zip(starts, seconds)]
    c = clean(text)
    prep_bare = det_noun = verb_bare = 0
    for s in S:
        for a, b in zip(s, s[1:]):
            if a in IMPER | {"need", "needs", "cite", "found", "has", "says"} and b not in DET and b not in FUNC \
                    and b not in PRON_OBJ and b not in ("X", ",") and not b.endswith(("s", "ing", "ly")) \
                    and b not in ("for", "out", "up", "whether", "how", "why", "where", "when", "if", "back", "again"):
                verb_bare += 1
            if a in PREP and b not in DET and b not in PRON_OBJ and b not in FUNC and b not in ("X", ",") \
                    and not b.endswith("ing"):
                prep_bare += 1
            if a in PREP:
                det_noun += 1
    q_sents = [s for s in re.split(r"(?<=[?])", c) if s.strip().endswith("?")]
    q_frag = 0
    for q in q_sents:
        toks = [w.lower() for w in re.findall(r"[A-Za-z']+", q.split("\n")[-1])]
        if len(toks) >= 2 and not (set(toks) & (AUX | SUBJ | {"what", "who", "which", "where", "when", "why", "how"})):
            q_frag += 1
    slen = [sum(w not in ("X", ",") for w in s) for s in prose]
    return dict(
        n_words=len(words),
        func=sum(w in FUNC for w in words) / n,
        art=sum(w in ARTICLES for w in words) / n,
        subj=sum(w in SUBJ for w in words) / n,
        aux=sum(w in AUX for w in words) / n,
        to=sum(w == "to" for w in words) / n,
        need_bare=100 * len(NEED_BARE.findall(c)) / n,
        due_bare=100 * len(DUE_BARE.findall(c)) / n,
        subjless_start=sum(subjless) / ns,
        imper_start=sum(w in IMPER for w in starts) / ns,
        modal_start=sum(modal) / ns,
        verb_bare=100 * verb_bare / n,
        list_share=len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", text)) / ns,
        letme=100 * len(LETME.findall(c)) / n,
        hedge=100 * sum(w in HEDGE for w in words) / n,
        end_not=100 * len(END_NOT.findall(c)) / n,
        bare_sent=sum(1 for s in prose if not (set(s) & (ARTICLES | AUX | SUBJ | {"to", "me", "let"}))) / ns,
        noart_sent=sum(1 for s in prose if not (set(s) & ARTICLES)) / ns,
        noverb_sent=sum(1 for s in prose if not (set(s) & AUX) and not any(w.endswith("ed") or w.endswith("s")
                                                                              for w in s if w not in FUNC)) / ns,
        prep_bare=prep_bare / max(1, det_noun),
        q_frag=q_frag / ns,
        sent_len=sum(slen) / len(slen) if slen else 0.0,
        short_sent=sum(1 for x in slen if x <= 4) / ns,
        mask_share=sum(w == "X" for s in S for w in s) / max(1, sum(len(s) for s in S)),
    )


ALL = ["func", "art", "subj", "aux", "to", "need_bare", "due_bare", "subjless_start", "imper_start", "modal_start",
       "verb_bare", "list_share", "letme", "hedge", "end_not", "bare_sent", "noart_sent", "noverb_sent", "prep_bare", "q_frag", "sent_len",
       "short_sent", "mask_share"]
