# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What language a text is written in (the `language` checker).

Only prose counts: code blocks, inline code, URLs, e-mail addresses, quoted text and technical tokens are set aside.
Words are counted per script (a run of Latin or Cyrillic letters is one word; a Han, kana or Hangul character is one
word). Languages that share a script are told apart: Latin-script languages by their common function words (English,
Spanish, Portuguese, French, German, Italian, Dutch, Polish, Turkish, Vietnamese, Indonesian), Russian and Ukrainian by
the letters only Ukrainian uses. `assess` says whether a verdict is clear; uncertain texts go to the judge. The user's
language (the value `user`) is read from the user's own words, with pasted output set aside (`detect`).
"""

import re
import unicodedata


_SCRIPT_BY_NAME = (
    ("LATIN", "latin"),
    ("FULLWIDTH LATIN", "latin"),
    ("CYRILLIC", "cyrillic"),
    ("CJK", "han"),
    ("HIRAGANA", "kana"),
    ("KATAKANA", "kana"),
    ("HALFWIDTH KATAKANA", "kana"),
    ("HANGUL", "hangul"),
    ("HALFWIDTH HANGUL", "hangul"),
    ("ARABIC", "arabic"),
    ("DEVANAGARI", "devanagari"),
    ("THAI", "thai"),
    ("HEBREW", "hebrew"),
    ("GREEK", "greek"),
)
_PER_CHARACTER = ("han", "kana", "hangul", "thai")
SCRIPTS = ("latin", "cyrillic", "han", "kana", "hangul", "arabic", "devanagari", "thai", "hebrew", "greek")

# language name -> (code, the scripts that count as the language)
LANGUAGES = {
    "english": ("en", ("latin",)),
    "spanish": ("es", ("latin",)),
    "portuguese": ("pt", ("latin",)),
    "french": ("fr", ("latin",)),
    "german": ("de", ("latin",)),
    "italian": ("it", ("latin",)),
    "dutch": ("nl", ("latin",)),
    "polish": ("pl", ("latin",)),
    "turkish": ("tr", ("latin",)),
    "vietnamese": ("vi", ("latin",)),
    "indonesian": ("id", ("latin",)),
    "russian": ("ru", ("cyrillic",)),
    "ukrainian": ("uk", ("cyrillic",)),
    "chinese": ("zh", ("han",)),
    "japanese": ("ja", ("kana", "han")),
    "korean": ("ko", ("hangul",)),
    "arabic": ("ar", ("arabic",)),
    "hindi": ("hi", ("devanagari",)),
    "thai": ("th", ("thai",)),
    "hebrew": ("he", ("hebrew",)),
    "greek": ("el", ("greek",)),
}
CODE_TO_NAME = {code: name for name, (code, _s) in LANGUAGES.items()}
# common function words, used to tell Latin-script languages apart (lower case)
STOPWORDS = {
    "en": "the and is are was were be been to of in that it for on with as this by not you we they have has had "
    "will would can could should from at an or but if then there their which what when where who how all any "
    "also into than these those here just only very more most some such been being do does did done",
    "es": "el la los las de del que y en un una es por con para no se lo como más pero sus le ya o este esta están "
    "está son fue ser hay muy también cuando sobre entre desde hasta sin porque donde qué cómo todo todos al "
    "nos les ese esa",
    "pt": "o a os as de do da dos das que e em um uma é por com para não se como mais mas seu sua ao foi ser há "
    "muito também quando sobre entre desde até sem porque onde isso este esta está estão são pelo pela você "
    "nos",
    "fr": "le la les de des du que et en un une est pour par avec pas ne se ce cette sont il elle ils au aux sur "
    "dans plus mais ou comme qui nous vous leur été être fait avoir tout très aussi quand sans car donc",
    "de": "der die das und ist nicht ein eine mit zu von den dem des auf für im sich es auch als an aus bei wird "
    "sind wurde noch nach wie oder aber nur so dass kann können haben hat ich wir sie ihr dieser diese",
    "it": "il lo la i gli le di del della che e è un una per con non si come più ma sono anche nel nella alla al da "
    "dei delle questo questa quando sulla sul ha hanno essere stato tutti molto",
    "nl": "de het een en van is dat op te in niet zijn met voor die er maar ook als bij aan wordt naar om dan dit "
    "kan worden heeft hebben werd nog wel geen",
    "pl": "i w na nie z się do to jest że o jak co ale za od po tak jest są już tylko przez dla może być czy jego "
    "jej tym oraz który która które",
    "tr": "ve bir bu için ile da de ne değil olarak çok daha gibi ama ya veya her kadar sonra olan oldu var yok ise "
    "bu şu mi mı biz siz onlar",
    "vi": "và của là có không được trong cho này với các những một người khi để đã cũng như từ thì nhưng đến về sẽ",
    "id": "dan yang di ini itu dengan untuk dari tidak ada dalam akan pada juga ke bisa atau saya kami kita mereka "
    "sudah karena oleh telah lebih",
}
STOP = {k: set(v.split()) for k, v in STOPWORDS.items()}
_UK_ONLY = set("іїєґІЇЄҐ")

_FENCED_CODE = re.compile(r"```.*?(?:```|\Z)", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"\b(?:https?|ftp)://\S+|\bwww\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|«[^»\n]*»|「[^」\n]*」|『[^』\n]*』')
_ASCII_RUN = re.compile(r"[!-~]+")
_TECHNICAL = re.compile(r"[/\\_=<>{}\[\]|$#@~^]|\w\.\w|\w\(|^-{1,2}[A-Za-z]|[a-z][A-Z]")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|[。！？；।\n]+")  # । ends a sentence in Hindi
CLEAR_SHARE = 0.9
# three or more plain Latin words in a row (not all capitals; technical tokens are already set aside): in a language
# not written in Latin script such a run can be a sentence in another language that no full stop sets apart (Thai has
# none), so a pass with one is not clear and the judge decides
_LATIN_RUN = re.compile(r"(?:\b[A-Za-z][a-z]+\b[ ,]+){2,}\b[A-Za-z][a-z]+\b")


def script_of(ch: str) -> str:
    """The script of a letter (SCRIPTS), from its Unicode name; "other" for any other character."""
    name = unicodedata.name(ch, "")
    for prefix, script in _SCRIPT_BY_NAME:
        if name.startswith(prefix):
            return script
    return "other"


def split_prose(text: str) -> tuple[str, str]:
    """(prose, code): code = fenced blocks and inline code; prose = the rest without URLs, e-mails, quoted text and
    technical tokens (paths, identifiers, options, calls)"""
    code = []

    def grab(m: re.Match) -> str:
        code.append(m.group(0))
        return " "

    rest = _INLINE_CODE.sub(grab, _FENCED_CODE.sub(grab, text))
    rest = _QUOTED.sub(" ", _EMAIL.sub(" ", _URL.sub(" ", rest)))
    prose = _ASCII_RUN.sub(lambda m: " " if _TECHNICAL.search(m.group(0)) else m.group(0), rest)
    return prose, "\n".join(code)


def word_counts(text: str) -> dict[str, int]:
    """{script: words}: a run of letters of one script is one word, and so is each Han, kana, Hangul or Thai
    letter."""
    counts, prev = {}, None
    for ch in text:
        if not ch.isalpha():
            prev = None
            continue
        sc = script_of(ch)
        if sc in _PER_CHARACTER or sc != prev:
            counts[sc] = counts.get(sc, 0) + 1
        prev = None if sc in _PER_CHARACTER else sc
    return counts


def latin_votes(text: str) -> dict[str, int]:
    """{language code: number of its function words} among the Latin words of a text"""
    words = [w.lower() for w in re.findall(r"[^\W\d_]+", text) if script_of(w[0]) == "latin"]
    return {code: sum(1 for w in words if w in stop) for code, stop in STOP.items()}


# What a user pastes rather than writes, set aside before the user's language is read: a Spanish user who writes
# "Me sale este error cuando corro las pruebas:" and pastes an English traceback writes Spanish. Each pattern matches a
# whole line; fenced code blocks are set aside whole.
_PASTED_LINE = re.compile(r"""(?x)^(?:
      [ ]{4,}\S | \t                                                    # indented by four spaces or a tab
    | \s*Traceback\ \(most\ recent\ call\ last\)                        # a Python traceback and its frames
    | \s*(?:During\ handling\ of\ the\ above\ exception|The\ above\ exception\ was\ the\ direct\ cause)
    | \s*File\ ".*",\ line\ \d+
    | \s*at\ [\w$.<>\[\]]+\ ?\(                                         # a JavaScript or Java stack frame
    | \s*(?:[\w$]+\.)*[A-Z]\w*(?:Error|Exception|Warning|Interrupt|Exit)(?:\[\w+\])?(?::|\s*$)   # KeyError: 'x'
    | \s*(?:error|fatal|warning|panic)(?:\[\w+\])?:\s                   # a tool's error line: error[E0308]: ...
    | \s*npm\ (?:ERR!|WARN)
    | \s*E\ {2,}\S                                                      # pytest's error lines
    | \s*(?:={3,}|_{3,}|-{3,}|\*{3,}|~{3,})                              # separator lines of test and build output
    | \s*(?:PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b                  # test results
    | \s*collected\ \d+\ items?\b
    | \s*\[?\d{4}-\d{2}-\d{2}[T\ ]\d{1,2}:\d{2}                          # a log line starting with a date and a time
    | \s*\[?\d{1,2}:\d{2}:\d{2}                                          # ... or with a time
    | \s*\[?(?:TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|CRITICAL|FATAL|SEVERE)\b   # ... or with a log level
    | \s*(?:\$|>>>|PS\ [^>\n]*>|[\w.-]+@[\w.-]+:[^\n$\#]*[$\#])\s        # a shell or Python prompt with its command
    | \s*>                                                              # a quoted line
    | \s*(?:diff\ --git|index\ [0-9a-f]+\.\.|---\ a/|\+\+\+\ b/|@@\ )    # a diff
    | \s*[\w./\\~-]+\.\w+:\d+                                           # compiler or linter output (path:line: ...)
    | \s*[\w./\\~-]+\.\w+::\w                                           # a test id (tests/test_x.py::test_y FAILED)
)""")
_FENCED_BLOCK = re.compile(r"(?ms)^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*\1[ \t]*$|\Z)")


def own_paragraphs(text: str | None) -> list[str]:
    """the paragraphs of a person's own words in a message: fenced code blocks and pasted lines (_PASTED_LINE:
    tracebacks, stack frames, error and log lines, command lines and test output, diffs, indented and quoted lines) set
    aside; a paragraph ends at a blank line and at every part set aside"""
    paras, cur = [], []
    for ln in _FENCED_BLOCK.sub("\n\n", text or "").split("\n"):
        if not ln.strip() or _PASTED_LINE.match(ln):
            if cur:
                paras.append("\n".join(cur))
            cur = []
        else:
            cur.append(ln)
    if cur:
        paras.append("\n".join(cur))
    return paras


def detect(text: str) -> tuple[str | None, str]:
    """(language name or None, detail): the language a person writes in, read from their own words: pasted parts are
    set aside (own_paragraphs), and the language is that of the first paragraph the code can tell, else that of all the
    person's own words read together; None when they cannot be told (a short "ok, sigue", or a message that is all
    pasted output). Used for the value `user` of the language checker."""
    paras = own_paragraphs(text)
    for k, p in enumerate(paras):
        lang, why = read_language(p)
        if lang:
            return lang, "%s (paragraph %d of %d of the user's own words)" % (why, k + 1, len(paras))
    if len(paras) > 1:
        lang, why = read_language("\n\n".join(paras))
        if lang:
            return lang, "%s (all of the user's own words)" % why
    if not paras:
        return None, "no words of the user's own: only pasted text (code, output)"
    return None, read_language("\n\n".join(paras))[1]


def read_language(text: str) -> tuple[str | None, str]:
    """(language name or None, detail): the language of the prose of a text, as far as code can tell"""
    prose, _ = split_prose(text)
    words = word_counts(prose) or word_counts(text)
    if not words:
        return None, "no letters"
    script = max(words, key=words.get)
    if script == "latin":
        votes = latin_votes(prose)
        best = max(votes, key=votes.get)
        if votes[best] < 2:
            return None, "Latin script, too few function words to tell the language"
        if sum(1 for v in votes.values() if v == votes[best]) > 1:
            # a tie is no reading: max() would pick whichever language STOP lists first (the French line "Le pipeline
            # CI échoue pendant l'analyse de sécurité" ties Spanish 2 to 2)
            return None, "Latin script, function words tie %s" % {k: v for k, v in votes.items() if v}
        return CODE_TO_NAME[best], "function words %s" % {k: v for k, v in votes.items() if v}
    if script == "cyrillic":
        return ("ukrainian" if any(ch in _UK_ONLY for ch in prose) else "russian"), "Cyrillic script"
    if script == "kana" or (script == "han" and words.get("kana")):
        return "japanese", "kana"
    for name, (_code, scripts) in LANGUAGES.items():
        if scripts == (script,):
            return name, "%s script" % script
    return None, "script %s" % script


def _words(value: str, text: str) -> tuple[dict[str, int], str, str]:
    """({script: words}, basis, prose) counted as check counts them"""
    scripts = LANGUAGES[value][1]
    prose, code = split_prose(text)
    words = word_counts(prose)
    in_code = word_counts(code)
    for k in scripts:
        if in_code.get(k):
            words[k] = words.get(k, 0) + in_code[k]
    if not words:
        return word_counts(text), "whole text", prose
    return words, "prose", prose


def check(value: str, text: str) -> tuple[bool | None, str, float]:
    """(ok, detail, share): is the text written in the language `value` (a name such as 'Spanish')"""
    value = str(value).lower()
    if value not in LANGUAGES:
        raise ValueError("language: unknown language %r" % (value,))
    code, scripts = LANGUAGES[value]
    words, basis, prose = _words(value, text)
    if basis != "prose" or not word_counts(prose):
        return None, "no prose (only code, paths or symbols): nothing to grade", 0.0
    total = sum(words.values())
    n_target = sum(words.get(k, 0) for k in scripts)
    share = n_target / total if total else 0.0
    other = sorted(k for k in SCRIPTS if k not in scripts and words.get(k) and k != "latin")
    if other:
        return False, "the %s mixes in %s" % (basis, ", ".join(other)), share
    if value == "japanese" and not words.get("kana"):
        return False, "no kana: not Japanese", share
    # only a Cyrillic text can be Russian instead: an English message fails on the share below, with a reason that says
    # so
    if value == "ukrainian" and words.get("cyrillic") and not any(ch in _UK_ONLY for ch in prose):
        return False, "Cyrillic without the letters only Ukrainian uses", share
    if value == "russian" and any(ch in _UK_ONLY for ch in prose):
        return False, "Ukrainian letters in the text", share
    if "latin" not in scripts and basis == "prose":
        for sentence in _SENTENCE_END.split(prose):
            latin = [
                w
                for w in re.findall(r"[^\W\d_]+", sentence)
                if script_of(w[0]) == "latin" and not (len(w) > 1 and w.isupper())
            ]
            if len(latin) >= 3 and not any(word_counts(sentence).get(k) for k in scripts):
                return False, "a sentence in a Latin-script language: %r" % " ".join(sentence.split())[:60], share
    if n_target * 2 <= total:
        return False, "%d of %d words in the script of %s (%s)" % (n_target, total, value, basis), share
    if scripts == ("latin",):
        votes = latin_votes(prose)
        mine, others = votes.get(code, 0), {k: v for k, v in votes.items() if k != code and v}
        best_other = max(others.values()) if others else 0
        for sentence in _SENTENCE_END.split(prose):
            sv = latin_votes(sentence)
            n_words = len(re.findall(r"[^\W\d_]+", sentence))
            rival = max(((k, v) for k, v in sv.items() if k != code), key=lambda kv: kv[1], default=(None, 0))
            if n_words >= 5 and sv.get(code, 0) == 0 and rival[1] >= 2:
                return False, "a sentence in %s: %r" % (CODE_TO_NAME[rival[0]], " ".join(sentence.split())[:60]), share
        if mine + sum(others.values()) < 3:
            return True, "too few function words to tell the language (%d)" % (mine + sum(others.values())), 0.5
        lshare = mine / (mine + sum(others.values())) if (mine + sum(others.values())) else 0.0
        if mine <= best_other:
            return False, "%s function words %d, another language %d" % (value, mine, best_other), lshare
        return True, "ok (%s function words %d of %d)" % (value, mine, mine + sum(others.values())), lshare
    return True, "ok (%d of %d words in %s)" % (n_target, total, value), share


# inline code that reads as a sentence: four or more plain words, with nothing in it that code has. The code sets
# inline code aside, so such a sentence in another language never counts against a pass; the judge, whose question says
# that a sentence of prose in backticks is prose, decides such a text ("已完成修复。`All of the split tests pass now.`")
_SENTENCE = re.compile(r"\s*[^\W\d_]+(?:['’-][^\W\d_]+)*(?:[ ,;:]+[^\W\d_]+(?:['’-][^\W\d_]+)*){3,}[.!?]?\s*")


def sentence_in_code(text: str) -> str | None:
    """an inline code span of the text (outside code blocks) that reads as a sentence of plain words, or None"""
    return next(
        (
            m.group(0)
            for m in _INLINE_CODE.finditer(_FENCED_CODE.sub(" ", text))
            if _SENTENCE.fullmatch(m.group(0)[1:-1])
        ),
        None,
    )


def assess(value: str, text: str) -> dict[str, bool | float | str | None]:
    """{ok, clear, share, detail}: a fail with no word of the language, or a pass with at least 90% of the words (or of
    the function words, for a Latin-script language) in it, is clear; every other text goes to the judge"""
    ok, detail, share = check(value, text)
    if ok is None:
        return {"ok": None, "clear": True, "share": 0.0, "detail": detail}
    clear = (not ok and share == 0) or (ok and share >= CLEAR_SHARE)
    if ok and clear and "latin" not in LANGUAGES[str(value).lower()][1] and _LATIN_RUN.search(split_prose(text)[0]):
        clear, detail = False, detail + "; a run of Latin words"
    if ok and clear and sentence_in_code(text):
        clear, detail = False, detail + "; a sentence of words in backticks"
    return {"ok": ok, "clear": clear, "share": round(share, 3), "detail": detail}
