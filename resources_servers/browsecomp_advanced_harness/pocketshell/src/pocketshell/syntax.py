# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lexer + parser for the pocket-shell grammar.

Scope is set by measurement, not ambition. Across 1.65M real agent-written
``bash_command`` calls sampled from BrowseComp evaluation and synthetic-data
rollouts, the cumulative coverage of each tier is::

    single command                       39.9 - 49.5 %
    + linear pipeline                    86.9 - 91.0 %
    + ; && ||                            97.5 - 98.6 %
    + variables / arithmetic             98.6 - 98.7 %
    + for/while/if                      100.00 - 100.00 %
    + command substitution, heredocs      0 calls / 22 calls (0.001%)

Command substitution, process substitution, heredocs and backgrounding are
therefore NOT implemented -- the upstream guard already rejected them, so they
never appear in real traffic. They raise :class:`ParseError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


__all__ = [
    "ParseError",
    "Word",
    "SimpleCommand",
    "Pipeline",
    "AndOrList",
    "Script",
    "ForLoop",
    "parse",
]


class ParseError(ValueError):
    """Syntax the pocket shell does not implement."""


# --------------------------------------------------------------------------
# Words carry their quoting so expansion can decide what to glob / split.
# --------------------------------------------------------------------------
@dataclass
class Part:
    text: str
    quote: str = ""  # "" bare, "'" single-quoted, '"' double-quoted


@dataclass
class Word:
    parts: list[Part] = field(default_factory=list)

    @property
    def raw(self) -> str:
        return "".join(p.text for p in self.parts)

    def has_unquoted(self, chars: str) -> bool:
        return any(p.quote == "" and any(c in p.text for c in chars) for p in self.parts)


@dataclass
class Redirect:
    fd: int  # 1 = stdout, 2 = stderr
    target: str  # only /dev/null is honoured; anything else is rejected


@dataclass
class SimpleCommand:
    words: list[Word] = field(default_factory=list)
    assignments: list[tuple[str, Word]] = field(default_factory=list)
    redirects: list[Redirect] = field(default_factory=list)


@dataclass
class Pipeline:
    commands: list = field(default_factory=list)  # SimpleCommand | ForLoop


@dataclass
class AndOrList:
    """pipelines joined by && / || ; operators[i] joins pipelines[i] and [i+1]."""

    pipelines: list[Pipeline] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)


@dataclass
class Script:
    items: list[AndOrList] = field(default_factory=list)


@dataclass
class ForLoop:
    var: str
    items: list[Word]
    body: "Script"


# --------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------
OPERATORS = ("&&", "||", ";;", ";", "|", "\n")
REDIR_START = ("2>", "1>", ">>", ">", "&>")

RESERVED = {"for", "do", "done", "in", "while", "until", "if", "then", "else", "elif", "fi", "case", "esac"}


@dataclass
class Token:
    kind: str  # "word" | "op" | "redirect"
    word: Word | None = None
    text: str = ""


def _lex(src: str) -> list[Token]:
    toks: list[Token] = []
    i, n = 0, len(src)
    cur: list[Part] = []
    bare: list[str] = []

    def flush_bare():
        """Coalesce runs of unquoted characters into ONE part.

        Emitting a Part per character would split `$f` into '$' and 'f', so
        variable expansion (which runs per part) would never see it.
        """
        nonlocal bare
        if bare:
            cur.append(Part("".join(bare), ""))
            bare = []

    def flush():
        nonlocal cur
        flush_bare()
        if cur:
            toks.append(Token("word", Word(cur)))
            cur = []

    while i < n:
        c = src[i]

        if c in " \t":
            flush()
            i += 1
            continue

        if c == "\\" and i + 1 < n:  # backslash escape outside quotes
            flush_bare()
            cur.append(Part(src[i + 1], '"'))
            i += 2
            continue

        if c == "'":
            j = src.find("'", i + 1)
            if j == -1:
                raise ParseError("unterminated single quote")
            flush_bare()
            cur.append(Part(src[i + 1 : j], "'"))
            i = j + 1
            continue

        if c == '"':
            j, buf = i + 1, []
            while j < n and src[j] != '"':
                if src[j] == "\\" and j + 1 < n and src[j + 1] in '"\\$`':
                    buf.append(src[j + 1])
                    j += 2
                    continue
                # Substitutions are LIVE inside double quotes. Detect them here or
                # they pass through as literal text and the command silently does
                # something other than what the agent wrote (fail-open).
                if src[j] == "`":
                    raise ParseError("backtick command substitution is not supported")
                if src.startswith("$(", j) and not src.startswith("$((", j):
                    raise ParseError("$(...) command substitution is not supported")
                buf.append(src[j])
                j += 1
            if j >= n:
                raise ParseError("unterminated double quote")
            flush_bare()
            cur.append(Part("".join(buf), '"'))
            i = j + 1
            continue

        if c == "`":
            raise ParseError("backtick command substitution is not supported")

        if src.startswith("$((", i):
            j = src.find("))", i)
            if j == -1:
                raise ParseError("unterminated $(( ))")
            flush_bare()
            cur.append(Part(src[i : j + 2], ""))
            i = j + 2
            continue

        if src.startswith("$(", i):
            raise ParseError("$(...) command substitution is not supported")

        if src.startswith("<(", i) or src.startswith(">(", i):
            raise ParseError("process substitution is not supported")

        if src.startswith("<<", i):
            raise ParseError("heredocs are not supported")

        # redirections
        if c in "><" or (c in "12" and i + 1 < n and src[i + 1] == ">"):
            m = None
            for r in ("2>>", "1>>", "2>", "1>", "&>", ">>", ">", "<"):
                if src.startswith(r, i):
                    m = r
                    break
            if m:
                # `2>&1` style fd-dup
                if src.startswith(m + "&", i):
                    k = i + len(m) + 1
                    while k < n and (src[k].isdigit() or src[k] == "-"):
                        k += 1
                    flush()
                    toks.append(Token("redirect", text=src[i:k]))
                    i = k
                    continue
                flush()
                toks.append(Token("redirect", text=m))
                i += len(m)
                continue

        if src.startswith("&&", i) or src.startswith("||", i):
            flush()
            toks.append(Token("op", text=src[i : i + 2]))
            i += 2
            continue

        if c in ";\n":
            flush()
            toks.append(Token("op", text=";"))
            i += 1
            continue

        if c == "|":
            flush()
            toks.append(Token("op", text="|"))
            i += 1
            continue

        if c == "&":
            raise ParseError("background execution (&) is not supported")

        bare.append(c)
        i += 1

    flush()
    return toks


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
class _P:
    def __init__(self, toks: list[Token]):
        self.t = toks
        self.i = 0

    def peek(self) -> Token | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def word_text(self) -> str | None:
        tok = self.peek()
        if tok and tok.kind == "word":
            return tok.word.raw
        return None

    def eat(self) -> Token:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect_word(self, text: str):
        if self.word_text() != text:
            raise ParseError(f"expected '{text}'")
        self.eat()

    # script := andor (';' andor)*
    def script(self, stop: set[str] | None = None) -> Script:
        stop = stop or set()
        sc = Script()
        while True:
            while self.peek() and self.peek().kind == "op" and self.peek().text == ";":
                self.eat()
            w = self.word_text()
            if self.peek() is None or (w is not None and w in stop):
                break
            sc.items.append(self.andor(stop))
            nxt = self.peek()
            if nxt is None:
                break
            if nxt.kind == "op" and nxt.text == ";":
                continue
            w = self.word_text()
            if w is not None and w in stop:
                break
            if nxt.kind == "op":
                raise ParseError(f"unexpected operator '{nxt.text}'")
        return sc

    def andor(self, stop: set[str]) -> AndOrList:
        node = AndOrList(pipelines=[self.pipeline(stop)])
        while True:
            tok = self.peek()
            if tok and tok.kind == "op" and tok.text in ("&&", "||"):
                node.operators.append(self.eat().text)
                node.pipelines.append(self.pipeline(stop))
                continue
            break
        return node

    def pipeline(self, stop: set[str]) -> Pipeline:
        pipe = Pipeline(commands=[self.command(stop)])
        while True:
            tok = self.peek()
            if tok and tok.kind == "op" and tok.text == "|":
                self.eat()
                pipe.commands.append(self.command(stop))
                continue
            break
        return pipe

    def command(self, stop: set[str]):
        w = self.word_text()
        if w == "for":
            return self.for_loop()
        if w in ("while", "until", "if", "case", "select", "function"):
            raise ParseError(f"'{w}' is not supported")
        return self.simple(stop)

    def for_loop(self) -> ForLoop:
        self.expect_word("for")
        var = self.word_text()
        if not var:
            raise ParseError("for: missing variable name")
        self.eat()
        items: list[Word] = []
        if self.word_text() == "in":
            self.eat()
            while True:
                tok = self.peek()
                if tok is None or tok.kind != "word" or tok.word.raw in ("do",):
                    break
                items.append(self.eat().word)
        while self.peek() and self.peek().kind == "op" and self.peek().text == ";":
            self.eat()
        self.expect_word("do")
        body = self.script(stop={"done"})
        self.expect_word("done")
        # `done 2>/dev/null | head -80` is common: consume redirections that
        # attach to the loop itself so the pipeline parser sees the `|` next.
        while self.peek() and self.peek().kind == "redirect":
            op = self.eat().text
            if "&" in op:
                continue
            tgt = self.peek()
            if tgt is None or tgt.kind != "word":
                raise ParseError("redirection: missing target")
            target = self.eat().word.raw
            if target != "/dev/null":
                raise ParseError("output redirection to a file (writes blocked)")
        return ForLoop(var=var, items=items, body=body)

    def simple(self, stop: set[str]) -> SimpleCommand:
        cmd = SimpleCommand()
        seen_word = False
        while True:
            tok = self.peek()
            if tok is None:
                break
            if tok.kind == "op":
                break
            if tok.kind == "redirect":
                op = self.eat().text
                if "&" in op:  # 2>&1 and friends: fd dup, no file involved
                    cmd.redirects.append(Redirect(fd=2 if op.startswith("2") else 1, target="&"))
                    continue
                if op == "<":
                    tgt = self.peek()
                    if tgt is None or tgt.kind != "word":
                        raise ParseError("input redirection: missing file")
                    cmd.redirects.append(Redirect(fd=0, target=self.eat().word.raw))
                    continue
                tgt = self.peek()
                if tgt is None or tgt.kind != "word":
                    raise ParseError("redirection: missing target")
                target = self.eat().word.raw
                if target != "/dev/null":
                    raise ParseError("output redirection to a file (writes blocked)")
                fd = 2 if op.startswith("2") else 1
                if op.startswith("&"):
                    cmd.redirects.append(Redirect(fd=1, target="/dev/null"))
                    fd = 2
                cmd.redirects.append(Redirect(fd=fd, target="/dev/null"))
                continue

            raw = tok.word.raw
            if not seen_word and raw in stop:
                break
            # leading VAR=value assignments
            if not seen_word and _is_assignment(tok.word):
                self.eat()
                name, _, _ = raw.partition("=")
                val = (
                    Word([Part(raw.partition("=")[2], p.quote) for p in tok.word.parts[:1]])
                    if len(tok.word.parts) == 1
                    else _strip_assign(tok.word)
                )
                cmd.assignments.append((name, val))
                continue
            seen_word = True
            cmd.words.append(self.eat().word)
        if not cmd.words and not cmd.assignments:
            raise ParseError("empty command")
        return cmd


def _is_assignment(w: Word) -> bool:
    raw = w.raw
    if "=" not in raw:
        return False
    name = raw.split("=", 1)[0]
    if not name or not (name[0].isalpha() or name[0] == "_"):
        return False
    if not all(ch.isalnum() or ch == "_" for ch in name):
        return False
    # the '=' must be unquoted for this to be an assignment
    seen = 0
    for p in w.parts:
        if p.quote == "" and "=" in p.text:
            return True
        seen += len(p.text)
        if seen > len(name):
            break
    return False


def _strip_assign(w: Word) -> Word:
    """Return the value side of a VAR=value word, preserving per-part quoting."""
    out: list[Part] = []
    remaining = w.raw.split("=", 1)[0] + "="
    budget = len(remaining)
    for p in w.parts:
        if budget <= 0:
            out.append(p)
        elif len(p.text) <= budget:
            budget -= len(p.text)
        else:
            out.append(Part(p.text[budget:], p.quote))
            budget = 0
    return Word(out)


def parse(src: str) -> Script:
    """Parse a keystrokes string into a :class:`Script`."""
    return _P(_lex(src)).script()
