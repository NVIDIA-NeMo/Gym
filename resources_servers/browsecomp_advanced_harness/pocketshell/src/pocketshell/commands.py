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
"""The read-only command set, implemented in-process.

Program selection is measurement-driven. Across 1.65M real calls these 12
programs cover 98.7-99.1% of all command-position occurrences::

    grep head cat sed echo ls wc tail tr printf sort cut

and these bring it past 99.9%: nl uniq strings cd find file diff.

Flag selection likewise: grep needs 12 flags for 99% coverage, sed needs 2,
and head/tail/wc/cat/ls/tr/sort need 2-5 each.

Every command has the signature ``fn(args, stdin, ctx) -> (stdout, stderr, code)``
where streams are ``str`` (files are read with ``errors='surrogateescape'`` so
arbitrary bytes round-trip) and ``code`` follows the usual conventions --
notably grep returns 1 when nothing matched, which ``&&`` / ``||`` rely on.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

from .fsview import PathEscape, Workspace
from .regex_xlate import compile_pattern


__all__ = ["COMMANDS", "CommandError", "Ctx", "run_command"]


class CommandError(Exception):
    def __init__(self, msg: str, code: int = 2):
        super().__init__(msg)
        self.code = code


@dataclass
class Ctx:
    ws: Workspace
    deadline: float | None = None

    def budget(self) -> float | None:
        """Seconds left, for the regex engine's per-match timeout."""
        if self.deadline is None:
            return None
        import time as _t

        return max(0.05, self.deadline - _t.monotonic())

    def tick(self) -> None:
        """Abort a long scan. Python's `re` BACKTRACKS where GNU grep's DFA does
        not, so a pathological agent pattern can run far longer here than it did
        under the subprocess. Checked periodically inside the per-line loops."""
        if self.deadline is not None:
            import time as _t

            if _t.monotonic() > self.deadline:
                raise TimeoutError("command timed out")


def _split_lines(text: str) -> list[str]:
    """Split like grep does: on \\n only, dropping the trailing empty field."""
    if text == "":
        return []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _join(lines: list[str]) -> str:
    return "".join(ln + "\n" for ln in lines)


def _byte_slice(text: str, n: int, tail: bool = False) -> str:
    raw = text.encode("utf-8", "surrogateescape")
    raw = raw[-n:] if tail else raw[:n]
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# argument helpers
# ---------------------------------------------------------------------------
def _parse_flags(args: list[str], takes_value: set[str], combinable: str, long_takes_value: set[str] | None = None):
    """Split argv into (flags dict, positional list). Supports -abc bundling."""
    flags: dict[str, str | bool] = {}
    pos: list[str] = []
    long_takes_value = long_takes_value or set()
    i = 0
    end = False
    while i < len(args):
        a = args[i]
        if end or a == "-" or not a.startswith("-") or len(a) == 1:
            pos.append(a)
            i += 1
            continue
        if a == "--":
            end = True
            i += 1
            continue
        if a.startswith("--"):
            name, eq, val = a.partition("=")
            if name in long_takes_value and not eq:
                if i + 1 < len(args):
                    flags[name] = args[i + 1]
                    i += 2
                    continue
            flags[name] = val if eq else True
            i += 1
            continue
        j = 1
        while j < len(a):
            ch = a[j]
            key = "-" + ch
            if ch in takes_value:
                rest = a[j + 1 :]
                if rest:
                    flags[key] = rest
                elif i + 1 < len(args):
                    flags[key] = args[i + 1]
                    i += 1
                else:
                    raise CommandError(f"option requires an argument -- '{ch}'")
                j = len(a)
                break
            if ch.isdigit() and combinable == "headtail":
                flags["-n"] = a[j:]
                flags["__obsolete_num"] = True
                j = len(a)
                break
            flags[key] = True
            j += 1
        i += 1
    return flags, pos


def _read_inputs(paths: list[str], stdin: str, ctx: Ctx, binary: bool = False):
    """Yield (label, text) for each input; falls back to stdin when no paths."""
    if not paths:
        yield None, stdin
        return
    for p in paths:
        if p == "-":
            yield None, stdin
            continue
        if binary:
            yield p, ctx.ws.read_bytes(p).decode("utf-8", "surrogateescape")
        else:
            yield p, ctx.ws.read_text(p)


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------
GREP_VALUE_FLAGS = set("emABCf")
GREP_LONG_VALUE = {
    "--regexp",
    "--max-count",
    "--after-context",
    "--before-context",
    "--context",
    "--include",
    "--exclude",
}


def cmd_grep(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, GREP_VALUE_FLAGS, "", GREP_LONG_VALUE)

    pattern = flags.get("-e") or flags.get("--regexp")
    if pattern is True:
        pattern = None
    if pattern is None:
        if not pos:
            raise CommandError("grep: no pattern")
        pattern = pos.pop(0)

    dialect = "bre"
    if flags.get("-E") or flags.get("--extended-regexp"):
        dialect = "ere"
    if flags.get("-P") or flags.get("--perl-regexp"):
        dialect = "pcre"
    if flags.get("-F") or flags.get("--fixed-strings"):
        dialect = "fixed"

    icase = bool(flags.get("-i") or flags.get("-y") or flags.get("--ignore-case"))
    invert = bool(flags.get("-v") or flags.get("--invert-match"))
    rx = compile_pattern(
        pattern,
        dialect=dialect,
        ignore_case=icase,
        word=bool(flags.get("-w")),
        line=bool(flags.get("-x")),
        timeout=ctx.budget(),
    )

    show_num = bool(flags.get("-n") or flags.get("--line-number"))
    only = bool(flags.get("-o") or flags.get("--only-matching"))
    count_only = bool(flags.get("-c") or flags.get("--count"))
    files_with = bool(flags.get("-l") or flags.get("--files-with-matches"))
    files_without = bool(flags.get("-L"))
    quiet = bool(flags.get("-q"))
    recursive = bool(flags.get("-r") or flags.get("-R") or flags.get("--recursive"))
    no_name = bool(flags.get("-h"))
    force_name = bool(flags.get("-H"))

    def _int(key, default=None):
        v = flags.get(key)
        if v in (None, True):
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    ctx_a = _int("-A", 0) or _int("--after-context", 0) or 0
    ctx_b = _int("-B", 0) or _int("--before-context", 0) or 0
    both = _int("-C", None)
    if both is None:
        both = _int("--context", None)
    if both is not None:
        ctx_a = ctx_b = both
    max_count = _int("-m", None) or _int("--max-count", None)

    # expand inputs (recursive walk when -r)
    targets: list[str] = list(pos)
    if recursive:
        expanded: list[str] = []
        for t in targets or ["."]:
            p = ctx.ws.resolve(t)
            if p.is_dir():
                for sub in sorted(p.rglob("*")):
                    if sub.is_file():
                        expanded.append(ctx.ws.display(sub))
            else:
                expanded.append(t)
        targets = expanded

    multi = len(targets) > 1
    show_name = (multi or force_name) and not no_name
    out: list[str] = []
    total_hits = 0
    errors: list[str] = []

    for label, text in _read_inputs(targets, stdin, ctx):
        try:
            lines = _split_lines(text)
        except Exception as exc:  # pragma: no cover
            errors.append(f"grep: {exc}")
            continue
        hits: list[int] = []
        for idx, line in enumerate(lines):
            if (idx & 0x3FF) == 0:
                ctx.tick()
            m = rx.search(line)
            if bool(m) != invert:
                hits.append(idx)
                if max_count and len(hits) >= max_count:
                    break
        total_hits += len(hits)

        if quiet:
            if hits:
                return "", "", 0
            continue
        if files_with:
            if hits and label:
                out.append(label)
            continue
        if files_without:
            if not hits and label:
                out.append(label)
            continue
        if count_only:
            prefix = f"{label}:" if show_name and label else ""
            out.append(f"{prefix}{len(hits)}")
            continue

        emitted: set[int] = set()
        hitset = set(hits)
        last_emitted = None
        # With context requested, GNU also separates one FILE's group from the
        # previous file's with `--`.
        if (ctx_a or ctx_b) and hits and out and not (count_only or files_with or files_without):
            out.append("--")
        for idx in hits:
            lo = max(0, idx - ctx_b)
            hi = min(len(lines) - 1, idx + ctx_a)
            for k in range(lo, hi + 1):
                if k in emitted:
                    continue
                # GNU grep prints a `--` separator between non-contiguous context
                # groups. Only when context was requested.
                if (ctx_a or ctx_b) and last_emitted is not None and k > last_emitted + 1:
                    out.append("--")
                emitted.add(k)
                last_emitted = k
                # GNU uses ':' after the filename and line number on MATCH lines
                # and '-' on context lines. Membership in the hit set, NOT
                # `k == idx` -- with overlapping context windows a line can be
                # emitted as another hit's context while still being a match.
                sep = ":" if (k in hitset or not (ctx_a or ctx_b)) else "-"
                prefix = ""
                if show_name and label:
                    prefix += f"{label}{sep}"
                if show_num:
                    prefix += f"{k + 1}{sep}"
                if only and k == idx:
                    for mo in rx.finditer(lines[k]):
                        out.append(prefix + mo.group(0))
                elif only:
                    continue
                else:
                    out.append(prefix + lines[k])

    if quiet:
        return "", "", 1
    code = 0 if total_hits else 1
    if errors:
        return _join(out), _join(errors), 2
    return _join(out), "", code


# ---------------------------------------------------------------------------
# cat / head / tail / nl / wc
# ---------------------------------------------------------------------------
def cmd_cat(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set(), "")
    number = bool(flags.get("-n"))
    squeeze = bool(flags.get("-s"))
    show_ends = bool(flags.get("-E") or flags.get("-A"))
    out, errs, code = [], [], 0
    n = 0
    for label, text in _read_inputs(pos, stdin, ctx):
        lines = _split_lines(text)
        prev_blank = False
        for ln in lines:
            if squeeze and ln == "" and prev_blank:
                continue
            prev_blank = ln == ""
            if show_ends:
                ln = ln + "$"
            if number:
                n += 1
                out.append(f"{n:>6}\t{ln}")
            else:
                out.append(ln)
    return _join(out), _join(errs), code


def cmd_head(args, stdin, ctx: Ctx):
    return _head_tail(args, stdin, ctx, tail=False)


def cmd_tail(args, stdin, ctx: Ctx):
    return _head_tail(args, stdin, ctx, tail=True)


def _head_tail(args, stdin, ctx: Ctx, tail: bool):
    flags, pos = _parse_flags(args, set("nc"), "headtail")
    quiet = bool(flags.get("-q"))
    nbytes = flags.get("-c")
    nlines = flags.get("-n")
    name = "tail" if tail else "head"

    def _num(v, default):
        if v in (None, True):
            return default, False
        s = str(v)
        plus = s.startswith("+")
        s = s.lstrip("+-")
        try:
            return int(s), plus
        except ValueError:
            raise CommandError(f"{name}: invalid number of lines: '{v}'")

    # GNU rejects the obsolete `tail -NUM` form when there are 2+ file operands
    # ("option used in invalid context"); `head -NUM` is fine. Match it so the
    # baseline this replaces stays byte-comparable.
    if tail and flags.get("__obsolete_num") and len(pos) > 1:
        raise CommandError(f"tail: option used in invalid context -- {str(flags.get('-n', ''))[0]}", code=1)
    multi = len(pos) > 1 and not quiet
    # -c slices raw bytes and must NOT invent a trailing newline (GNU head -c 5
    # of "Alpha Beta" prints exactly "Alpha"), so build the output as raw text
    # rather than going through the line joiner.
    chunks: list[str] = []
    first = True
    for label, text in _read_inputs(pos, stdin, ctx):
        if multi and label:
            # GNU separates subsequent files with a BLANK line before the header.
            chunks.append(f"==> {label} <==\n" if first else f"\n==> {label} <==\n")
        first = False
        if nbytes not in (None, True):
            cnt, _ = _num(nbytes, 10)
            chunks.append(_byte_slice(text, cnt, tail=tail))
            continue
        cnt, from_start = _num(nlines, 10)
        lines = _split_lines(text)
        if tail:
            sel = lines[cnt - 1 :] if from_start else (lines[-cnt:] if cnt else [])
        else:
            sel = lines[:cnt]
        chunks.append(_join(sel))
    return "".join(chunks), "", 0


def cmd_nl(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set("bs"), "")
    out, n = [], 0
    for label, text in _read_inputs(pos, stdin, ctx):
        for ln in _split_lines(text):
            if ln.strip() == "" and flags.get("-b") != "a":
                out.append("       " + ln)
                continue
            n += 1
            out.append(f"{n:>6}\t{ln}")
    return _join(out), "", 0


def cmd_wc(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set(), "")
    want_l = bool(flags.get("-l"))
    want_w = bool(flags.get("-w"))
    want_c = bool(flags.get("-c"))
    want_m = bool(flags.get("-m"))
    want_L = bool(flags.get("-L"))
    if not any([want_l, want_w, want_c, want_m, want_L]):
        want_l = want_w = want_c = True
    # GNU wc does NOT pad a single input; with several inputs it right-aligns
    # every column to the widest value (including the `total` row).
    rows: list[tuple[list[int], str | None]] = []
    tot = [0, 0, 0, 0, 0]
    multi = len(pos) > 1
    for label, text in _read_inputs(pos, stdin, ctx):
        lines = _split_lines(text)
        nl_ = text.count("\n")
        nw = len(text.split())
        nc = len(text.encode("utf-8", "surrogateescape"))
        nm = len(text)
        nL = max((len(x) for x in lines), default=0)
        vals = []
        for want, v, idx in ((want_l, nl_, 0), (want_w, nw, 1), (want_c, nc, 2), (want_m, nm, 3), (want_L, nL, 4)):
            if want:
                vals.append(v)
                tot[idx] = max(tot[idx], v) if idx == 4 else tot[idx] + v
        rows.append((vals, label))
    if multi:
        rows.append(([tot[i] for i, want in enumerate([want_l, want_w, want_c, want_m, want_L]) if want], "total"))
    width = max((len(str(v)) for vals, _ in rows for v in vals), default=1) if multi else 0
    out = []
    for vals, label in rows:
        row = " ".join(f"{v:>{width}}" if width else str(v) for v in vals)
        out.append(f"{row} {label}" if label else row)
    return _join(out), "", 0


# ---------------------------------------------------------------------------
# sed  (79% of real uses are -n 'A,Bp'; s/// is ~13-16%)
# ---------------------------------------------------------------------------
_SED_ADDR = r"(?:\d+|\$|/(?:\\.|[^/])*/)"


def _sed_parse(script: str):
    """Parse the sed subset: [addr[,addr]]{p,d}  and  s/re/rep/flags."""
    cmds = []
    for raw in re.split(r"\s*;\s*|\n", script):
        s = raw.strip()
        if not s:
            continue
        m = re.match(rf"^({_SED_ADDR})?(?:\s*,\s*({_SED_ADDR}))?\s*(p|d|q)$", s)
        if m:
            cmds.append(("range", m.group(1), m.group(2), m.group(3)))
            continue
        m = re.match(rf"^(?:({_SED_ADDR})(?:\s*,\s*({_SED_ADDR}))?\s*)?s(.)", s)
        if m:
            delim = m.group(3)
            body = s[m.end() :]
            parts, cur, esc = [], [], False
            for ch in body:
                if esc:
                    cur.append("\\" + ch if ch != delim else ch)
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == delim:
                    parts.append("".join(cur))
                    cur = []
                else:
                    cur.append(ch)
            parts.append("".join(cur))
            if len(parts) < 2:
                raise CommandError("sed: unterminated `s' command")
            cmds.append(("subst", m.group(1), m.group(2), parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
            continue
        raise CommandError(f"sed: unsupported command: {s!r}")
    return cmds


def _sed_match_addr(addr, idx, total, line, ere, budget=None):
    if addr is None:
        return None
    if addr == "$":
        return idx == total - 1
    if addr.isdigit():
        return idx + 1 == int(addr)
    rx = compile_pattern(addr[1:-1], dialect="ere" if ere else "bre", timeout=budget)
    return bool(rx.search(line))


def cmd_sed(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set("e"), "")
    if flags.get("-i") or flags.get("--in-place"):
        raise CommandError("sed: -i (in-place edit) is blocked -- the workspace is read-only")
    quiet = bool(flags.get("-n") or flags.get("--quiet") or flags.get("--silent"))
    ere = bool(flags.get("-E") or flags.get("-r"))
    script = flags.get("-e")
    if script in (None, True):
        if not pos:
            raise CommandError("sed: no script")
        script = pos.pop(0)
    cmds = _sed_parse(script)

    out: list[str] = []
    # GNU sed treats multiple operands as ONE continuous stream: line numbers and
    # `$` run across the concatenation, not per file. Processing per file makes
    # `sed -n '1,5p' pages/*` print 5 lines from EVERY file instead of 5 total.
    separate = bool(flags.get("-s") or flags.get("--separate"))
    groups = (
        [_split_lines(t) for _, t in _read_inputs(pos, stdin, ctx)]
        if separate
        else [[ln for _, t in _read_inputs(pos, stdin, ctx) for ln in _split_lines(t)]]
    )
    for lines in groups:
        total = len(lines)
        active = {}
        for idx, line in enumerate(lines):
            printed = False
            cur = line
            deleted = False
            for c in cmds:
                kind = c[0]
                a1, a2 = c[1], c[2]
                if a2 is not None:
                    key = id(c)
                    inside = active.get(key, False)
                    if not inside:
                        if _sed_match_addr(a1, idx, total, cur, ere, ctx.budget()):
                            inside = True
                            active[key] = True
                    if (
                        inside
                        and _sed_match_addr(a2, idx, total, cur, ere, ctx.budget())
                        and not (a1 is not None and a1.isdigit() and a2.isdigit() and int(a1) == int(a2))
                    ):
                        active[key] = False if idx + 1 > 0 else True
                    sel = inside
                    if inside and a2 is not None and a2.isdigit() and idx + 1 >= int(a2):
                        active[key] = False
                elif a1 is not None:
                    sel = bool(_sed_match_addr(a1, idx, total, cur, ere, ctx.budget()))
                else:
                    sel = True
                if not sel:
                    continue
                if kind == "range":
                    if c[3] == "p":
                        out.append(cur)
                        printed = True
                    elif c[3] == "d":
                        deleted = True
                    elif c[3] == "q":
                        if not quiet and not printed:
                            out.append(cur)
                        return _join(out), "", 0
                else:
                    _, _, _, pat, rep, sflags = c
                    rx = compile_pattern(
                        pat,
                        dialect="ere" if ere else "bre",
                        ignore_case="i" in sflags or "I" in sflags,
                        timeout=ctx.budget(),
                    )
                    count = 0 if "g" in sflags else 1
                    py_rep = re.sub(r"\\(\d)", r"\\\1", rep.replace("\\", "\\\\"))
                    py_rep = py_rep.replace("\\\\\\", "\\")
                    try:
                        cur = rx.sub(lambda m, r=rep: _sed_expand(m, r), cur, count=count)
                    except re.error as exc:
                        raise CommandError(f"sed: {exc}")
            if deleted:
                continue
            if not quiet and not printed:
                out.append(cur)
    return _join(out), "", 0


def _sed_expand(m, rep: str) -> str:
    """Expand a sed replacement: & = whole match, \\1..\\9 = groups."""
    out, i, n = [], 0, len(rep)
    while i < n:
        c = rep[i]
        if c == "\\" and i + 1 < n:
            d = rep[i + 1]
            if d.isdigit():
                try:
                    out.append(m.group(int(d)) or "")
                except (IndexError, error_types):
                    out.append("")
            elif d == "n":
                out.append("\n")
            elif d == "t":
                out.append("\t")
            else:
                out.append(d)
            i += 2
            continue
        if c == "&":
            out.append(m.group(0))
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


error_types = re.error


# ---------------------------------------------------------------------------
# ls / find / file / diff
# ---------------------------------------------------------------------------
def cmd_ls(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set(), "")
    long_fmt = bool(flags.get("-l"))
    show_all = bool(flags.get("-a") or flags.get("-A"))
    by_time = bool(flags.get("-t"))
    by_size = bool(flags.get("-S"))
    reverse = bool(flags.get("-r"))
    dirs_only = bool(flags.get("-d"))
    recurse = bool(flags.get("-R"))

    targets = pos or ["."]
    out, errs, code = [], [], 0

    # GNU ls lists non-directory operands FIRST (no header), then each directory
    # operand. A `name:` header appears only for directories, and only when there
    # is more than one operand overall.
    files: list[Path] = []
    dirs: list[tuple[str, Path]] = []
    for t in targets:
        try:
            p = ctx.ws.resolve(t)
        except PathEscape as exc:
            errs.append(f"ls: {exc}")
            code = 2
            continue
        if not p.exists():
            errs.append(f"ls: cannot access '{t}': No such file or directory")
            code = 2
            continue
        if p.is_dir() and not dirs_only:
            dirs.append((t, p))
        else:
            files.append(p)

    def _sorted(entries: list[Path]) -> list[Path]:
        if by_time:
            return sorted(entries, key=lambda c: _safe_stat(c).st_mtime, reverse=not reverse)
        if by_size:
            return sorted(entries, key=lambda c: _safe_stat(c).st_size, reverse=not reverse)
        return sorted(entries, key=lambda c: c.name, reverse=reverse)

    def _emit(entries: list[Path], display, with_total: bool):
        if long_fmt and with_total:
            blocks = sum(getattr(_safe_stat(c), "st_blocks", 0) for c in entries)
            out.append(f"total {blocks // 2}")
        width = max((len(str(_safe_stat(c).st_size)) for c in entries), default=1) if long_fmt else 0
        for c in entries:
            if long_fmt:
                out.append(_long_row(c, display(c), width))
            else:
                out.append(display(c))

    if files:
        _emit(_sorted(files), lambda c: ctx.ws.display(c), with_total=False)

    header = len(targets) > 1 or (files and dirs)
    for n, (t, p) in enumerate(dirs):
        if files or n:
            out.append("")
        if header:
            out.append(f"{t}:")
        try:
            entries = [c for c in p.iterdir() if show_all or not c.name.startswith(".")]
        except OSError as exc:
            errs.append(f"ls: {exc}")
            code = 2
            continue
        listing = _sorted(entries)
        names = {}
        if show_all and not flags.get("-A"):
            # `p / "."` normalises away, so carry the display name explicitly.
            names = {p.resolve(): ".", p.parent.resolve(): ".."}
            listing = [p, p.parent] + listing
        _emit(listing, lambda c: names.get(c.resolve(), c.name) if names else c.name, with_total=True)
        if recurse:
            for sub in _sorted([x for x in entries if x.is_dir()]):
                sub_out, _, _ = cmd_ls([*[a for a in args if a.startswith("-")], ctx.ws.display(sub)], stdin, ctx)
                out.append("")
                out.append(f"{ctx.ws.display(sub)}:")
                out.extend(_split_lines(sub_out))
    return _join(out), _join(errs), code


def _long_row(c: Path, name: str, width: int = 6) -> str:
    import grp
    import pwd
    import stat as statmod
    import time as timemod

    st = _safe_stat(c)
    mode = getattr(st, "st_mode", None)
    perms = statmod.filemode(mode) if mode else ("drwxr-xr-x" if c.is_dir() else "-rw-r--r--")
    try:
        user = pwd.getpwuid(st.st_uid).pw_name
    except (KeyError, AttributeError):
        user = str(getattr(st, "st_uid", "agent"))
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except (KeyError, AttributeError):
        group = str(getattr(st, "st_gid", "agent"))
    nlink = getattr(st, "st_nlink", 1)
    when = timemod.strftime("%b %e %H:%M", timemod.localtime(getattr(st, "st_mtime", 0)))
    return f"{perms} {nlink} {user} {group} {st.st_size:>{width}} {when} {name}"


class _FakeStat:
    st_size = 0
    st_mtime = 0.0


def _safe_stat(p: Path):
    try:
        return p.stat()
    except OSError:
        return _FakeStat()


def cmd_find(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set(), "")
    paths, preds = [], []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("-"):
            break
        paths.append(a)
        i += 1
    while i < len(args):
        a = args[i]
        if a in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprintf", "-fprint"):
            raise CommandError(f"find: {a} is blocked -- the workspace is read-only")
        if a in ("-name", "-iname", "-path", "-type", "-maxdepth", "-mindepth"):
            preds.append((a, args[i + 1] if i + 1 < len(args) else ""))
            i += 2
            continue
        i += 1
    out = []
    for t in paths or ["."]:
        root = ctx.ws.resolve(t)
        cands = [root, *sorted(root.rglob("*"))] if root.is_dir() else [root]
        maxdepth = next((int(v) for k, v in preds if k == "-maxdepth" and v.isdigit()), None)
        for c in cands:
            rel = ctx.ws.display(c)
            if maxdepth is not None:
                depth = len(Path(rel).parts) - (1 if rel not in (".", "") else 1)
                if depth > maxdepth:
                    continue
            ok = True
            for k, v in preds:
                if k == "-name" and not fnmatch.fnmatch(c.name, v):
                    ok = False
                elif k == "-iname" and not fnmatch.fnmatch(c.name.lower(), v.lower()):
                    ok = False
                elif k == "-path" and not fnmatch.fnmatch(str(rel), v):
                    ok = False
                elif k == "-type" and ((v == "f" and not c.is_file()) or (v == "d" and not c.is_dir())):
                    ok = False
            if ok:
                out.append(rel if rel else ".")
    return _join(out), "", 0


def cmd_file(args, stdin, ctx: Ctx):
    _flags, pos = _parse_flags(args, set(), "")
    out = []
    for t in pos:
        p = ctx.ws.resolve(t)
        if p.is_dir():
            out.append(f"{t}: directory")
            continue
        raw = ctx.ws.read_bytes(t)[:4096]
        if b"\x00" in raw:
            out.append(f"{t}: data")
        elif not raw:
            out.append(f"{t}: empty")
        else:
            out.append(f"{t}: ASCII text" if all(c < 128 for c in raw) else f"{t}: UTF-8 Unicode text")
    return _join(out), "", 0


def cmd_diff(args, stdin, ctx: Ctx):
    import difflib

    _flags, pos = _parse_flags(args, set(), "")
    if len(pos) != 2:
        raise CommandError("diff: need exactly two files")
    a = _split_lines(ctx.ws.read_text(pos[0]))
    b = _split_lines(ctx.ws.read_text(pos[1]))
    delta = list(difflib.unified_diff(a, b, pos[0], pos[1], lineterm=""))
    return _join(delta), "", (1 if delta else 0)


# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------
def cmd_sort(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set("kt"), "")
    numeric = bool(flags.get("-n") or flags.get("-g"))
    reverse = bool(flags.get("-r"))
    unique = bool(flags.get("-u"))
    ignore_case = bool(flags.get("-f"))
    sep = flags.get("-t") if isinstance(flags.get("-t"), str) else None
    key = flags.get("-k") if isinstance(flags.get("-k"), str) else None

    lines: list[str] = []
    for label, text in _read_inputs(pos, stdin, ctx):
        lines.extend(_split_lines(text))

    def keyfn(s: str):
        v = s
        if key:
            fields = s.split(sep) if sep else s.split()
            try:
                idx = int(str(key).split(",")[0].split(".")[0]) - 1
                v = fields[idx] if 0 <= idx < len(fields) else ""
            except ValueError:
                v = s
        if ignore_case:
            v = v.lower()
        if numeric:
            m = re.match(r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", v)
            return (float(m.group(1)) if m else float("-inf"),)
        return (v,)

    lines.sort(key=keyfn, reverse=reverse)
    if unique:
        seen, dedup = set(), []
        for ln in lines:
            k = keyfn(ln)
            if k in seen:
                continue
            seen.add(k)
            dedup.append(ln)
        lines = dedup
    return _join(lines), "", 0


def cmd_uniq(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set(), "")
    count = bool(flags.get("-c"))
    only_dup = bool(flags.get("-d"))
    only_uniq = bool(flags.get("-u"))
    icase = bool(flags.get("-i"))
    lines: list[str] = []
    for label, text in _read_inputs(pos, stdin, ctx):
        lines.extend(_split_lines(text))
    out, prev, n = [], None, 0

    def flush():
        if prev is None:
            return
        if only_dup and n < 2:
            return
        if only_uniq and n > 1:
            return
        out.append(f"{n:>7} {prev}" if count else prev)

    for ln in lines:
        k = ln.lower() if icase else ln
        pk = prev.lower() if (icase and prev is not None) else prev
        if prev is not None and k == pk:
            n += 1
            continue
        flush()
        prev, n = ln, 1
    flush()
    return _join(out), "", 0


def cmd_cut(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set("dfc"), "")
    delim = flags.get("-d")
    delim = "\t" if delim in (None, True) else str(delim)
    fields = flags.get("-f")
    chars = flags.get("-c")
    only_delim = bool(flags.get("-s"))

    def spec(v):
        parts = []
        for chunk in str(v).split(","):
            if "-" in chunk:
                a, _, b = chunk.partition("-")
                parts.append((int(a) if a else 1, int(b) if b else 10**9))
            else:
                parts.append((int(chunk), int(chunk)))
        return parts

    out = []
    for label, text in _read_inputs(pos, stdin, ctx):
        for ln in _split_lines(text):
            if chars not in (None, True):
                sel = spec(chars)
                out.append("".join(ln[a - 1 : b] for a, b in sel))
            elif fields not in (None, True):
                if delim not in ln:
                    if not only_delim:
                        out.append(ln)
                    continue
                fs = ln.split(delim)
                sel = spec(fields)
                picked = []
                for a, b in sel:
                    picked.extend(fs[a - 1 : min(b, len(fs))])
                out.append(delim.join(picked))
            else:
                raise CommandError("cut: you must specify a list of bytes, characters, or fields")
    return _join(out), "", 0


def cmd_tr(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set(), "")
    delete = bool(flags.get("-d"))
    squeeze = bool(flags.get("-s"))
    complement = bool(flags.get("-c") or flags.get("-C"))

    def expand(s: str) -> str:
        out, i = [], 0
        while i < len(s):
            if s.startswith("[:", i):
                end = s.find(":]", i)
                if end != -1:
                    name = s[i + 2 : end]
                    mapping = {
                        "alpha": [chr(c) for c in range(256) if chr(c).isalpha()],
                        "digit": list("0123456789"),
                        "alnum": [chr(c) for c in range(256) if chr(c).isalnum()],
                        "space": list(" \t\n\r\f\v"),
                        "upper": [chr(c) for c in range(256) if chr(c).isupper()],
                        "lower": [chr(c) for c in range(256) if chr(c).islower()],
                        "punct": [chr(c) for c in range(33, 127) if not chr(c).isalnum()],
                    }
                    out.extend(mapping.get(name, []))
                    i = end + 2
                    continue
            if s[i] == "\\" and i + 1 < len(s):
                out.append({"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(s[i + 1], s[i + 1]))
                i += 2
                continue
            if i + 2 < len(s) and s[i + 1] == "-":
                for c in range(ord(s[i]), ord(s[i + 2]) + 1):
                    out.append(chr(c))
                i += 3
                continue
            out.append(s[i])
            i += 1
        return "".join(out)

    text = stdin
    if not pos:
        raise CommandError("tr: missing operand")
    set1 = expand(pos[0])
    set2 = expand(pos[1]) if len(pos) > 1 else ""

    if delete:
        keep = (lambda ch: ch in set1) if complement else (lambda ch: ch not in set1)
        text = "".join(ch for ch in text if keep(ch))
    elif set2:
        if len(set2) < len(set1):
            set2 = set2 + set2[-1] * (len(set1) - len(set2))
        table = {ord(a): b for a, b in zip(set1, set2)}
        text = text.translate(table)
    if squeeze:
        target = set2 or set1
        out, prev = [], None
        for ch in text:
            if ch == prev and ch in target:
                continue
            out.append(ch)
            prev = ch
        text = "".join(out)
    return text, "", 0


def cmd_strings(args, stdin, ctx: Ctx):
    flags, pos = _parse_flags(args, set("n"), "")
    minlen = flags.get("-n")
    try:
        minlen = int(minlen) if minlen not in (None, True) else 4
    except ValueError:
        minlen = 4
    out = []
    for label, text in _read_inputs(pos, stdin, ctx, binary=True):
        raw = text.encode("utf-8", "surrogateescape")
        for m in re.finditer(rb"[\x20-\x7e\t]{%d,}" % minlen, raw):
            out.append(m.group(0).decode("ascii", "replace"))
    return _join(out), "", 0


def cmd_echo(args, stdin, ctx: Ctx):
    no_newline = False
    interpret = False
    i = 0
    while i < len(args) and args[i] in ("-n", "-e", "-E", "-ne", "-en"):
        if "n" in args[i][1:]:
            no_newline = True
        if "e" in args[i][1:]:
            interpret = True
        i += 1
    text = " ".join(args[i:])
    if interpret:
        text = text.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    return text + ("" if no_newline else "\n"), "", 0


def cmd_printf(args, stdin, ctx: Ctx):
    if not args:
        raise CommandError("printf: usage: printf format [arguments]")
    fmt = args[0].replace("\\n", "\n").replace("\\t", "\t")
    rest = args[1:]
    nspec = len(re.findall(r"%[-+ #0-9.]*[sdifeguxXoc]", fmt))
    if nspec == 0:
        return fmt * (1 if not rest else 1), "", 0
    out = []
    idx = 0
    while True:
        chunk = fmt
        vals = []
        for m in re.finditer(r"%[-+ #0-9.]*([sdifeguxXoc])", fmt):
            v = rest[idx] if idx < len(rest) else ""
            idx += 1
            if m.group(1) in "dioxXuc":
                try:
                    v = int(float(v)) if v else 0
                except ValueError:
                    v = 0
            elif m.group(1) in "feg":
                try:
                    v = float(v) if v else 0.0
                except ValueError:
                    v = 0.0
            vals.append(v)
        try:
            chunk = fmt % tuple(vals)
        except (TypeError, ValueError):
            chunk = fmt
        out.append(chunk)
        if idx >= len(rest):
            break
    return "".join(out), "", 0


def cmd_cd(args, stdin, ctx: Ctx):
    target = args[0] if args else "."
    ctx.ws.chdir(target)
    return "", "", 0


def cmd_true(args, stdin, ctx: Ctx):
    return "", "", 0


def cmd_false(args, stdin, ctx: Ctx):
    return "", "", 1


COMMANDS = {
    "grep": cmd_grep,
    "egrep": lambda a, s, c: cmd_grep(["-E", *a], s, c),
    "fgrep": lambda a, s, c: cmd_grep(["-F", *a], s, c),
    "cat": cmd_cat,
    "head": cmd_head,
    "tail": cmd_tail,
    "nl": cmd_nl,
    "wc": cmd_wc,
    "sed": cmd_sed,
    "ls": cmd_ls,
    "find": cmd_find,
    "file": cmd_file,
    "diff": cmd_diff,
    "sort": cmd_sort,
    "uniq": cmd_uniq,
    "cut": cmd_cut,
    "tr": cmd_tr,
    "strings": cmd_strings,
    "echo": cmd_echo,
    "printf": cmd_printf,
    "cd": cmd_cd,
    "true": cmd_true,
    "false": cmd_false,
    ":": cmd_true,
}


def run_command(name: str, args: list[str], stdin: str, ctx: Ctx):
    fn = COMMANDS.get(name)
    if fn is None:
        raise CommandError(
            f"{name}: command not found (read-only shell implements: {' '.join(sorted(COMMANDS))})", code=127
        )
    return fn(args, stdin, ctx)
