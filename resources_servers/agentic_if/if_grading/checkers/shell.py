# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading a bash command the way the shell does, for the checkers of bash calls.

Which text is data (quoted strings, here-document bodies, comments), where one simple command ends and the next
begins, which files a command writes, creates or deletes, and whether it runs a test suite. A best-effort reader of the
common forms (a pattern read over the raw command matched `|`, `&&`, `rm` or `> nul` inside a grep pattern, an echo or
a Python string); it expands nothing, and a command substitution inside double quotes stays data.
"""

import posixpath
import re
import shlex


HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][\w.-]*)\1")
# a separator between simple commands: && || |& | & ; ( ) and line ends, not the & of 2>&1 or &> and not the | of >|
SEPARATOR_RE = re.compile(r"&&|\|\||\|&|(?<![>\\])\|(?!\|)|(?<![<>\\])&(?![>&])|(?<!\\);|\n|(?<!\\)[()]")


def drop_heredocs(command: str) -> str:
    """The command without here-document bodies (their text is data, not shell)."""
    out, lines, i = [], command.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        ends = [m.group(2) for m in HEREDOC_RE.finditer(line)]
        i += 1
        for delim in ends:
            while i < len(lines) and lines[i].strip() != delim:
                i += 1
            i += 1
    return "\n".join(out)


def _quote_end(text: str, i: int) -> int:
    """The index after the quoted string at text[i] ('...', "..." or $'...'); len(text) if it never closes."""
    ansi = text[i] == "$"  # $'...' honours backslash escapes, like "..."; '...' does not
    if ansi:
        i += 1
    q, j = text[i], i + 1
    while j < len(text):
        if text[j] == "\\" and (q == '"' or ansi):
            j += 2
            continue
        if text[j] == q:
            return j + 1
        j += 1
    return len(text)


def segments(command: str) -> list[tuple[str, bool]]:
    """[(text, quoted)] of a command without its here-document bodies and comments: each quoted string ('...', "...",
    $'...', quotes included) apart from the shell text around it. A backslash-escaped character is shell text."""
    text = drop_heredocs(command.replace("\\\n", " "))
    out, cur, i = [], [], 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            cur.append(text[i : i + 2])
            i += 2
            continue
        if ch == "#" and (i == 0 or text[i - 1] in " \t\n;&|()"):  # a comment, to the end of the line
            j = text.find("\n", i)
            i = len(text) if j < 0 else j
            continue
        if ch in "'\"" or (ch == "$" and text[i + 1 : i + 2] == "'"):
            if cur:
                out.append(("".join(cur), False))
                cur = []
            j = _quote_end(text, i)
            out.append((text[i:j], True))
            i = j
            continue
        cur.append(ch)
        i += 1
    if cur:
        out.append(("".join(cur), False))
    return out


def unquoted(command: str) -> str:
    """The shell text of a command: every quoted string becomes "", here-document bodies and comments are left out."""
    return "".join('""' if quoted else t for t, quoted in segments(command))


def command_parts(command: str) -> list[str]:
    """The simple commands of a shell command, split at ; && || | & ( ) and line ends outside quotes; quoted text
    stays in its part as written."""
    parts, cur = [], ""
    for t, quoted in segments(command):
        if quoted:
            cur += t
            continue
        pieces = SEPARATOR_RE.split(t)
        cur += pieces[0]
        for p in pieces[1:]:
            parts.append(cur)
            cur = p
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


REDIRECTS = {">", ">>", "&>", "&>>", ">|"}
PREFIXES = {"sudo", "env", "command", "nohup", "time", "exec", "builtin"}
# options whose next word is their argument, not a file (in `touch -d '2026-09-05 11:10:26' /app/telemetry.db`
# the date is no file the command creates; `touch -r REF` reads REF, it does not write it)
OPTIONS_WITH_ARG = {"touch": ("-d", "-t", "-r", "--date", "--reference"), "mkdir": ("-m", "--mode", "--context")}


def _operands(args: list[str], with_arg: tuple[str, ...] = ()) -> list[str]:
    """The operands among a command's arguments: no option, nor the argument of an option that takes one; after
    `--`, every word."""
    out, k = [], 0
    while k < len(args):
        x = args[k]
        if x == "--":
            return out + args[k + 1 :]
        if x in with_arg:
            k += 2
            continue
        if not x.startswith("-"):
            out.append(x)
        k += 1
    return out


def bash_write_targets(command: str, workdir: str | None = None) -> list[tuple[str, str]]:
    """[(path, kind)] of the files a shell command plainly writes, creates or deletes.

    The kinds: `redirect` (>, >>, &>), `tee`, `in_place` (sed -i), `copy` (the destination of cp, mv, install and ln)
    or `copy_into` when that destination is a folder (cp -t DIR, or a destination ending in /), `create` (touch),
    `folder` (mkdir), `delete` (rm, rmdir, unlink) and `truncate`. Quoted text and here-document bodies are not read as
    shell (also a quoted text over several lines, such as a python -c script); /dev/* is not a file; a relative target
    after `cd DIR` in the same command is read as DIR/target. A write made inside a program (python -c, a here-script)
    is not seen here: inline_write_targets reads the common literal forms. The argument of touch's -d, -t and -r and of
    mkdir's -m is no target.
    """
    targets, cwd = [], workdir
    for part in command_parts(command):
        try:
            lex = shlex.shlex(part, posix=True, punctuation_chars=True)
            lex.whitespace_split = True
            cmd = list(lex)
        except ValueError:
            continue
        words, i = [], 0
        while i < len(cmd):  # redirects, anywhere in the command
            t = cmd[i]
            if t in REDIRECTS or (t.endswith((">", ">>")) and t.rstrip(">").isdigit() and t.rstrip(">")):
                if i + 1 < len(cmd):
                    tgt = cmd[i + 1]
                    if not tgt.startswith(("&", "/dev/")):
                        targets.append((tgt, cwd, "redirect"))
                i += 2
                continue
            if t in (">&", "<", "<<", "<<<", "<&", "<<-"):
                i += 2
                continue
            if re.fullmatch(r"\d", t) and i + 1 < len(cmd) and cmd[i + 1] in REDIRECTS | {">&"}:
                i += 1  # the "2" of "2>" or "2>&1"
                continue
            words.append(t)
            i += 1
        while words and (re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]) or words[0] in PREFIXES):
            words = words[1:]
        if not words:
            continue
        name, args = posixpath.basename(words[0]), words[1:]
        plain = [x for x in args if not x.startswith("-")]
        if name == "cd":
            d = plain[0] if plain else None
            if d and d != "-":
                cwd = d if d.startswith("/") or not cwd else posixpath.join(cwd, d)
        elif name == "tee":
            targets += [(x, cwd, "tee") for x in plain if not x.startswith("/dev/")]
        elif name == "sed" and any(
            x == "--in-place"
            or x.startswith(("-i", "--in-place="))
            or (x.startswith("-") and not x.startswith("--") and "i" in x[1:])
            for x in args
        ):
            rest, script_seen, k = (
                [],
                any(x in ("-e", "-f") or x.startswith(("--expression", "--file")) for x in args),
                0,
            )
            while k < len(args):
                x = args[k]
                if x in ("-e", "-f"):
                    k += 2
                    continue
                if not x.startswith("-"):
                    if script_seen:
                        rest.append(x)
                    else:
                        script_seen = True
                k += 1
            targets += [(x, cwd, "in_place") for x in rest]
        elif name in ("cp", "mv", "install", "ln"):
            if "-t" in args and args.index("-t") + 1 < len(args):
                targets.append((args[args.index("-t") + 1], cwd, "copy_into"))
            elif len(plain) >= 2:
                targets.append((plain[-1], cwd, "copy_into" if plain[-1].endswith("/") else "copy"))
        elif name in ("touch", "mkdir"):
            targets += [
                (x, cwd, "folder" if name == "mkdir" else "create") for x in _operands(args, OPTIONS_WITH_ARG[name])
            ]
        elif name in ("rm", "rmdir", "unlink"):
            targets += [(x, cwd, "delete") for x in plain]
        elif name == "truncate":
            k, files = 0, []
            while k < len(args):
                if args[k] in ("-s", "--size", "-r", "--reference"):
                    k += 2
                    continue
                if not args[k].startswith("-"):
                    files.append(args[k])
                k += 1
            targets += [(x, cwd, "truncate") for x in files]
    out = []
    for t, d, kind in targets:
        if t.startswith("/dev/") or not t:
            continue
        out.append((t if t.startswith(("/", "~")) or not d else posixpath.normpath(posixpath.join(d, t)), kind))
    return out


# an inline script in a shell command: python -c / -m with code after it, a here-document fed to python, or node -e
# and ruby -e
INLINE_SCRIPT_RE = re.compile(
    r"\b(?:python[\d.]*|node|ruby)\b[^\n|;&]*?(?:\s-[A-Za-z]*[ce]\b|\s-(?=\s|$)|<<)"
    r"|\|\s*(?:\S*/)?(?:python[\d.]*|node|ruby)(?:\s+-)?[ \t]*(?:\n|$)"
)
_LIT = r"""(?:'[^'\n]*'|"[^"\n]*")"""
_ARG = r"(?:%s|[A-Za-z_][\w.]*)" % _LIT
INLINE_WRITE_RES = (
    # open(<path>, 'w') and the like: a mode with w, a, x or + (the second argument, or mode=)
    re.compile(r'\bopen\(\s*(?P<arg>%s)\s*,\s*(?:mode\s*=\s*)?(?P<q>[\'"])[rbt]*[wax+][rwabtx+]*(?P=q)' % _ARG),
    re.compile(r"\bPath\(\s*(?P<arg>%s)\s*\)\s*\.\s*write_(?:text|bytes)\s*\(" % _ARG),
    re.compile(r"(?<![\w.])(?P<arg>[A-Za-z_]\w*)\s*\.\s*write_(?:text|bytes)\s*\("),
    re.compile(r"\b(?:writeFileSync|appendFileSync|writeFile|appendFile)\(\s*(?P<arg>%s)\s*," % _ARG),
    re.compile(r"\bFile\.write\(\s*(?P<arg>%s)\s*," % _ARG),
)


# A file opened for writing through a library (such as /tmp/test_empty_index.nc with h5netcdf.File(..., 'w') or
# /tmp/t.h5 with h5py.File(..., 'w')).
# Matched in general, not by a list of libraries: a call whose first argument is a literal path (or a name set to one)
# and whose mode, the second argument or `mode=` later in the call, is a write mode: w, a or x, with b, t, + or -
# (h5py's w-), r+ (read and write), and tarfile's w:gz or w|bz2. A read mode, or no mode at all (h5py.File(path),
# netCDF4.Dataset(path) and zipfile.ZipFile(path) read by default), is no write.
_WRITE_MODE = r"""(?P<q>['"])(?:[wax][bt+-]*|r[bt]*\+[bt+]*)(?:[:|][a-z0-9]*)?(?P=q)"""
# the rest of a call's arguments before `mode=`: no closing parenthesis outside quotes and one level of nested calls
_CALL_ARGS = r"""(?:[^()'"\n]|'[^'\n]*'|"[^"\n]*"|\((?:[^()'"\n]|'[^'\n]*'|"[^"\n]*")*\))*?"""
_CALLEE = r"(?<![\w.])(?P<fn>[A-Za-z_][\w.]*)\s*\(\s*(?P<arg>%s)\s*," % _ARG
LIBRARY_MODE_RES = (
    re.compile(r"%s\s*(?:mode\s*=\s*)?%s\s*[,)]" % (_CALLEE, _WRITE_MODE)),
    re.compile(r"%s%s\bmode\s*=\s*%s" % (_CALLEE, _CALL_ARGS, _WRITE_MODE)),
)
PATH_OPEN_RE = re.compile(
    r"\bPath\(\s*(?P<arg>%s)\s*\)\s*\.\s*open\(\s*(?:mode\s*=\s*)?%s" % (_ARG, _WRITE_MODE)
)  # pathlib
# a first argument and a mode-like second one that are no file write: os.path.join('/tmp', 'x'), s.replace('.py', 'a'),
# d.get('a.txt', 'w'), the source of a copy or a move, a member added to an archive
NOT_FILE_OPENERS = {
    "join",
    "joinpath",
    "Path",
    "PurePath",
    "PosixPath",
    "PurePosixPath",
    "replace",
    "sub",
    "subn",
    "get",
    "getenv",
    "setdefault",
    "pop",
    "split",
    "rsplit",
    "startswith",
    "endswith",
    "format",
    "print",
    "relpath",
    "copy",
    "copy2",
    "copyfile",
    "copytree",
    "move",
    "rename",
    "symlink",
    "link",
    "write",
    "writestr",
    "add",
    "glob",
    "fnmatch",
    "match",
    "search",
    "compile",
}
# library writers that take the path first and need no mode: numpy's save, savez, savez_compressed and savetxt; the
# pandas and xarray .to_* writers; and, for a path-like literal only (the method names are generic), .save(path) (a
# font, an image, a workbook) and .savefig(path)
NAMED_WRITER_RES = (
    re.compile(r"\b(?:np|numpy)\s*\.\s*(?:save|savez|savez_compressed|savetxt)\s*\(\s*(?P<arg>%s)" % _ARG),
    re.compile(r"\.\s*to_(?:csv|parquet|json|excel|netcdf|hdf|feather|pickle|zarr)\s*\(\s*(?P<arg>%s)" % _ARG),
)
PATHLIKE_WRITER_RE = re.compile(r"\.\s*(?:save|savefig)\s*\(\s*(?P<arg>%s)" % _ARG)
# sqlite3.connect('<path>') writes when the script changes the database: a statement that writes, executescript, or a
# commit; a connection that only queries is a read, as are ':memory:', '' and a read-only URI (mode=ro, immutable=1)
SQLITE_CONNECT_RE = re.compile(r"\bsqlite3\s*\.\s*connect\s*\(\s*(?P<arg>%s)" % _ARG)
SQL_WRITE_RE = re.compile(
    r'(?i)\b(?:insert\s+(?:or\s+\w+\s+)?into|update\s+[\w."`\[\]]+\s+set|delete\s+from|replace\s+into'
    r"|create\s+(?:temp\w*\s+|unique\s+|virtual\s+)?(?:table|index|view|trigger)|drop\s+(?:table|index|view|trigger)"
    r"|alter\s+table|vacuum)\b|\.\s*(?:commit|executescript)\s*\("
)
_URL_RE = re.compile(r"^[A-Za-z][\w+.-]*://")
# a write inside an inline script, whatever its target: open(..., 'w'), Path.write_text, Node's writeFileSync,
# File.write (a library write counts when inline_write_targets names its file)
INLINE_WRITE_ANY_RE = re.compile(
    r'\bopen\s*\((?:[^()]|\([^()]*\))*?[\'"][rwabt+]*[wax+][rwabt+]*[\'"]|\.write_(?:text|bytes)\s*\('
    r"|\b(?:write|append)FileSync\s*\(|\bfs\.(?:promises\.)?(?:writeFile|appendFile)\s*\(|\bFile\.write\s*\("
)


def _pathlike(s: str | None) -> bool:
    """A literal that reads as a file path: no space and no URL, a slash, a leading ~ or a file extension."""
    return (
        bool(s)
        and not re.search(r"\s", s)
        and not _URL_RE.match(s)
        and ("/" in s or s.startswith("~") or re.search(r"\.[A-Za-z0-9]{1,8}$", s) is not None)
    )


def library_write_paths(command: str, names: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """[(literal path, how)] of the files an inline script opens for writing through a library (see
    LIBRARY_MODE_RES, NAMED_WRITER_RES, PATHLIKE_WRITER_RE and SQLITE_CONNECT_RE); `names` maps a name the script
    sets to a literal path."""
    names = names or {}

    def value(arg: str) -> str | None:
        return arg[1:-1] if arg[:1] in "'\"" else names.get(arg)

    out = []
    for rx in LIBRARY_MODE_RES:
        for m in rx.finditer(command):
            p = value(m.group("arg"))
            if m.group("fn").split(".")[-1] not in NOT_FILE_OPENERS and _pathlike(p):
                out.append((p, "mode"))
    for m in PATH_OPEN_RE.finditer(command):
        p = value(m.group("arg"))
        if p:
            out.append((p, "mode"))
    for rx in NAMED_WRITER_RES:
        for m in rx.finditer(command):
            p = value(m.group("arg"))
            if p and not re.search(r"\s", p) and not _URL_RE.match(p):
                out.append((p, "writer"))
    for m in PATHLIKE_WRITER_RE.finditer(command):
        p = value(m.group("arg"))
        if _pathlike(p):
            out.append((p, "save"))
    if SQL_WRITE_RE.search(command):
        for m in SQLITE_CONNECT_RE.finditer(command):
            p = value(m.group("arg"))
            if not p or p == ":memory:" or re.search(r"[?&](?:mode=ro|immutable=1)\b", p):
                continue
            if p.startswith("file:"):  # a URI: file:<path>?<options>
                p = p[len("file:") :].split("?", 1)[0]
            if p:
                out.append((p, "sqlite"))
    return out


def inline_write_targets(command: object, workdir: str | None = None) -> list[str]:
    """The files an inline script in a shell command writes, where the script names them plainly.

    open('<path>', 'w') (also a mode a, x or +), Path('<path>').write_text(...), Node's writeFileSync('<path>', ...),
    Ruby's File.write, a file a library opens for writing (library_write_paths: h5py.File('<path>', 'w'),
    np.save('<path>', ...), df.to_csv('<path>'), font.save('<path>'), sqlite3.connect('<path>') in a script that
    changes the database), or the same with a name the script sets to a literal path first (p = 'src/a.py' ...
    open(p, 'w')). A path the script computes is not seen. A relative path is read from the folder of a `cd` before it
    in the command (a scratch file an inline script writes under /tmp is not file work, a source file it rewrites is).
    """
    if not isinstance(command, str) or not INLINE_SCRIPT_RE.search(command):
        return []
    names = {}
    for m in re.finditer(r"(?<![\w.$])([A-Za-z_]\w*)\s*=(?!=)\s*(?:Path\(\s*)?(%s)" % _LIT, command):
        names.setdefault(m.group(1), m.group(2)[1:-1])
    cwd = workdir
    for part in command_parts(command):  # the folder the shell is in when the script starts
        if INLINE_SCRIPT_RE.search(part) or part.startswith(("python", "node", "ruby")):
            break
        m = re.match(r"cd\s+(\S+)$", part.strip())
        if m and m.group(1) != "-":
            d = m.group(1).strip("'\"")
            cwd = d if d.startswith("/") or not cwd else posixpath.join(cwd, d)
    found = []
    for rx in INLINE_WRITE_RES:
        for m in rx.finditer(command):
            arg = m.group("arg")
            found.append(arg[1:-1] if arg[:1] in "'\"" else names.get(arg))
    found += [p for p, _how in library_write_paths(command, names)]
    out = []
    for path in found:
        if not path:
            continue
        path = path if path.startswith(("/", "~")) or not cwd else posixpath.normpath(posixpath.join(cwd, path))
        if path not in out:
            out.append(path)
    return out


TEST_RUNNERS = {"pytest", "py.test", "tox", "nox", "ctest", "nosetests", "trial"}
WRAPPERS = {"timeout", "xvfb-run", "nice", "stdbuf"}


def _plain_words(part: str) -> list[str]:
    """The words of one simple command, without redirections (2>/dev/null, > out.txt) and leading assignments and
    prefixes (sudo, env, VAR=1)."""
    try:
        lex = shlex.shlex(part, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        return []
    words, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in REDIRECTS or t in (">&", "<", "<<", "<<<", "<&"):
            i += 2
            continue
        if re.fullmatch(r"\d", t) and i + 1 < len(toks) and toks[i + 1] in REDIRECTS | {">&"}:
            i += 1
            continue
        words.append(t)
        i += 1
    while words and (re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]) or words[0] in PREFIXES):
        words = words[1:]
    return words


def runs_tests(command: str) -> bool:
    """A simple command of the shell command runs a test suite: pytest, python -m pytest or unittest, tox, nox,
    make test or check, npm test, go test, cargo test and the like (also after timeout, uv run, poetry run, coverage
    run -m). Used to tell the files a test run leaves in the repository (test outputs) from changes."""
    for part in command_parts(command):
        w = _plain_words(part)
        while w and (posixpath.basename(w[0]) in WRAPPERS or re.fullmatch(r"\d+[smh]?", w[0]) or w[0].startswith("-")):
            w = w[1:]  # timeout 300 pytest, nice -n 5 pytest
        while len(w) > 1 and w[0] in ("uv", "poetry", "pipenv", "hatch", "pdm") and w[1] == "run":
            w = w[2:]
        if not w:
            continue
        name = posixpath.basename(w[0])
        if name in TEST_RUNNERS:
            return True
        if re.fullmatch(r"python[\d.]*|coverage", name) and "-m" in w:
            k = w.index("-m")
            if k + 1 < len(w) and w[k + 1] in ("pytest", "unittest", "nose", "nose2", "tox"):
                return True
        if name == "make" and any(x in ("test", "tests", "check") for x in w[1:]):
            return True
        if name in ("npm", "yarn", "pnpm", "bun") and "test" in w[1:3]:
            return True
        if name in ("go", "cargo", "dotnet", "mvn", "gradle", "./gradlew") and len(w) > 1 and w[1] == "test":
            return True
    return False


def bash_delete_targets(command: str, workdir: str | None = None) -> list[str]:
    """What a shell command deletes recursively: the targets of rm -r (-R, --recursive, -rf ...) and the start folders
    of find ... -delete. A relative target after `cd DIR` in the same command is read as DIR/target
    (`cd /tmp && rm -rf rectest` deletes /tmp/rectest)."""
    targets, cwd = [], workdir
    for part in command_parts(command):
        words = _plain_words(part)
        if not words:
            continue
        name, args = posixpath.basename(words[0]), words[1:]
        plain = [x for x in args if not x.startswith("-")]
        if name == "cd":
            d = plain[0] if plain else None
            if d and d != "-":
                cwd = d if d.startswith("/") or not cwd else posixpath.join(cwd, d)
        elif name == "rm" and any(x == "--recursive" or re.fullmatch(r"-[A-Za-z]*[rR][A-Za-z]*", x) for x in args):
            targets += [(x, cwd) for x in plain]
        elif name == "find" and "-delete" in args:
            roots = []
            for x in args:
                if x.startswith(("-", "(", "!")):
                    break
                roots.append(x)
            targets += [(x, cwd) for x in roots or ["."]]
    return [t if t.startswith(("/", "~")) or not d else posixpath.normpath(posixpath.join(d, t)) for t, d in targets]


def command_allowed(part: str, allowed: list[str]) -> bool:
    """A simple command starts with one of the allowed commands: `git log` allows git log -3 and git -C <dir>
    --no-pager log (git's global options are skipped), `cat` allows cat with any arguments; redirections, variable
    settings and prefixes such as sudo or env before the command are set aside."""
    words = _plain_words(part)
    if not words:
        return True
    for entry in allowed:
        want = str(entry).split()
        if not want or posixpath.basename(words[0]) != want[0]:
            continue
        rest = words[1:]
        if want[0] == "git":
            rest = _after_git_options(rest)
        if rest[: len(want) - 1] == want[1:]:
            return True
    return False


def _after_git_options(words: list[str]) -> list[str]:
    """The words after git's global options (-C <dir>, -c <key=value>, --git-dir <dir>, --no-pager, -P ...)."""
    i = 0
    while i < len(words) and words[i].startswith("-"):
        i += 2 if words[i] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path") else 1
    return words[i:]
