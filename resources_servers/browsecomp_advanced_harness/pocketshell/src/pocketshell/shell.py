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
"""The executor: expansion, globbing, pipelines, and the public :func:`run` API.

Replaces ``asyncio.create_subprocess_exec("bash", "-c", ...)``. Nothing is
spawned, so there is no sandbox to maintain: a command that is not implemented
simply does not exist, and every path argument is confined by
:class:`~pocketshell.fsview.Workspace`.
"""

from __future__ import annotations

import ast
import operator
import re
import time
from dataclasses import dataclass

from .commands import CommandError, Ctx, run_command


# Exit status GNU uses when an input file cannot be read. Measured, not guessed:
#   grep/sed/ls/sort -> 2 ; cat/head/tail/wc/cut/nl -> 1
# These feed `&&` / `||`, and the harness shows them to the model as
# [exit_code=N], so a wrong value is both visible and behaviourally live.
_FILE_ERROR_CODE = {
    "grep": 2,
    "egrep": 2,
    "fgrep": 2,
    "sed": 2,
    "ls": 2,
    "sort": 2,
    "find": 1,
    "diff": 2,
    "strings": 1,
    "file": 1,
}
from .fsview import PathEscape, Workspace
from .syntax import AndOrList, ForLoop, ParseError, Pipeline, Script, SimpleCommand, Word, parse


__all__ = ["run", "Result", "PocketShellError"]

DEFAULT_TIMEOUT = 30.0
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_LOOP_ITERATIONS = 10_000


class PocketShellError(Exception):
    pass


@dataclass
class Result:
    stdout: str
    stderr: str
    exit_code: int


# ---------------------------------------------------------------------------
# arithmetic:  $(( ... ))  -- integer only, exactly as bash gives you
# ---------------------------------------------------------------------------
_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: lambda a, b: a // b if b else 0,
    ast.FloorDiv: lambda a, b: a // b if b else 0,
    ast.Mod: lambda a, b: a % b if b else 0,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def eval_arith(expr: str, env: dict[str, str]) -> int:
    """Evaluate a bash arithmetic expansion body. Integer semantics, no eval()."""
    expr = expr.strip()
    if not expr:
        return 0

    # bare names are variable references in arithmetic context
    def _sub(m):
        name = m.group(0)
        if name in ("True", "False", "None"):
            return "0"
        val = env.get(name, "0")
        try:
            return str(int(str(val).strip() or 0))
        except ValueError:
            return "0"

    expr = re.sub(r"\$?\b[A-Za-z_]\w*\b", _sub, expr)
    expr = expr.replace("&&", " and ").replace("||", " or ")

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return int(node.value)
            if isinstance(node.value, (int, float)):
                return int(node.value)
            raise PocketShellError("arithmetic: non-numeric constant")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return int(_BIN[type(node.op)](_walk(node.left), _walk(node.right)))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -_walk(node.operand)
            if isinstance(node.op, ast.UAdd):
                return _walk(node.operand)
            if isinstance(node.op, ast.Not):
                return int(not _walk(node.operand))
            if isinstance(node.op, ast.Invert):
                return ~_walk(node.operand)
        if isinstance(node, ast.BoolOp):
            vals = [_walk(v) for v in node.values]
            return int(all(vals) if isinstance(node.op, ast.And) else any(vals))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
            return int(_CMP[type(node.ops[0])](_walk(node.left), _walk(node.comparators[0])))
        if isinstance(node, ast.IfExp):
            return _walk(node.body) if _walk(node.test) else _walk(node.orelse)
        raise PocketShellError("arithmetic: unsupported expression")

    try:
        return _walk(ast.parse(expr, mode="eval"))
    except PocketShellError:
        raise
    except Exception as exc:
        raise PocketShellError(f"arithmetic: cannot evaluate {expr!r}") from exc


# ---------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------
_VAR = re.compile(r"\$\{([A-Za-z_]\w*)(?::?-([^}]*))?\}|\$([A-Za-z_]\w*)|\$\{(\d+)\}|\$(\d)")


def expand_text(text: str, env: dict[str, str]) -> str:
    """Expand $((...)), ${VAR}, $VAR in one pass (no command substitution)."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("$((", i):
            depth, j = 0, i + 2
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            body = text[i + 3 : j]
            out.append(str(eval_arith(body, env)))
            i = j + 2
            continue
        if text.startswith("$[", i):
            j = text.find("]", i)
            if j != -1:
                out.append(str(eval_arith(text[i + 2 : j], env)))
                i = j + 1
                continue
        m = _VAR.match(text, i)
        if m:
            name = m.group(1) or m.group(3) or m.group(4) or m.group(5)
            default = m.group(2)
            val = env.get(name)
            if val is None or (val == "" and default is not None):
                val = default if default is not None else ""
            out.append(val)
            i = m.end()
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


_BRACE_SEQ = re.compile(r"^(-?\d+)\.\.(-?\d+)(?:\.\.(-?\d+))?$")


def expand_braces(text: str) -> list[str]:
    """bash brace expansion: ``a{1,2}b`` and ``pages/00{36..40}_*``.

    Runs before globbing, as in bash. Agents use the numeric-range form to sweep
    page indices, and without it the brace survives into the glob and matches
    nothing.
    """
    depth = start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth <= 0:
                start = i
                depth = 1
            else:
                depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                pre, body, post = text[:start], text[start + 1 : i], text[i + 1 :]
                m = _BRACE_SEQ.match(body)
                if m:
                    lo, hi = int(m.group(1)), int(m.group(2))
                    step = abs(int(m.group(3))) if m.group(3) else 1
                    step = step or 1
                    width = 0
                    for g in (m.group(1), m.group(2)):
                        if len(g.lstrip("-")) > 1 and g.lstrip("-").startswith("0"):
                            width = max(width, len(g))
                    rng = range(lo, hi + 1, step) if lo <= hi else range(lo, hi - 1, -step)
                    items = [str(v).zfill(width) if width else str(v) for v in rng]
                elif "," in body:
                    items, d, cur = [], 0, []
                    for ch2 in body:
                        if ch2 == "{":
                            d += 1
                        elif ch2 == "}":
                            d -= 1
                        if ch2 == "," and d == 0:
                            items.append("".join(cur))
                            cur = []
                            continue
                        cur.append(ch2)
                    items.append("".join(cur))
                else:
                    return [text]
                out = []
                for it in items:
                    out.extend(expand_braces(pre + it + post))
                return out
    return [text]


def _glob(word_raw: str, patterned: bool, ws: Workspace) -> list[str]:
    """Expand a glob against the workspace. Unmatched globs stay literal (bash default)."""
    if not patterned or not any(c in word_raw for c in "*?["):
        return [word_raw]
    from pathlib import Path

    p = Path(word_raw)
    anchor = ws.cwd if not p.is_absolute() else Path("/")
    pattern = word_raw if not p.is_absolute() else word_raw.lstrip("/")
    try:
        matches = sorted(anchor.glob(pattern))
    except (ValueError, OSError, IndexError):
        return [word_raw]
    # A leading `*` never matches a dotfile in the shell; pathlib does match them.
    seg_starts_dot = [s.startswith(".") for s in pattern.split("/")]
    kept = []
    for m in matches:
        rel_parts = m.parts[len(anchor.parts) :] if not p.is_absolute() else m.parts
        hidden = any(
            part.startswith(".") and not (i < len(seg_starts_dot) and seg_starts_dot[i])
            for i, part in enumerate(rel_parts)
        )
        if not hidden:
            kept.append(m)
    matches = kept
    out = []
    for m in matches:
        try:
            ws.resolve(str(m))
        except PathEscape:
            continue
        out.append(ws.display(m) if not p.is_absolute() else str(m))
    return out or [word_raw]


def expand_word(w: Word, env: dict[str, str], ws: Workspace) -> list[str]:
    """Expand one word into zero or more argv entries."""
    pieces: list[tuple[str, bool]] = []  # (text, may_glob_and_split)
    for part in w.parts:
        if part.quote == "'":
            pieces.append((part.text, False))
        elif part.quote == '"':
            pieces.append((expand_text(part.text, env), False))
        else:
            pieces.append((expand_text(part.text, env), True))

    # field-split only the unquoted pieces, keeping quoted ones glued
    fields: list[list[tuple[str, bool]]] = [[]]
    for text, bare in pieces:
        if not bare:
            fields[-1].append((text, False))
            continue
        chunks = re.split(r"[ \t\n]+", text)
        for k, ch in enumerate(chunks):
            if k:
                fields.append([])
            if ch:
                fields[-1].append((ch, True))
    out: list[str] = []
    for field in fields:
        if not field:
            continue
        raw = "".join(t for t, _ in field)
        can_glob = any(g for _, g in field)
        for braced in expand_braces(raw) if can_glob else [raw]:
            out.extend(_glob(braced, can_glob, ws))
    return out


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
class _Exec:
    def __init__(self, ws: Workspace, timeout: float):
        self.deadline = time.monotonic() + timeout
        self.ctx = Ctx(ws, deadline=self.deadline)
        self.env: dict[str, str] = {}
        self.err: list[str] = []

    def _tick(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError("command timed out")

    def script(self, sc: Script, stdin: str = "") -> tuple[str, int]:
        out, code = [], 0
        for item in sc.items:
            text, code = self.andor(item, stdin)
            if text:
                out.append(text)
        return "".join(out), code

    def andor(self, node: AndOrList, stdin: str) -> tuple[str, int]:
        out, code = [], 0
        for i, pipe in enumerate(node.pipelines):
            if i:
                op = node.operators[i - 1]
                if op == "&&" and code != 0:
                    break
                if op == "||" and code == 0:
                    break
            text, code = self.pipeline(pipe, stdin)
            if text:
                out.append(text)
        return "".join(out), code

    def pipeline(self, pipe: Pipeline, stdin: str) -> tuple[str, int]:
        data, code = stdin, 0
        for cmd in pipe.commands:
            self._tick()
            if isinstance(cmd, ForLoop):
                data, code = self.for_loop(cmd, data)
            else:
                data, code = self.simple(cmd, data)
            if len(data) > MAX_OUTPUT_BYTES:
                data = data[:MAX_OUTPUT_BYTES]
                self.err.append("[output truncated at 16 MB]")
        return data, code

    def for_loop(self, loop: ForLoop, stdin: str) -> tuple[str, int]:
        items: list[str] = []
        for w in loop.items:
            items.extend(expand_word(w, self.env, self.ctx.ws))
        out, code = [], 0
        if len(items) > MAX_LOOP_ITERATIONS:
            raise PocketShellError(f"for: too many iterations ({len(items)})")
        for val in items:
            self._tick()
            self.env[loop.var] = val
            text, code = self.script(loop.body, stdin)
            if text:
                out.append(text)
        return "".join(out), code

    def simple(self, cmd: SimpleCommand, stdin: str) -> tuple[str, int]:
        for name, wval in cmd.assignments:
            self.env[name] = "".join(expand_word(wval, self.env, self.ctx.ws)) if wval.parts else ""
        if not cmd.words:
            return "", 0

        argv: list[str] = []
        for w in cmd.words:
            argv.extend(expand_word(w, self.env, self.ctx.ws))
        if not argv:
            return "", 0

        name = argv[0].rsplit("/", 1)[-1]
        drop_out = any(r.fd == 1 and r.target == "/dev/null" for r in cmd.redirects)
        drop_err = any(r.fd == 2 and r.target == "/dev/null" for r in cmd.redirects)
        merge_err = any(r.target == "&" for r in cmd.redirects)

        for r in cmd.redirects:
            if r.fd == 0:
                try:
                    stdin = self.ctx.ws.read_text(r.target)
                except (OSError, PathEscape) as exc:
                    self.err.append(f"{name}: {exc}")
                    return "", 1

        try:
            out, err, code = run_command(name, argv[1:], stdin, self.ctx)
        except CommandError as exc:
            if not drop_err:
                self.err.append(f"{name}: {exc}" if not str(exc).startswith(name) else str(exc))
            return "", exc.code
        except PathEscape as exc:
            if not drop_err:
                self.err.append(f"{name}: {exc}")
            return "", _FILE_ERROR_CODE.get(name, 1)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError) as exc:
            if not drop_err:
                msg = getattr(exc, "filename", None) or exc
                self.err.append(
                    f"{name}: {msg}: No such file or directory"
                    if isinstance(exc, FileNotFoundError)
                    else f"{name}: {exc}"
                )
            return "", _FILE_ERROR_CODE.get(name, 1)
        except (UnicodeError, ValueError, re.error) as exc:
            if not drop_err:
                self.err.append(f"{name}: {exc}")
            return "", 2

        if err and not drop_err:
            self.err.append(err.rstrip("\n"))
        if merge_err and err and not drop_err:
            out = out + err
        return ("" if drop_out else out), code


def run(keystrokes: str, workspace: str, timeout: float = DEFAULT_TIMEOUT) -> Result:
    """Execute ``keystrokes`` read-only inside ``workspace``.

    Never spawns a process. Never reads outside ``workspace``. Returns the same
    (stdout, stderr, exit_code) triple the subprocess implementation did, so the
    harness's response formatting is unchanged.
    """
    try:
        ws = Workspace(workspace)
    except (OSError, PathEscape) as exc:
        return Result("", f"[workspace error: {exc}]", -2)

    try:
        script = parse(keystrokes or "")
    except ParseError as exc:
        return Result("", f"[blocked: {exc}]", -3)

    ex = _Exec(ws, timeout)
    try:
        out, code = ex.script(script)
    except TimeoutError:
        return Result("", f"[command timed out after {timeout:.0f}s]", -1)
    except PocketShellError as exc:
        return Result("", f"[error: {exc}]", 2)
    except RecursionError:
        return Result("", "[error: expression too deeply nested]", 2)

    return Result(out, "\n".join(x for x in ex.err if x), code)
