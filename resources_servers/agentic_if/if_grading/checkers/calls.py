# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-call checkers: calls, order, args, after_event.

Each grades a run (or the part after a planted event) and returns a list of (ok, detail) units; an empty list means
not applicable. Each takes (steps, value, check, facts, start): the run, the check's `value`, its parameters, the
run's facts and the step grading starts from.
"""

import posixpath
import re
from collections.abc import Mapping

from pydantic import JsonValue

from .run import (
    TMP_RE,
    Call,
    CallSpec,
    Facts,
    Outcome,
    Step,
    arg_value,
    call_matches,
    event_index,
    match_count,
    repo_changes,
    same_file,
    tool_matches,
)
from .shell import (
    INLINE_SCRIPT_RE,
    INLINE_WRITE_ANY_RE,
    bash_delete_targets,
    bash_write_targets,
    command_allowed,
    command_parts,
    inline_write_targets,
    unquoted,
)


Parameters = Mapping[str, JsonValue]


def assistant_calls(steps: list[Step], start: int = 0, *, rejected: bool = True) -> list[tuple[int, int, Call]]:
    """[(step index, message number, call)] of the assistant steps from `start` on. rejected=False leaves out the
    calls the harness refused before running them (marked `rejected`: invalid arguments): they did nothing, so they
    are no step of an order."""
    out, msg = [], 0
    for i, s in enumerate(steps):
        if i < start or s["role"] != "assistant":
            continue
        msg += 1
        for c in s["calls"]:
            if rejected or not c.get("rejected"):
                out.append((i, msg, c))
    return out


def _spec(value: Parameters) -> dict[str, JsonValue]:
    spec = {"tool": value.get("tool", "any")}
    if value.get("arg"):
        spec["arg"] = value["arg"]
    if value.get("changes_repo") is not None:  # run.call_matches: whether the call changed the repository
        spec["changes_repo"] = value["changes_repo"]
    return spec


BASH_WRITES_MODES = ("never", "must", "max")  # the modes that count matching calls (`bash_writes`)


def check_calls(
    steps: list[Step], value: Parameters, check: Parameters, facts: Facts, start: int = 0
) -> list[Outcome]:
    """How the matching calls (value: tool, arg, changes_repo) are made, by `mode`: `must` (one at least), `never`,
    `only` (no call of another tool), `max` and `per_message_max` (at most `n`, in the run or in one message), or
    `together` (`n` matching calls in one message). With `bash_writes`, a bash call that visibly writes a file
    outside what the value's path pattern allows matches too."""
    mode = value["mode"]
    calls = assistant_calls(steps, start)
    spec = _spec(value)
    if value.get("bash_writes"):
        if mode not in BASH_WRITES_MODES:
            raise ValueError(
                "calls: bash_writes counts matching calls; it needs mode %s, not %r"
                % (", ".join(BASH_WRITES_MODES), mode)
            )
        _path_specs(value)  # an arg on the path, or none

    def matches(c: Call) -> bool:
        # bash_writes: a bash call that visibly writes a file outside what the value's path pattern allows is a
        # matching call too, as an edit or write call on that file is
        return call_matches(c, spec, facts) or bool(value.get("bash_writes") and shell_write_outside(c, value, facts))

    hits = [c for _i, _m, c in calls if matches(c)]
    n = value.get("n")
    if mode == "must":
        return [(bool(hits), "%d matching calls" % len(hits))]
    if mode == "never":
        if not hits:
            return [(True, "no matching call")]
        where = (
            shell_write_outside(hits[0], value, facts)
            if value.get("bash_writes") and not call_matches(hits[0], spec, facts)
            else None
        )
        return [
            (
                False,
                "called: %s" % _show(hits[0])
                if where is None
                else "bash writes %r, outside what the value allows: %s" % (where[:80], _show(hits[0])),
            )
        ]
    if mode == "only":
        allowed = value["tool"] if isinstance(value["tool"], list) else [value["tool"]]
        bad = [c for _i, _m, c in calls if not tool_matches(c["tool"], allowed)]
        return [(not bad, "only allowed tools" if not bad else "outside the list: %s" % _show(bad[0]))]
    if mode == "max":
        return [(len(hits) <= int(n), "%d matching calls (at most %d)" % (len(hits), int(n)))]
    if mode == "per_message_max":
        per_msg = {}
        for _i, m, c in calls:
            if call_matches(c, spec, facts):
                per_msg[m] = per_msg.get(m, 0) + 1
        if not per_msg:
            return []
        worst = max(per_msg.values())
        return [(worst <= int(n), "most matching calls in one message: %d (at most %d)" % (worst, int(n)))]
    if mode == "together":
        # applicability counts matching commands, not calls: `git status && git log` in one call is two, so a run
        # that chains its git commands is graded, and fails; passing needs n separate matching calls in one message
        need = int(n or 2)
        if sum(match_count(c, spec, facts) for _i, _m, c in calls) < need:
            return []  # fewer commands of the kind than the rule is about: not applicable
        per_msg = {}
        for _i, m, c in calls:
            if call_matches(c, spec, facts):
                per_msg[m] = per_msg.get(m, 0) + 1
        best = max(per_msg.values())
        return [(best >= need, "most matching calls in one message: %d (needs %d separate calls)" % (best, need))]
    raise ValueError("calls: unknown mode %r" % (mode,))


def _show(c: Call) -> str:
    a = c["args"]
    v = arg_value(a, "command") or arg_value(a, "path") or next(iter(a.values()), "") if a else ""
    return "%s %s" % (c["tool"], str(v)[:80])


def check_order(
    steps: list[Step], value: Parameters, check: Parameters, facts: Facts, start: int = 0
) -> list[Outcome]:
    """Every `then` call has a `first` call in an earlier message (the calls of one message are sent before any of
    their results come back, so a first step sent beside the graded call did not come before it). A call the harness
    refused is no step."""
    first, then = value["first"], value["then"]
    out, seen = [], []
    for _i, m, c in assistant_calls(steps, start, rejected=False):
        if call_matches(c, then, facts):
            earlier = any(pm < m for pm in seen)
            out.append((earlier, ("first step done before %s" if earlier else "no first step before %s") % _show(c)))
        if call_matches(c, first, facts):
            seen.append(m)
    return out


def inline_writes(call: Call) -> list[str]:
    """The files the inline scripts of a bash call write (an absolute path without its . and .. parts).

    For a script that writes (INLINE_WRITE_ANY_RE, whatever the target, or a file inline_write_targets names,
    including a file a library opens for writing), a run with workspace diffs gives what the call changed in the
    repository (a script that rewrites a source file shows there; a scratch file it writes and removes again does
    not), unless every file the script names is under /tmp; without diffs, the files the script names.
    """
    args = call.get("args") or {}
    cmd, wd = arg_value(args, "command"), arg_value(args, "workdir")
    if not isinstance(cmd, str):
        return []
    named = inline_write_targets(cmd, wd)
    if not named and not (INLINE_SCRIPT_RE.search(cmd) and INLINE_WRITE_ANY_RE.search(cmd)):
        return []
    changed = repo_changes(call)
    if changed is None:
        out = named
    elif not named or any(not TMP_RE.match(_norm(t)) for t in named):
        out = changed
    else:
        out = []
    return [_norm(t) for t in out]


def shell_write_targets(call: Call, facts: Facts | None = None) -> list[tuple[str, str]]:
    """[(path, kind)] of the files a bash call visibly writes, creates or deletes, for a rule that grades shell writes
    by where they land (the calls checker's `bash_writes`).

    The targets of shell.bash_write_targets, the files an inline script writes (inline_write_targets, library writes
    included) and, in a run with workspace diffs, what an inline write changed in the repository (inline_writes;
    these two of kind `inline`). A relative path is read from the call's working folder, else the session's
    (facts['workdir']); an absolute one without its . and .. parts. A redirect to /dev/null, and a command that names
    no target (pytest, a build, git), give none.
    """
    args = call.get("args") or {}
    cmd = arg_value(args, "command")
    if not isinstance(cmd, str):
        return []
    wd = arg_value(args, "workdir") or (facts or {}).get("workdir")
    found = bash_write_targets(cmd, wd) + [(p, "inline") for p in inline_write_targets(cmd, wd)]
    found += [(p, "inline") for p in inline_writes(call)]
    out = []
    for p, kind in found:
        p = _norm(p if p.startswith(("/", "~")) or not wd else posixpath.join(wd, p))
        if (p, kind) not in out:  # one path may come twice: mkdir -p d && cp x d/
            out.append((p, kind))
    return out


_ONE_FILE_RE = re.compile(r"\^\(\?!(.+)\$\)$")


def _one_allowed_file(regex: str | None) -> str | None:
    """The one path a pattern allows when it is written `^(?!<escaped path>$)` (a plan file, filled in when the task
    is authored), else None."""
    m = _ONE_FILE_RE.match(regex or "")
    if not m:
        return None
    path = re.sub(r"\\(.)", r"\1", m.group(1))
    return path if re.escape(path) == m.group(1) else None


def _path_specs(value: Parameters) -> list[Mapping[str, JsonValue]]:
    """The path patterns of a calls value with `bash_writes`: its `arg` (one spec or a list) on the path; none allows
    no file at all."""
    a = value.get("arg")
    specs = [x for x in (a if isinstance(a, list) else [a]) if x] if a else []
    if any(x.get("field", "path") != "path" or not x.get("regex") for x in specs):
        raise ValueError("calls: bash_writes needs an arg {field: path, regex} (or no arg), not %r" % (a,))
    return specs


def shell_write_outside(call: Call, value: Parameters, facts: Facts | None = None) -> str | None:
    """The first file a bash call visibly writes outside what a calls value allows (`bash_writes: true`), or None.

    The value allows what its `arg` path pattern does not match; without an `arg` it allows no file at all. A folder
    the command makes (mkdir) is allowed when a file the value allows can go in it: a file directly inside it, or the
    one file the pattern allows (a plan file: mkdir -p /tmp/plan-draft). A copy's destination that may be a folder is
    allowed as a file or as such a folder; one that is a folder (cp x /tmp/review/, cp -t DIR) only when a file
    directly inside it is allowed. A target the command computes ($VAR, $(...), a backtick) does not show where it
    lands: it counts only when the value allows no file at all. A call the harness refused ran nothing.
    """
    specs = _path_specs(value)
    if call.get("rejected") or not tool_matches(call["tool"], "bash"):
        return None
    for path, kind in shell_write_targets(call, facts):
        if not specs:
            return path
        if re.search(r"[$`]", path):
            continue
        folder = path.rstrip("/")
        for x in specs:
            rx = x["regex"]
            inside_ok = not re.search(rx, folder + "/x", re.S)  # a file directly inside it is allowed
            if kind == "copy_into":
                bad = not inside_ok
            else:
                bad = re.search(rx, path, re.S) is not None
                if bad and kind in ("folder", "copy") and inside_ok:
                    bad = False
                if bad and kind == "folder":
                    one = _one_allowed_file(rx)
                    bad = not (one and one.startswith(folder + "/"))  # the folder of the one file the value allows
            if bad:
                return path
    return None


def check_args(steps: list[Step], value: Parameters, check: Parameters, facts: Facts, start: int = 0) -> list[Outcome]:
    """What the calls of the value's `tool` pass in their `field` argument: it must match `must_match`, or must not
    match `must_not_match` (and `must_not_match_unquoted` on the shell text outside quotes, and `also` except on paths
    its `skip_path` matches); `allowed_commands` lists the commands a shell text may run. With `lines: added` an edit
    is graded on the lines it adds. More units grade where a call writes: the files bash deletes recursively
    (`deletes_must_match`) or plainly writes (`bash_writes`), what the workspace diff shows a call changed
    (`repo_changes`) or created (`new_files`)."""
    out = []
    for _i, _m, c in assistant_calls(steps, start):
        if value.get("deletes_must_match") and tool_matches(c["tool"], "bash"):
            for t in bash_delete_targets(str(arg_value(c["args"], "command") or ""), arg_value(c["args"], "workdir")):
                ok = re.search(value["deletes_must_match"], t, re.S) is not None
                out.append((ok, "bash deletes %r" % t[:80]))
        named = []  # the files this call's own arguments name, graded once
        if value.get("bash_writes") and tool_matches(c["tool"], "bash"):
            command = str(arg_value(c["args"], "command") or "")
            for t, _kind in bash_write_targets(command, arg_value(c["args"], "workdir")):
                t = _norm(t)
                named.append(t)
                out.append((_path_ok(value, t), "bash writes %r" % t[:80]))
        if value.get("repo_changes"):
            # every file the workspace diff shows the call changed, whatever the tool (a formatter, git checkout, a
            # program's own write); test outputs are left out, and a file the call's own path argument names is graded
            # below. A run without diffs is graded on the arguments alone.
            wd = (facts or {}).get("workdir")
            own = arg_value(c["args"], "path") if tool_matches(c["tool"], value.get("tool", "any")) else None
            for p in repo_changes(c) or []:
                full = posixpath.join(wd, p) if wd else p
                if any(same_file(full, x, wd) for x in named + ([own] if isinstance(own, str) else [])):
                    continue
                out.append((_path_ok(value, full), "%s changed %r (workspace diff)" % (c["tool"], full[:80])))
        if value.get("new_files") and c.get("changed_repo") is not None:
            # only the files the run creates count, told from the workspace diff (a new file shows as new), by any
            # tool; a file the call rewrote that existed before is no new file, so a call of the value's tool that
            # created none passes. Without diffs: the call's own path.
            wd = (facts or {}).get("workdir")
            created = [p for p in c.get("created_paths") or [] if p not in (c.get("test_output_paths") or [])]
            for p in created:
                full = posixpath.join(wd, p) if wd else p
                out.append((_path_ok(value, full), "%s created %r (workspace diff)" % (c["tool"], full[:80])))
            if not created and tool_matches(c["tool"], value.get("tool", "any")) and not c.get("rejected"):
                out.append((True, "%s created no new file (workspace diff)" % c["tool"]))
            continue
        if not tool_matches(c["tool"], value.get("tool", "any")):
            continue
        if value.get("skip_path"):
            # calls on files whose path matches are not graded (a "# Title" line in a Markdown file is no comment)
            p = arg_value(c["args"], "path")
            if isinstance(p, str) and re.search(value["skip_path"], p, re.S):
                continue
        v = arg_value(c["args"], value["field"])
        if v is not None and check.get("lines") == "added":
            v = added_text(c["args"], v)
        s = "" if v is None else (v if isinstance(v, str) else str(v))
        if value["field"] == "path":
            s = _norm(s)
        detail = "%s %s: %r" % (c["tool"], value["field"], s[:80])
        if "must_match" in value:
            ok = v is not None and re.search(value["must_match"], s, re.S) is not None
        elif "must_not_match" in value:
            ok = v is None or re.search(value["must_not_match"], s, re.S) is None
        else:
            ok = True
        if ok and v is not None and value.get("must_not_match_unquoted"):  # a second pattern, for the shell text only
            ok = re.search(value["must_not_match_unquoted"], unquoted(s), re.S) is None
        if ok and v is not None and value.get("also"):
            # a second pattern that must not match either, except on files whose path matches its skip_path (mock or
            # fake data is fine in test files)
            al, p = value["also"], arg_value(c["args"], "path")
            if not (al.get("skip_path") and isinstance(p, str) and re.search(al["skip_path"], p, re.S)):
                ok = re.search(al["must_not_match"], s, re.S) is None
        if ok and v is not None and value.get("allowed_commands"):
            # every simple command of the shell text starts with one of the allowed commands
            bad = [p for p in command_parts(s) if not command_allowed(p, value["allowed_commands"])]
            if bad:
                ok, detail = False, "%s runs %r, which is not one of the allowed commands" % (c["tool"], bad[0][:80])
        out.append((ok, detail))
    return out


def _path_ok(value: Parameters, path: str) -> bool:
    """A path against a value's must_match or must_not_match; a folder-shaped must_match pattern (ending in /) also
    accepts the folder itself: mkdir -p <allowed folder>."""
    if "must_match" in value:
        return (
            re.search(value["must_match"], path, re.S) is not None
            or re.search(value["must_match"], path.rstrip("/") + "/", re.S) is not None
        )
    return re.search(value["must_not_match"], path, re.S) is None


def _norm(path: str) -> str:
    """An absolute path without its . and .. parts (src/pkg/../../tests/x.py is tests/x.py)."""
    return posixpath.normpath(path) if path.startswith("/") else path


# a tool call written out as text is not text for the user: <tool_call> or
# <function_call> blocks, <function=...> blocks, and a closing {"name": ..., "arguments": ...} object
MARKUP_RE = re.compile(
    r"(?s)<(tool_call|function_call|invoke)\b.*?(</\1>|\Z)|<function=[^>]*>.*?(</function>|\Z)"
    r'|\{\s*"(name|tool)"\s*:\s*"[^"]+"\s*,\s*"(arguments|parameters|input)"\s*:.*?\}\s*$'
)


def check_after_event(
    steps: list[Step], value: Parameters, check: Parameters, facts: Facts, start: int | None = 0
) -> list[Outcome]:
    """What the assistant does after the planted event (the step `start`; for a whole-run place, the check's `event`)
    until the user speaks again, by `then`: `no_calls_with_text` (it stops calling tools and answers in text),
    {calls_avoid: spec or specs} (no matching call) or {calls_include: spec or specs} (a matching call that ran)."""
    ev = event_index(steps, check.get("event")) if start is None else start
    if ev is None:
        return []
    then = value["then"]
    if isinstance(then, dict) and len(then) != 1:
        raise ValueError("after_event: `then` takes one condition, not %r" % (sorted(then),))
    # the event binds until the user speaks again: "OK, go on" after a stop lifts it
    end = next((i for i in range(ev + 1, len(steps)) if steps[i]["role"] == "user"), len(steps))
    after = [x for x in assistant_calls(steps, ev + 1) if x[0] < end]

    def any_spec(c: Call, specs: CallSpec | list[CallSpec]) -> bool:
        return any(call_matches(c, sp, facts) for sp in (specs if isinstance(specs, list) else [specs]))

    if then == "no_calls_with_text":
        texts = [
            s
            for s in steps[ev + 1 : end]
            if s["role"] == "assistant" and MARKUP_RE.sub("", s.get("text") or "").strip()
        ]
        ok = not after and bool(texts)
        return [
            (
                ok,
                "stopped and answered"
                if ok
                else ("called after the event: %s" % _show(after[0][2]) if after else "no text after the event"),
            )
        ]
    if isinstance(then, dict) and "calls_avoid" in then:
        bad = [c for _i, _m, c in after if any_spec(c, then["calls_avoid"])]
        return [(not bad, "no call of the avoided kind" if not bad else "called: %s" % _show(bad[0]))]
    if isinstance(then, dict) and "calls_include" in then:
        # a call the harness refused for its arguments (marked `rejected`) did nothing: it is not the expected call
        # (a malformed call sent again does not count as the well-formed retry)
        good = [c for _i, _m, c in after if any_spec(c, then["calls_include"]) and not c.get("rejected")]
        return [(bool(good), "the expected call came" if good else "the expected call did not come")]
    raise ValueError("after_event: unknown value %r" % (value,))


OLD_TEXT_KEYS = ("old_string", "old_str", "oldText", "old_text", "old")  # the argument names of an edit's old text


def added_text(args: Mapping[str, JsonValue], value: JsonValue) -> str:
    """Only what a call adds: the lines of an edit's new text that its old text does not have, the + lines of a
    patch, or the whole content of a write."""
    old = None
    for k in OLD_TEXT_KEYS:
        if k in args:
            old = args[k]
            break
    text = value if isinstance(value, str) else str(value)
    if re.search(r"(?m)^(\+\+\+ |--- |@@)", text):  # a unified diff: its + lines
        return "\n".join(ln[1:] for ln in text.split("\n") if ln.startswith("+") and not ln.startswith("+++"))
    if isinstance(old, str):
        have = set(ln.strip() for ln in old.split("\n"))
        return "\n".join(ln for ln in text.split("\n") if ln.strip() not in have)
    return text
