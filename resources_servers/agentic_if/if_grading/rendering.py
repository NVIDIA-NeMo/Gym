# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The views of a session the judge is shown, with the lengths of `judge.yaml` `contexts`."""

import difflib
import json
import re
from collections.abc import Mapping

from pydantic import JsonValue

from .checkers.calls import OLD_TEXT_KEYS
from .checkers.run import Call, Step, arg_value, facts_from_steps, tool_matches


def _short(s: str | None, n: int) -> str:
    """s, or its first 70% and last 30% of n characters with a "[... n characters cut ...]" marker between."""
    s = s or ""
    return (
        s if len(s) <= n else s[: int(n * 0.7)] + " [... %d characters cut ...] " % (len(s) - n) + s[-int(n * 0.3) :]
    )


def changed_files(steps: list[Step], index: int) -> list[str]:
    """The paths of the write and edit calls from the start of the run up to and including step `index`."""
    return facts_from_steps(steps[: index + 1])["changed_files"]


def render_options(check: Mapping[str, JsonValue] | None) -> dict[str, bool]:
    """The render settings a rule asks for in its parameters: `show_calls: false` leaves a message's own calls out of
    the message contexts; `edits: diff` shows each write and edit call of a run context as its change."""
    check = check or {}
    return {"calls": check.get("show_calls", True), "edits": check.get("edits") == "diff"}


PATCH_RE = re.compile(r"(?m)^(?:@@|\+\+\+ |\*\*\* (?:Begin Patch|Update File|Add File|Delete File))")
DIFF_MARK = {" ": "kept:    ", "-": "removed: ", "+": "added:   "}


def _change(c: Call, cap: int) -> str | None:
    """A write or edit call as its path and its change (`edits: diff`): an edit as its changed lines, each marked
    kept, removed or added (two kept lines around each change), a write as its whole content (every line added), a
    patch as written; None for any other call. The marks are words, not diff signs, which a judge can misread: a kept
    line's leading space reads as an addition."""
    a = c.get("args") or {}
    new = arg_value(a, "content")
    if not tool_matches(c["tool"], ["write", "edit"]) or not isinstance(new, str):
        return None
    old = next((a[k] for k in OLD_TEXT_KEYS if isinstance(a.get(k), str)), None)
    if PATCH_RE.search(new):
        how, body = "a patch", new.split("\n")
    elif old is None:
        how, body = "the whole new content", [DIFF_MARK["+"] + ln for ln in new.split("\n")]
    else:
        how = "its change, line by line"
        diff = list(difflib.unified_diff(old.split("\n"), new.split("\n"), lineterm="", n=2))[
            3:
        ]  # no file header, no first @@
        body = ["..." if ln.startswith("@@") else DIFF_MARK.get(ln[:1], "") + ln[1:] for ln in diff] or [
            "(no change: the new text is the old text)"
        ]
    return "  CALL %s %s (%s):\n%s" % (
        c.get("model_tool") or c["tool"],
        arg_value(a, "path") or "",
        how,
        _short("\n".join("    " + ln for ln in body), cap),
    )


def _call(c: Call, caps: Mapping[str, int], *, edits: bool = False) -> str:
    """One tool call of a run context: the tool and its arguments, cut to call_chars; with `edits`, a write or edit
    call is shown as its change."""
    return (_change(c, caps["edit_chars"]) if edits else None) or "  CALL %s %s" % (
        c.get("model_tool") or c["tool"],
        _short(
            json.dumps(c["model_args"] if c.get("model_args") is not None else c["args"], ensure_ascii=False),
            caps["call_chars"],
        ),
    )


def _result_call(pending: list[tuple[str | None, Call]], step: Step) -> Call | dict:
    """Match parallel/child results by identity, never by their arrival order."""
    identity = {key: step[key] for key in ("invocation_id", "call_id") if step.get(key) is not None}
    matches = [
        (session, call)
        for session, call in pending
        if session == step.get("session_id")
        and (
            all(call.get(key) == value for key, value in identity.items())
            if identity
            else call["tool"] == step.get("tool")
        )
    ]
    if len(matches) != 1:
        return {}
    pending.remove(matches[0])
    return matches[0][1]


def _run_lines(
    steps: list[Step],
    start: int,
    last: int | None,
    settings: Mapping[str, JsonValue],
    caps: Mapping[str, int],
    *,
    edits: bool,
) -> tuple[list[str], list[str]]:
    """(lines before the last assistant message, lines from it on) of the steps from `start` on, each cut to
    `caps`."""
    lines, tail, pending = [], [], []
    multi_session = len({s.get("session_id") for s in steps}) > 1
    for step in steps[:start]:
        if step["role"] == "assistant":
            pending.extend((step.get("session_id"), call) for call in step.get("calls") or [])
        elif step["role"] == "tool":
            _result_call(pending, step)
    for i in range(start, len(steps)):
        s = steps[i]
        out = tail if last is not None and i >= last else lines
        if multi_session:
            out.append("[session=%s]" % s.get("session_id"))
        if s["role"] == "assistant":
            # every assistant message gets its own line, so the judge sees where one message ends and the next begins
            cap = settings.get("final_chars", settings["assistant_chars"]) if i == last else caps["assistant_chars"]
            out.append("ASSISTANT: " + (_short(s["text"], cap) if (s.get("text") or "").strip() else "(no text)"))
            pending.extend((s.get("session_id"), call) for call in s.get("calls") or [])
            for c in s.get("calls") or []:
                out.append(_call(c, caps, edits=edits))
            continue
        # a planted step (the instruction or the event a rule is about) is cut to planted_chars only, whatever the
        # run's length, so a notice appended to a long tool output stays in view
        full = s["text"] if s["role"] in ("user", "system") else s.get("output") or ""
        shown = _short(
            full,
            settings["planted_chars"]
            if s.get("planted")
            else caps["user_chars"]
            if s["role"] in ("user", "system")
            else caps["result_chars"],
        )
        if s["role"] in ("user", "system"):
            out.append(s["role"].upper() + (" (planted): " if s.get("planted") else ": ") + shown)
            continue
        # the call this result answers; one the harness refused for its arguments (marked `rejected` in the evidence)
        # never ran, which a judge counting attempts must know
        c = _result_call(pending, s)
        state = ", error, refused before it ran" if c.get("rejected") else ", error" if s.get("error") else ""
        out.append(
            "  RESULT (%s%s%s): %s"
            % (c.get("model_tool") or s.get("tool"), state, ", planted" if s.get("planted") else "", shown)
        )
    return lines, tail


def render(
    steps: list[Step],
    kind: str,
    index: int | None,
    config: Mapping[str, JsonValue],
    *,
    calls: bool = True,
    edits: bool = False,
) -> str:
    """The context text of a judge call: the graded message, the message and the request, the whole run, or the run
    from the event on; lengths from the config's `contexts`.

    calls=False leaves a message's own tool calls out of the message contexts (a rule's `show_calls: false`: for rules
    about how the text refers to files or code, the arguments of a call are not references and the judge must not
    grade them). In run contexts, edits=True shows write and edit calls as their change (`_change`). `render_options`
    reads both from a rule's parameters.
    """
    contexts = config["contexts"]
    if kind == "message":
        settings = contexts["message"]
        s = steps[index]
        shown = (
            "".join(
                "\n[call] %s %s"
                % (
                    c.get("model_tool") or c["tool"],
                    _short(
                        json.dumps(
                            c["model_args"] if c.get("model_args") is not None else c["args"], ensure_ascii=False
                        ),
                        settings["call_chars"],
                    ),
                )
                for c in s.get("calls") or []
            )
            if calls
            else ""
        )
        return _short(s.get("text") or "", settings["max_chars"]) + shown
    if kind == "message_with_changed_files":
        # the files changed BEFORE this message: a message's text comes before its own calls, so a note that announces
        # an edit names a file not changed yet (grade skips a message with none)
        settings = contexts["message_with_changed_files"]
        files = changed_files(steps, index - 1)
        listed = "\n".join("- " + _short(f, settings["path_chars"]) for f in files[: settings["max_files"]])
        more = (
            "\n- [... %d more files]" % (len(files) - settings["max_files"])
            if len(files) > settings["max_files"]
            else ""
        )
        return "%s\n%s%s\n\n%s\n%s" % (
            settings["files_label"],
            listed,
            more,
            settings["message_label"],
            render(steps, "message", index, config, calls=calls),
        )
    if kind == "message_and_request":
        users = [
            s for s in steps[:index] if s["role"] == "user" and s.get("session_id") == steps[index].get("session_id")
        ]
        req = users[-1]["text"] if users else "(no user message)"
        return "USER:\n%s\n\nASSISTANT:\n%s" % (
            _short(req, contexts["message_and_request"]["request_chars"]),
            render(steps, "message", index, config, calls=calls),
        )
    if kind == "messages_after_request":
        settings = contexts[kind]
        session = steps[index].get("session_id")
        users = [i for i in range(index) if steps[i]["role"] == "user" and steps[i].get("session_id") == session]
        request = _short(steps[users[-1]]["text"], settings["request_chars"]) if users else "(no user message)"
        first = users[-1] + 1 if users else 0
        messages = [
            step
            for step in steps[first:index]
            if step["role"] == "assistant" and step.get("session_id") == session and (step.get("text") or "").strip()
        ]
        count = len(messages) + 1
        parts = [
            f"ASSISTANT (message {position + 1} of {count}):\n" + _short(step["text"], settings["assistant_chars"])
            for position, step in enumerate(messages)
        ]
        earlier = _short("\n\n".join(parts), settings["max_chars"]) + "\n\n" if parts else ""
        return f"USER:\n{request}\n\n{earlier}ASSISTANT (message {count} of {count}, the last):\n" + render(
            steps, "message", index, config, calls=False
        )
    if kind == "message_and_conversation":
        settings = contexts[kind]
        session = steps[index].get("session_id")
        users = [
            "USER:\n" + _short(step["text"], settings["user_chars"])
            for step in steps[:index]
            if step["role"] == "user" and step.get("session_id") == session
        ]
        conversation = _short("\n\n".join(users), settings["max_chars"]) if users else "(no user message)"
        return (
            conversation
            + "\n\n"
            + settings["reply_label"]
            + "\nASSISTANT:\n"
            + render(steps, "message", index, config, calls=calls)
        )
    run = contexts["run"]
    lines = []
    start = 0
    if kind == "after_event" and index is not None:
        event = contexts["after_event"]
        event_session = steps[index].get("session_id")
        first_user = next((s for s in steps if s["role"] == "user" and s.get("session_id") == event_session), None)
        if first_user is not None and steps.index(first_user) < index:
            lines.append("USER (the task): " + _short(first_user["text"], event["task_chars"]))
        prev = [
            i for i in range(index) if steps[i]["role"] == "assistant" and steps[i].get("session_id") == event_session
        ]
        if prev:  # the message (and call) that led to the event, for comparison
            p = steps[prev[-1]]
            lines.append("--- the assistant message just before the event ---")
            lines.append(
                "ASSISTANT: "
                + (_short(p["text"], event["before_chars"]) if (p.get("text") or "").strip() else "(no text)")
            )
            for c in p.get("calls") or []:
                lines.append(_call(c, run, edits=edits))
        lines.append("--- the event, and everything after it ---")
        start = index
    n_steps = len(steps) - start
    caps = {
        "assistant_chars": run["assistant_chars"],
        "call_chars": run["call_chars"],
        "edit_chars": run["edit_chars"],
        "user_chars": run["user_chars"],
        "result_chars": run["result_chars"] if n_steps < run["long_run_steps"] else run["long_run_result_chars"],
    }  # long runs: results cut harder
    # the last assistant message (usually the final report) is shown whole, up to final_chars, and the part before it
    # is what gets cut to fit max_chars: a rule about the final report must see all of it
    last = max((i for i in range(start, len(steps)) if steps[i]["role"] == "assistant"), default=None)
    body_lines, tail = _run_lines(steps, start, last, run, caps, edits=edits)
    tail_text = "\n".join(tail)
    room = run["max_chars"] if not tail else max(run["max_chars"] - len(tail_text) - 1, run["max_chars"] // 4)
    if len("\n".join(lines + body_lines)) > room and run.get("over_max"):
        # too long: every step is shown again with the tighter caps of over_max before the middle is cut, so that no
        # message drops out of the view
        caps = {k: min(v, run["over_max"].get(k, v)) for k, v in caps.items()}
        body_lines, tail = _run_lines(steps, start, last, run, caps, edits=edits)
    body = "\n".join(lines + body_lines)
    return _short(body, room) if not tail else (tail_text if not body else _short(body, room) + "\n" + tail_text)
