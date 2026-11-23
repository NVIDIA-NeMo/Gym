# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`first-request NATIVE_DIR HARBOR_DIR`: compare the first agent request of each task on the two paths (G6).

Each folder holds the `*.capture.jsonl` files a run writes with `++observability_enabled=true
++model_call_capture_dir=DIR`. A file's first agent request is its earliest request that offers tools (OpenCode's
title call offers none). Tasks are paired by the first user message, exactly, or with OpenCode's argv quoting undone
and CRLF or CR read as LF (Gym reads instruction.md in text mode, so a native CRLF prompt arrives with LF); either is
a note, not a failure.
Each pair compares the model, the tool names, every other top-level request key but `messages`, `stream` and
`stream_options` (the sampling settings, `tool_choice` among them) and the message roles. A contamination flag
(`AGENTS.md`, `CLAUDE.md`, `Instructions from`, `<available_skills>`), counted over every message but the assistant's,
fails a pair when its count differs between the sides or it sits on a system-prompt line that differs; OpenCode's
built-in prompt carries some of them identically on both paths. Zero pairs fails.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

from scripts.harbor_conversion_checks.messages import InputError, iter_jsonl, lf_newlines, message_text, unquote_argv


NOT_SAMPLING = ("messages", "tools", "stream", "stream_options", "model")
FLAGS = ("AGENTS.md", "CLAUDE.md", "Instructions from", "<available_skills>")
DIFF_LINES = 40


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("native_dir", type=Path, help="model-call capture folder of the native run")
    parser.add_argument("harbor_dir", type=Path, help="model-call capture folder of the Harbor run")
    parser.add_argument("--out", type=Path, help="write each pair's system prompt and user message here")
    parser.add_argument("--json", type=Path, help="write the report as JSON")


def first_requests(directory: Path) -> tuple[dict[str, dict], list[str]]:
    """First agent request per task (keyed by its first user message; the first file per key wins), and the capture
    files that hold no agent request."""
    files = sorted(directory.glob("*.capture.jsonl")) if directory.is_dir() else []
    if not files:
        raise InputError(f"{directory}: no *.capture.jsonl files")
    found: dict[str, dict] = {}
    no_agent: list[str] = []
    for path in files:
        best, started = None, None
        for _, row in iter_jsonl(path):
            request = row.get("request")
            if isinstance(request, dict) and request.get("tools"):
                at = row.get("started_at") or 0
                if started is None or at < started:
                    best, started = request, at
        if best is None:
            no_agent.append(path.name)
            continue
        users = [message_text(m.get("content")) for m in _messages(best) if m.get("role") == "user"]
        found.setdefault(users[0] if users else "", {"file": path.name, "request": best})
    return found, no_agent


def _messages(request: dict) -> list[dict]:
    messages = request.get("messages")
    return [m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []


def describe(request: dict) -> dict:
    messages = _messages(request)
    system = "\n".join(message_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer"))
    seen = "\n".join(message_text(m.get("content")) for m in messages if m.get("role") != "assistant")
    tools = request.get("tools") if isinstance(request.get("tools"), list) else []
    return {
        "model": request.get("model"),
        "tools": sorted(str((t.get("function") or t).get("name")) for t in tools if isinstance(t, dict)),
        "sampling": {key: value for key, value in request.items() if key not in NOT_SAMPLING},
        "roles": [m.get("role") for m in messages],
        "flags": {flag: seen.count(flag) for flag in FLAGS if flag in seen},
        "system": system,
    }


def _flag_problems(a: dict, b: dict, diff: list[str]) -> tuple[list[str], list[str]]:
    """A flag is a problem when its count differs between the sides or it is on a system-prompt line that differs.

    OpenCode's own system prompt names AGENTS.md ("proactively suggest writing it to AGENTS.md") and lists its
    built-in skills in `<available_skills>` on both paths: identical lines are noted, not failed.
    """
    changed = "\n".join(line for line in diff[2:] if not line.startswith("@@"))  # past the ---/+++ header
    problems, notes = [], []
    for flag in FLAGS:
        native, harbor = a["flags"].get(flag, 0), b["flags"].get(flag, 0)
        if flag in changed:
            problems.append(f"flag {flag} on a system-prompt line that differs (native {native}, harbor {harbor})")
        elif native != harbor:
            problems.append(f"flag {flag}: native {native}, harbor {harbor}")
        elif native:
            notes.append(f"flag {flag}: {native} on both sides, on identical lines")
    return problems, notes


def compare(native_key: str, harbor_key: str, native: dict, harbor: dict) -> dict:
    a, b = describe(native["request"]), describe(harbor["request"])
    problems = [f"{name} differs" for name in ("model", "tools", "sampling") if a[name] != b[name]]
    problems += ["roles differ"] if a["roles"] != b["roles"] else []
    diff = list(
        difflib.unified_diff(a["system"].splitlines(), b["system"].splitlines(), "native", "harbor", n=0, lineterm="")
    )
    flag_problems, notes = _flag_problems(a, b, diff)
    problems += flag_problems
    if lf_newlines(native_key) != lf_newlines(harbor_key):
        notes.append("argv_prompt_quoting: the user messages differ by OpenCode's argv quoting")
    if native_key != harbor_key and ("\r" in native_key or "\r" in harbor_key):
        notes.append("newlines: the user messages differ by CRLF or CR vs LF (record it as a difference)")
    notes += [f"system prompt: {len(diff)} diff line(s)"] if diff else []
    return {
        "key": hashlib.sha256(harbor_key.encode()).hexdigest()[:12],
        "files": {"native": native["file"], "harbor": harbor["file"]},
        "native": {k: v for k, v in a.items() if k != "system"},
        "harbor": {k: v for k, v in b.items() if k != "system"},
        "problems": problems,
        "notes": notes,
        "system_diff": diff[:DIFF_LINES],
        "_texts": {"native": (a["system"], native_key), "harbor": (b["system"], harbor_key)},
    }


def _plain(key: str) -> str:
    """A user message with OpenCode's argv quoting undone (when it is quoted) and its newlines read as LF."""
    unquoted = unquote_argv(key)
    return lf_newlines(key if unquoted is None else unquoted)


def run(args: argparse.Namespace) -> int:
    (native, native_none), (harbor, harbor_none) = first_requests(args.native_dir), first_requests(args.harbor_dir)
    pairs, harbor_left = [], dict(harbor)
    native_left = []
    for key, entry in native.items():
        if key in harbor_left:
            pairs.append(compare(key, key, entry, harbor_left.pop(key)))
        else:
            native_left.append(key)
    plain = {}
    for key in native_left:
        plain.setdefault(_plain(key), key)
    for harbor_key in list(harbor_left):
        native_key = plain.pop(_plain(harbor_key), None)
        if native_key is not None:
            native_left.remove(native_key)
            pairs.append(compare(native_key, harbor_key, native[native_key], harbor_left.pop(harbor_key)))
    for pair in pairs:
        files = pair["files"]
        status = "DIFFERENT" if pair["problems"] else "same"
        print(f"{pair['key']} native {files['native']} / harbor {files['harbor']}: {status}")
        for line in pair["problems"] + pair["notes"]:
            print(f"  {line}")
        for line in pair["system_diff"]:
            print(f"    {line[:200]}")
        texts = pair.pop("_texts")
        if args.out:
            folder = args.out / pair["key"]
            folder.mkdir(parents=True, exist_ok=True)
            for side, (system, user) in texts.items():
                (folder / f"{side}.system.txt").write_text(system)
                (folder / f"{side}.user.txt").write_text(user)
    unpaired = {
        "native": [native[key]["file"] for key in native_left],
        "harbor": [entry["file"] for entry in harbor_left.values()],
    }
    for side, names in unpaired.items():
        for name in names:
            print(f"unpaired: only on {side}: {name}")
    # A file without a request that offers tools (a rollout that died before its first agent call) is listed, not
    # counted as unpaired.
    no_agent = {"native": native_none, "harbor": harbor_none}
    for side, names in no_agent.items():
        for name in names:
            print(f"no agent request: {side}: {name}")
    if not pairs:
        print("first-request: no pair to compare")
    differing = sum(bool(pair["problems"]) for pair in pairs)
    count = len(unpaired["native"]) + len(unpaired["harbor"])
    print(f"first-request: {len(pairs)} paired, {count} unpaired, {differing} differing")
    if args.json:
        report = {"pairs": pairs, "unpaired": unpaired, "no_agent_request": no_agent}
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 1 if differing or count or not pairs else 0
