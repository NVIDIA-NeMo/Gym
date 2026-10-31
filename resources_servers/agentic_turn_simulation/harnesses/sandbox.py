# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sandbox harness: an agent with 11 tools in a Linux sandbox whose home is /home/user.

File and media tools, and bash commands that only print a known file, are answered in code where the result is
knowable (a workspace model replays the history); other shell commands, web and unknown files go to the simulator LLM,
which answers in the tools' JSON. Nothing is rewritten: the simulator and the judge see the paths the policy sees, so
rows must use the sandbox's home /home/user.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import re
from typing import Any, Callable, Collection, Mapping

from resources_servers.agentic_turn_simulation.harnesses.base import USER, Harness, ObstacleTexts, Requester
from resources_servers.agentic_turn_simulation.items import MALFORMED_ARGS_KEY, Items, json_object, parse_args
from resources_servers.agentic_turn_simulation.workspace import (
    BROAD_MUTATION,
    MUTATING,
    SHOWN_FILE_CHARS,
    SHOWN_FILES,
    WINDOWS_MUTATING,
    ReadBack,
    Workspace,
    mentions,
    plain_shell_args,
    render_files,
)


HOME = "/home/user"
KNOWN_TOOLS = frozenset(
    "bash read_file write_file edit_file present_file web_search fetch_page image_search generate_image "
    "generate_speech ask_user".split()
)
# read_file returns at most this many characters of a file (and then says the result is truncated).
READ_CAP = 131_072
# bash cuts each output stream at this many characters (and sets its *_truncated flag).
BASH_STREAM_CAP = 20_000
# The sandbox's answer to malformed arguments, unknown tools and schema violations. It may also appear for some
# valid calls; that noise is not simulated, and injected obstacles use the web tools' own HTTP errors instead.
HARNESS_ERROR = {"status": "error", "message": "Something went wrong. Please try again."}


def _search_result(result: dict[str, Any], status: str) -> Callable[[str, str | None], str]:
    """An injected web_search result in the format of the search results the conversation shows. Its results carry
    a `status`, but the simulator's format has none (its prompt's), and it adds one only when the conversation shows
    results with it; so the result takes a `status` only when the latest search result shown has one."""

    def render(query: str, shown: str | None) -> str:
        return json.dumps({**result, "status": status} if "status" in (json_object(shown) or {}) else result)

    return render


def web_failed(output: str) -> bool:
    """Whether a fetch_page or web_search result brought nothing: an error, or a search without results."""
    result = json_object(output)
    return result is None or "error" in result or result.get("status") == "error" or result.get("results") == []


# Its web tools' failure texts, for injected obstacles.
OBSTACLE_TEXTS = {
    "fetch_page": ObstacleTexts(
        target="url",
        failed=web_failed,
        transient=json.dumps({"error": "Failed to fetch page (HTTP 500)", "status": "error"}),
        blocked_route=json.dumps({"error": "Failed to fetch page (HTTP 403)", "status": "error"}),
    ),
    "web_search": ObstacleTexts(
        target="query",
        search=True,
        failed=web_failed,
        transient=_search_result({"error": "Search failed (HTTP 500)"}, "error"),
        empty_search=_search_result({"results": []}, "success"),
    ),
}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
IMAGE_READ = {"data": "(image returned as visible content; not viewable in this simulation)"}
# Images and speech clips per turn. The image limit's message says "for this session": the sandbox's own wording.
GENERATION_LIMIT = 10
IMAGE_LIMIT_MESSAGE = (
    "Failed to generate image: Image generation limit of 10 reached for this session. "
    "Use CSS gradients, SVG, or placeholder images for remaining visuals."
)
SPEECH_LIMIT_MESSAGE = (
    "Failed to generate speech: Speech generation limit of 10 reached for this turn. "
    "Reuse an already-generated clip or deliver the text instead."
)
# Fields the exact answers read, which must be strings when present.
_STRING_FIELDS = {
    "write_file": ("content",),
    "edit_file": ("old_text", "new_text"),
    "generate_image": ("file_path", "prompt"),
    "generate_speech": ("file_path", "text", "language"),
    "present_file": ("path",),
}


def file_hash(s: str | None) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256((s or "").encode("utf-8", "replace")).digest()).decode().rstrip("=")


def norm_path(p: str | None) -> str:
    """An absolute path in the sandbox: relative paths and ~ resolve under the home directory."""
    p = (p or "").strip()
    if p == "~":
        p = HOME
    elif p.startswith("~/"):
        p = HOME + p[1:]
    elif not p.startswith("/"):
        p = HOME + "/" + p
    return os.path.normpath(p)


def fuzzy_replace(text: str, old: str, new: str) -> str | None:
    """edit_file semantics: first match, tolerant of whitespace and indentation differences."""
    if not old:
        return None
    if old in text:
        return text.replace(old, new, 1)
    toks = old.split()
    if not toks:
        return None
    m = re.search(r"\s+".join(re.escape(t) for t in toks), text)
    return None if not m else text[: m.start()] + new + text[m.end() :]


def _read_result(content: str) -> str:
    """read_file's result for a file whose content is known."""
    out: dict[str, Any] = {
        "content": content[:READ_CAP],
        "kind": "text",
        "lines": content.count("\n") + (0 if (not content or content.endswith("\n")) else 1),
        "size": len(content.encode("utf-8", "replace")),
        "status": "success",
    }
    if len(content) > READ_CAP:
        out["truncated"] = True
    return json.dumps(out)


class SandboxWorkspace(Workspace):
    """The sandbox's files by absolute path, and the generations made this turn."""

    def __init__(self) -> None:
        super().__init__()
        self.images = 0
        self.speech = 0

    def new_turn(self) -> None:
        self.images = 0  # both generation limits are per turn
        self.speech = 0

    def observe(self, call: dict[str, Any], output: Any) -> None:
        name, args = call.get("name"), parse_args(call) or {}
        try:
            out = json.loads(output) if isinstance(output, str) else (output or {})
        except json.JSONDecodeError:
            out = {}
        ok = isinstance(out, dict) and out.get("status") in ("success", "completed")
        path = norm_path(args["path"]) if isinstance(args.get("path"), str) else None
        if name == "write_file" and ok and path:
            self.files[path] = args.get("content", "")
        elif name == "edit_file" and ok and path:
            known = self.files.get(path)
            if isinstance(known, str):
                self.files[path] = fuzzy_replace(known, args.get("old_text", ""), args.get("new_text", ""))
        elif name == "read_file" and ok and path and isinstance(out.get("content"), str):
            # Only an untruncated text read shows the whole file.
            if not out.get("truncated") and out.get("kind", "text") == "text":
                self.files[path] = out["content"]
        elif name in ("generate_image", "generate_speech") and ok:
            if name == "generate_image":
                self.images += 1
            else:
                self.speech += 1
            if isinstance(args.get("file_path"), str):
                self.files[norm_path(args["file_path"])] = None
        elif name == "bash":
            self.forget_mutated(args.get("command", ""))

    def forget_mutated(self, cmd: Any) -> None:
        """Forget what a bash command may have changed: every known content after a broad mutation (a formatter or
        fixer, a version-control operation), else the files a mutating command names. The names match as plain
        substrings of the path or its basename, looser than `mentions` on purpose: a command that may have changed a
        file had better forget too much than too little."""
        cmd = cmd if isinstance(cmd, str) else ""
        broad = BROAD_MUTATION.search(cmd) is not None
        if not broad and not (MUTATING.search(cmd) or WINDOWS_MUTATING.search(cmd)):
            return
        for p in list(self.files):
            if isinstance(self.files[p], str) and (broad or p in cmd or os.path.basename(p) in cmd):
                self.files[p] = None

    def shown_files(self, call: dict[str, Any]) -> str:
        text = json.dumps(parse_args(call) or {})
        hits = [p for p, c in self.files.items() if isinstance(c, str) and mentions(p, text)]
        if call.get("name") == "bash" and re.search(r"\b(npm|npx|yarn|pnpm|node)\b", text):
            # A package manager reads package.json whether or not the command names it.
            hits += [
                p
                for p, c in self.files.items()
                if isinstance(c, str) and os.path.basename(p) == "package.json" and p not in hits
            ]
        return render_files(((p, self.files[p]) for p in hits[:SHOWN_FILES]), SHOWN_FILE_CHARS)

    def answer(self, call: dict[str, Any], tool_names: frozenset[str]) -> str | None:
        name, args = call.get("name"), parse_args(call)
        if args is None or name not in tool_names:
            return json.dumps(HARNESS_ERROR)
        if any(k in args and not isinstance(args[k], str) for k in _STRING_FIELDS.get(name, ())):
            return json.dumps(HARNESS_ERROR)  # a schema violation (e.g. content as an object)
        path = norm_path(args["path"]) if isinstance(args.get("path"), str) else None
        if name == "write_file":
            return self._write(path, args.get("content", "")) if path else None
        if name in ("read_file", "edit_file"):
            known = self.files.get(path) if path else None
            if not isinstance(known, str):
                return None
            return _read_result(known) if name == "read_file" else self._edit(path, known, args)
        if name == "bash":
            return self._read_back(args)
        if name == "present_file":
            return json.dumps({"path": args.get("path", ""), "status": "success"})
        if name == "generate_image":
            return self._generate_image(args)
        if name == "generate_speech":
            return self._generate_speech(args)
        return None

    def _read_back(self, args: Mapping[str, Any]) -> str | None:
        """A bash command that only prints a known file (`ReadBack`), by an absolute path, one under ~, or one
        relative to `cwd` (default: the home directory). Output over the stream cap goes to the LLM."""
        read_back = ReadBack.parse(args.get("command"))
        if read_back is None or not plain_shell_args(args, "command", ("cwd", "timeout")):
            return None
        path, cwd = read_back.path, args.get("cwd")
        if path.startswith("~") and not path.startswith("~/"):
            return None  # another user's home
        if path.startswith("~/"):
            # bash expands ~ before the command runs, so wc prints the expanded path.
            path = HOME + path[1:]
            read_back = dataclasses.replace(read_back, path=path)
        elif not path.startswith("/"):
            if cwd is not None and (not isinstance(cwd, str) or not cwd.strip()):
                return None
            path = f"{norm_path(cwd)}/{path}"
        known = self.files.get(norm_path(path))
        if not isinstance(known, str):
            return None
        stdout = read_back.stdout(known)
        if len(stdout) > BASH_STREAM_CAP:
            return None
        # The duration is not knowable: a stable value in a typical range for read-backs.
        duration = 75 + int(hashlib.sha256(args["command"].encode()).hexdigest(), 16) % 25
        return json.dumps(
            {
                "duration_ms": duration,
                "exit_code": 0,
                "status": "completed",
                "stderr": "",
                "stderr_truncated": False,
                "stdout": stdout,
                "stdout_truncated": False,
            },
            ensure_ascii=False,
        )

    def _write(self, path: str, content: str) -> str:
        self.files[path] = content
        return json.dumps({"hash": file_hash(content), "status": "success"})

    def _edit(self, path: str, known: str, args: Mapping[str, Any]) -> str:
        after = fuzzy_replace(known, args.get("old_text", ""), args.get("new_text", ""))
        if after is None:
            return json.dumps(
                {"message": "Context not found. Read the file to verify the text exists.", "status": "error"}
            )
        self.files[path] = after
        return json.dumps({"hash": file_hash(after), "message": f"Edited {path}.", "status": "success"})

    def _generate_image(self, args: Mapping[str, Any]) -> str:
        if self.images >= GENERATION_LIMIT:
            return json.dumps({"message": IMAGE_LIMIT_MESSAGE, "status": "error"})
        self.images += 1
        file_path = args.get("file_path", "")
        self.files[norm_path(file_path)] = None
        return json.dumps(
            {"file_path": file_path, "hash": file_hash(file_path + args.get("prompt", "")), "status": "success"}
        )

    def _generate_speech(self, args: Mapping[str, Any]) -> str:
        if self.speech >= GENERATION_LIMIT:
            return json.dumps({"message": SPEECH_LIMIT_MESSAGE, "status": "error"})
        self.speech += 1
        file_path = args.get("file_path", "")
        self.files[norm_path(file_path)] = None
        attribution = {
            "language": args.get("language", "en"),
            "locale_match": "supported",
            "model": "gemini-3.1-flash-tts-preview",
            "provider": "google",
        }
        return json.dumps(
            {
                "attribution": attribution,
                "file_path": file_path,
                "hash": file_hash(file_path + args.get("text", "")),
                "status": "success",
            }
        )


def image_read(call: dict[str, Any], history: Items) -> str | None:
    """read_file on an image the conversation already mentions (made, listed or given).

    The sandbox shows the image to the model; a text-only rollout cannot, so the result says it was returned. An
    image never mentioned still goes to the LLM, which decides whether it exists.
    """
    if call.get("name") != "read_file":
        return None
    args = parse_args(call)
    path = args.get("path") if args else None
    if not isinstance(path, str) or not path.lower().endswith(IMAGE_EXT):
        return None
    base = os.path.basename(path.rstrip("/"))
    if base and any(base in json.dumps(x, ensure_ascii=False) for x in history if x is not call):
        return json.dumps(IMAGE_READ)
    return None


def last_json_object(text: str | None) -> dict[str, Any] | None:
    """The whole text when it is one JSON object, else the last top-level object that looks like a tool result (has
    status, results or content): the model may think inline first."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    best = None
    pos = 0
    while (start := text.find("{", pos)) != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(value, dict) and ("status" in value or "results" in value or "content" in value):
            best = value
        pos = end  # skip the object's inside: nested objects are not top-level answers
    return best


def parse_json_reply(message: dict[str, Any], finish_reason: str | None = None) -> str | None:
    """A reply holding one JSON object (the tool's result): that object, re-serialised; None when there is none or
    the reply was cut off by the output budget (an object in cut-off text may be a draft or an inner fragment)."""
    if finish_reason == "length":
        return None
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if "</think>" in content:
        # Gym's vLLM server returns the reasoning inline: <think>...</think> before the answer.
        inline, content = content.rsplit("</think>", 1)
        reasoning = reasoning or inline.replace("<think>", "", 1)
    found = last_json_object(content)
    if found is None and not content.strip():
        # An empty reply: the answer sometimes lands in the reasoning channel instead.
        found = last_json_object(reasoning)
    return None if found is None else json.dumps(found, ensure_ascii=False)


def sandbox_shell_failures(items: Items, tool_names: Collection[str] = ()) -> int:
    """bash calls that timed out or exited non-zero, from their JSON results (`status`, `exit_code`)."""
    calls = {x.get("call_id"): x for x in items if x.get("type") == "function_call"}
    n = 0
    for x in items:
        if x.get("type") != "function_call_output" or calls.get(x.get("call_id"), {}).get("name") != "bash":
            continue
        out = json_object(x.get("output") or "{}")
        if out is not None and (out.get("status") == "timeout" or out.get("exit_code") not in (0, None)):
            n += 1
    return n


class SandboxHarness(Harness):
    """The sandbox's own workspace model and answers in code, its JSON replies and a user behind every row."""

    def new_workspace(self, metadata: Mapping[str, Any]) -> SandboxWorkspace:
        return SandboxWorkspace()

    def exact_answer(
        self, call: dict[str, Any], *, history: Items, workspace: Workspace | None, tool_names: frozenset[str]
    ) -> str | None:
        """An image read the history mentions, else the workspace's answer."""
        image = image_read(call, history)
        if image is not None:
            return image
        return super().exact_answer(call, history=history, workspace=workspace, tool_names=tool_names)

    def parse_reply(self, message: dict[str, Any], finish_reason: str | None) -> str | None:
        """The tools answer in one JSON object: the reply's, re-serialised."""
        return parse_json_reply(message, finish_reason)

    def requester(self, metadata: Mapping[str, Any]) -> Requester:
        """Its rows answer a user even when the turn runs in a subagent."""
        return USER


SANDBOX = SandboxHarness(
    name="sandbox",
    simulator_prompt="tool-simulator-sandbox.md",
    judge_prompt="judge-system-sandbox.md",
    stop_tools=("ask_user",),
    shell_failures=sandbox_shell_failures,
    # The judge frame describes the sandbox, so the prompt is hidden by default.
    show_system_prompt=False,
    default_tools=KNOWN_TOOLS,
    malformed_argument_keys=(MALFORMED_ARGS_KEY,),
    gates_missing_arguments=True,
    obstacle_texts=OBSTACLE_TEXTS,
)
