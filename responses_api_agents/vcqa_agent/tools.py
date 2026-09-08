# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool surface exposed to the model during VCQA rollouts.

Every tool returns a dict with `result` and `error` keys. Bad arguments,
escapes from `/codebase`, non-zero exits, and timeouts all produce an
`error` payload; they never raise out of the dispatch function. This
matches NeMo Gym guideline 4.a (tool / model errors propagate back to the
model rather than crashing the env).

`responses_create_params.tools` for the model is built from
`build_tool_definitions()` so vLLM / OpenAI both see a JSON schema for each
tool and can produce well-formed `function_call` items.
"""

from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from responses_api_agents.vcqa_agent.sandbox import Sandbox


MAX_TOOL_OUTPUT_BYTES = 32 * 1024  # 32 KiB per tool call


class _BaseToolArgs(BaseModel):
    model_config = {"extra": "forbid"}


class ReadFileArgs(_BaseToolArgs):
    path: str = Field(description="Path to the file to read, relative to /codebase.")
    max_bytes: Optional[int] = Field(default=None, description="Optional cap on returned bytes. Defaults to 32 KiB.")


class GrepArgs(_BaseToolArgs):
    pattern: str = Field(description="Regex pattern to search for.")
    path: Optional[str] = Field(default=None, description="Optional file or directory under /codebase to search in.")
    fixed_strings: bool = Field(default=False, description="Treat pattern as a literal string, not a regex.")
    case_insensitive: bool = Field(default=False, description="Case-insensitive match.")
    max_count: Optional[int] = Field(default=200, description="Maximum number of matches to return.")


class GlobArgs(_BaseToolArgs):
    pattern: str = Field(description="Glob/fd pattern to match filenames.")
    path: Optional[str] = Field(
        default=None, description="Directory under /codebase to search in. Defaults to the repo root."
    )


class ListDirArgs(_BaseToolArgs):
    path: str = Field(description="Directory under /codebase to list.")


class WriteTodosArgs(_BaseToolArgs):
    items: List[str] = Field(description="Todo lines to append to the model's per-rollout scratchpad.")


class BashArgs(_BaseToolArgs):
    command: str = Field(description="Shell command to run inside /codebase.")
    timeout: int = Field(default=30, ge=1, le=120, description="Seconds before the command is killed.")


# Distractor tool arg models
#
# Plausible-looking tools that are off-task for a read-only
# code-investigation question. Three tiers, ordered by how plausibly the
# model might still pick them:
#
#   Tier 1 (mutation / comms; clearly off-persona, return an error string):
#     install_package, send_pr_review
#   Tier 2 (succeeds-but-ungrounded; tempting shortcut):
#     websearch
#   Tier 3 (agent-shape; wrong workflow):
#     ask_user
#
# Surfacing these is controlled by `build_tool_definitions(include_distractors=...)`
# so callers can A/B against the no-distractor surface. The grader does NOT
# currently penalize distractor usage; their value is in measuring how the
# model handles tempting off-task shortcuts under load.


class InstallPackageArgs(_BaseToolArgs):
    name: str = Field(description="Package name to install.")
    manager: Optional[str] = Field(
        default=None,
        description="Package manager (pip, npm, apt, cargo, ...). Optional; autodetected by default.",
    )


class SendPrReviewArgs(_BaseToolArgs):
    comments: str = Field(description="Review text to post.")
    verdict: Optional[str] = Field(default=None, description="Optional verdict: approve / request_changes / comment.")


class WebSearchArgs(_BaseToolArgs):
    query: str = Field(description="Search query.")


class AskUserArgs(_BaseToolArgs):
    question: str = Field(description="Question to ask the user.")


_REAL_TOOL_ARG_MODELS: Dict[str, type[_BaseToolArgs]] = {
    "read_file": ReadFileArgs,
    "grep": GrepArgs,
    "glob": GlobArgs,
    "list_dir": ListDirArgs,
    "write_todos": WriteTodosArgs,
    "bash": BashArgs,
}

_DISTRACTOR_TOOL_ARG_MODELS: Dict[str, type[_BaseToolArgs]] = {
    "install_package": InstallPackageArgs,
    "send_pr_review": SendPrReviewArgs,
    "websearch": WebSearchArgs,
    "ask_user": AskUserArgs,
}

DISTRACTOR_TOOL_NAMES = frozenset(_DISTRACTOR_TOOL_ARG_MODELS)

_TOOL_ARG_MODELS: Dict[str, type[_BaseToolArgs]] = {
    **_REAL_TOOL_ARG_MODELS,
    **_DISTRACTOR_TOOL_ARG_MODELS,
}


def build_tool_definitions(include_distractors: bool = True) -> List[Dict[str, Any]]:
    """OpenAI Responses API `function` tool defs.

    The six real tools are always surfaced. The four distractor tools (see
    `_DISTRACTOR_TOOL_ARG_MODELS`) are only surfaced when `include_distractors`
    is true.
    """
    descriptions: Dict[str, str] = {
        "read_file": "Read a file from the repo working tree at /codebase.",
        "grep": "Search for a pattern in the repo working tree using ripgrep.",
        "glob": "Find files matching a pattern using fd.",
        "list_dir": "List the contents of a directory under /codebase.",
        "write_todos": "Append todo notes to the model's per-rollout scratchpad.",
        "bash": "Run a shell command inside the sandbox (e.g. for `git log`, `git show`).",
        "install_package": "Install a package (pip, npm, apt, etc.) into the sandbox.",
        "send_pr_review": "Post a review comment on the upstream pull request.",
        "websearch": "Search the web for general information.",
        "ask_user": "Ask the user a clarifying question.",
    }
    arg_models: Dict[str, type[_BaseToolArgs]] = dict(_REAL_TOOL_ARG_MODELS)
    if include_distractors:
        arg_models.update(_DISTRACTOR_TOOL_ARG_MODELS)

    tools: List[Dict[str, Any]] = []
    for name, model_cls in arg_models.items():
        schema = model_cls.model_json_schema()
        schema.pop("title", None)
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": descriptions[name],
                "parameters": schema,
                # OpenAI's Responses API requires `strict` on every function
                # tool. `False` is required because some argument schemas
                # have Optional fields with defaults, which OpenAI's strict
                # mode rejects (it requires `additionalProperties: false`
                # AND every property listed in `required`).
                "strict": False,
            }
        )
    return tools


def _make_error(message: str) -> Dict[str, Any]:
    return {"result": None, "error": message}


def _truncate(s: str, limit: int = MAX_TOOL_OUTPUT_BYTES) -> Dict[str, Any]:
    encoded = s.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return {"text": s, "truncated": False}
    return {
        "text": encoded[:limit].decode("utf-8", errors="replace"),
        "truncated": True,
        "original_bytes": len(encoded),
    }


def _resolve_under_codebase(path: str, codebase_path: str = "/codebase") -> Optional[str]:
    """Map a model-supplied path to an absolute path under `codebase_path`.

    Returns None if the model tries to escape the working tree (absolute
    paths outside `codebase_path`, or `..` segments that would walk above it).
    """
    codebase_path = _normpath(codebase_path)
    if not path:
        return codebase_path
    if path.startswith("/"):
        normalized = _normpath(path)
        if not _is_under_path(normalized, codebase_path):
            return None
        return normalized
    normalized = _normpath(f"{codebase_path}/{path}")
    if not _is_under_path(normalized, codebase_path):
        return None
    return normalized


def _is_under_path(path: str, parent: str) -> bool:
    if parent == "/":
        return path.startswith("/")
    return path == parent or path.startswith(parent + "/")


def _normpath(path: str) -> str:
    """Same semantics as posixpath.normpath but written out so it is
    obvious there are no symlink-following or filesystem touches.
    """
    parts: List[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            if not parts and path.startswith("/"):
                parts.append("")
            continue
        if segment == "..":
            if parts and parts[-1] not in ("", ".."):
                parts.pop()
            elif not path.startswith("/"):
                parts.append("..")
            continue
        parts.append(segment)
    if path.startswith("/"):
        return "/" + "/".join(p for p in parts if p)
    return "/".join(parts) or "."


async def _read_file(args: ReadFileArgs, sandbox: Sandbox) -> Dict[str, Any]:
    abs_path = _resolve_under_codebase(args.path, sandbox.codebase_path)
    if abs_path is None:
        return _make_error(f"path '{args.path}' resolves outside the working tree")
    cap = args.max_bytes if args.max_bytes else MAX_TOOL_OUTPUT_BYTES
    cmd = f"head -c {int(cap)} -- {shlex.quote(abs_path)}"
    res = await sandbox.exec(cmd)
    if res.timed_out:
        return _make_error("read_file timed out")
    if res.exit_code != 0:
        return _make_error(f"read_file failed (exit={res.exit_code}): {res.stderr.strip()[:512]}")
    truncated = _truncate(res.stdout, cap)
    return {"result": truncated, "error": None}


async def _grep(args: GrepArgs, sandbox: Sandbox) -> Dict[str, Any]:
    flags = ["--line-number", "--with-filename", "--no-heading"]
    if args.fixed_strings:
        flags.append("--fixed-strings")
    if args.case_insensitive:
        flags.append("--ignore-case")
    if args.max_count:
        flags.append(f"--max-count={int(args.max_count)}")
    target = sandbox.codebase_path
    if args.path:
        resolved = _resolve_under_codebase(args.path, sandbox.codebase_path)
        if resolved is None:
            return _make_error(f"path '{args.path}' resolves outside the working tree")
        target = resolved
    cmd = f"rg {' '.join(shlex.quote(f) for f in flags)} -- {shlex.quote(args.pattern)} {shlex.quote(target)}"
    res = await sandbox.exec(cmd)
    if res.timed_out:
        return _make_error("grep timed out")
    if res.exit_code not in (0, 1):
        return _make_error(f"grep failed (exit={res.exit_code}): {res.stderr.strip()[:512]}")
    return {"result": _truncate(res.stdout), "error": None}


async def _glob(args: GlobArgs, sandbox: Sandbox) -> Dict[str, Any]:
    target = sandbox.codebase_path
    if args.path:
        resolved = _resolve_under_codebase(args.path, sandbox.codebase_path)
        if resolved is None:
            return _make_error(f"path '{args.path}' resolves outside the working tree")
        target = resolved
    cmd = f"fd --hidden --no-ignore --glob -- {shlex.quote(args.pattern)} {shlex.quote(target)}"
    res = await sandbox.exec(cmd)
    if res.timed_out:
        return _make_error("glob timed out")
    if res.exit_code != 0:
        return _make_error(f"glob failed (exit={res.exit_code}): {res.stderr.strip()[:512]}")
    return {"result": _truncate(res.stdout), "error": None}


async def _list_dir(args: ListDirArgs, sandbox: Sandbox) -> Dict[str, Any]:
    abs_path = _resolve_under_codebase(args.path, sandbox.codebase_path)
    if abs_path is None:
        return _make_error(f"path '{args.path}' resolves outside the working tree")
    cmd = f"ls -la --color=never -- {shlex.quote(abs_path)}"
    res = await sandbox.exec(cmd)
    if res.timed_out:
        return _make_error("list_dir timed out")
    if res.exit_code != 0:
        return _make_error(f"list_dir failed (exit={res.exit_code}): {res.stderr.strip()[:512]}")
    return {"result": _truncate(res.stdout), "error": None}


async def _write_todos(args: WriteTodosArgs, sandbox: Sandbox) -> Dict[str, Any]:
    if not args.items:
        return _make_error("write_todos requires at least one item")
    payload = "\n".join(f"- {item}" for item in args.items) + "\n"
    cmd = f"printf %s {shlex.quote(payload)} >> {shlex.quote(sandbox.todos_path)}"
    res = await sandbox.exec(cmd)
    if res.exit_code != 0:
        return _make_error(f"write_todos failed (exit={res.exit_code}): {res.stderr.strip()[:512]}")
    return {"result": {"appended": len(args.items)}, "error": None}


async def _bash(args: BashArgs, sandbox: Sandbox) -> Dict[str, Any]:
    res = await sandbox.exec(args.command, timeout_s=args.timeout)
    truncated_out = _truncate(res.stdout)
    truncated_err = _truncate(res.stderr)
    return {
        "result": {
            "exit_code": res.exit_code,
            "stdout": truncated_out,
            "stderr": truncated_err,
            "timed_out": res.timed_out,
        },
        "error": None,
    }


# Distractor tool handlers
#
# install_package / send_pr_review return a clear error string so the model
# learns these are off-persona for a read-only investigation. websearch
# returns a plausible-but-ungrounded blurb (the "tempting shortcut" tier).
# ask_user reflects the question back without an answer. All of them
# succeed at the framework level (`error=None`); the model needs to read
# the response text to learn the tool was useless.


async def _install_package(args: InstallPackageArgs, sandbox: Sandbox) -> Dict[str, Any]:
    del sandbox
    text = (
        "Error: package installation is not available in this environment. "
        "The codebase ships with its existing dependencies; you cannot add new ones."
    )
    return {"result": {"text": text}, "error": None}


async def _send_pr_review(args: SendPrReviewArgs, sandbox: Sandbox) -> Dict[str, Any]:
    del sandbox
    text = "Error: this environment is offline and has no GitHub access; PR reviews cannot be posted."
    return {"result": {"text": text}, "error": None}


async def _websearch(args: WebSearchArgs, sandbox: Sandbox) -> Dict[str, Any]:
    del sandbox
    query = args.query.strip() or "(empty query)"
    text = (
        f"# Web search: {query}\n\n"
        f"Top results discuss general background on `{query}`, common usage "
        f"patterns, and typical pitfalls. Most results link to official docs, "
        f"Stack Overflow threads, and blog posts. Specific implementation "
        f"details vary by project; consult the source you're actually working "
        f"with for authoritative behavior."
    )
    return {"result": {"text": text}, "error": None}


async def _ask_user(args: AskUserArgs, sandbox: Sandbox) -> Dict[str, Any]:
    del sandbox
    question = args.question.strip()
    preview = (question[:80] + "...") if len(question) > 80 else question
    text = (
        f"(No response received; the user is not available to answer "
        f"follow-up questions in this environment. Question was: {preview!r})"
    )
    return {"result": {"text": text}, "error": None}


_DISPATCH = {
    "read_file": _read_file,
    "grep": _grep,
    "glob": _glob,
    "list_dir": _list_dir,
    "write_todos": _write_todos,
    "bash": _bash,
    "install_package": _install_package,
    "send_pr_review": _send_pr_review,
    "websearch": _websearch,
    "ask_user": _ask_user,
}


async def dispatch_tool(name: str, raw_arguments: Dict[str, Any], sandbox: Sandbox) -> Dict[str, Any]:
    """Validate args + run the named tool. Always returns a dict, never raises."""
    if name not in _DISPATCH:
        return _make_error(f"unknown tool: {name!r}")
    arg_model = _TOOL_ARG_MODELS[name]
    try:
        parsed = arg_model.model_validate(raw_arguments)
    except ValidationError as e:
        return _make_error(f"invalid arguments for {name}: {e.errors()}")

    try:
        return await _DISPATCH[name](parsed, sandbox)
    except Exception as e:
        return _make_error(f"{name} raised {type(e).__name__}: {e}")
