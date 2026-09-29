# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Responses API function-tool specs for the codex_tools server.

Descriptions are copied from openai/codex (Apache-2.0, Copyright OpenAI).

Names, descriptions, and parameters follow Codex (openai/codex codex-rs, core/src/tools/handlers/*_spec.rs
at 44fe510ce3). Approval/sandbox parameters are omitted because this server has neither. ``apply_patch``
uses Codex's JSON function variant (codex-rs/tools/src/apply_patch_tool.rs, removed upstream after
April 2026) since the freeform grammar tool needs custom-tool support that Gym's agents and model
servers lack.
"""

from typing import Any


def _function_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


_MAX_OUTPUT_TOKENS = {
    "type": "number",
    "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy.",
}

EXEC_COMMAND_TOOL = _function_tool(
    "exec_command",
    "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
    {
        "cmd": {"type": "string", "description": "Shell command to execute."},
        "login": {
            "type": "boolean",
            "description": "True runs the shell with -l/-i semantics; false disables them. Defaults to true.",
        },
        "max_output_tokens": _MAX_OUTPUT_TOKENS,
        "shell": {"type": "string", "description": "Shell binary to launch. Defaults to the user's default shell."},
        "tty": {
            "type": "boolean",
            "description": "True allocates a PTY for the command; false or omitted uses plain pipes.",
        },
        "workdir": {
            "type": "string",
            "description": "Working directory for the command. Defaults to the turn cwd.",
        },
        "yield_time_ms": {
            "type": "number",
            "description": "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms.",
        },
    },
    ["cmd"],
)

WRITE_STDIN_TOOL = _function_tool(
    "write_stdin",
    "Writes characters to an existing unified exec session and returns recent output.",
    {
        "chars": {
            "type": "string",
            "description": "Bytes to write to stdin. Defaults to empty, which polls without writing.",
        },
        "max_output_tokens": _MAX_OUTPUT_TOKENS,
        "session_id": {"type": "number", "description": "Identifier of the running unified exec session."},
        "yield_time_ms": {
            "type": "number",
            "description": (
                "Wait before yielding output. Non-empty writes default to 250 ms and cap at 30000 ms; "
                "empty polls wait 5000-300000 ms by default."
            ),
        },
    },
    ["session_id"],
)

APPLY_PATCH_DESCRIPTION = """Use the `apply_patch` tool to edit files.
Your patch language is a stripped‑down, file‑oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high‑level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more “hunks”, each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change’s [context_after] lines in the second change’s [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ 	 def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
- File references can only be relative, NEVER ABSOLUTE.
"""

APPLY_PATCH_TOOL = _function_tool(
    "apply_patch",
    APPLY_PATCH_DESCRIPTION,
    {"input": {"type": "string", "description": "The entire contents of the apply_patch command"}},
    ["input"],
)

UPDATE_PLAN_TOOL = _function_tool(
    "update_plan",
    "Updates the task plan.\n"
    "Provide an optional explanation and a list of plan items, each with a step and status.\n"
    "At most one step can be in_progress at a time.\n",
    {
        "explanation": {"type": "string", "description": "Optional explanation for this plan update."},
        "plan": {
            "type": "array",
            "description": "The list of steps",
            "items": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "Step status.",
                    },
                    "step": {"type": "string", "description": "Task step text."},
                },
                "required": ["step", "status"],
                "additionalProperties": False,
            },
        },
    },
    ["plan"],
)

CODEX_TOOLS = [EXEC_COMMAND_TOOL, WRITE_STDIN_TOOL, APPLY_PATCH_TOOL, UPDATE_PLAN_TOOL]

DEFAULT_INSTRUCTIONS = """You are a coding agent working non-interactively in a software repository. \
Nobody will answer questions, so complete the task autonomously and end your turn with a concise \
final message describing what you changed and how you verified it.

- The repository is your current working directory. Use relative paths; tools default to it.
- Use `exec_command` to inspect the repository and run commands (prefer `rg` for search), and \
`write_stdin` to poll or interact with commands that are still running.
- Edit files with `apply_patch`. Keep changes focused on the task and consistent with the \
surrounding code.
- Run the relevant tests or checks when practical, and report what you actually ran.
- Use `update_plan` to keep a short checklist for multi-step work.
"""


def environment_context(*, shell: str = "bash") -> str:
    """Codex-style environment context block for the first user turn."""
    return f"<environment_context>\n<cwd>.</cwd>\n<shell>{shell}</shell>\n</environment_context>"
