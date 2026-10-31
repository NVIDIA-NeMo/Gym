# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write rows for sandboxed harnesses from OfficeEnv's sandbox example tasks.

Each row carries the task's TaskSpec and a system and user message; it has no tool list, since a
sandboxed harness takes the episode's tools from its MCP grant and refuses request tools. The
OfficeEnv that runs this script is the one the rows are stamped with.

    python resources_servers/officeenv/sandbox/prepare_rows.py [output.jsonl] [--exposure tools|cli|api_search]
        [--tool-prefix <prefix>]

``--exposure`` sets how the harness reaches the apps (the server's SANDBOX_SURFACES): one MCP tool
per op (``tools``, the default, written to data/sandbox_pilot.jsonl), one command per app (``cli``),
or endpoints found over MCP and called with curl (``api_search``); those two are written to
data/sandbox_pilot_<exposure>_prepare.jsonl unless an output is given. The system prompt says which,
naming MCP tools as the harness lists them: ``--tool-prefix company_`` (the default) for Pi, which
lists configs/officeenv_sandbox.yaml's MCP endpoint as ``company``, ``mcp__company__`` for Codex. It
names the company's API the way the sandbox does (app.py's API_URL_ENV, API_TOKEN_ENV and
GUIDE_NAME), never the environment. A task that needs more than the shell, curl and Python in its
sandbox (its example's ``sandbox_needs``: documents, office) keeps that in its row, for the server
to give its episode an image that has them.
"""

import argparse
import json
from importlib import resources
from pathlib import Path

from officeenv import __version__ as officeenv_version


# How Pi lists the tools of configs/officeenv_sandbox.yaml's MCP endpoint: <mcp_server_name>_<tool>
TOOL_PREFIX = "company_"

SANDBOX = (
    "You work for the user from a Linux sandbox: bash, Python 3.12 with pandas and openpyxl, and curl, in /workspace."
)
FILES = """Move file content between the workspace and the apps with the file API described in /workspace/README.md \
(its URL and token are in the environment variables COMPANY_API_URL and COMPANY_API_TOKEN; in a shell without them, \
run `. /etc/company-api.env` first), and attach workspace files to email by their path, rather than pasting file \
content into tool calls; do data work with code."""
# Per surface: how the apps are reached, and how the task is completed
SURFACES = {
    "tools": (
        "The company's apps (Google Drive, Gmail and others) are your tools.",
        "call the {prefix}user_complete_task tool with a one-line summary",
    ),
    "cli": (
        "The company's apps (Google Drive, Gmail and others) are commands here, one per app "
        "(slack post-message --channel='#sales' --text='hi'; `<app> --help` lists an app's commands); "
        "/workspace/README.md lists them.",
        "run `user complete-task --answer='<a one-line summary>'`",
    ),
    "api_search": (
        "The company's apps (Google Drive, Gmail and others) are an HTTP API: find an endpoint with your "
        "{prefix}api_search tool, read its parameters with your {prefix}api_fetch tool, and call it with curl "
        "as /workspace/README.md shows.",
        "call the user/complete_task endpoint with a one-line summary as its answer",
    ),
}

DATA = Path(__file__).resolve().parents[1] / "data"


def system_prompt(exposure: str = "tools", tool_prefix: str = TOOL_PREFIX) -> str:
    apps, done = (text.format(prefix=tool_prefix) for text in SURFACES[exposure])
    return f"{SANDBOX} {apps} {FILES} When the task is done, {done}."


SYSTEM_PROMPT = system_prompt()


def default_output(exposure: str) -> Path:
    return DATA / ("sandbox_pilot.jsonl" if exposure == "tools" else f"sandbox_pilot_{exposure}_prepare.jsonl")


def rows(exposure: str = "tools", tool_prefix: str = TOOL_PREFIX, folder=None) -> list[dict]:
    """A row per example task in ``folder`` (OfficeEnv's sandbox examples by default)."""
    out = []
    folder = folder or resources.files("officeenv.examples") / "sandbox"
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".json"):
            continue
        task = json.loads(entry.read_text(encoding="utf-8"))
        spec = task["spec"]
        if spec.get("exposure", {}).get("mode", "tools") != exposure:
            spec["exposure"] = {**spec.get("exposure", {}), "mode": exposure}
        row = {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": system_prompt(exposure, tool_prefix)},
                    {"role": "user", "content": task["prompt"]},
                ]
            },
            "officeenv": spec,
            "officeenv_version": officeenv_version,
            "task_id": spec["task_id"],
        }
        if task.get("sandbox_needs"):
            row["sandbox_needs"] = task["sandbox_needs"]
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write rows for sandboxed harnesses")
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--exposure", choices=list(SURFACES), default="tools", help="how the harness reaches the apps")
    parser.add_argument(
        "--tool-prefix",
        default=TOOL_PREFIX,
        help="how the harness names the MCP tools (Pi: company_, Codex: mcp__company__)",
    )
    args = parser.parse_args()
    output = args.output or default_output(args.exposure)
    written = rows(args.exposure, args.tool_prefix)
    output.write_text("".join(json.dumps(row) + "\n" for row in written), encoding="utf-8")
    print(f"wrote {len(written)} rows ({officeenv_version}, {args.exposure}) to {output}")


if __name__ == "__main__":
    main()
