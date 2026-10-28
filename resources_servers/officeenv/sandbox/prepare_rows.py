# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write rows for sandboxed harnesses from OfficeEnv's sandbox example tasks.

Each row carries the task's TaskSpec and a system and user message; it has no tool list, since a
sandboxed harness takes the episode's tools from its MCP grant and refuses request tools. The
OfficeEnv that runs this script is the one the rows are stamped with.

    python resources_servers/officeenv/sandbox/prepare_rows.py [output.jsonl]
"""

import json
import sys
from importlib import resources
from pathlib import Path

from officeenv import __version__ as officeenv_version


# How Pi lists the tools of configs/officeenv_sandbox.yaml's MCP endpoint: <mcp_server_name>_<tool>
TOOL_PREFIX = "company_"

SYSTEM_PROMPT = f"""You work for the user from a Linux sandbox: bash, Python 3.12 with pandas and openpyxl, \
and curl, in /workspace. The company's apps (Google Drive, Gmail and others) are your tools. Move file \
content between the workspace and Google Drive with the file API described in /workspace/README.md (its \
URL and token are in the environment variables COMPANY_API_URL and COMPANY_API_TOKEN), not by pasting \
it into tool calls, and do data work with code. When the task is done, call the {TOOL_PREFIX}user_complete_task \
tool with a one-line summary."""

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "sandbox_pilot.jsonl"


def rows() -> list[dict]:
    out = []
    folder = resources.files("officeenv.examples") / "sandbox"
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".json"):
            continue
        task = json.loads(entry.read_text(encoding="utf-8"))
        spec = task["spec"]
        out.append(
            {
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task["prompt"]},
                    ]
                },
                "officeenv": spec,
                "officeenv_version": officeenv_version,
                "task_id": spec["task_id"],
            }
        )
    return out


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    written = rows()
    output.write_text("".join(json.dumps(row) + "\n" for row in written), encoding="utf-8")
    print(f"wrote {len(written)} rows ({officeenv_version}) to {output}")


if __name__ == "__main__":
    main()
