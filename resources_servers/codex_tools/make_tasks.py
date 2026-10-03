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
"""Build codex_tools task rows (instructions, environment context, prompt, and tool specs).

    python resources_servers/codex_tools/make_tasks.py --prompt "Fix the failing test in ..." \\
        [--check-command "pytest -q tests/test_x.py"] [--repo-path /path/to/repo] [--base-ref main] -o task.jsonl

With no --prompt, writes the example tasks (which target this repository) to data/example.jsonl.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from resources_servers.codex_tools.tools import CODEX_TOOLS, DEFAULT_INSTRUCTIONS, environment_context


EXAMPLE_TASKS = [
    {
        "task_id": "find_chat_completions_provider_server",
        "prompt": (
            "Which directory under responses_api_models/ contains the model server that serves hosted "
            "OpenAI-compatible Chat Completions providers such as Fireworks and Together.ai by converting "
            "Responses API requests to Chat Completions? Write just that directory name (for example "
            "`openai_model`) followed by a newline to ANSWER.txt at the repository root."
        ),
        "check_command": 'test "$(cat ANSWER.txt)" = inference_provider',
    },
    {
        "task_id": "implement_roman_numerals",
        "prompt": (
            "Create a standard-library-only Python module scratch/roman.py with two functions: "
            "to_roman(n: int) -> str for 1 <= n <= 3999, and from_roman(s: str) -> int, which must raise "
            "ValueError for invalid numerals such as 'IIII' or 'IC'. Add pytest tests in "
            "scratch/test_roman.py and run them."
        ),
        "check_command": """python3 - <<'EOF'
import sys

sys.path.insert(0, "scratch")
from roman import from_roman, to_roman

assert [to_roman(n) for n in (1, 4, 9, 14, 40, 90, 400, 1994, 3999)] == [
    "I", "IV", "IX", "XIV", "XL", "XC", "CD", "MCMXCIV", "MMMCMXCIX"
]
assert all(from_roman(to_roman(n)) == n for n in range(1, 4000))
for numeral in ("IIII", "IC", "VV", "", "MMMM"):
    try:
        from_roman(numeral)
    except ValueError:
        continue
    raise SystemExit(f"from_roman({numeral!r}) did not raise ValueError")
EOF""",
    },
]


def make_row(
    prompt: str,
    *,
    check_command: Optional[str] = None,
    repo_path: Optional[str] = None,
    base_ref: Optional[str] = None,
    task_id: Optional[str] = None,
    instructions: str = DEFAULT_INSTRUCTIONS,
    shell: str = "bash",
) -> dict[str, Any]:
    metadata = {"repo_path": repo_path, "base_ref": base_ref, "check_command": check_command}
    row: dict[str, Any] = {
        "responses_create_params": {
            "input": [
                {"role": "developer", "content": instructions},
                {"role": "user", "content": environment_context(shell=shell)},
                {"role": "user", "content": prompt},
            ],
            "tools": CODEX_TOOLS,
        },
        "verifier_metadata": {key: value for key, value in metadata.items() if value is not None},
    }
    if task_id is not None:
        row["task_id"] = task_id
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", help="Task for the agent; omit to write the example tasks.")
    parser.add_argument("--check-command", help="Shell command run in the workspace by verify; exit 0 = reward 1.")
    parser.add_argument("--repo-path", help="Repository to work on (default: the server's repo_path).")
    parser.add_argument("--base-ref", help="Commit-ish for the workspace (default: the server's base_ref).")
    parser.add_argument("--task-id")
    parser.add_argument("-o", "--output", type=Path, default=Path(__file__).parent / "data" / "example.jsonl")
    args = parser.parse_args()

    if args.prompt is None:
        rows = [
            make_row(task["prompt"], check_command=task["check_command"], task_id=task["task_id"])
            for task in EXAMPLE_TASKS
        ]
    else:
        rows = [
            make_row(
                args.prompt,
                check_command=args.check_command,
                repo_path=args.repo_path,
                base_ref=args.base_ref,
                task_id=args.task_id,
            )
        ]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Wrote {len(rows)} task(s) to {args.output}")


if __name__ == "__main__":
    main()
