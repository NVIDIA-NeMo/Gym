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

"""
CLI client for the GDPVal agent.

Two modes:
  prepare  — Convert the HuggingFace openai/gdpval dataset into JSONL rows
             compatible with ng_collect_rollouts.
  run      — Post tasks directly to a running GDPVal agent (for dev/debug).

Prerequisites (for 'run' mode):
    The head server, model server, resources server, and agent server must all
    be running.  See the configs/ directory for example YAML configurations.

Usage:
    python client.py prepare --output-jsonl data/train.jsonl [--split train] [--limit N] [--task-ids id1,id2]
    python client.py run [--task-ids id1,id2] [--limit N] [--output-dir /tmp/gdpval_output]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from nemo_gym.server_utils import ServerClient, get_response_json


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Tool definitions exposed to the model.
# NOTE: session_id and output_dir are injected by the agent — they should NOT
# appear in these definitions since the model doesn't manage sessions.
TOOLS = [
    {
        "type": "function",
        "name": "run_command",
        "description": "Execute a bash command in the sandbox environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum seconds to wait for the command (default 30).",
                },
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "finish",
        "description": (
            "Mark the task as complete and end the session. "
            "Optionally provide a list of file paths (relative to the sandbox working directory) "
            "to save as permanent output files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths in the sandbox to save as output.",
                },
            },
            "required": [],
        },
    },
]

SYSTEM_PROMPT = (
    "You are a capable coding assistant with access to a bash sandbox. "
    "Use the run_command tool to execute bash commands. "
    "When you have completed the task, call the finish tool. "
    "If you created output files you want to keep, pass their paths to finish."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hf_dataset(split: str):
    """Lazily import datasets and load the openai/gdpval dataset."""
    from datasets import load_dataset
    return load_dataset("openai/gdpval", split=split)


def _filter_dataset(dataset, task_ids: list[str] | None, limit: int | None):
    """Filter dataset rows by task_ids and/or limit."""
    rows = list(dataset)
    if task_ids is not None:
        task_id_set = set(task_ids)
        rows = [r for r in rows if r["task_id"] in task_id_set]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _build_run_request(row: dict, output_dir: str) -> dict:
    """Build a GDPValAgentRunRequest-compatible dict from an HF dataset row.

    Note: instruction_prompt_template is intentionally omitted — the agent
    loads it from its own prompts/ directory when the field is None.
    """
    return {
        "responses_create_params": {
            "input": "",
            "tools": TOOLS,
        },
        "task_prompt": row["prompt"],
        "system_prompt": SYSTEM_PROMPT,
        "output_dir": output_dir,
        "task_id": row["task_id"],
        "reference_file_urls": row.get("reference_file_urls", []) or [],
        "reference_files_to_save": row.get("reference_files", []) or [],
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    """Convert HF dataset rows to JSONL for ng_collect_rollouts."""
    dataset = _load_hf_dataset(args.split)
    rows = _filter_dataset(dataset, args.task_ids, args.limit)

    if not rows:
        print("No matching rows found.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for row in rows:
            request = _build_run_request(row, args.output_dir)
            f.write(json.dumps(request) + "\n")

    print(f"Wrote {len(rows)} row(s) to {output_path}")

    if args.validate:
        _validate_jsonl(output_path)


def _validate_jsonl(path: Path) -> None:
    """Validate that each line in a JSONL file parses as a GDPValAgentRunRequest."""
    from responses_api_agents.gdpval_agent.app import GDPValAgentRunRequest

    errors = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            try:
                GDPValAgentRunRequest.model_validate_json(line)
            except Exception as e:
                print(f"  Row {i}: INVALID — {e}", file=sys.stderr)
                errors += 1

    if errors:
        print(f"Validation: {errors} error(s) found.", file=sys.stderr)
        sys.exit(1)
    else:
        print("Validation: all rows OK.")


async def _cmd_run_async(args: argparse.Namespace) -> None:
    """Run tasks directly against the agent server (for dev/debug)."""
    dataset = _load_hf_dataset(args.split)
    rows = _filter_dataset(dataset, args.task_ids, args.limit)

    if not rows:
        print("No matching rows found.", file=sys.stderr)
        sys.exit(1)

    server_client = ServerClient.load_from_global_config()

    for i, row in enumerate(rows, 1):
        task_id = row["task_id"]
        print(f"\n[{i}/{len(rows)}] Running task {task_id} ...")

        request = _build_run_request(row, args.output_dir)
        response = await server_client.post(
            server_name="bash_sandbox_agent",
            url_path="/run",
            json=request,
        )
        result = await get_response_json(response)

        reward = result.get("reward", "N/A")
        output_files = result.get("output_files", [])
        print(f"  Reward: {reward}")
        if output_files:
            print(f"  Saved {len(output_files)} file(s):")
            for f in output_files:
                print(f"    -> {f.get('output_path', 'unknown')} ({f.get('size', '?')} bytes)")
        else:
            print("  No output files saved.")

    print(f"\nDone. Ran {len(rows)} task(s).")


def cmd_run(args: argparse.Namespace) -> None:
    asyncio.run(_cmd_run_async(args))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _csv_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list."""
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="GDPVal agent client — prepare JSONL or run tasks directly.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- prepare --
    p_prepare = subparsers.add_parser(
        "prepare",
        help="Convert HF openai/gdpval dataset to JSONL for ng_collect_rollouts.",
    )
    p_prepare.add_argument(
        "--output-jsonl", required=True,
        help="Path to write the output JSONL file.",
    )
    p_prepare.add_argument("--split", default="train", help="HF dataset split (default: train).")
    p_prepare.add_argument("--limit", type=int, default=None, help="Max number of rows to emit.")
    p_prepare.add_argument(
        "--task-ids", type=_csv_list, default=None,
        help="Comma-separated list of task_ids to include.",
    )
    p_prepare.add_argument(
        "--output-dir", default="/tmp/gdpval_output",
        help="Value for output_dir in each JSONL row (default: /tmp/gdpval_output).",
    )
    p_prepare.add_argument(
        "--validate", action="store_true",
        help="Validate each JSONL row against GDPValAgentRunRequest after writing.",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    # -- run --
    p_run = subparsers.add_parser(
        "run",
        help="Run tasks directly against the agent server (dev/debug).",
    )
    p_run.add_argument("--split", default="train", help="HF dataset split (default: train).")
    p_run.add_argument("--limit", type=int, default=None, help="Max number of tasks to run.")
    p_run.add_argument(
        "--task-ids", type=_csv_list, default=None,
        help="Comma-separated list of task_ids to run.",
    )
    p_run.add_argument(
        "--output-dir", default="/tmp/gdpval_output",
        help="Output directory for saved files (default: /tmp/gdpval_output).",
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
