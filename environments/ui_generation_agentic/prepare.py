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
"""Turn a UI-generation JSONL into ui_generation_agentic task rows.

Accepted input rows (one JSON object per line):

- chat shape with a container: ``{"id", "messages": [{"role": "user", ...}, ...], "container": {"image": "<stack>"}}``;
  only the
  first user message and the stack are read, and any assistant message (a reference answer) is ignored and never
  written;
- generic shape: ``{"id", "prompt" | "messages", "stack"}``.

Rows are written flat (``task_id`` is ``<stack>-<id>``), or, with ``--materialize-taskset NAME``, as materialized task
envelopes that ``gym eval run --input`` routes by taskset. The output path is always supplied by the caller: the
data stays outside the repository. A ``<output>.manifest.json`` records counts and checksums, never content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path


HERE = Path(__file__).parent
DEFAULT_TEMPLATE = HERE / "prompts" / "task_prompt.txt"

STACKS = {
    "ui-next": {
        "workdir": "/app",
        "entry_path": "app/sample.tsx",
        "description": "a Next.js 15 (app router) + React 19 + TypeScript + Tailwind CSS 3 project.",
        "contract": (
            "It must default-export a React component; the home page renders it client-side "
            '(start the file with `"use client";`).'
        ),
        "check": "`cd /app && ./node_modules/.bin/next build` must succeed.",
    },
    "ui-vite": {
        "workdir": "/app",
        "entry_path": "src/App.tsx",
        "description": "a Vite 7 + React 19 + TypeScript + Tailwind CSS 3 project.",
        "contract": "It must default-export the `App` component, which `src/main.tsx` renders.",
        "check": "`cd /app && ./node_modules/.bin/vite build` must succeed.",
    },
    "ui-static": {
        "workdir": "/usr/share/nginx/html",
        "entry_path": "index.html",
        "description": "a static web root served by nginx.",
        "contract": (
            "It must be one complete, self-contained HTML document (inline CSS and JavaScript; "
            "libraries only from public CDNs)."
        ),
        "check": "the file must be a complete HTML document (`<!doctype html>` or `<html>`, and `</html>`).",
    },
}


TEMPLATE_KEYS = ("user_prompt", "stack_description", "workdir", "entry_file", "entry_contract", "check_command")


class RowError(ValueError):
    pass


def _render(template: str, values: dict[str, str]) -> str:
    """Fill the template's ``{{ key }}`` placeholders in one pass, so inserted values (a prompt that itself contains
    ``{{ ... }}``, as Vue or Handlebars code does) stay literal. An unknown or unclosed placeholder in the template
    raises ``RowError``."""
    parts = template.split("{{ ")
    rendered = [parts[0]]
    for part in parts[1:]:
        key, closed, rest = part.partition(" }}")
        if not closed or key not in values:
            raise RowError(f"unfilled placeholder in the prompt template: {part[:40]}")
        rendered.extend((values[key], rest))
    return "".join(rendered)


def convert(row: dict, line_no: int, template: str) -> dict:
    """Return one flat task row, or raise ``RowError`` with the reason."""
    source_id = row.get("id")
    if not isinstance(source_id, str) or not source_id:
        raise RowError(f"line {line_no}: missing id")
    stack = row.get("stack") or (row.get("container") or {}).get("image")
    if stack not in STACKS:
        raise RowError(f"line {line_no}: unknown stack {stack!r}")
    prompt = row.get("prompt")
    if prompt is None:
        users = [m for m in row.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
        prompt = users[0].get("content") if users else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise RowError(f"line {line_no}: no user prompt")
    spec = STACKS[stack]
    entry_file = f"{spec['workdir'].rstrip('/')}/{spec['entry_path']}"
    rendered = _render(
        template,
        {
            "user_prompt": prompt.strip(),
            "stack_description": spec["description"],
            "workdir": spec["workdir"],
            "entry_file": entry_file,
            "entry_contract": spec["contract"],
            "check_command": spec["check"],
        },
    )
    return {
        "task_id": f"{stack}-{source_id}",
        "source_id": source_id,
        "stack": stack,
        "entry_path": spec["entry_path"],
        "user_prompt": prompt.strip(),
        "responses_create_params": {"input": [{"role": "user", "content": rendered}]},
    }


def materialize(row: dict, taskset: str) -> dict:
    params = row["responses_create_params"]
    task_data = {k: v for k, v in row.items() if k != "responses_create_params"}
    return {
        "task_id": {"taskset": taskset, "task_id": row["task_id"]},
        "task_input": {"responses_create_params": params, "task_data": task_data},
    }


def _parse_per_stack(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    counts = {}
    for part in value.split(","):
        stack, _, n = part.partition("=")
        if stack not in STACKS or not n.isdigit():
            raise SystemExit(f"--per-stack: bad entry {part!r}")
        counts[stack] = int(n)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.environ.get("UI_GEN_SOURCE_JSONL"), help="source JSONL")
    ap.add_argument(
        "--output", default=os.environ.get("UI_GEN_PREPARED_JSONL"), help="output JSONL (outside the repo)"
    )
    ap.add_argument("--stacks", default=",".join(STACKS), help="comma-separated stacks to keep")
    ap.add_argument("--per-stack", default=None, help="e.g. ui-next=2,ui-vite=2,ui-static=1 (after shuffling)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--ids-file", type=Path, default=None, help="keep only these source ids (one per line)")
    ap.add_argument("--exclude-ids-file", type=Path, default=None, help="drop these source ids (one per line)")
    ap.add_argument("--materialize-taskset", default=None, help="write task envelopes for this taskset")
    ap.add_argument("--prompt-template", type=Path, default=DEFAULT_TEMPLATE)
    args = ap.parse_args()
    if not args.input or not args.output:
        ap.error("--input and --output are required (or UI_GEN_SOURCE_JSONL / UI_GEN_PREPARED_JSONL)")

    template = args.prompt_template.read_text()
    keep_stacks = set(args.stacks.split(","))
    keep_ids = set(args.ids_file.read_text().split()) if args.ids_file else None
    drop_ids = set(args.exclude_ids_file.read_text().split()) if args.exclude_ids_file else set()
    _render(template, dict.fromkeys(TEMPLATE_KEYS, ""))  # a template error stops the run before any row
    rows, seen, skipped = [], set(), 0
    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = convert(json.loads(line), line_no, template)
                # The task id holds the stack, so one source id may appear once per stack.
                if row["task_id"] in seen:
                    raise RowError(f"line {line_no}: duplicate id {row['source_id']!r} for stack {row['stack']}")
            except (RowError, json.JSONDecodeError) as error:
                print(f"skipping line {line_no}: {error}", file=sys.stderr)
                skipped += 1
                continue
            seen.add(row["task_id"])
            if row["stack"] not in keep_stacks or row["source_id"] in drop_ids:
                continue
            if keep_ids is not None and row["source_id"] not in keep_ids:
                continue
            rows.append(row)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(rows)
    per_stack = _parse_per_stack(args.per_stack)
    if per_stack is not None:
        taken: dict[str, int] = {}
        picked = []
        for row in rows:
            if taken.get(row["stack"], 0) < per_stack.get(row["stack"], 0):
                taken[row["stack"]] = taken.get(row["stack"], 0) + 1
                picked.append(row)
        rows = picked
    if args.limit is not None:
        rows = rows[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with open(partial, "w", encoding="utf-8") as f:
        for row in rows:
            record = materialize(row, args.materialize_taskset) if args.materialize_taskset else row
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    partial.replace(out)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stack"]] = counts.get(row["stack"], 0) + 1
    manifest = {
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "rows": len(rows),
        "skipped_rows": skipped,
        "per_stack": counts,
        "materialized_taskset": args.materialize_taskset,
        "shuffle_seed": args.shuffle_seed,
    }
    out.with_name(out.name + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} rows {counts} -> {out} (skipped {skipped} invalid rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
