# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a JSONL file of UI generation tasks into ui_generation rows.

Each input line is one task in one of two shapes:

    {"id": "task-1", "prompt": "Build a ...", "reference_answer": "```html\\n...\\n```", "category": "html"}

    {"id": "task-2", "category": "react",
     "messages": [{"role": "user", "content": "Build a ..."},
                  {"role": "assistant", "content": "```tsx\\n...\\n```"}]}

- ``prompt`` (string) is the user request. Alternatively ``messages`` (a list of ``{"role", "content"}``
  with roles ``system``, ``user`` or ``assistant`` and nonempty string content) holds the conversation. When
  it ends with an assistant message, that message is the reference answer and leaves the prompt; otherwise
  ``reference_answer`` must be given. A task with both is refused, so an answer never stays in the policy's
  prompt.
- ``reference_answer`` (string) is a good answer to the task, usually Markdown with fenced code. The judge
  uses it as a guideline; it is never shown to the policy.
- ``id`` (string, optional) must be unique when present; it becomes the row's top-level ``task_id``, the
  stable task identity Gym uses for resume and for comparing runs (rows without one get positional ids). ``category`` (string, optional) labels the task,
  for example by stack; metrics report mean scores per category. ``--id-field`` and ``--category-field``
  read them from other fields, including nested ones (``meta.stack``).

Usage:
    python environments/ui_generation/prepare.py --input tasks.jsonl --output resources_servers/ui_generation/data/train.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional


ROLES = ("system", "user", "assistant")


class TaskFormatError(ValueError):
    """An input line that is not a usable task."""


def _field(row: dict, dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def convert_task(row: dict, line_number: int, *, id_field: str = "id", category_field: str = "category") -> dict:
    """One Gym row: the prompt messages as input, the reference answer and labels as verifier metadata."""
    if ("prompt" in row) == ("messages" in row):
        raise TaskFormatError(f"line {line_number}: give exactly one of 'prompt' or 'messages'")
    reference = row.get("reference_answer")
    if "prompt" in row:
        messages = [{"role": "user", "content": row["prompt"]}]
    else:
        messages = row["messages"]
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise TaskFormatError(f"line {line_number}: 'messages' must be a list of objects")
        messages = [{"role": m.get("role"), "content": m.get("content")} for m in messages]
        if messages and messages[-1]["role"] == "assistant":
            if reference is not None:
                raise TaskFormatError(f"line {line_number}: final assistant message conflicts with 'reference_answer'")
            reference = messages.pop()["content"]
    if not all(m["role"] in ROLES for m in messages):
        raise TaskFormatError(f"line {line_number}: message roles must be one of {ROLES}")
    if not any(m["role"] == "user" for m in messages):
        raise TaskFormatError(f"line {line_number}: the prompt needs a user message")
    if not all(isinstance(m["content"], str) and m["content"].strip() for m in messages):
        raise TaskFormatError(f"line {line_number}: prompt messages must have nonempty string content")
    if not isinstance(reference, str) or not reference.strip():
        raise TaskFormatError(f"line {line_number}: missing reference answer")

    labels: dict[str, str] = {}
    for key, path in (("id", id_field), ("category", category_field)):
        value = _field(row, path)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise TaskFormatError(f"line {line_number}: '{path}' must be a nonempty string")
            labels[key] = value
    metadata: dict[str, Any] = {"reference_answer": reference}
    if "category" in labels:
        metadata["category"] = labels["category"]
    task_id = {"task_id": labels["id"]} if "id" in labels else {}
    return task_id | {"responses_create_params": {"input": messages}, "verifier_metadata": metadata}


def prepare(
    input_path: Path,
    output_path: Path,
    *,
    limit: Optional[int] = None,
    id_field: str = "id",
    category_field: str = "category",
) -> Path:
    """Convert ``input_path`` into ``output_path`` and return it; refuses duplicate ids and unusable lines."""
    rows, seen = [], set()
    with open(input_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break
            try:
                task = json.loads(line)
            except json.JSONDecodeError as error:
                raise TaskFormatError(f"line {line_number}: invalid JSON ({error.msg})") from error
            if not isinstance(task, dict):
                raise TaskFormatError(f"line {line_number}: expected a JSON object")
            row = convert_task(task, line_number, id_field=id_field, category_field=category_field)
            task_id = row.get("task_id")
            if task_id is not None:
                if task_id in seen:
                    raise TaskFormatError(f"line {line_number}: duplicate id {task_id!r}")
                seen.add(task_id)
            rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    categories = Counter(row["verifier_metadata"].get("category", "(none)") for row in rows)
    print(f"Wrote {len(rows)} rows to {output_path}; categories: {dict(categories)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="JSONL of tasks in the format above")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the ui_generation rows")
    parser.add_argument("--limit", type=int, help="Only convert the first N tasks")
    parser.add_argument("--id-field", default="id", help="Field (dotted for nested) holding the task id")
    parser.add_argument("--category-field", default="category", help="Field (dotted for nested) holding the label")
    args = parser.parse_args()
    prepare(args.input, args.output, limit=args.limit, id_field=args.id_field, category_field=args.category_field)


if __name__ == "__main__":
    main()
