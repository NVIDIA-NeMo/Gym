# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
"""Turn a loaded suite into NeMo Gym JSONL rows.

Each row follows the NeMo Gym data contract: ``responses_create_params`` in OpenAI Responses
API shape (system + user messages, function tools; ``system`` rather than ``developer`` because DeepSeek, Gemini and other OpenAI-compatible providers reject the ``developer`` role), plus top-level fields the resources
server reads back at ``/seed_session`` and ``/verify`` (``task_id`` above all). Rows carry no
answers, verifiers, or seed data; those stay inside the task package on the server side.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .suite import Suite, TaskPackage


__all__ = ["DEFAULT_DEVELOPER_PROMPT", "build_row", "build_rows", "developer_prompt", "split_rows", "write_jsonl"]

DEFAULT_DEVELOPER_PROMPT = (
    "You are an employee operating your company's business systems through the tools provided. "
    "Complete the assignment the way a careful colleague would: investigate the connected systems "
    "before changing anything, change only what the assignment requires, read your writes back to "
    "confirm they persisted, and submit your final answer with the submit tool before you finish. "
    "Tool names use double underscores where system documentation shows dots "
    "(for example benchmark__submit_answer for benchmark.submit_answer)."
)

DATA_LICENSE = "CC-BY-4.0"


def _humanize(identifier: str) -> str:
    return identifier.replace("_", " ").replace("-", " ").strip()


def developer_prompt(task: TaskPackage, base: str | None = None) -> str:
    lead = ""
    role = _humanize(task.role) if task.role else None
    if role and task.company:
        lead = f"You are the {role} at {task.company}. "
    elif role:
        lead = f"You are the {role}. "
    elif task.company:
        lead = f"You work at {task.company}. "
    # Harbor runs expose the task identity through the container; here it has to travel in the
    # prompt, because tools such as benchmark.get_task and benchmark.submit_answer take it.
    tail = f" Your task id is {task.task_id}; pass it wherever a tool asks for a task identifier."
    return lead + (base or DEFAULT_DEVELOPER_PROMPT) + tail


def build_row(
    task: TaskPackage,
    suite_name: str,
    *,
    base_prompt: str | None = None,
    include_tools: bool = True,
    source_url: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "input": [
            {"role": "system", "content": developer_prompt(task, base_prompt)},
            {"role": "user", "content": task.instruction},
        ],
        "parallel_tool_calls": False,
    }
    if include_tools:
        params["tools"] = [tool.as_responses_tool() for tool in task.tools]
    row: dict[str, Any] = {
        "responses_create_params": params,
        "task_id": task.task_id,
        "suite": suite_name,
        "benchmark": task.benchmark,
        "world_id": task.world_id or f"{task.benchmark.lower()}-world",
        "license": DATA_LICENSE,
    }
    for key in ("category", "difficulty", "metric", "project_code"):
        if task.metadata.get(key) is not None:
            row[key] = task.metadata[key]
    if source_url:
        row["source"] = source_url
    return row


def build_rows(suite: Suite, **kwargs: Any) -> list[dict[str, Any]]:
    return [build_row(task, suite.name, **kwargs) for task in suite.tasks]


def split_rows(
    rows: Iterable[dict[str, Any]], *, validation_fraction: float = 0.1, salt: str = "blobfish"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic train/validation split keyed on the task id, stable across rebuilds."""
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in rows:
        digest = hashlib.sha256(f"{salt}:{row['task_id']}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        (validation if bucket < validation_fraction else train).append(row)
    return train, validation


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
