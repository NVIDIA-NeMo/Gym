# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hash-bind original task records and every input image to prepared Gym rows."""

import hashlib
import json
from pathlib import Path

from responses_api_agents.agentvista_agent.runtime import HARNESS_VERSION


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inventory(dataset: str | Path, image_folder: str | Path, selector: str = "all") -> dict:
    path, images = Path(dataset).resolve(strict=True), Path(image_folder).resolve(strict=True)
    text = path.read_text()
    records = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if path.suffix == ".jsonl"
        else json.loads(text)
    )
    if not isinstance(records, list):
        raise ValueError("Dataset must contain a list of task records")
    tasks = {}
    for index, record in enumerate(records):
        task_id = str(record.get("doc_id", record.get("question_id", f"sample_{index}")))
        if task_id in tasks:
            raise ValueError(f"Duplicate task ID: {task_id}")
        question = record.get("problem", record.get("question"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question: {task_id}")
        paths, hashes = [], {}
        for name in record.get("images", []):
            image = (images / name).resolve(strict=True)
            if not image.is_relative_to(images) or not image.is_file():
                raise ValueError(f"Image outside dataset: {name}")
            paths.append(str(image))
            hashes[name] = digest(image.read_bytes())
        binding = digest(json.dumps({"record": record, "images": hashes}, sort_keys=True, ensure_ascii=False).encode())
        tasks[task_id] = {
            "record": record,
            "question": question,
            "images": paths,
            "source_sha256": binding,
        }
    selected = list(tasks) if selector == "all" else selector.replace(",", " ").split()
    if not selected or len(selected) != len(set(selected)) or set(selected) - tasks.keys():
        raise ValueError("Empty, duplicate or unknown task selection")
    return {key: tasks[key] for key in selected}


def dataset_rows(tasks: dict) -> list[dict]:
    rows = [
        {
            "responses_create_params": {"input": [{"role": "user", "content": task["question"]}]},
            "verifier_metadata": {
                "task_id": task_id,
                "source_sha256": task["source_sha256"],
                "category_id": str(task["record"].get("category", "agentvista")),
                "harness_version": HARNESS_VERSION,
            },
        }
        for task_id, task in tasks.items()
    ]

    for row in rows:
        row["agent_context"] = {"agent": {"values": {"verifier_metadata": dict(row["verifier_metadata"])}}}
    return rows
