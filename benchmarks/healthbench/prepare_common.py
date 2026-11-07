# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared preparation utilities for HealthBench-style rubric benchmarks."""

from __future__ import annotations

import json
import math
import uuid
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_: Any):
        return iterable


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected object on {path}:{line_number}, got {type(record).__name__}")
            yield record


def records_from_zip(path: Path, source_filename: str) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == source_filename]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {source_filename} in {path}, found {len(matches)}")
        with archive.open(matches[0]) as stream:
            for line in stream:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("Expected a JSON object in source archive")
                yield record


def iter_local_records(data_dir: str | Path, source_filename: str) -> Iterator[dict[str, Any]] | None:
    source = Path(data_dir).expanduser()
    if not source.exists():
        return None

    if source.is_file() and source.suffix == ".zip":
        return records_from_zip(source, source_filename)
    if source.is_file():
        return load_jsonl(source)

    candidate = source / source_filename
    if not candidate.exists():
        matches = list(source.rglob(source_filename))
        if not matches:
            raise FileNotFoundError(f"{source_filename} not found under {source}")
        candidate = matches[0]
    return load_jsonl(candidate)


def normalize_conversation(record: dict[str, Any]) -> list[dict[str, str]]:
    conversation = record.get("conversation") or record.get("messages") or record.get("prompt")
    if isinstance(conversation, dict):
        conversation = conversation.get("messages")
    if not isinstance(conversation, list):
        raise TypeError("HealthBench row is missing a conversation/messages/prompt list")

    messages: list[dict[str, str]] = []
    for index, message in enumerate(conversation):
        if not isinstance(message, dict):
            raise TypeError(f"conversation[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"conversation[{index}] has unsupported role {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"conversation[{index}].content must be a string")
        messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Conversation must end with a user message")
    return messages


def normalize_rubrics(record: dict[str, Any]) -> list[dict[str, Any]]:
    rubrics = record.get("rubrics") or record.get("rubric_items") or record.get("criteria")
    if not isinstance(rubrics, list) or not rubrics:
        raise TypeError("HealthBench row is missing non-empty rubrics/rubric_items/criteria")

    normalized = []
    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, dict):
            raise TypeError(f"rubrics[{index}] must be an object")
        criterion = rubric.get("criterion") or rubric.get("criterion_text") or rubric.get("rubric")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"rubrics[{index}] is missing criterion text")
        points = rubric.get("points")
        if points is None:
            points = rubric.get("score")
        normalized_rubric = {
            key: value
            for key, value in rubric.items()
            if key not in {"criterion", "criterion_text", "rubric", "points", "score"}
        }
        if isinstance(points, bool) or not isinstance(points, (int, float)) or not math.isfinite(points):
            raise ValueError(f"rubrics[{index}].points must be finite numeric points")
        normalized_rubric.update({"criterion": criterion, "points": points})
        normalized.append(normalized_rubric)
    if not any(rubric["points"] > 0 for rubric in normalized):
        raise ValueError("Rubrics have no positive points")
    return normalized


def to_gym_row(record: dict[str, Any], row_number: int, *, benchmark_name: str) -> dict[str, Any]:
    source_id = str(record.get("id") or record.get("prompt_id") or f"row-{row_number}")
    conversation = normalize_conversation(record)
    rubrics = normalize_rubrics(record)
    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "conversation",
            "messages",
            "prompt",
            "rubrics",
            "rubric_items",
            "criteria",
        }
    }
    metadata["source_id"] = source_id
    metadata["benchmark_name"] = benchmark_name

    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{benchmark_name}/{source_id}")),
        "source_id": source_id,
        "conversation": conversation,
        "rubrics": rubrics,
        "metadata": metadata,
        "responses_create_params": {
            "input": conversation,
            "parallel_tool_calls": False,
            "tools": [],
        },
    }


def write_gym_jsonl(
    records: Iterable[dict[str, Any]],
    *,
    output: Path,
    benchmark_name: str,
    limit: int | None = None,
) -> Path:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_fpath = output.with_suffix(".jsonl.tmp")
    count = 0
    with tmp_fpath.open("w", encoding="utf-8") as stream:
        for row_number, record in enumerate(tqdm(records), 1):
            if limit is not None and count >= limit:
                break
            stream.write(
                json.dumps(to_gym_row(record, row_number, benchmark_name=benchmark_name), ensure_ascii=False) + "\n"
            )
            count += 1

    tmp_fpath.replace(output)
    print(f"Wrote {count} {benchmark_name} examples to {output}")
    return output
