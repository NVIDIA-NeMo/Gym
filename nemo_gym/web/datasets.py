# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless upstream-row adapters for Gym web benchmark JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from nemo_gym.web.models import (
    WebActionProfile,
    WebBenchmark,
    WebObservationProfile,
    WebTask,
)


def _start_urls(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(" |AND| ") if part.strip()]
    if isinstance(value, list):
        return [str(part) for part in value if part]
    return [str(value)]


def _auth_profile(record: Mapping[str, Any]) -> str | None:
    """Keep a real storage-state reference without stringifying JSON null."""

    storage_state = record.get("storage_state")
    if not record.get("require_login") or not storage_state:
        return None
    return str(storage_state)


def gym_row(task: WebTask) -> dict[str, Any]:
    """Wrap a normalized task in the standard request shape consumed by WebAgent."""

    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": task.intent}],
            "metadata": {
                "benchmark": task.benchmark.value,
                "task_id": task.task_id,
            },
        },
        "web_task": task.model_dump(mode="json"),
    }


def adapt_webarena_record(record: Mapping[str, Any]) -> dict[str, Any]:
    task = WebTask(
        benchmark=WebBenchmark.WEBARENA,
        task_id=record["task_id"],
        intent=str(record.get("intent") or ""),
        start_urls=_start_urls(record.get("start_url")),
        sites=[str(site) for site in record.get("sites") or []],
        observation_profile=WebObservationProfile.A11Y,
        verifier_profile="browsergym_webarena",
        auth_profile=_auth_profile(record),
        original_metadata=dict(record),
    )
    return gym_row(task)


def adapt_visualwebarena_records(
    partitions: Iterable[tuple[str, Iterable[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """Concatenate and globally re-index VWA partitions like libvisualwebarena.

    BrowserGym/libvisualwebarena 0.0.15 orders the official partitions as
    Classifieds, Reddit (including its cross-site tasks), then Shopping, and
    assigns the resulting 910 rows global task ids 0..909.
    """

    rows: list[dict[str, Any]] = []
    for partition, records in partitions:
        for record in records:
            original_metadata = dict(record)
            original_metadata["_source_partition"] = partition
            original_metadata["_source_task_id"] = record.get("task_id")
            task = WebTask(
                benchmark=WebBenchmark.VISUALWEBARENA,
                task_id=len(rows),
                intent=str(record.get("intent") or ""),
                start_urls=_start_urls(record.get("start_url")),
                sites=[str(site) for site in record.get("sites") or []],
                observation_profile=WebObservationProfile.SOM,
                verifier_profile="browsergym_visualwebarena",
                auth_profile=_auth_profile(record),
                original_metadata=original_metadata,
            )
            rows.append(gym_row(task))
    return rows


def adapt_webvoyager_record(record: Mapping[str, Any]) -> dict[str, Any]:
    task = WebTask(
        benchmark=WebBenchmark.WEBVOYAGER,
        task_id=record.get("id"),
        intent=str(record.get("ques") or record.get("intent") or ""),
        start_urls=_start_urls(record.get("web") or record.get("start_url")),
        sites=[str(record.get("web_name"))] if record.get("web_name") else [],
        observation_profile=WebObservationProfile.SOM,
        action_profile=WebActionProfile.WEBVOYAGER_LEGACY,
        verifier_profile="webvoyager_llm_judge",
        original_metadata=dict(record),
    )
    return gym_row(task)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{source} must contain a JSON array or JSONL stream of objects")
    return records


def write_jsonl(rows: Iterable[Mapping[str, Any]], output: str | Path) -> int:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
