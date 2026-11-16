# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless upstream-row adapters for Gym web benchmark JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from nemo_gym.web.evaluation_collision import build_collision_plans
from nemo_gym.web.models import (
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


def gym_row(task: WebTask) -> dict[str, Any]:
    """Wrap a model-independent task in the request shape consumed by WebAgent.

    The policy adapter owns instructions, tools, image folding, and output
    parsing. Keeping those fields out of prepared data lets the exact same
    immutable 552-task population run with Nano Omni, Qwen, or another policy.
    """

    return {
        "responses_create_params": {
            "input": [],
            "metadata": {
                "benchmark": task.benchmark.value,
                "task_id": task.task_id,
            },
        },
        "web_task": task.model_dump(mode="json"),
    }


def _string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item]


def _webarena_task_id(record: Mapping[str, Any]) -> Any:
    source_id = record.get("id", record.get("task_id"))
    if source_id is None:
        raise ValueError("WebArena record requires id or task_id")
    if isinstance(source_id, str) and source_id.startswith("webarena-"):
        suffix = source_id.removeprefix("webarena-")
        if suffix.isdigit():
            return int(suffix)
    return source_id


def adapt_webarena_record(
    record: Mapping[str, Any],
    *,
    collision_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt one maintained WebArena row to the shared visual-browser contract."""

    task = WebTask(
        benchmark=WebBenchmark.WEBARENA,
        task_id=_webarena_task_id(record),
        intent=str(record.get("ques") or record.get("intent") or ""),
        start_urls=_start_urls(record.get("web") or record.get("start_url")),
        sites=_string_list(record.get("web_name") or record.get("sites")),
        input_images=_string_list(record.get("image") or record.get("images")),
        observation_profile=WebObservationProfile.SCREENSHOT,
        verifier_profile="webarena_classic",
        task_kwargs={
            "collision_plan": dict(collision_plan) if collision_plan is not None else {},
        },
        original_metadata=dict(record),
    )
    return gym_row(task)


def adapt_webarena_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Adapt a selection while planning protection for shared mutable targets."""

    source_records = [dict(record) for record in records]
    collision_plans = build_collision_plans(source_records)
    return [
        adapt_webarena_record(record, collision_plan=plan)
        for record, plan in zip(source_records, collision_plans, strict=True)
    ]


def adapt_webvoyager_record(record: Mapping[str, Any]) -> dict[str, Any]:
    source_id = record.get("id", record.get("task_id"))
    if source_id is None:
        raise ValueError("WebVoyager record requires id or task_id")
    site_value = record.get("web_name") or record.get("sites") or []
    sites = [site_value] if isinstance(site_value, str) else [str(site) for site in site_value if site]
    image_value = record.get("image") or record.get("images") or []
    input_images = [image_value] if isinstance(image_value, str) else [str(image) for image in image_value if image]
    task = WebTask(
        benchmark=WebBenchmark.WEBVOYAGER,
        task_id=source_id,
        intent=str(record.get("ques") or record.get("intent") or ""),
        start_urls=_start_urls(record.get("web") or record.get("start_url")),
        sites=sites,
        input_images=input_images,
        observation_profile=WebObservationProfile.SCREENSHOT,
        verifier_profile="webvoyager_gemini",
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
