# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize official AgentDojo task selectors for the external agent adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BENCHMARK_VERSION = "v1.2.2"
SUITE_COUNTS = {
    "banking": (16, 9),
    "slack": (21, 5),
    "travel": (20, 7),
    "workspace": (40, 14),
}
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "agentdojo_benchmark.jsonl"


def _row(
    *,
    suite: str,
    user_task_id: str,
    injection_task_id: str | None = None,
) -> dict[str, Any]:
    attacked = injection_task_id is not None
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": (
                        f"AgentDojo {suite} {user_task_id}. The authoritative upstream task prompt is loaded "
                        "by the adapter."
                    ),
                }
            ],
        },
        "suite": suite,
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "attack": "important_instructions" if attacked else None,
        "defense": None,
        "benchmark_version": BENCHMARK_VERSION,
        "condition": "attacked" if attacked else "clean",
    }


def prepare() -> Path:
    rows: list[dict[str, Any]] = []
    for suite_name, (user_count, injection_count) in SUITE_COUNTS.items():
        for user_index in range(user_count):
            user_task_id = f"user_task_{user_index}"
            rows.append(_row(suite=suite_name, user_task_id=user_task_id))
            rows.extend(
                _row(
                    suite=suite_name,
                    user_task_id=user_task_id,
                    injection_task_id=f"injection_task_{injection_index}",
                )
                for injection_index in range(injection_count)
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} AgentDojo selectors to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
