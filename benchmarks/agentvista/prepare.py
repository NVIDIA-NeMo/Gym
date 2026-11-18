# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare hash-bound AgentVista rows without any tool/model calls."""

import os
from pathlib import Path


def prepare(output_fpath: str | Path | None = None, **kwargs: object) -> Path:
    import json

    from responses_api_agents.agentvista_agent.runtime.data import dataset_rows, inventory

    tasks = inventory(
        os.environ["AGENTVISTA_DATASET"],
        os.environ["AGENTVISTA_IMAGES"],
        os.environ.get("AGENTVISTA_TASKS", "all"),
    )
    destination = Path(output_fpath or Path(__file__).parent / "data/agentvista_benchmark.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dataset_rows(tasks)))
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    prepare()
