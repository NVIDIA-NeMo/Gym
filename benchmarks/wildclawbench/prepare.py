# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize selected tasks with pinned source and media bindings; no model calls."""

import json
import os
from pathlib import Path

from responses_api_agents.wildclawbench_agent.data import prepare_rows, selected_tasks


def prepare(output_fpath: str | Path | None = None, **kwargs: object) -> Path:
    output = Path(output_fpath or os.environ["WILDCLAW_DATASET_PATH"])
    tasks = selected_tasks(
        os.environ.get("WILDCLAW_TASKS", "all"),
        shard_index=int(os.environ.get("WILDCLAW_SHARD_INDEX", "0")),
        shard_count=int(os.environ.get("WILDCLAW_SHARD_COUNT", "1")),
    )
    rows = prepare_rows(Path(os.environ["WILDCLAW_ROOT"]), tasks)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(output)
    return output
