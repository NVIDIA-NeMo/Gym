# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained osworld_agent (no resources server).

Committed rows (benchmarks/osworld) nest everything inside ``verifier_metadata``, but the agent
accepts both placements (``metadata.get("osworld_task") or body.model_extra.get(...)`` in
app.py), so no field carries a ``legacy_location`` marker. ``osworld_task`` is the full OSWorld
task spec handed to the runner; ``task_id``/``domain`` label logs and provenance and fall back to
values inside the spec when absent.
"""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="OSWorld task UUID; log/provenance label (falls back to osworld_task.id).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    domain: str = Field(
        description="OSWorld task domain (e.g. 'chrome'); log/provenance label.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    osworld_task: Dict[str, Any] = Field(
        description=(
            "Full OSWorld task spec (id, snapshot, instruction, config, evaluator, ...) passed "
            "verbatim to the OSWorld runner."
        ),
        json_schema_extra={"consumed_by": ["run"]},
    )
