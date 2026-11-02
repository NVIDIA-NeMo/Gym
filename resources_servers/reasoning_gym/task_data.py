# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the reasoning_gym server.

Task fields ride at the row top level; there is no verifier_metadata. Mirrors
``ReasoningGymVerifyRequest`` (app.py): all three fields are wire-required (``answer`` is
required-but-nullable).

Deliberately OPEN on ``metadata``: rows are heterogeneous per source dataset —
scripts/create_dataset.py emits whatever the external ``reasoning_gym`` package puts in
``entry["metadata"]`` for any of its ~100 task datasets (composite datasets mix families within
one file), and verify() forwards the whole dict opaquely to the dataset-specific score function.
Only ``metadata["source_dataset"]`` is a stable key (verify() raises ValueError without it — it
selects the scorer), so the payload stays ``Dict[str, Any]``; do not chase closure on the
per-dataset shapes.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(
        description=(
            "Task question text, rebuilt into the entry dict handed to the reasoning_gym score "
            "function (and the prompt source at dataset-creation time)."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    answer: Optional[str] = Field(
        description=(
            "Gold answer string for the entry dict; nullable because some reasoning_gym datasets "
            "score procedurally from metadata alone. Wire-required (the key must be present, its "
            "value may be null)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Dict[str, Any] = Field(
        description=(
            "Opaque per-dataset payload forwarded whole to the reasoning_gym scorer. "
            "'source_dataset' (str) is the only stable key — it selects the score function and "
            "verify() raises ValueError if it is missing. Everything else varies by source dataset "
            "(e.g. knights_knaves rows carry source_index, statements, solution, names, "
            "knight_knave_terms, difficulty)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
