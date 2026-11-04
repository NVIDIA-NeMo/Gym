# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the spider2_lite server.

All task fields are top-level (no ``verifier_metadata``). Two dataset variants exist: the
committed ``data/example.jsonl`` carries ``gold_sql`` (verify() executes it against the
local SQLite database named by ``db_id``), while ``scripts/prepare_dataset.py`` also emits an
uncommitted validation file whose rows carry ``gold_result`` (a list of acceptable result sets)
instead. ``verify()`` branches on whichever is present and raises when both are absent. There is
no in-band discriminator field, so this is a single model with both golds Optional, mirroring
``Spider2LiteVerifyRequest`` in app.py (``extra="allow"``). ``uuid`` and ``metadata`` are
accepted by the wire but never emitted by the prepare script.

Rows may set ``engine: bigquery`` to opt into BigQuery execution against a Google Cloud project
read from the user's Application Default Credentials file. The default ``engine: sqlite`` keeps
all existing rows on the current local SQLite path.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description="Optional row identifier; accepted by the wire but absent from committed data.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    instance_id: Optional[str] = Field(
        default=None,
        description="Spider2-lite instance identifier (e.g. 'local022').",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    db_id: str = Field(
        description=(
            "Name of the database the query runs against (e.g. 'IPL' for SQLite, "
            "or a BigQuery dataset/project reference for engine='bigquery')."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    question: str = Field(
        description="Natural-language question; wire-required, echoed into the verify response.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    engine: Literal["sqlite", "bigquery"] = Field(
        default="sqlite",
        description=(
            "Execution engine for the predicted SQL. 'sqlite' (default) runs against the "
            "local Spider 2.0 Lite SQLite databases. 'bigquery' submits a query job to the "
            "user's Google Cloud project (BigQuery Job User required)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    gold_sql: Optional[str] = Field(
        default=None,
        description="Reference SQL executed to produce the gold result (variant 1; committed example.jsonl).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    gold_result: Optional[List[List[List[Any]]]] = Field(
        default=None,
        description=(
            "List of acceptable result sets, each a list of rows (variant 2; uncommitted validation file). "
            "verify() requires gold_sql or gold_result."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    ignore_order: bool = Field(
        default=True,
        description="Whether row order is ignored when comparing predicted vs gold results.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    condition_cols: Optional[List[Any]] = Field(
        default=None,
        description="Column indices to restrict result comparison to; empty list in all committed rows.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Untyped passthrough; accepted by the wire but absent from committed data.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
