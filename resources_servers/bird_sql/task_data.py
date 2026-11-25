# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the bird_sql server.

All task data rides at the top level; there is no verifier_metadata. Mirrors
``BirdSqlVerifyRequest`` (app.py): ``question``/``gt_sql``/``db_id`` are wire-required,
``difficulty``/``id`` are wire-Optional. ``sql_context`` (a large CREATE TABLE DDL string used
only as prompt context) is UNDECLARED on today's wire model — it rides through
``extra="allow"`` and is echoed back — so it is declared here as an optional prompt field. So are the
fields only some prepared files carry: ``evidence_section`` (the schema-only prompt) and ``question_id``/
``bird_version`` (a named BIRD dev version, or the schema-only prompt); see benchmarks/birdbench/prepare.py.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="Stable upstream task identifier, when the benchmark provides one (BIRD: the source `question_id`).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: str = Field(
        description=(
            "Natural-language question the SQL must answer; wire-required but not read for grading — "
            "only echoed into the verify response."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    gt_sql: str = Field(
        description="Gold SQL whose execution result the extracted model SQL is compared against.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    db_id: str = Field(
        description="BIRD dev database name; resolves the per-database SQLite file both queries run on.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="BIRD difficulty label (e.g. 'simple', 'moderate', 'challenging'); used for subset metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    id: Optional[int] = Field(
        default=None,
        description="Numeric BIRD task id; logging and echo only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    sql_context: Optional[str] = Field(
        default=None,
        description=(
            "Database schema DDL (CREATE TABLE statements, can be ~14KB) shown to the model as prompt "
            "context. Not read by verify(); undeclared on today's wire model (flows via extra='allow')."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    evidence_section: Optional[str] = Field(
        default=None,
        description=(
            "Schema-only prompt only (prepare_script_args.prompt_format=schema_only): BIRD's evidence as a "
            "'### Evidence' section, or '' when the question has none. Not read by verify()."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    question_id: Optional[int] = Field(
        default=None,
        description=(
            "BIRD's own question id, on rows of a named BIRD dev version or the schema-only prompt; the same "
            "value as task_id, as an integer. Logging and echo only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    bird_version: Optional[str] = Field(
        default=None,
        description=(
            "Which dated release of BIRD dev the row comes from (e.g. 'dev_20251106'), on rows of a named version "
            "or the schema-only prompt. Logging and echo only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
