# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the beaver_sql server.

All task data rides at the top level; there is no verifier_metadata. Mirrors ``BeaverSqlVerifyRequest`` (app.py):
``question``/``gt_sql``/``db_id`` are wire-required, ``id``/``category`` wire-Optional, and undeclared fields ride
through ``extra="allow"`` as on the wire. The other fields are the ones ``benchmarks/beaver/prepare.py`` writes:
prompt inputs (``evidence_section``, ``sql_context``) and provenance. tests/test_task_data.py keeps the wire fields'
types and requiredness equal to the request model's.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="Stable task id: BEAVER's question id, which names its domain (dw_12, dw_real_12, nova_3).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: str = Field(
        description="Natural-language question the SQL must answer; shown in the prompt, echoed in the verify response.",
        json_schema_extra={"consumed_by": ["prompt", "provenance"]},
    )
    gt_sql: str = Field(
        description="Gold MySQL query whose result the model's query is compared against.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    db_id: str = Field(
        description="BEAVER MySQL database both queries run on: dw (dw and dw_real questions), nova or neutron.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id: Optional[str] = Field(
        default=None,
        description="BEAVER's question id as in the source data; logging and echo only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    category: Optional[str] = Field(
        default=None,
        description="BEAVER question category with spaces as underscores (e.g. 'complex_query'); subset metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    domain: Optional[str] = Field(
        default=None,
        description="BEAVER query split the question comes from: dw, dw_real, nova or neutron.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    detailed_category: Optional[str] = Field(
        default=None,
        description="BEAVER's finer question category, as in the source data.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    evidence: Optional[str] = Field(
        default=None,
        description="The question's domain knowledge, one item per line; empty when prepared without evidence.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    evidence_section: Optional[str] = Field(
        default=None,
        description="The evidence as the prompt shows it (a '### Evidence' section), or empty.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    sql_context: Optional[str] = Field(
        default=None,
        description="The database's CREATE TABLE statements, shown in the prompt. Not read by verify(), which drops it.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
