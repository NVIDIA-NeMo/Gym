# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the text_to_sql server.

Text-to-SQL generation graded by an LLM judge (with an optional swapped second pass). All task
fields ride at the row top level and are already typed on the wire (``TextToSqlRunRequest`` at
app.py, ``extra="allow"``): ``sql``, ``sql_dialect`` and ``sql_prompt`` are wire-required,
``sql_context`` defaults to ``""``, and ``uuid``/``metadata`` are optional pass-through. verify()
normalizes ``sql_dialect`` through alias mapping (postgres/pg -> postgresql, sqlite3 -> sqlite)
and raises for anything that does not resolve to mysql/postgresql/sqlite.
"""

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    sql: str = Field(
        description="Ground-truth SQL query the judge compares the model's extracted SQL against.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    sql_dialect: str = Field(
        description=(
            "SQL dialect; must normalize to mysql/postgresql/sqlite (aliases postgres, pg, sqlite3 accepted), "
            "otherwise verify() raises ValueError. Embedded in the judge prompt."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    sql_prompt: str = Field(
        description="Natural-language question the SQL answers; embedded in the judge prompt.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    sql_context: str = Field(
        default="",
        description="Database schema DDL (CREATE/INSERT statements) embedded in the judge prompt.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description="Task identifier (str in committed data, e.g. 'weather-station-001'); echoed in the verify response.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Free-form pass-through metadata echoed in the verify response; absent from committed data.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
