# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the decision_index server.

Each row carries one Decision Index 0.3 request (``di_row``: the kit's row with its questions, options, expected
answers and scoring spec) inside ``verifier_metadata``. The verifier scores the policy's answer against it with the
kit's official scorers.
"""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    di_row: Dict[str, Any] = Field(
        description="The Decision Index request: id, benchmark, questions with options, expected answers, scoring.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
