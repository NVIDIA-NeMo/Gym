# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the canonical NOOA capability example.

The resources server verifies ``expected_result``. The remaining fields define the
copied capability input and immutable provenance used by native-vs-Gym parity tests.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_result: Any = Field(
        description="Expected typed result compared by the exact-match verifier.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id: str = Field(
        description="Stable source capability case identifier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    a: int = Field(
        description="First operand supplied to the copied calculate capability.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    b: int = Field(
        description="Second operand supplied to the copied calculate capability.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    calculation: str = Field(
        description="Natural-language calculation instruction mapped to the NOOA entrypoint.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    capability_metadata: dict[str, Any] = Field(
        description="Immutable source-manifest and case-index provenance.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
