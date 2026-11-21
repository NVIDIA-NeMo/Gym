# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OCSR row fields: the gold SMILES the verifier scores against, plus the row's identity and label source."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    gold_smiles: str = Field(
        min_length=1,
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify"]},
    )
    id: str | None = Field(
        default=None,
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["provenance"]},
    )
    source_of_label: str | None = Field(
        default=None,
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["provenance"]},
    )
