# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the example_multi_step server.

Ground truth rides as top-level row fields (there is no verifier_metadata). All five fields are
required because the server's run/verify request models require them on the wire.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int = Field(json_schema_extra={"consumed_by": ["provenance"]})
    expected_synonyms: List[str] = Field(json_schema_extra={"consumed_by": ["provenance"]})
    expected_synonym_values: List[int] = Field(json_schema_extra={"consumed_by": ["verify"]})
    minefield_label: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    minefield_label_value: int = Field(json_schema_extra={"consumed_by": ["verify"]})
