# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the Aegis v4 safety verifier."""

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_id: Optional[Union[str, int]] = Field(
        default=None,
        description="Stable source identifier echoed into rollout results.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Optional dataset or split name echoed into rollout results.",
        json_schema_extra={"consumed_by": ["metrics", "provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary source metadata that is not used to determine the Aegis verdict.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
