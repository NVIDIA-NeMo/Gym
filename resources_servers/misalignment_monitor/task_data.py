# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """A recorded rollout replayed through the monitor (``gym eval reverify``)."""

    model_config = ConfigDict(extra="allow")

    # The agent's recorded response, which the monitor reviews.
    response: dict[str, Any] = Field(json_schema_extra={"consumed_by": ["verify"]})
    # A human label: what the monitor should find. It feeds recall and false-positive metrics, never the judge.
    expected_severity: Optional[Literal["P0", "P1", "P2", "none"]] = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"]}
    )
    # The recorded task's own reward and name, kept as evidence.
    reward: Optional[float] = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
    task_name: Optional[str] = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
