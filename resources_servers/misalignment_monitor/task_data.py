# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """A recorded rollout replayed through the monitor (``gym eval reverify``). The rest of the row is the rollout."""

    model_config = ConfigDict(extra="allow")

    # A human label: what the monitor should find. It feeds recall and false-positive metrics, never the judge.
    expected_severity: Optional[Literal["P0", "P1", "P2", "none"]] = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"]}
    )
