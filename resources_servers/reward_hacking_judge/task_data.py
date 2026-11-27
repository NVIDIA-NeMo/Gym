# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """A saved rollout re-judged by MFN's reward-hacking judge (gym eval reverify)."""

    model_config = ConfigDict(extra="allow")

    # The agent's saved response, which the judge reads as its transcript.
    response: dict[str, Any] = Field(json_schema_extra={"consumed_by": ["verify"]})
    # The source run's reward for this rollout, shown to the judge; read from source_rollouts_jsonl_fpath when absent.
    source_reward: Optional[float] = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
