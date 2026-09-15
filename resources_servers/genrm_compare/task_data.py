# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the genrm_compare server.

Task-owned data contains the optional provenance label ``dataset``. Multi-member
verification also requires caller-owned ``_ng_group_id`` and ``_ng_rollout_index``
coordinates. The group ID must distinguish runs and prompt occurrences; rollout
indices are local group slots 0..N-1. A shared ``_ng_group_attempt`` distinguishes
replacement groups and defaults to zero. Prompt/task labels alone cannot isolate
independent runs. See the GenRM cohort guide for the wire contract and migration.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset: Optional[str] = Field(
        default=None,
        description=(
            "Source-dataset label (e.g. 'hs3'). Never read by any server code; passes through "
            "the wire only because GenRMCompareVerifyRequest sets extra='allow'."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
