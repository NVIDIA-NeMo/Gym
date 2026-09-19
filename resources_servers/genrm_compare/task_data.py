# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the genrm_compare server.

Task-owned data contains the optional provenance label ``dataset``. Multi-member
verification requires ``_ng_rollout_index`` slots 0..N-1. Caller-owned
``_ng_group_id`` and shared ``_ng_group_attempt`` coordinates support isolated
replacement attempts and completed reward replay. Legacy task/prompt grouping
remains available for sequential runs, without reliable late-request isolation.
See the GenRM cohort guide for the wire contract and retention limits.
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
