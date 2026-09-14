# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the genrm_compare server.

Task-owned data contains the optional provenance label ``dataset``. Cohort verification
also requires caller-owned identity: a group ID, task index or prompt_id, and a member
slot. Callers supply group IDs, shared attempts and local member indices; the existing
rollout index is a fallback when it already identifies a slot within this group.
Anonymous prompt-only verification is unsupported because transport retries cannot be counted as new group members.
These wire fields remain on ``GenRMCompareVerifyRequest``; see the GenRM cohort guide
for the distinction between a global rollout index and a comparison-group slot.
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
