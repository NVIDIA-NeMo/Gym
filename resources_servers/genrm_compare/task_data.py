# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the genrm_compare server.

Rows carry no verify-consumed task data at all: verify() is a cohort-buffered pairwise GenRM
comparison driven entirely by ``responses_create_params.input`` (framework key), the rollout
``response``, and server config. Every verify-specific wire field (``principle``,
``task_index``/``rollout_index`` via the ``_ng_*`` aliases, ``prompt_id``) is injected by the
agent/harness at verify time, never read from dataset rows, so those stay declared on
``GenRMCompareVerifyRequest`` in app.py only. The single task-owned column in committed data is
``dataset``, a provenance label that survives transit purely via the wire model's
``extra="allow"``.
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
