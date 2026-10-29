# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""Task-data schema for the chemcost server."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="The ChemCost reaction id, as the task identity.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    reaction_id: str = Field(
        description="ChemCost reaction id, e.g. CHEMCOST-0001.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    type: Literal["single_step", "multi_step"] = Field(
        description="A single-step reaction or a multi-step route.",
        json_schema_extra={"consumed_by": ["metrics"], **_VM},
    )
    procurement_cost_usd_per_g_product: Optional[float] = Field(
        default=None,
        description="Ground-truth procurement cost in USD per gram of product; a row without a positive one is masked.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
