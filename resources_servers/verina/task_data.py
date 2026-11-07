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
# limitations under the License.
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class VerinaTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    args: list[str] = Field(description="Lean terms for the function's arguments, each with a type ascription.")
    expected: str = Field(description="Lean term the function's result must equal (compared with `==`).")


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="Verina task identifier, e.g. `verina_basic_1`.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    lean_template: str = Field(
        description=(
            "The task's Lean file with the solution regions (`code_aux`, `code`, `proof_aux`, `proof`) emptied and "
            "their `-- !benchmark` markers kept; the verifier fills them with the model's answer."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    function_name: str = Field(
        description="Name of the function the model implements; the tests call it.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    theorem_name: str = Field(
        description="Name of the specification theorem whose axioms decide whether the proof is complete.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    tests: list[VerinaTest] = Field(
        description="Hidden test cases from Verina, with the precondition already satisfied.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    difficulty: Optional[Literal["basic", "advanced"]] = Field(
        default=None,
        description="Verina subset the task comes from.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
