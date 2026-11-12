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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One design task: a suite name and one target from it. The suite file sets scoring, gates and budget."""

    model_config = ConfigDict(extra="allow")

    suite: str = Field(
        description="A file stem under suites/, for example rnainvbench_eterna100_v1.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    target: dict[str, Any] = Field(
        description="The suite's own target block: name, length, target_structure and any constraints.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
