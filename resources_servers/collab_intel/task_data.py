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
from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    dimension: str = Field(description="Teamwork dimension of the scenario; the score averages these equally.")
    scenario_path: str = Field(
        description="Scenario folder (relative to the release dir) whose test version is scored."
    )
    gate_path: str = Field(description="Scenario folder whose alone version gates the row.")
    gate_seed: int = Field(default=0, description="Seed of the alone (gate) run.")
    seed: int = Field(default=0, description="Seed of the scored test run.")
