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
"""Host adapter for the reflection workflow executed inside its task sandbox."""

from typing import ClassVar

from pydantic import Field

from responses_api_agents.langgraph_agent.app import LangGraphAgentAdapter, LangGraphAgentConfig


class ReflectionAgentConfig(LangGraphAgentConfig):
    max_reflections: int = Field(default=2, ge=0)


class ReflectionAgent(LangGraphAgentAdapter):
    config: ReflectionAgentConfig
    workflow: ClassVar[str] = "reflection"
    calls_model_from_sandbox: ClassVar[bool] = True


if __name__ == "__main__":
    ReflectionAgent.run_webserver()
