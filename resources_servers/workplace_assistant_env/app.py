# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""workplace_assistant reimplemented as an Env.

verifier_metadata fields:
  available_tools: List[str]  — toolkit names to load (required)
  ground_truth:   list        — expected function calls for scoring
"""

import json
from typing import Any, Dict, Optional

from pydantic import Field

from nemo_gym.envs import Env
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.workplace_assistant.utils import get_tools, is_correct


class WorkplaceAssistantEnv(Env):
    session_tools: Dict[str, Any] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        tool_list = metadata.get("available_tools") or [
            "email",
            "calendar",
            "analytics",
            "project_management",
            "customer_relationship_manager",
        ]
        self.session_tools[session_id] = get_tools(tool_list)
        return None, {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        tool_calls = [o for o in action.output if o.type == "function_call"]

        if tool_calls:
            tool_env = self.session_tools.get(session_id, {})
            results = []
            for call in tool_calls:
                args = {k: v for k, v in json.loads(call.arguments).items() if v is not None}
                try:
                    result = tool_env["functions"][call.name](**args)
                except Exception as e:
                    result = f"Error executing '{call.name}': {e}"
                results.append(f"{call.name} -> {result}")
            return "\n".join(results), 0.0, False, False, {}

        ground_truth = metadata.get("ground_truth", [])
        predicted = [m.model_dump() for m in action.output if m.type == "function_call"]
        return None, is_correct(predicted, ground_truth, None) * 1.0, True, False, {}


if __name__ == "__main__":
    WorkplaceAssistantEnv.run_webserver()
