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

"""Scripted multi-turn GymnasiumServer example.

Replays follow-up messages from verifier_metadata. Turn index tracked per session_id.

verifier_metadata fields:
  follow_ups:      List[str]   follow-up messages sent after each assistant turn
  expected_answer: str         substring expected in the final assistant response
"""

from typing import Dict, Optional

from pydantic import Field

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer


def _extract_text(response: NeMoGymResponse) -> str:
    for item in response.output:
        if item.type == "message":
            content = item.content
            if isinstance(content, str):
                return content
            for c in content:
                if c.type == "output_text":
                    return c.text
    return ""


class ScriptedMultiTurnEnv(GymnasiumServer):
    session_turns: Dict[str, int] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        self.session_turns[session_id] = 0
        return None, {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        follow_ups = metadata.get("follow_ups", [])
        turn = self.session_turns.get(session_id, 0)

        if turn < len(follow_ups):
            self.session_turns[session_id] = turn + 1
            return follow_ups[turn], 0.0, False, False, {}

        expected = metadata.get("expected_answer", "")
        text = _extract_text(action)
        reward = 1.0 if expected and expected.lower() in text.lower() else 0.0
        return None, reward, True, False, {}


if __name__ == "__main__":
    ScriptedMultiTurnEnv.run_webserver()
