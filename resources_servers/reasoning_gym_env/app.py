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

"""reasoning_gym reimplemented as an Env.

Single-step: step() always returns terminated=True after the first model response.

verifier_metadata fields:
  question: str
  answer: Optional[str]
  metadata: dict  (must contain source_dataset key)
"""

import re
from typing import Optional

import reasoning_gym
from reasoning_gym.utils import extract_answer

from resources_servers.gymnasium import GymnasiumServer
from nemo_gym.openai_utils import NeMoGymResponse


def _extract_answer_from_response(response: NeMoGymResponse) -> str:
    parts = []
    for item in response.output:
        if item.type != "message":
            continue
        content = item.content
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(c.text for c in content if c.type == "output_text")

    text = "".join(parts)
    extracted = extract_answer(text, tag_name="answer")
    if extracted is not None:
        return extracted
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    return text.strip()


class ReasoningGymEnv(GymnasiumServer):
    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        model_answer = _extract_answer_from_response(action)
        task_name = metadata.get("metadata", {}).get("source_dataset")
        if not task_name:
            return None, 0.0, True, False, {}
        entry = {
            "question": metadata.get("question"),
            "answer": metadata.get("answer"),
            "metadata": metadata.get("metadata", {}),
        }
        try:
            score = float(reasoning_gym.get_score_answer_fn(task_name)(answer=model_answer, entry=entry))
        except Exception:
            score = 0.0
        return None, score, True, False, {"extracted_answer": model_answer, "task_name": task_name}


if __name__ == "__main__":
    ReasoningGymEnv.run_webserver()
