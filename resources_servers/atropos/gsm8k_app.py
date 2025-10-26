# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from pathlib import Path
from typing import Any, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

answer_parse_cfg = LatexExtractionConfig(
    normalization_config=NormalizationConfig(
        nits=False,
        malformed_operators=False,
        basic_latex=True,
        equations=True,
        boxed="all",
        units=True,
    ),
    boxed_match_priority=0,
    try_extract_without_anchor=False,
)

atropos_path = Path(__file__).parent.parent.parent / "atropos"
if str(atropos_path) not in sys.path:
    sys.path.insert(0, str(atropos_path))

from environments.gsm8k_server import GSM8kEnv, system_prompt

from resources_servers.atropos.app import AtroposResourcesServer


class GSM8kAtroposServer(AtroposResourcesServer[GSM8kEnv]):
    system_prompt: Optional[str] = system_prompt
    item_message_field: Optional[str] = "question"
    _shared_env: Optional[GSM8kEnv] = None

    async def env_factory(self, task_idx: int) -> GSM8kEnv:
        if self._shared_env is None:
            env_config, server_configs = GSM8kEnv.config_init()
            self._shared_env = GSM8kEnv(
                config=env_config,
                server_configs=server_configs,
                slurm=False,
                testing=True,
            )
            await self._shared_env.setup()
        return self._shared_env

    async def score_response(
        self,
        env: GSM8kEnv,
        item: Any,
        response: str,
        messages: list,
    ) -> tuple[float, bool, Optional[dict]]:
        gold_answer = item["answer"].split("#")[-1].strip().replace(",", "")
        gold_parsed = parse(
            "\\boxed{" + gold_answer + "}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        response_after_think = response.split("</think>")[-1] if "</think>" in response else response
        answer_parsed = parse(
            response_after_think,
            extraction_config=[answer_parse_cfg],
            extraction_mode="first_match",
        )

        correct = verify(answer_parsed, gold_parsed) if gold_parsed else False
        reward = 1.0 if correct else 0.0

        return reward, True, {
            "correct": correct,
            "gold_answer": gold_answer,
            "gold_parsed": str(gold_parsed) if gold_parsed else None,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "response_after_think": response_after_think,
        }


if __name__ == "__main__":
    GSM8kAtroposServer.run_webserver()
