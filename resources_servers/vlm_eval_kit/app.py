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
from asyncio import Event
from collections import defaultdict
from pathlib import Path
from subprocess import run
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class VlmEvalKitResourcesServerConfig(BaseResourcesServerConfig):
    pass


class VLMEvalKitVerifyRequest(BaseVerifyRequest):
    # We allow extra inputs here since there are many VLMEvalKit benchmarks that are run through the same resources server.
    model_config = ConfigDict(extra="allow")

    benchmark_name: str
    category: str
    answer: Any


class VLMEvalKitVerifyResponse(VLMEvalKitVerifyRequest, BaseVerifyResponse):
    pass


class Coordinator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rewards: List[int] = Field(default_factory=list)
    event: Event = Field(default_factory=Event)


class VlmEvalKitResourcesServer(SimpleResourcesServer):
    config: VlmEvalKitResourcesServerConfig

    MMBench_DEV_EN_V11_sets: Dict[str, Coordinator] = Field(default_factory=lambda: defaultdict(Coordinator))

    def setup_webserver(self):
        self.setup_VLMEvalKit()

        return super().setup_webserver()

    def setup_VLMEvalKit(self) -> None:
        this_dir = Path(__file__).parent.absolute()
        # We freeze the commit SHA for now.
        # We pip install with no-deps since we have the deps in the pyproject.toml already.
        setup_command = f"""cd {this_dir} \
&& source .venv/bin/activate \
&& if [ ! -d VLMEvalKit ]; then git clone https://github.com/open-compass/VLMEvalKit/; fi \
&& cd VLMEvalKit \
&& git checkout 00804217f868058f871f5ff252a7b9623c3475d9 \
&& uv pip install '-e .' --no-deps \
&& sed -i '' 's/import clip/# import clip/' vlmeval/dataset/utils/SArena/FID.py
"""
        print(f"Running VLMEvalKit setup command: {setup_command}")
        run(setup_command, shell=True, check=True)

        # Dummy import to load ahead of time
        import vlmeval.utils.matching_util

        vlmeval.utils.matching_util

    async def verify(self, body: VLMEvalKitVerifyRequest) -> VLMEvalKitVerifyResponse:
        score_fn = getattr(self, f"_score_{body.benchmark_name}")

        score_dict = await score_fn(body)

        return VLMEvalKitVerifyResponse(**body.model_dump(), **score_dict)

    # For each of the scoring functions, we copy it over in a nicer way since the original functions
    # couple together reading from an input file path, LLM as judge, etc. It's just easier to reimplement and test e2e accuracy.
    async def _score_OCRBench(self, body: BaseVerifyRequest) -> Dict[str, Any]:
        # Reformatted from https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/image_vqa.py#L505
        reward = 0.0

        predict = body.response.output_text
        answers = body.answer
        category = body.category
        if category == "Handwritten Mathematical Expression Recognition":
            for j in range(len(answers)):
                answer = answers[j].strip().replace("\n", " ").replace(" ", "")
                predict = predict.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    reward = 1.0
                    break
        else:
            for j in range(len(answers)):
                answer = answers[j].lower().strip().replace("\n", " ")
                predict = predict.lower().strip().replace("\n", " ")
                if answer in predict:
                    reward = 1.0
                    break

        return {f"OCRBench/{category}": reward, "OCRBench": reward, "reward": reward}

    async def _score_MMBench_DEV_EN_V11(self, body: BaseVerifyRequest) -> Dict[str, Any]:
        # Reformatted from https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/image_mcq.py#L294
        # Each example is run 4 times and we only output score 1 if all examples are correct.
        from vlmeval.utils.matching_util import can_infer

        predict = body.response.output_text
        answer = body.answer
        category = body.category

        # Choices looks like https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/utils/multiple_choice.py#L337
        prediction = can_infer(predict, body.choices)
        this_reward = int(prediction == answer)

        coordinator = self.MMBench_DEV_EN_V11_sets[body.group]
        coordinator.rewards.append(this_reward)
        if len(coordinator.rewards) == body.group_size:
            coordinator.rewards = [int(all(coordinator.rewards))]
            self.MMBench_DEV_EN_V11_sets.pop(body.group)
            coordinator.event.set()
        else:
            await coordinator.event.wait()

        # Just take the first one since that's what we set
        reward = coordinator.rewards[0]

        # We need to return a group-level reward. Here we mark the returned reward as unweighted.
        return {f"MMBench_DEV_EN_V11/unweighted/{category}": reward, "reward": reward}

    def _aggregate_MMBench_DEV_EN_V11(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        grouped_tasks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for group in tasks:
            for task in group:
                if task["benchmark_name"] == "MMBench_DEV_EN_V11":
                    grouped_tasks[task["group"]].append(task)

        if not grouped_tasks:
            return dict()

        # All rewards are the same for items within a group
        rewards = [group[0]["reward"] for group in grouped_tasks.values()]
        return {
            "MMBench_DEV_EN_V11": sum(rewards) / len(rewards),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return self._aggregate_MMBench_DEV_EN_V11(tasks)

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "mean/OCRBench",
            "MMBench_DEV_EN_V11",
        ]
        return {k: agent_metrics[k] for k in keys}


if __name__ == "__main__":
    VlmEvalKitResourcesServer.run_webserver()
