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
import asyncio
from contextlib import nullcontext
from typing import List, Optional

from fastapi import FastAPI
from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json


class ExampleMultiStepResourcesServerConfig(BaseResourcesServerConfig):
    # The auxiliary model in the verification loop — a judge, reward model, or subagent. It is
    # wired by name to a model server you pass at run time (see configs/<name>.yaml), exactly like
    # `policy_model` is for the agent.
    judge_model_server: ModelServerRef
    # Base Responses API params for the auxiliary model; `input` is filled in per verify() call.
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # Optional system prompt and the per-task user prompt. Placeholders: {question}, {answer}.
    judge_system_message: Optional[str] = None
    judge_prompt_template: str = (
        "Question:\n{question}\n\nAnswer:\n{answer}\n\n"
        "Reply with a single number from 0 to 1 scoring how correct the answer is."
    )
    # Bound concurrent calls to the auxiliary model so a large rollout batch can't overwhelm it.
    # Set to None to disable limiting.
    judge_endpoint_max_concurrency: Optional[int] = Field(default=64)


class ExampleMultiStepResourcesServer(SimpleResourcesServer):
    config: ExampleMultiStepResourcesServerConfig

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_concurrency = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_concurrency = nullcontext()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def _score_with_judge(self, question: str, answer: str) -> float:
        """Call the auxiliary model with the per-task prompt and read a reward from its reply."""
        params = self.config.judge_responses_create_params.model_copy(deep=True)

        messages: List[NeMoGymEasyInputMessage] = []
        if self.config.judge_system_message:
            messages.append(NeMoGymEasyInputMessage(role="system", content=self.config.judge_system_message))
        prompt = self.config.judge_prompt_template.format(question=question, answer=answer)
        messages.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        params.input = messages

        async with self._judge_concurrency:
            response = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=params,
            )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        return self._parse_reward(judge_response)

    @staticmethod
    def _parse_reward(judge_response: NeMoGymResponse) -> float:
        """Read a clamped 0..1 score from the judge's reply.

        This default expects the judge to reply with a leading number; replace it with your own
        parsing (a verdict label, a JSON field, etc.).
        """
        try:
            text = judge_response.output[-1].content[-1].text
            return max(0.0, min(1.0, float(text.strip().split()[0])))
        except (AttributeError, IndexError, ValueError):
            return 0.0

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # TODO: pull the question and the model's answer for this task out of `body`
        # (e.g. the last user message in body.responses_create_params.input and the model's
        # final output), then score them with the auxiliary model.
        question = ""
        answer = ""
        reward = await self._score_with_judge(question, answer)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    ExampleMultiStepResourcesServer.run_webserver()
