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

import re
from typing import Any, Optional

import reasoning_gym
from fastapi import FastAPI
from reasoning_gym.utils import extract_answer

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ReasoningGymResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ReasoningGymVerifyRequest(BaseVerifyRequest):
    question: str
    answer: Optional[str]
    metadata: dict[str, Any]


class ReasoningGymVerifyResponse(BaseVerifyResponse):
    task_name: str
    score: float
    extracted_answer: Optional[str]


class ReasoningGymResourcesServer(SimpleResourcesServer):
    config: ReasoningGymResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: ReasoningGymVerifyRequest) -> ReasoningGymVerifyResponse:
        """Uses reasoning gym verifier"""
        model_answer = self._extract_answer_from_response(body.response)

        task_name = body.metadata.get("source_dataset")

        if not task_name:
            raise ValueError(f"No task name found in metadata: {body.metadata}")

        entry = {
            "question": body.question,
            "answer": body.answer,
            "metadata": body.metadata,
        }
        try:
            score_fn = reasoning_gym.get_score_answer_fn(task_name)
            score = float(score_fn(answer=model_answer, entry=entry))
        except Exception as e:
            print(f"Error scoring answer for task {task_name}: {e}")
            score = 0.0

        return ReasoningGymVerifyResponse(
            **body.model_dump(),
            reward=score,
            task_name=task_name,
            score=score,
            extracted_answer=model_answer,
        )

    def _extract_answer_from_response(self, response) -> str:
        assistant_responses = []
        for output_item in response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                assistant_responses.append(content_item.text)

        full_text = "".join(assistant_responses)

        # Try <answer> tags first (reasoning gym default)
        extracted = extract_answer(full_text, tag_name="answer")
        if extracted is not None:
            return extracted

        # Try \boxed{} if <answer> tags fail
        # this could be a slight instruction following issue, if model is prompted to use <answer> but uses boxed instead
        # found for deepseek-distill-qwen-1.5b it fails to use <answer> tags in favor of boxed, hence this fallback
        # may advise commenting this out for large models who follow instructions to use <answer> well
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", full_text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # return full text if <answer> or \boxed{} fail
        return full_text.strip() if full_text.strip() else ""


if __name__ == "__main__":
    ReasoningGymResourcesServer.run_webserver()
