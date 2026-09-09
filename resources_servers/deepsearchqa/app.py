# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


class DeepSearchQAConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class DeepSearchQARunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    example_id: str
    problem: str
    answer: str
    answer_type: str
    problem_category: str


class DeepSearchQAVerifyRequest(DeepSearchQARunRequest, BaseVerifyRequest):
    pass


class DeepSearchQAVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    precision: float
    recall: float
    f1: float
    fully_correct: float
    fully_incorrect: float
    correct_with_extraneous: float
    judge_output: dict[str, Any]


def response_text(response: NeMoGymResponse) -> str:
    texts = []
    for output in response.output:
        if output.type == "message" and output.role == "assistant":
            texts.extend(part.text for part in output.content if getattr(part, "text", None))
    return "\n".join(texts).strip()


def parse_judge(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].removesuffix("```").strip()
    result = json.loads(text)["Answer Correctness"]
    details = result["Correctness Details"]
    excessive = result.get("Excessive Answers", [])
    if (
        not isinstance(details, dict)
        or not details
        or not all(isinstance(key, str) and isinstance(value, bool) for key, value in details.items())
    ):
        raise ValueError("judge Correctness Details must be a non-empty string-to-boolean object")
    if not isinstance(excessive, list) or not all(isinstance(item, str) for item in excessive):
        raise ValueError("judge Excessive Answers must be a list of strings")
    return result


class DeepSearchQAServer(SimpleResourcesServer):
    config: DeepSearchQAConfig

    def model_post_init(self, context: Any) -> None:
        self._judge_prompt = Path(__file__).with_name("judge_prompt.txt").read_text()
        super().model_post_init(context)

    async def verify(self, body: DeepSearchQAVerifyRequest) -> DeepSearchQAVerifyResponse:
        prompt = self._judge_prompt.format(
            problem=body.problem,
            answer_type=body.answer_type,
            answer=body.answer,
            response=response_text(body.response),
        )
        params = self.config.judge_responses_create_params.model_copy(
            update={"input": [NeMoGymEasyInputMessage(role="user", content=prompt)]},
            deep=True,
        )
        judged = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )
        judge_output = parse_judge(response_text(judged))
        matched = sum(judge_output["Correctness Details"].values())
        expected = len(judge_output["Correctness Details"])
        excessive = len(judge_output.get("Excessive Answers", []))
        precision = matched / (matched + excessive) if matched + excessive else 0.0
        recall = matched / expected
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return DeepSearchQAVerifyResponse(
            **body.model_dump(),
            reward=f1,
            precision=precision,
            recall=recall,
            f1=f1,
            fully_correct=float(matched == expected and excessive == 0),
            fully_incorrect=float(matched == 0),
            correct_with_extraneous=float(matched == expected and excessive > 0),
            judge_output=judge_output,
        )


if __name__ == "__main__":
    DeepSearchQAServer.run_webserver()
