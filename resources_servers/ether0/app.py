# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import Any, Optional

from ether0.model_prompts import extract_answer_loose
from ether0.models import RewardFunctionInfo
from ether0.rewards import EVAL_FUNCTIONS

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


logger = logging.getLogger(__name__)


class Ether0RunRequest(BaseRunRequest):
    verifier_metadata: Optional[dict[str, Any]] = None


class Ether0VerifyRequest(Ether0RunRequest, BaseVerifyRequest):
    pass


class Ether0VerifyResponse(BaseVerifyResponse):
    extracted_answer: Optional[str] = None
    eval_function: Optional[str] = None
    problem_type: Optional[str] = None


class Ether0ResourcesServer(SimpleResourcesServer):
    config: BaseResourcesServerConfig

    async def verify(self, body: Ether0VerifyRequest) -> Ether0VerifyResponse:
        text = _extract_last_assistant_text(body)

        meta = body.verifier_metadata or {}
        solution_str = meta.get("solution", "")
        problem_type = meta.get("problem_type", "")

        try:
            reward_info = RewardFunctionInfo.model_validate(solution_str)
        except Exception:
            logger.warning("Malformed solution string: %r", solution_str)
            return _response(body, 0.0, None, None, problem_type)

        eval_fn_name = reward_info.fxn_name
        answer_info = reward_info.answer_info

        text = text.replace("<|answer_start|>", "<answer>").replace("<|answer_end|>", "</answer>")
        answer = (extract_answer_loose(text) or "").strip() or None
        if answer is None:
            return _response(body, 0.0, None, eval_fn_name, problem_type)

        eval_fn = EVAL_FUNCTIONS.get(eval_fn_name)
        if eval_fn is None:
            logger.warning("Unknown eval function %r", eval_fn_name)
            return _response(body, 0.0, answer, eval_fn_name, problem_type)

        try:
            reward = eval_fn(answer, answer_info)
        except Exception:
            logger.exception("Error in %r for problem_type=%r", eval_fn_name, problem_type)
            reward = 0.0

        return _response(body, reward, answer, eval_fn_name, problem_type)


def _response(
    body: Ether0VerifyRequest,
    reward: float,
    extracted_answer: Optional[str],
    eval_function: Optional[str],
    problem_type: str,
) -> Ether0VerifyResponse:
    return Ether0VerifyResponse(
        **body.model_dump(exclude={"extracted_answer", "eval_function", "problem_type"}),
        reward=reward,
        extracted_answer=extracted_answer,
        eval_function=eval_function,
        problem_type=problem_type,
    )


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


if __name__ == "__main__":
    Ether0ResourcesServer.run_webserver()
