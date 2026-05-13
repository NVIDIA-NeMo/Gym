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
import json
import logging
from typing import Any, Union

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.ifeval.if_functions import IF_FUNCTIONS_MAP

logger = logging.getLogger(__name__)


class IFEvalResourcesServerConfig(BaseResourcesServerConfig):
    pass


class IFEvalVerifyRequest(BaseVerifyRequest):
    # Constraint specification: a JSON-encoded string, a single dict, or a list of dicts.
    # Each constraint dict must contain a "func_name" key referencing IF_FUNCTIONS_MAP,
    # plus the kwargs accepted by that function.
    ground_truth: Union[str, dict, list]


class IFEvalVerifyResponse(BaseVerifyResponse):
    follow_all_constraints: bool
    follow_constraint_list: list[bool]
    verification_failed: bool


def _verify_single_constraint(answer: str, constraint_dict: dict[str, Any]) -> bool:
    constraint_dict = dict(constraint_dict)
    if "func_name" not in constraint_dict:
        logger.warning("constraint missing func_name: %s", constraint_dict)
        return False

    func_name = constraint_dict.pop("func_name")
    if func_name not in IF_FUNCTIONS_MAP:
        logger.warning("unknown func_name: %s", func_name)
        return False
    func = IF_FUNCTIONS_MAP[func_name]

    non_none_args = {k: v for k, v in constraint_dict.items() if v is not None}
    result = func(answer) if not non_none_args else func(answer, **non_none_args)
    # Some validators return (bool, extra). Coerce to bool.
    if isinstance(result, tuple):
        result = result[0]
    return bool(result)


def _parse_constraint(constraint: Union[str, dict, list]) -> Union[dict, list]:
    if isinstance(constraint, str):
        return json.loads(constraint)
    return constraint


def _extract_answer(text: str) -> str:
    """Strip a leading <think>...</think> block if present."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


class IFEvalResourcesServer(SimpleResourcesServer):
    config: IFEvalResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: IFEvalVerifyRequest) -> IFEvalVerifyResponse:
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                final_response_text = last_output.content[0].text

        # Incomplete reasoning trace: <think> opened but never closed -> reward 0, not a verifier failure.
        if "<think>" in final_response_text and "</think>" not in final_response_text:
            return IFEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                follow_all_constraints=False,
                follow_constraint_list=[],
                verification_failed=False,
            )

        answer = _extract_answer(final_response_text)

        verification_failed = False
        follow_list: list[bool] = []
        try:
            parsed = _parse_constraint(body.ground_truth)
            constraints = parsed if isinstance(parsed, list) else [parsed]
            for c in constraints:
                follow_list.append(_verify_single_constraint(answer, c))
        except json.JSONDecodeError as e:
            logger.warning("IFEval verification failed (JSON parse): %s", e)
            verification_failed = True
        except Exception as e:
            logger.warning("IFEval verification failed (other): %s", e)
            verification_failed = True

        follow_all = bool(follow_list) and all(follow_list)
        reward = 1.0 if follow_all else 0.0

        return IFEvalVerifyResponse(
            **body.model_dump(),
            reward=reward,
            follow_all_constraints=follow_all,
            follow_constraint_list=follow_list,
            verification_failed=verification_failed,
        )


if __name__ == "__main__":
    IFEvalResourcesServer.run_webserver()
