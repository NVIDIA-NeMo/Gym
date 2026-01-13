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
import re
from enum import Enum
from typing import Any, Literal, Optional

from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.terminal_pivot.schemas import TERMINUS_1_SCHEMA, TERMINUS_2_SCHEMA


SCHEMA_MAP = {
    'terminus_1': TERMINUS_1_SCHEMA,
    'terminus_2': TERMINUS_2_SCHEMA,
}


class FailureCode(str, Enum):
    JSON_PARSING_FAILED = "JSON_PARSING_FAILED"
    SCHEMA_CHECK_FAILED = "SCHEMA_CHECK_FAILED"
    TASK_COMPLETE_CHECK_FAILED = "TASK_COMPLETE_CHECK_FAILED"
    COMMAND_CORRECTNESS_FAILED = "COMMAND_CORRECTNESS_FAILED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    UNKNOWN_HARNESS = "UNKNOWN_HARNESS"



class TBResourcesServerConfig(BaseResourcesServerConfig):
    pass


class TBRunRequest(BaseRunRequest):
    uuid: Optional[str] = None
    # Preferred dataset format: top-level `metadata` carries arbitrary data and
    # is not interpreted by the verifier. Only the fields below are used for
    # grading.
    expected_answer: Optional[str] = None
    # Optional additional metadata for the request; if provided, may contain
    # fields like options/expected_answer as an alternative location.
    metadata: Optional[dict[str, Any]] = None


class TBVerifyRequest(TBRunRequest, BaseVerifyRequest):
    pass


class TBVerifyResponse(BaseVerifyResponse):
    uuid: Optional[str] = None
    expected_answer: str
    model_output: str
    failure_reason: Optional[FailureCode] = None
    metadata: Optional[dict[str, Any]] = None


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    # body.response.output is a list of union types; we only want assistant message texts
    # TODO: @fsoares should we just assume we are always receiving the last message only? Not sure if this is always true.
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            # Each message has content which can be text parts; normalize to string
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def check_task_complete(pred: dict, expected_answer: dict) -> bool:
    if 'task_complete' in expected_answer and expected_answer['task_complete']:
        if 'task_complete' not in pred or not pred['task_complete']:
            return False
    elif 'is_task_complete' in expected_answer and expected_answer['is_task_complete']:
        if 'is_task_complete' not in pred or not pred['is_task_complete']:
            return False
    return True


def check_schema(pred: dict, expected_answer: dict) -> bool:
    required_keys = expected_answer.keys()
    for key in required_keys:
        if key not in pred or pred[key] is None:
            return False
    if not isinstance(pred['commands'], list):
        return False
    for each_command in pred['commands']:
        if not isinstance(each_command, dict):
            return False
        if 'keystrokes' not in each_command:
            return False
        if not isinstance(each_command['keystrokes'], str):
            return False
    return True


def check_command_correctness(pred: dict, expected_answer: dict) -> bool:
    if len(pred['commands']) != len(expected_answer['commands']):
        return False
    for i in range(len(pred['commands'])):
        if pred['commands'][i]['keystrokes'] != expected_answer['commands'][i]['keystrokes']:
            return False
    return True


class TBResourcesServer(SimpleResourcesServer):
    config: TBResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: TBVerifyRequest) -> TBVerifyResponse:
        text = _extract_last_assistant_text(body)
        expected_answer = json.loads(body.expected_answer)
        is_correct = True
        failure_reason = None
        try:
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            pred = json.loads(text)
            harness = body.metadata.get('harness', None)
            if harness is None or harness not in ['terminus_1', 'terminus_2']:
                failure_reason = FailureCode.UNKNOWN_HARNESS
                is_correct = False
            if is_correct:
                try:
                    validate_against_schema_openapi(pred, SCHEMA_MAP[harness])
                except Exception as e:
                    failure_reason = FailureCode.SCHEMA_CHECK_FAILED
                    is_correct = False
            if is_correct and not check_task_complete(pred, expected_answer):
                failure_reason = FailureCode.TASK_COMPLETE_CHECK_FAILED
                is_correct = False
            if is_correct and not check_command_correctness(pred, expected_answer):
                failure_reason = FailureCode.COMMAND_CORRECTNESS_FAILED
                is_correct = False
        except json.JSONDecodeError:
            failure_reason = FailureCode.JSON_PARSING_FAILED
            is_correct = False
        except Exception as e:
            failure_reason = FailureCode.UNKNOWN_ERROR
            is_correct = False

        reward = 1.0 if is_correct else 0.0

        return TBVerifyResponse(
            **body.model_dump(),
            reward=reward,
            model_output=text,
            failure_reason=failure_reason,
        )


if __name__ == "__main__":
    TBResourcesServer.run_webserver()
