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
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi
from difflib import SequenceMatcher

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


def extract_keystrokes(data: Dict[str, Any]) -> List[str]:
    """Extract all keystrokes from commands list."""
    commands = data.get('commands', [])
    return [cmd.get('keystrokes', '') for cmd in commands if 'keystrokes' in cmd]


def text_similarity(s1: str, s2: str) -> float:
    """Compute similarity ratio between two strings using SequenceMatcher.
    Returns value between 0 (no similarity) and 1 (identical).
    """
    return SequenceMatcher(None, s1, s2).ratio()


def command_similarity(
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    separator: str = ''
) -> float:
    """
    Compute text similarity between commands in gt and pred.
    
    Concatenates all commands in sequence before comparing, preserving
    the sequential execution order.
    
    Args:
        gt: Ground truth dictionary with 'commands' key
        pred: Prediction dictionary with 'commands' key
        separator: String to join commands with (default: empty string)
    
    Returns:
        Similarity score between 0 (no similarity) and 1 (identical)
    """
    gt_keystrokes = extract_keystrokes(gt)
    pred_keystrokes = extract_keystrokes(pred)
    
    if not gt_keystrokes and not pred_keystrokes:
        return 1.0  # Both empty = identical
    
    if not gt_keystrokes or not pred_keystrokes:
        return 0.0  # One empty, one not = no similarity
    
    # Concatenate commands in sequence
    gt_concat = separator.join(gt_keystrokes)
    pred_concat = separator.join(pred_keystrokes)
    
    # Compare concatenated command sequences
    return text_similarity(gt_concat, pred_concat)


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
    expected_answer: str
    # Additional metadata for the request; must contain 'harness' field.
    metadata: dict[str, Any]
    threshold: Optional[float] = None


class TBVerifyRequest(TBRunRequest, BaseVerifyRequest):
    pass


class TBVerifyResponse(BaseVerifyResponse):
    uuid: Optional[str] = None
    expected_answer: str
    model_output: str
    similarity: float
    failure_reason: Optional[FailureCode] = None
    metadata: dict[str, Any]
    threshold: Optional[float] = None


class TBResourcesServer(SimpleResourcesServer):
    config: TBResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: TBVerifyRequest) -> TBVerifyResponse:
        text = _extract_last_assistant_text(body)
        is_correct = True
        failure_reason = None
        sim_score = -1.0
        try:
            threshold = body.threshold
            expected_answer = json.loads(body.expected_answer)
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
            if is_correct:
                sim_score = command_similarity(expected_answer, pred)
                if threshold is not None and sim_score < threshold:
                    failure_reason = FailureCode.COMMAND_CORRECTNESS_FAILED
                    is_correct = False
        except json.JSONDecodeError:
            failure_reason = FailureCode.JSON_PARSING_FAILED
            is_correct = False
        except Exception as e:
            failure_reason = FailureCode.UNKNOWN_ERROR
            is_correct = False

        reward = sim_score if is_correct else 0.0

        return TBVerifyResponse(
            **body.model_dump(),
            reward=reward,
            model_output=text,
            similarity=sim_score,
            failure_reason=failure_reason,
        )


if __name__ == "__main__":
    TBResourcesServer.run_webserver()
