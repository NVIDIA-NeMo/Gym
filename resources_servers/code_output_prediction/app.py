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
"""Code-output-prediction resources server.

Port of NeMo-RLVR's `nemo_rl/environments/code_output_prediction_environment.py`.

Given a code snippet in the prompt, the model must predict its stdout / return
value. Verification extracts the LAST ``{...}`` block from the model output,
parses it as a Python literal, and compares its ``output`` field to the
ground-truth's ``ground_truth`` field. Pure scoring — no tools, no sandbox,
no LLM judge.
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Union

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> tuple[str, bool]:
    """Strip a closed ``<think>...</think>`` prefix.

    Returns ``(text, ok)``. ``ok=False`` when ``<think>`` was opened but never
    closed — the caller should treat this as reward 0 without flagging a
    verifier failure.
    """
    if "<think>" in text and "</think>" not in text:
        return "", False
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text, True


def _parse_last_json_object(text: str) -> Union[dict, None]:
    """Find the LAST ``{...}`` substring in ``text`` and parse it as a Python literal.

    Mirrors RLVR's behavior:
    - Uses ``re.findall(r'\\{.*?\\}', text, re.DOTALL)`` (non-greedy braces).
    - Falls back to a brace-rebalanced substring on parse failure.
    - Returns ``None`` if nothing parses to a dict.
    """
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return None
    last = matches[-1].strip()
    try:
        parsed = ast.literal_eval(last)
    except (ValueError, SyntaxError):
        start = last.find("{")
        end = last.rfind("}") + 1
        try:
            parsed = ast.literal_eval(last[start:end])
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


def verify_code_output_prediction_sample(model_output: str, ground_truth_answer: str) -> bool:
    """Compare the model's predicted output against ``ground_truth.ground_truth``.

    Returns True iff the LAST ``{...}`` block in ``model_output`` parses to a
    dict whose ``output`` field equals ``ground_truth_answer``'s ``ground_truth``.
    """
    text, ok = _strip_thinking(model_output)
    if not ok:
        return False

    model_json = _parse_last_json_object(text)
    if model_json is None:
        return False

    model_output_value = model_json.get("output", "")

    # ``ground_truth_answer`` is itself a stringified Python-literal dict in RLVR.
    try:
        gt_json = ast.literal_eval(ground_truth_answer) if isinstance(ground_truth_answer, str) else ground_truth_answer
    except (ValueError, SyntaxError):
        return False
    if not isinstance(gt_json, dict):
        return False
    expected = gt_json.get("ground_truth", "")

    return model_output_value == expected


class CodeOutputPredictionResourcesServerConfig(BaseResourcesServerConfig):
    pass


class CodeOutputPredictionVerifyRequest(BaseVerifyRequest):
    # A Python-literal-encoded dict (or already-parsed dict) carrying a
    # ``ground_truth`` key. Mirrors RLVR's ``metadata["ground_truth"]`` shape:
    # e.g. ``"{'ground_truth': '42'}"`` or ``{"ground_truth": "42"}``.
    ground_truth: Union[str, dict]


class CodeOutputPredictionVerifyResponse(BaseVerifyResponse):
    verification_failed: bool


class CodeOutputPredictionResourcesServer(SimpleResourcesServer):
    config: CodeOutputPredictionResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(
        self, body: CodeOutputPredictionVerifyRequest
    ) -> CodeOutputPredictionVerifyResponse:
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                final_response_text = last_output.content[0].text

        # Normalize ``ground_truth`` to a string for the legacy verifier;
        # passing dicts and JSON-strings should both work.
        gt = body.ground_truth
        if isinstance(gt, dict):
            gt_str = repr(gt)
        else:
            gt_str = gt

        verification_failed = False
        reward = 0.0
        try:
            verified = verify_code_output_prediction_sample(final_response_text, gt_str)
            reward = 1.0 if verified else 0.0
        except Exception as e:
            logger.warning("Code-output-prediction verification failed: %s", e)
            verification_failed = True
            reward = 0.0

        return CodeOutputPredictionVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verification_failed=verification_failed,
        )


if __name__ == "__main__":
    CodeOutputPredictionResourcesServer.run_webserver()
