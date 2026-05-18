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
"""math_rlvr resources server.

Port of NeMo-RLVR's `nemo_rl/environments/math_environment.py` to NeMo-Gym.

Per-row dispatch of four verifier types via the ``verifier_type`` field:

- ``math``                    — Hugging Face's ``math-verify`` library
                                (LaTeX / SymPy-aware grader; the default).
- ``math500``                 — strict ``\\boxed{}`` extraction +
                                ``is_equiv`` (Hendrycks MATH-style normalizer).
- ``english_multichoice``     — regex ``Answer:\\s*([A-Z])``
                                + ``answer_parsing.normalize_*``.
- ``multilingual_multichoice``— multilingual ``Answer\\s*:`` regex bank
                                covering Korean / Bengali / Arabic / etc.

Pure scoring — no tools, no LLM judge, no sandbox. The optional
``followup`` (teacher-feedback retry loop) feature from RLVR is **not**
ported; that's a multi-turn agent in Gym, not a verifier concern.
"""
from __future__ import annotations

import contextlib
import io
import logging
import re
from typing import Optional

from fastapi import FastAPI
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.math_rlvr import answer_parsing
from resources_servers.math_rlvr.math_utils import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)

logger = logging.getLogger(__name__)

VerifierType = str


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


@contextlib.contextmanager
def _mute_output():
    """Silence math-verify's chatty stdout/stderr during scoring."""
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


def _strip_thinking(text: str) -> tuple[str, bool]:
    """Strip a closed ``<think>...</think>`` prefix.

    Returns ``(text, ok)``. ``ok=False`` when ``<think>`` was opened but never
    closed — caller should treat as reward 0 with no verifier failure.
    """
    if "<think>" in text and "</think>" not in text:
        return "", False
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text, True


# ----------------------------------------------------------------------------
# Verifier implementations
# ----------------------------------------------------------------------------


# Module-level cache of the math-verify metric. Constructing it is expensive
# (it imports SymPy), so reuse one instance across requests.
_LIBRARY_VERIFIER = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(),
    ),
)


def _verify_math(model_output: str, ground_truth: str) -> tuple[float, Optional[str]]:
    """``math`` verifier — math-verify on the post-``</think>`` text.

    Returns ``(reward, extracted_answer)``. ``extracted_answer`` is currently
    always ``None`` (math-verify exposes a richer answer-extraction API but the
    RLVR worker did not surface it; we keep the same surface).
    """
    text, ok = _strip_thinking(model_output)
    if not ok:
        return 0.0, None

    ground_truth_parsable = "\\boxed{" + ground_truth + "}"
    with _mute_output():
        try:
            ret_score, _extracted = _LIBRARY_VERIFIER([ground_truth_parsable], [text])
        except (Exception, TimeoutException) as e:
            logger.debug("math-verify call failed: %s", e)
            ret_score = 0.0
    reward = 1.0 if ret_score and ret_score > 0.0 else 0.0
    return reward, None


def _verify_math500(model_output: str, ground_truth: str) -> tuple[float, Optional[str]]:
    """``math500`` verifier — Hendrycks MATH ``\\boxed{}`` + normalizer."""
    text, ok = _strip_thinking(model_output)
    if not ok:
        return 0.0, None

    try:
        gt_clean = remove_boxed(last_boxed_only_string(ground_truth))
        answer = remove_boxed(last_boxed_only_string(text))
        verified = is_equiv(answer, gt_clean)
        return (1.0 if verified else 0.0), answer
    except Exception as e:
        logger.debug("math500 verifier failed: %s", e)
        return 0.0, None


def _verify_english_multichoice(
    model_output: str, ground_truth: str
) -> tuple[float, Optional[str]]:
    """``english_multichoice`` — extract ``Answer: X`` (case-insensitive)."""
    response = answer_parsing.normalize_response(model_output)
    ground_truth = answer_parsing.normalize_response(ground_truth)

    if "<think>" in response and "</think>" not in response:
        return 0.0, response
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    extracted_answer = None
    match = re.search(r"(?i)Answer\s*:[ \t]*([A-Z])", response)
    if match:
        extracted_answer = answer_parsing.normalize_extracted_answer(match.group(1))

    score = 1.0 if extracted_answer == ground_truth else 0.0
    return score, extracted_answer


def _verify_multilingual_multichoice(
    model_output: str, ground_truth: str
) -> tuple[float, Optional[str]]:
    """``multilingual_multichoice`` — try each multilingual ``Answer:`` regex."""
    response = answer_parsing.normalize_response(model_output)

    if "<think>" in response and "</think>" not in response:
        return 0.0, response
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    extracted_answer = None
    for answer_regex in answer_parsing.MULTILINGUAL_ANSWER_REGEXES:
        regex = answer_parsing.MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response)
        if match:
            extracted_answer = answer_parsing.normalize_extracted_answer(match.group(1))
            break

    score = 1.0 if extracted_answer == ground_truth else 0.0
    return score, extracted_answer


VERIFIERS = {
    "math": _verify_math,
    "math500": _verify_math500,
    "english_multichoice": _verify_english_multichoice,
    "multilingual_multichoice": _verify_multilingual_multichoice,
}


def verify_sample(
    model_output: str, ground_truth: str, verifier_type: str = "math"
) -> tuple[float, Optional[str]]:
    """Top-level dispatcher: pick the verifier by name and run it."""
    fn = VERIFIERS.get(verifier_type)
    if fn is None:
        logger.warning(
            "Unknown verifier_type %r; falling back to 'math'", verifier_type
        )
        fn = _verify_math
    return fn(model_output, ground_truth)


# ----------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------


class MathRlvrResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MathRlvrVerifyRequest(BaseVerifyRequest):
    # The expected answer text. For ``math``: the latex/expression to wrap in
    # ``\\boxed{...}`` and feed to math-verify. For ``math500``: a string
    # containing a ``\\boxed{...}`` block. For ``english_multichoice`` /
    # ``multilingual_multichoice``: a single letter A-D (case-insensitive).
    ground_truth: str
    # One of: ``math`` (default), ``math500``, ``english_multichoice``,
    # ``multilingual_multichoice``. Unknown values fall back to ``math``.
    verifier_type: str = "math"


class MathRlvrVerifyResponse(BaseVerifyResponse):
    verifier_type: str
    extracted_answer: Optional[str]
    verification_failed: bool


class MathRlvrResourcesServer(SimpleResourcesServer):
    config: MathRlvrResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: MathRlvrVerifyRequest) -> MathRlvrVerifyResponse:
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                final_response_text = last_output.content[0].text

        verification_failed = False
        reward = 0.0
        extracted: Optional[str] = None
        try:
            reward, extracted = verify_sample(
                final_response_text, body.ground_truth, body.verifier_type
            )
        except (Exception, TimeoutException) as e:
            logger.warning(
                "math_rlvr verification raised (%s): %s",
                body.verifier_type,
                e,
            )
            verification_failed = True
            reward = 0.0
            extracted = None

        return MathRlvrVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted,
            verification_failed=verification_failed,
        )


if __name__ == "__main__":
    MathRlvrResourcesServer.run_webserver()
