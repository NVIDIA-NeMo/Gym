# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from nemo_gym.openai_utils import NeMoGymResponse


WRONG_PREDICTION_TYPES = [
    "API error",
    "Extraction error",
    "False positive",
    "False negative",
    "Incorrect value",
]


@dataclass(frozen=True)
class OspreyEvalResult:
    score: int
    is_correct: bool
    wrong_prediction_type: Optional[str]
    clean_prediction: Optional[Any] = None


def maybe_clean_prediction_of_empty(prediction: Any, ground_truth: Any) -> Any:
    if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
        return prediction

    clean_prediction = deepcopy(prediction)
    for key in prediction:
        prediction_is_empty = prediction[key] is None or prediction[key] == ""
        ground_truth_is_empty = key not in ground_truth
        if prediction_is_empty and ground_truth_is_empty:
            clean_prediction.pop(key)

    return clean_prediction


def evaluate_response(response: Optional[NeMoGymResponse], ground_truth: Any) -> OspreyEvalResult:
    if response is None or response.error is not None:
        return OspreyEvalResult(score=0, is_correct=False, wrong_prediction_type="API error")

    result = _extract_result_payload(response)
    if result is None or "extraction" not in result:
        return OspreyEvalResult(score=0, is_correct=False, wrong_prediction_type="Extraction error")

    prediction = result["extraction"]
    clean_prediction = maybe_clean_prediction_of_empty(prediction, ground_truth)

    if ground_truth is None and clean_prediction is None:
        return OspreyEvalResult(
            score=1, is_correct=True, wrong_prediction_type=None, clean_prediction=clean_prediction
        )
    if ground_truth is None and clean_prediction is not None:
        return OspreyEvalResult(
            score=0,
            is_correct=False,
            wrong_prediction_type="False positive",
            clean_prediction=clean_prediction,
        )
    if ground_truth is not None and clean_prediction is None:
        return OspreyEvalResult(
            score=0,
            is_correct=False,
            wrong_prediction_type="False negative",
            clean_prediction=clean_prediction,
        )
    if ground_truth != clean_prediction:
        return OspreyEvalResult(
            score=0,
            is_correct=False,
            wrong_prediction_type="Incorrect value",
            clean_prediction=clean_prediction,
        )

    return OspreyEvalResult(score=1, is_correct=True, wrong_prediction_type=None, clean_prediction=clean_prediction)


def _extract_result_payload(response: NeMoGymResponse) -> Optional[Any]:
    for output_item in response.output:
        if output_item.type != "function_call":
            continue

        arguments = output_item.arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            return None

        return parsed

    return None
