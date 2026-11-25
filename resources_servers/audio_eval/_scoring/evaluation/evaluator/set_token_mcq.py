# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the ``string_match`` function of the MMAU evaluation script (Apache-2.0):
# https://github.com/Sakshi113/MMAU at commit 110127f54c0dfba3faa5ec9feee4a7e4148679c5, ``evaluation.py``.
# The MMAR evaluation script (https://github.com/ddlBoJack/MMAR at commit 3bce090a967db576c8ad433b290a8da582d6d2a0,
# ``code/evaluation.py``; no license file) defines the same decision rule; no MMAR source text is used here.
# Modifications: type annotations, restructured conditions, and a per-row evaluator class in place of the script's
# file-level loop.

"""Set-token multiple-choice evaluator used by MMAR.

A prediction is correct when it contains every word of the reference answer and none of the words that occur only
in the other options.
"""

import re
from typing import Any

from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def official_set_token_match(answer: str, prediction: str, choices: list[str]) -> bool:
    """Return whether the prediction matches the answer under the MMAR/MMAU set-token rule."""
    prediction_tokens = _tokenize(prediction)
    answer_tokens = _tokenize(answer)
    if not prediction_tokens:
        return False

    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = _tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(incorrect_tokens)


class SetTokenMCQEvaluator(BaseEvaluator):
    """Grade a generation against the reference answer and the option list of its row."""

    def __init__(self, config: dict[str, Any], num_parallel_requests: int = 10):
        if config:
            raise ValueError(f"The set-token-mcq evaluator takes no ++eval_config settings, got {sorted(config)}")
        super().__init__(config, num_parallel_requests)

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        generation = data_point.get("generation")
        prediction = generation if isinstance(generation, str) else ""
        answer = data_point.get("expected_answer")
        choices = data_point.get("choices")
        if not isinstance(answer, str):
            raise TypeError("set-token-mcq row requires a string expected_answer")
        if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
            raise TypeError("set-token-mcq row requires a list of string choices")

        is_correct = official_set_token_match(answer, prediction, choices)
        return {
            "answer_prediction": prediction,
            "predicted_answer": prediction.strip() or None,
            "symbolic_correct": is_correct,
            "match": int(is_correct),
        }
