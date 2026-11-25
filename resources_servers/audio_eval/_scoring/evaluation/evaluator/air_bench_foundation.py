# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reimplemented from the published scoring protocol of OFA-Sys/AIR-Bench (``score_foundation.py`` at commit
# 983247666e2a25519b5c122810c2bcd91f26fd8d, https://github.com/OFA-Sys/AIR-Bench; its README states Apache-2.0 but the
# repository has no license file). No upstream source text is copied; only the behavior (answer-letter parsing,
# denominator rule) is reproduced.

"""AIR-Bench Foundation evaluator.

Follows the published AIR-Bench scoring protocol. Parse failures are flagged and excluded from each leaf's
accuracy denominator, as in the reference scorer; they remain visible through ``fail_num``.
"""

from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator


def official_foundation_prediction(response: str) -> str | None:
    """Read the answer letter from the first character, or else from the second-to-last one."""
    predict = response.strip().replace("\n", "")
    if predict == "None" or not predict:
        return None
    if predict[0] in "ABCD":
        return predict[0]
    # Covers replies that end with the letter and a full stop, such as ``The answer is A.``.
    if len(predict) > 1 and predict[-2] in "ABCD":
        return predict[-2]
    return None


def official_foundation_ground_truth(data_point: dict) -> str:
    """Map the answer text to the letter of the first choice equal to it (A to D)."""
    answer = data_point.get("answer_gt")
    for letter in "abcd":
        if answer == data_point.get(f"choice_{letter}"):
            return letter.upper()
    raise ValueError(f"AIR-Bench answer_gt does not match any choice: {answer!r}")


def evaluate_air_bench_foundation_sample(data_point: dict) -> dict:
    generation = data_point.get("generation")
    response = generation if isinstance(generation, str) else ""
    predicted_answer = official_foundation_prediction(response)
    ground_truth = official_foundation_ground_truth(data_point)
    parse_valid = predicted_answer is not None
    is_correct = parse_valid and predicted_answer == ground_truth
    leaf = f"{data_point['task_name']}_{data_point['dataset_name']}"
    return {
        "response": response,
        "predicted_answer": predicted_answer,
        "official_ground_truth": ground_truth,
        "official_parse_valid": parse_valid,
        "symbolic_correct": is_correct,
        "match": int(is_correct),
        "air_bench_foundation_leaf": leaf,
        "subset_for_metrics": leaf,
    }


class AIRBenchFoundationEvaluator(BaseEvaluator):
    """Grade one AIR-Bench Foundation response with the published parser."""

    async def eval_single(self, data_point: dict) -> dict:
        return evaluate_air_bench_foundation_sample(data_point)
