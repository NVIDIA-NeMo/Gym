# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-label evaluator for FLEURS language identification."""

from resources_servers.audio_eval._scoring.dataset.fleurs.languages import (
    FLEURS_LANGUAGES,
    LID_LOCALES,
    get_lang_group,
)
from resources_servers.audio_eval._scoring.evaluation.evaluator.language_id import (
    LanguageIDEvaluator,
    evaluate_language_id_sample,
)


# Chinese is often answered with the short form ``zh_cn`` although the prompt lists the FLEURS label ``cmn_hans_cn``;
# the alias is graded as the canonical label. It is the only alias.
LID_ALIASES = {"zh_cn": "cmn_hans_cn"}


def evaluate_fleurs_lid_sample(data_point: dict) -> dict:
    predicted = (data_point.get("generation") or "").strip()
    if predicted in LID_ALIASES:
        data_point = {**data_point, "generation": LID_ALIASES[predicted]}
    # Any FLEURS locale is a valid reference (a default run prepares all 102); a prediction counts as a valid label
    # when it is one of the 36 language-ID locales or the reference itself.
    expected = str(data_point["expected_answer"])
    if expected not in FLEURS_LANGUAGES:
        raise ValueError(f"Non-canonical language-ID reference label: {expected!r}")
    result = evaluate_language_id_sample(data_point, valid_labels={*LID_LOCALES, expected})
    result["locale"] = result["label"]
    result["lang_group"] = get_lang_group(result["label"])
    result["label_group"] = result["lang_group"]
    return result


class FleursLIDEvaluator(LanguageIDEvaluator):
    async def eval_single(self, data_point: dict) -> dict:
        return evaluate_fleurs_lid_sample(data_point)
