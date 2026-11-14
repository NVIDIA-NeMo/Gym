# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.verification import JudgeError
from resources_servers.opencode_multimodal_benchmarks.agentx import judge_score


@pytest.mark.parametrize(
    "reply,query_type",
    [
        ('{"verdict":"different","verdict":"equivalent"}', "objective"),
        ('{"verdict":"equivalent","verdict":"equivalent"}', "objective"),
        ('{"score":0.1,"score":0.9}', "subjective"),
        ('{"score":' + "9" * 400 + "}", "subjective"),
        ('{"score":NaN}', "subjective"),
        ('{"score":Infinity}', "subjective"),
        ('{"score":-Infinity}', "subjective"),
    ],
)
def test_invalid_json_and_oversized_numbers_are_typed_judge_errors(reply, query_type):
    with pytest.raises(JudgeError):
        judge_score(reply, query_type)
