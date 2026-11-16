# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.officeqa.verifiers.officeqa_pro import extract_final_answer, score_answer


@pytest.mark.parametrize(
    "gold,answer,reward",
    [
        ("42", "<REASONING>computed</REASONING><FINAL_ANSWER>42</FINAL_ANSWER>", 1.0),
        ("42", "<FINAL_ANSWER>41</FINAL_ANSWER><FINAL_ANSWER>42</FINAL_ANSWER>", 1.0),
        ("42", "42", 1.0),
        ("42", "", 0.0),
        ("42", "Unable to determine", 0.0),
        ("42", "42 and 43", 0.0),
        ("42", "42\n43", 0.0),
        ("-1234", "(1,234)", 1.0),
        ("-1234", "−1,234", 1.0),
        ("543 million", "543", 1.0),
        ("543 million", "543 billion", 0.0),
        ("543 million", "543000000", 0.0),
        ("[21, 34]", "[21,34]", 1.0),
        ("[21, 34]", "[34,21]", 0.0),
        ("[8, 152260]", "[8,152,260]", 1.0),
        ("March 1977", "March 1977", 1.0),
        ("March 1977", "April 1977", 0.0),
        ("March 1977", "1977", 0.0),
        ("Trust Fund (TF)", "Trust Fund", 1.0),
        ("Trust Fund", "The Trust Fund", 0.0),
        ("0", "0", 1.0),
        ("0", "1", 0.0),
    ],
)
def test_historical_official_scoring_contract(gold, answer, reward):
    assert score_answer(gold, answer) == reward


def test_final_extraction_uses_last_tag_without_reasoning_numbers():
    assert extract_final_answer("<REASONING>41 plus 1</REASONING><FINAL_ANSWER>42</FINAL_ANSWER>") == "42"


def test_tolerance_is_explicit_and_does_not_change_default_scoring():
    assert score_answer("100", "100.1") == 0
    assert score_answer("100", "100.1", tolerance=0.01) == 1
