# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numb3rs scoring: the category-based normalizer and its use by ASR_LEADERBOARD rows."""

import asyncio
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from resources_servers.asr_with_pc.app import (
    ASRWithPCConfig,
    ASRWithPCResourcesServer,
    ASRWithPCVerifyRequest,
    evaluate_asr,
    row_normalizer,
)
from resources_servers.asr_with_pc.numb3rs_normalizer import normalize_numb3rs_text


@pytest.mark.parametrize(
    "category,reference,prediction",
    [
        ("DIGIT", "o four", "Oh, four."),
        ("TELEPHONE", "o o seven", "oh oh seven"),
        ("TIME", "seven p m", "Seven p.m."),
        ("TIME", "eight forty five u t c", "Eight forty five UTC."),
        ("ADDRESS", "s r one thirteen", "SR one thirteen."),
    ],
)
def test_equivalent_spellings(category, reference, prediction):
    assert normalize_numb3rs_text(reference, category) == normalize_numb3rs_text(prediction, category)


@pytest.mark.parametrize(
    "reference,prediction",
    [
        ("$1.5", "$15"),
        ("1/5", "15"),
        ("-15", "15"),
        ("$100", "100"),
        ("10%", "10"),
        ("1:15", "115"),
        ("one five", "fifteen"),
        ("o o seven", "o seven"),
        ("one hundred", "one hundred and"),
    ],
)
def test_number_errors_remain_errors(reference, prediction):
    assert normalize_numb3rs_text(reference, "DIGIT") != normalize_numb3rs_text(prediction, "DIGIT")


def test_initialisms_do_not_change_prose_or_chemical_names():
    assert normalize_numb3rs_text("I am here", "PLAIN") == "i am here"
    assert normalize_numb3rs_text("aluminium oxide", "PLAIN") != normalize_numb3rs_text("aluminum oxide", "PLAIN")


def test_normalization_is_idempotent():
    for category, text in (("TIME", "Two P.M. UTC"), ("TELEPHONE", "Oh, oh seven! 555-0100"), ("MONEY", "$1,000.50")):
        once = normalize_numb3rs_text(text, category)
        assert normalize_numb3rs_text(once, category) == once


def request(generation: str, **fields) -> ASRWithPCVerifyRequest:
    return ASRWithPCVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": ""}]},
            "response": {
                "id": "r",
                "created_at": 0,
                "model": "m",
                "object": "response",
                "output": [
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": generation, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "task_type": "ASR_LEADERBOARD",
            "reference_fields": ["text_tn", "text_itn"],
            "expected_answer": "two p m",
            "text_itn": "two p m",
            "text_tn": "2 pm",
            "category": "TIME",
            **fields,
        }
    )


def server() -> ASRWithPCResourcesServer:
    config = ASRWithPCConfig(name="numb3rs", host="127.0.0.1", port=8100, entrypoint="app.py")
    return ASRWithPCResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_row_normalizer_defaults_to_whisper():
    assert row_normalizer({"category": "TIME"})("Two PM.") == "2 pm"
    assert row_normalizer({"category": "TIME", "normalization": "numb3rs"})("Two PM.") == "two p m"


def test_evaluate_asr_reports_word_error_counts():
    result = evaluate_asr("a b c d", "a x c", normalize=lambda text: text)
    assert (result["wer_errors"], result["wer_ref_words"]) == (2, 4)
    assert (result["wer_substitutions"], result["wer_insertions"], result["wer_deletions"]) == (1, 0, 1)


def test_verify_scores_with_the_numb3rs_normalizer_and_keeps_the_fixed_reference():
    svc = server()
    fixed = asyncio.run(svc.verify(request("two PM.", normalization="numb3rs")))
    assert fixed.wer == 0 and fixed.wer_errors == 0 and fixed.wer_ref_words == 3
    assert fixed.wer_itn == 0 and fixed.wer_tn > 0
    default = asyncio.run(svc.verify(request("two PM.")))
    assert default.wer > 0  # Whisper turns "two p m" into digits on the reference side only


def test_aggregate_reports_corpus_wer_per_category_and_profile():
    svc = server()
    rows = []
    for index, (category, generation) in enumerate(
        (("TIME", "two PM."), ("TIME", "two pm now"), ("DIGIT", "oh four"))
    ):
        fields = {
            "category": category,
            "normalization": "numb3rs",
            "subset_for_metrics": f"numb3rs_{category}",
            "benchmark_profile": "numb3rs",
        }
        if category == "DIGIT":
            fields |= {"expected_answer": "o four", "text_itn": "o four", "text_tn": "04"}
        row = asyncio.run(svc.verify(request(generation, **fields))).model_dump(mode="json")
        rows.append([row | {"_ng_task_index": index, "_ng_rollout_index": 0}])
    metrics = svc.compute_metrics(rows)
    # Three reference words in each TIME row and two in the DIGIT row; one inserted word overall.
    assert metrics["corpus_wer@k=1"] == pytest.approx(100 * 1 / 8)
    assert metrics["wer::subset::numb3rs_TIME@k=1"] == pytest.approx(100 * 1 / 6)
    assert metrics["wer::subset::numb3rs_DIGIT@k=1"] == 0
    assert metrics["wer::profile::numb3rs@k=1"] == pytest.approx(100 * 1 / 8)
    assert metrics["wer_itn@k=1"] == pytest.approx(100 * 1 / 8)


@pytest.mark.parametrize(
    ("category", "reference", "prediction"),
    [
        ("CARDINAL", "twenty five people", "twenty-five people"),
        ("CARDINAL", "a five year old", "a five-year-old"),
        ("CARDINAL", "twenty five", "Twenty\u2013five"),
    ],
)
def test_a_dash_between_words_is_a_separator(category, reference, prediction):
    assert normalize_numb3rs_text(reference, category) == normalize_numb3rs_text(prediction, category)


def test_dashes_that_are_not_between_words_keep_their_old_handling():
    assert normalize_numb3rs_text("-15", "CARDINAL") == "-15"
    assert normalize_numb3rs_text("555-0100", "TELEPHONE") == "555-0100"
    assert normalize_numb3rs_text("well -", "CARDINAL") == "well"
