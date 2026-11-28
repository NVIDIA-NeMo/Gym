# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Malformed rows answer 422, scoring options stay out of the response, and the verifier fixture runs."""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.asr_with_pc.app import VERIFIER_FIXTURE, ASRWithPCVerifyRequest
from resources_servers.asr_with_pc.tests.test_speech_task_types import _response, _verify


async def _rejected(task_type: str, **row) -> HTTPException:
    with pytest.raises(HTTPException) as error:
        await _verify(task_type, "twenty five people", "twenty five people", **row)
    assert error.value.status_code == 422
    return error.value


async def test_numb3rs_normalization_needs_a_category() -> None:
    row = dict(normalization="numb3rs")
    assert "requires a category" in (await _rejected("ASR_LEADERBOARD", **row)).detail
    # An explicit null is the same malformed row, not a silent fall-back to no category rules.
    await _rejected("ASR_LEADERBOARD", category=None, **row)
    await _rejected("ASR_LEADERBOARD", category="", **row)


async def test_numb3rs_dash_between_number_words_scores_as_a_space() -> None:
    result = await _verify(
        "ASR_LEADERBOARD", "twenty five people", "twenty-five people", normalization="numb3rs", category="CARDINAL"
    )
    assert result["wer"] == 0.0 and result["wer_errors"] == 0


async def test_chunk_rows_need_a_valid_chunk_index() -> None:
    await _rejected("OpenASR", parent_id="call")
    await _rejected("OpenASR", parent_id="call", chunk_index=-1)


def test_an_unknown_cer_whitespace_mode_fails_at_parse_time() -> None:
    with pytest.raises(ValidationError):
        ASRWithPCVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}]},
            response=_response("x"),
            expected_answer="x",
            cer_whitespace_mode="bogus",
        )


async def test_scoring_options_are_not_echoed_into_the_response() -> None:
    result = await _verify(
        "Multilingual-ASR", "你好", "你好", language="cmn_hans_cn", use_cer=True, remove_diacritics=False,
        normalize_compound=False, merge_compounds=False, cer_whitespace_mode="all",
    )  # fmt: skip
    for option in ("use_cer", "remove_diacritics", "normalize_compound", "merge_compounds", "cer_whitespace_mode"):
        assert option not in result
    assert result["language"] == "cmn_hans_cn" and result["error_unit"] == "char"


async def test_language_codes_split_on_dashes_as_well_as_underscores() -> None:
    # "en-US" takes the English path (no compound normalization): "any way" and "anyway" differ.
    assert (await _verify("Multilingual-ASR", "any way", "anyway", language="en-US"))["wer_errors"] == 2
    assert (await _verify("Multilingual-ASR", "fuß ball", "fußball", language="de-DE"))["wer_errors"] == 0


async def test_verifier_fixture() -> None:
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )
    names = {result.name for result in results}
    assert {"open asr exact transcript", "multilingual exact transcript"} <= names
    assert {result.kind for result in results} >= {"full_reward", "zero_reward", "malformed"}
