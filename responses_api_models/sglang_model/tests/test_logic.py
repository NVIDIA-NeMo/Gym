# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from responses_api_models.sglang_model._logic import extract_generated_tokens_and_logprobs


def test_extracts_tuple_entries() -> None:
    result = {
        "meta_info": {
            "output_ids": [11, 12],
            "output_token_logprobs": [
                [-0.1, 11, "a"],
                (-0.2, 12, "b"),
            ],
        }
    }

    assert extract_generated_tokens_and_logprobs(result) == ([11, 12], [-0.1, -0.2])


def test_extracts_mapping_entries() -> None:
    result = {
        "meta_info": {
            "output_ids": [21, 22],
            "output_token_logprobs": [
                {"token_id": 21, "logprob": -0.3},
                {"id": 22, "logprob": -0.4},
            ],
        }
    }

    assert extract_generated_tokens_and_logprobs(result) == ([21, 22], [-0.3, -0.4])


@pytest.mark.parametrize("location", ["meta", "result"])
def test_extracts_split_value_index_fallback(location: str) -> None:
    fields = {
        "output_ids": [31, 32],
        "output_token_logprobs_val": [-0.5, -0.6],
        "output_token_logprobs_idx": [31, 32],
    }
    result = {"meta_info": fields} if location == "meta" else fields

    assert extract_generated_tokens_and_logprobs(result) == ([31, 32], [-0.5, -0.6])


def test_extracts_output_ids_fallback() -> None:
    result = {
        "output_ids": [41, 42],
        "meta_info": {"output_token_logprobs_val": [-0.7, -0.8]},
    }

    assert extract_generated_tokens_and_logprobs(result) == ([41, 42], [-0.7, -0.8])


@pytest.mark.parametrize(
    "result",
    [
        {"meta_info": {"output_ids": [], "output_token_logprobs": []}},
        {
            "meta_info": {
                "output_ids": [],
                "output_token_logprobs_val": [],
                "output_token_logprobs_idx": [],
            }
        },
    ],
)
def test_present_empty_arrays_are_valid_zero_token_completion(result: dict) -> None:
    assert extract_generated_tokens_and_logprobs(result) == ([], [])


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"meta_info": {"output_token_logprobs": [[-0.1]]}},
        {"meta_info": {"output_token_logprobs": [{"token_id": 1}]}},
        {
            "meta_info": {
                "output_token_logprobs_val": [-0.1, -0.2],
                "output_token_logprobs_idx": [1],
            }
        },
        {
            "meta_info": {
                "output_ids": [1],
                "output_token_logprobs_val": [],
            }
        },
    ],
)
def test_rejects_missing_or_malformed_training_data(result: dict) -> None:
    with pytest.raises(RuntimeError):
        extract_generated_tokens_and_logprobs(result)


@pytest.mark.parametrize(
    "result",
    [
        {
            "meta_info": {
                "output_ids": [1, 2],
                "output_token_logprobs": [{"token_id": 1, "logprob": -0.1}],
            }
        },
        {
            "meta_info": {
                "output_ids": [1],
                "output_token_logprobs": [{"token_id": 2, "logprob": -0.1}],
            }
        },
        {
            "meta_info": {
                "output_ids": [1],
                "output_token_logprobs_val": [-0.1],
                "output_token_logprobs_idx": [2],
            }
        },
    ],
)
def test_rejects_logprob_ids_that_do_not_match_output_ids(result: dict) -> None:
    with pytest.raises(RuntimeError, match="do not match output_ids"):
        extract_generated_tokens_and_logprobs(result)
