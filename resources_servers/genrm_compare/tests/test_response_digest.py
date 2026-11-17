# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response fingerprints preserve exact retry identity across serialization paths."""

import pytest

from resources_servers.genrm_compare.app import GenRMCompareResourcesServer
from resources_servers.genrm_compare.tests.test_cohort_storage import training_member


def test_digest_matches_model_and_reordered_dictionary():
    response = training_member(0).response
    response.metadata = {"z": "雪", "a": "first"}
    response.extra_a = {"y": 1e-6, "b": 2}
    response.extra_z = [1, 2]
    payload = response.model_dump(mode="json")
    reordered = dict(reversed(list(payload.items())))
    reordered["metadata"] = dict(reversed(list(payload["metadata"].items())))
    reordered["extra_a"] = dict(reversed(list(payload["extra_a"].items())))
    digest = GenRMCompareResourcesServer._response_digest
    assert digest(response) == digest(payload) == digest(reordered)


@pytest.mark.parametrize("field", ["generation_token_ids", "generation_log_probs", "routed_experts"])
def test_digest_detects_training_data_changes(field):
    response = training_member(0).response
    changed = response.model_copy(deep=True)
    setattr(
        changed.output[1],
        field,
        {"generation_token_ids": [999], "generation_log_probs": [-9.99], "routed_experts": [[[999]]]}[field],
    )
    digest = GenRMCompareResourcesServer._response_digest
    assert digest(response) != digest(changed)


@pytest.mark.parametrize("value", ["\ud800", 2**80, -(2**80), float("nan"), float("inf"), float("-inf")])
def test_digest_preserves_nonstandard_python_values(value):
    digest = GenRMCompareResourcesServer._response_digest
    assert digest({"value": value}) == digest({"value": value})
    assert digest({"value": value}) != digest({"value": None})


@pytest.mark.parametrize("output_index", [0, 1])
def test_digest_keeps_nonfinite_model_values_distinct_and_matches_dictionary(output_index):
    digest = GenRMCompareResourcesServer._response_digest
    digests = set()
    for value in (float("nan"), float("inf"), float("-inf"), -0.5):
        response = training_member(0).response
        response.output[output_index].generation_log_probs = [value]
        assert digest(response) == digest(response.model_dump(mode="json"))
        digests.add(digest(response))
    assert len(digests) == 4


def test_digest_rejects_cyclic_dictionary():
    payload = {"id": "response"}
    payload["self"] = payload
    with pytest.raises(ValueError, match="Circular reference"):
        GenRMCompareResourcesServer._response_digest(payload)
