# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Curation annotations must not change actor input or the scoring contract."""

from copy import deepcopy

import pytest

from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_service import contract_hash
from resources_servers.partner_if.llm import request_payload
from resources_servers.partner_if.package import validate_rows
from resources_servers.partner_if.schema import VerifierMetadata
from resources_servers.partner_if.tests.test_shared_adapter import shared_request


def metadata():
    return {
        "task_id": "smoke",
        "provenance": "modified",
        "source": "https://example.invalid/source",
        "bug_ids": None,
        "taxonomy_revision": "a" * 40,
        "rubric_families": {"deterministic": ["required_element"], "semantic": ["required_element"]},
        "evaluations": [
            {
                "model": "test/policy",
                "passed": 1,
                "attempts": 3,
                "valid_attempts": 3,
                "stage": "discovery",
                "actor_sha256": "b" * 64,
                "rubric_hash": "c" * 64,
                "settings": {"temperature": 1.0},
                "receipts": [{"uri": "https://example.invalid/pinned-receipt.json", "sha256": "d" * 64}],
            }
        ],
    }


def test_curation_survives_preparation_without_changing_prompts_hash_or_judge_payload():
    body = shared_request()
    original = body.verifier_metadata.partner_if
    row = {
        "responses_create_params": body.responses_create_params.model_dump(),
        "verifier_metadata": {"partner_if": original.model_dump(), "curation": metadata()},
    }
    before = deepcopy(row)
    assert validate_rows([row], allow_synthetic=True) == [before]
    parsed = VerifierMetadata.model_validate(row["verifier_metadata"])
    assert parsed.model_dump()["curation"]["evaluations"][0]["passed"] == 1
    assert contract_hash(parsed.partner_if) == contract_hash(original)
    bundle = EvidenceBundle(answer="HELLO")
    assert request_payload(parsed.partner_if.rubrics, bundle, max_tokens=1024, temperature=0) == request_payload(
        original.rubrics, bundle, max_tokens=1024, temperature=0
    )


@pytest.mark.parametrize(
    "change",
    [
        {"provenance": "inspired"},
        {"task_id": "different-task"},
        {"rubric_families": {"invented-rubric": ["coverage"]}},
    ],
)
def test_curation_rejects_unknown_provenance_or_misattributed_task_and_rubrics(change):
    body = shared_request()
    with pytest.raises(ValueError):
        VerifierMetadata.model_validate(
            {"partner_if": body.verifier_metadata.partner_if.model_dump(), "curation": metadata() | change}
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"passed": 2, "valid_attempts": 1},
        {"passed": None, "valid_attempts": 3},
        {"passed": 4},
        {"attempts": 5},
    ],
)
def test_model_pass_count_requires_three_valid_attempts(changes):
    annotations = metadata()
    annotations["evaluations"][0].update(changes)
    with pytest.raises(ValueError):
        VerifierMetadata.model_validate(
            {"partner_if": shared_request().verifier_metadata.partner_if.model_dump(), "curation": annotations}
        )


def test_untested_metadata_is_not_zero_of_three():
    annotations = metadata()
    annotations["evaluations"][0].update(passed=None, valid_attempts=0, receipts=[])
    parsed = VerifierMetadata.model_validate(
        {"partner_if": shared_request().verifier_metadata.partner_if.model_dump(), "curation": annotations}
    )
    assert parsed.curation.evaluations[0].passed is None


def test_historical_tasks_without_curation_still_validate():
    body = shared_request()
    parsed = VerifierMetadata.model_validate({"partner_if": body.verifier_metadata.partner_if.model_dump()})
    assert parsed.partner_if == body.verifier_metadata.partner_if
