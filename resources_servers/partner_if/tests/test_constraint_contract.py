# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only the explicit shared verifier contract is accepted at the runtime boundary."""

import copy
import json
from pathlib import Path

import pytest

from resources_servers.partner_if.schema import VerifierMetadata


# CI runs this suite from the server directory, so data paths must not depend on the working directory.
DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.mark.parametrize("legacy_rules", [False, True])
def test_old_text_contract_requires_explicit_curation_not_runtime_migration(legacy_rules):
    old = {
        "task_id": "old",
        "origin": "synthetic",
        "source": "authored test",
        "rubric_version": "old",
        "audit_signoff": False,
        "data_kind": "synthetic_smoke",
        "rubrics": [
            {
                "id": "hello",
                "category": "always",
                "instruction_text": "Say HELLO.",
                "source_role": "user",
                "applicability_reason": "Explicit output instruction.",
                "constraint": "The answer contains HELLO.",
                "taxonomy": ["IF-FORMAT"],
                "evidence": {
                    "source": "synthetic",
                    "source_id": "old",
                    "instruction": "Say HELLO.",
                    "similarity": "Synthetic only",
                    "difference": "Not task data",
                },
            }
        ],
    }
    if legacy_rules:
        old["rubrics"][0].pop("constraint")
        old["rubrics"][0].update(pass_rule="Say HELLO", fail_rule="Missing HELLO")
    before = copy.deepcopy(old)
    with pytest.raises(ValueError):
        VerifierMetadata.model_validate({"partner_if": old})
    assert old == before


def test_all_synthetic_examples_use_explicit_registered_verifiers():
    from resources_servers.partner_if.checks import resolve
    from resources_servers.partner_if.grading_contract import GradingContract

    rows = [json.loads(line) for line in (DATA_DIR / "example.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    for row in rows:
        contract = VerifierMetadata.model_validate(row["verifier_metadata"]).partner_if
        assert isinstance(contract, GradingContract)
        for rubric in contract.checks:
            resolve(rubric.verifier)


def test_example_receipts_match_current_contracts_and_grounded_judge_protocol():
    from nemo_gym.openai_utils import NeMoGymResponse
    from resources_servers.partner_if.evidence import EvidenceBundle
    from resources_servers.partner_if.grading_service import contract_hash
    from resources_servers.partner_if.llm import PROTOCOL, parse_verdict

    inputs = [json.loads(line) for line in (DATA_DIR / "example.jsonl").read_text().splitlines()]
    receipts = [json.loads(line) for line in (DATA_DIR / "example_rollouts.jsonl").read_text().splitlines()]
    assert len(inputs) == len(receipts) == 5
    by_id = {row["verifier_metadata"]["partner_if"]["task_id"]: row for row in receipts}
    assert len(by_id) == 5
    for row in inputs:
        contract = VerifierMetadata.model_validate(row["verifier_metadata"]).partner_if
        receipt = by_id[contract.task_id]
        assert receipt["verifier_metadata"]["partner_if"] == contract.model_dump()
        assert receipt["if_result"]["rubric_hash"] == contract_hash(contract)
        assert receipt["responses_create_params"]["input"] == [
            message | {"type": "message", "phase": None} for message in row["responses_create_params"]["input"]
        ]
        assert receipt["judge_protocol"] == PROTOCOL and not receipt["mask_sample"]
        answer = NeMoGymResponse.model_validate(receipt["response"]).output_text
        evidence = EvidenceBundle(context=receipt["responses_create_params"]["input"], answer=answer)
        judge = NeMoGymResponse.model_validate(receipt["judge_response"])
        verdicts = parse_verdict(judge.output_text, contract.rubrics, evidence)
        assert [v.passed for v in verdicts] == [v["raw_passed"] for v in receipt["grading_result"]["items"]]
