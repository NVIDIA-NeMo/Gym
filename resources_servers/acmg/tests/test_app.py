# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.verification import exercise_verifier_fixture

from ..app import VERIFIER_FIXTURE, AcmgResourcesServer, AcmgResourcesServerConfig, AcmgVerifyRequest
from ..task_data import TaskData


SERVER_DIR = Path(__file__).resolve().parents[1]
EXAMPLES = [json.loads(line) for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()]


def _server() -> AcmgResourcesServer:
    config = AcmgResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return AcmgResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _example(case_id: str, arm: str) -> dict:
    return next(r for r in EXAMPLES if r["verifier_metadata"]["record_id"] == f"{case_id}:{arm}")


def _verify(text: str, case_id: str, arm: str):
    row = _example(case_id, arm)
    body = AcmgVerifyRequest(
        responses_create_params=row["responses_create_params"],
        verifier_metadata=row["verifier_metadata"],
        response=_response(text),
    )
    return asyncio.run(_server().verify(body))


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


def test_examples_validate_and_keep_the_answer_key_out_of_the_prompt() -> None:
    assert len(EXAMPLES) == 6
    for row in EXAMPLES:
        task = TaskData.model_validate(row["verifier_metadata"])
        messages = row["responses_create_params"]["input"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert ("specification" in messages[0]["content"]) == (task.arm == "with_rules")
        assert task.reference.label not in messages[1]["content"]
        assert "CRITERIA" not in messages[1]["content"]


def test_exact_answer_earns_full_reward_and_reports_components() -> None:
    result = _verify("CRITERIA: PVS1 VeryStrong; PM2 Supporting\nCLASSIFICATION: LP", "example_lp", "with_rules")
    assert result.reward == 1.0
    assert result.predicted_label == "LP"
    assert result.predicted_criteria == {"PVS1": "VeryStrong", "PM2": "Supporting"}
    assert result.components["label"] == 1.0
    assert result.components["criterion_f1"] == 1.0
    assert result.components["self_consistent"] == 1.0
    assert result.components["evidence_score"] == 1.0
    assert (result.returned_n, result.returned_acceptable, result.required_n, result.required_hit) == (2, 2, 2, 2)
    assert result.self_consistency_assessable


def test_right_criteria_wrong_combination_scores_only_the_criteria() -> None:
    # PVS1 + PM2_Supporting cannot reach Pathogenic, so the label is wrong and not self-consistent.
    result = _verify("CRITERIA: PVS1 VeryStrong; PM2 Supporting\nCLASSIFICATION: P", "example_lp", "without_rules")
    assert result.predicted_label == "P"
    assert result.components["label"] == 0.0
    assert result.components["criterion_f1"] == 1.0
    assert result.components["self_consistent"] == 0.0
    assert result.reward == pytest.approx(0.3)


def test_extra_criterion_lowers_precision_and_evidence_score() -> None:
    result = _verify(
        "CRITERIA: PVS1 VeryStrong; PM2 Supporting; PP3 Supporting\nCLASSIFICATION: LP", "example_lp", "with_rules"
    )
    assert result.components["label"] == 1.0
    assert result.components["criterion_precision"] == pytest.approx(2 / 3)
    assert result.components["criterion_recall"] == 1.0
    assert result.components["exact_criteria"] == 0.0
    assert result.components["evidence_score"] == pytest.approx(1 - 2 / 8)
    assert 0.0 < result.reward < 1.0


def test_excluded_label_drops_out_of_the_scalar() -> None:
    result = _verify("CRITERIA: PM2 Supporting\nCLASSIFICATION: LP", "example_excluded", "without_rules")
    assert result.components["label"] is None
    # PM2_Supporting alone combines to VUS, so LP is not self-consistent: 0.3 x F1 renormalised over 0.5.
    assert result.components["self_consistent"] == 0.0
    assert result.reward == pytest.approx(0.6)


def test_reasoning_blocks_are_not_graded() -> None:
    text = "<think>CRITERIA: BA1 StandAlone\nCLASSIFICATION: B</think>\nCRITERIA: PM2 Supporting\nCLASSIFICATION: VUS"
    result = _verify(text, "example_excluded", "with_rules")
    assert result.predicted_label == "VUS"
    assert result.predicted_criteria == {"PM2": "Supporting"}
    assert result.reward == 1.0


def test_missing_classification_is_an_output_failure() -> None:
    result = _verify("The packet is insufficient to classify this variant.", "example_b", "without_rules")
    assert result.reward == 0.0
    assert result.predicted_label is None
    assert result.components["answered"] == 0.0
    assert not result.self_consistency_assessable


def test_metrics_are_reported_per_arm() -> None:
    server = _server()
    rows = [
        _verify("CRITERIA: PVS1 VeryStrong; PM2 Supporting\nCLASSIFICATION: LP", "example_lp", "with_rules"),
        _verify("CRITERIA: BA1 StandAlone\nCLASSIFICATION: B", "example_b", "with_rules"),
        _verify("No answer.", "example_lp", "without_rules"),
        _verify("CRITERIA: PM2 Supporting\nCLASSIFICATION: VUS", "example_excluded", "without_rules"),
    ]
    metrics = server.compute_metrics([[row.model_dump()] for row in rows])
    assert metrics["acmg/with_rules/responses"] == 2
    assert metrics["acmg/with_rules/label_accuracy"] == 1.0
    assert metrics["acmg/with_rules/criterion_f1"] == 1.0
    assert metrics["acmg/with_rules/output_failure_rate"] == 0.0
    # The excluded-label case leaves the label denominator but still counts for criteria and output failures.
    assert metrics["acmg/without_rules/label_scorable"] == 1
    assert metrics["acmg/without_rules/label_accuracy"] == 0.0
    assert metrics["acmg/without_rules/output_failure_rate"] == 0.5
    assert metrics["acmg/without_rules/criterion_recall"] == pytest.approx(1 / 3)
    assert metrics["acmg/without_rules/self_consistency_assessable"] == 1
    assert metrics["acmg/without_rules/self_consistency"] == 1.0
    assert server.compute_repeat_metrics([[row.model_dump()] for row in rows]) == metrics
    key = server.get_key_metrics(metrics)
    assert set(key) == {
        f"acmg/{arm}/{name}"
        for arm in ("with_rules", "without_rules")
        for name in ("label_accuracy", "criterion_f1", "self_consistency", "official_reward")
    }
    assert "acmg/all/label_accuracy" not in metrics
