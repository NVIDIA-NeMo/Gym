# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.core import NeMoGymResponse
from resources_servers.mcqa.app import (
    MCQAResourcesServer,
    MCQAResourcesServerConfig,
    MCQAVerifyRequest,
    extract_boxed_and_normalized_answers,
)


OPTIONS = [{"A": "first option"}, {"B": "second option"}, {"C": "third option"}, {"D": "fourth option"}]


def server():
    return MCQAResourcesServer(
        config=MCQAResourcesServerConfig(
            host="127.0.0.1", port=12345, entrypoint="app.py", name="dual", grading_mode="boxed_and_normalized"
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )


def request(text, gold="B"):
    response = NeMoGymResponse(
        id="r",
        created_at=0,
        model="test",
        object="response",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            {"type": "reasoning", "id": "thought", "summary": [{"type": "summary_text", "text": r"\boxed{A}"}]},
            {
                "type": "message",
                "id": "m",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        ],
    )
    return MCQAVerifyRequest(
        responses_create_params={"input": []}, response=response, options=OPTIONS, expected_answer=gold
    )


@pytest.mark.parametrize(
    "text, strict, normalized",
    [
        (r"Explanation. \boxed{B}", "B", "B"),
        (r"\boxed{\text{B}}", "B", "B"),
        (r"\boxed{B: second option}", "B", "B"),
        (r"\boxed{A} then \boxed{B}", "B", "B"),
        (r"\boxed{b}", None, "B"),
        (r"\boxed { B }", None, "B"),
        (r"\boxed{Ｂ}", None, "B"),
        ("B", None, "B"),
        (" **(b).** ", None, "B"),
        ("Ｂ", None, "B"),
        ("Answer: B", None, "B"),
        ("**Answer:** **B**", None, "B"),
        ("The answer is (B).", None, "B"),
        ("Answer: (B) because D is incorrect.", None, "B"),
        ("Answer: (b) because D is incorrect.", None, "B"),
        ("Answer: (E) because D is incorrect.", None, None),
        ("Answer: A\nAnswer: B", None, "B"),
        ("Answer: A\nAnswer: B or C", None, None),
        ("B or C", None, None),
        (r"\boxed{B or C}", None, None),
        (r"Answer: B\n\boxed{A}", "A", "A"),
        (r"\boxed{B} then \boxed{C}", "C", "C"),
        (r"\boxed{B} then \boxed{A or C}", None, None),
        (r"\boxed{B", None, None),
        (r"Answer: B\n\boxed{E}", None, None),
        (r"\boxed{second option}", None, "B"),
        ("Answer: SECOND   OPTION", None, "B"),
        ("second option", None, "B"),
        ("Explanation mentions A, B, C, and D without a final answer.", None, None),
        ("A proposition is being considered.", None, None),
        ("", None, None),
    ],
)
async def test_dual_scores_reuse_boxed_parser_and_normalize_final_answer(text, strict, normalized):
    result = await server().verify(request(text))
    assert result.strict_boxed_answer == strict
    assert result.normalized_answer == normalized
    assert result.reward == result.strict_boxed_reward == float(strict == "B")
    assert result.normalized_reward == float(normalized == "B")
    assert result.reward_components == {"strict_boxed": float(strict == "B"), "normalized": float(normalized == "B")}


@pytest.mark.parametrize("text", ["বড় নদী", "Answer: বড় নদী", r"\boxed{বড় নদী}"])
async def test_normalization_matches_exact_bengali_option_before_replacing_labels(text):
    body = request(text).model_copy(update={"options": [{"A": "first"}, {"B": "বড় নদী"}, {"C": "third"}]})
    result = await server().verify(body)
    assert result.strict_boxed_answer is None
    assert result.normalized_answer == "B"
    assert result.reward == result.strict_boxed_reward == 0
    assert result.normalized_reward == 1


def test_normalization_does_not_guess_option_substrings_or_duplicate_text():
    assert extract_boxed_and_normalized_answers(r"\boxed{12}", options=[{"A": "2"}, {"B": "3"}]) == (None, None)
    assert extract_boxed_and_normalized_answers("same", options=[{"A": "same"}, {"B": "same"}]) == (None, None)
    assert extract_boxed_and_normalized_answers("বড় নদী", options=[{"A": "বড় নদী"}, {"B": "বড় নদী"}]) == (None, None)
    assert extract_boxed_and_normalized_answers("Answer: বাংলা", options=[{"A": "first"}, {"B": "বাংলা"}]) == (None, "B")
    assert extract_boxed_and_normalized_answers("Answer: বাংলা", options=OPTIONS) == (None, None)
    assert extract_boxed_and_normalized_answers("Answer: A priori", options=[{"A": "first"}, {"B": "A priori"}]) == (
        None,
        "B",
    )


@pytest.mark.parametrize("gold", ["A", "B", "C", "D"])
async def test_answer_selection_is_independent_of_gold(gold):
    result = await server().verify(request(r"Answer: B. \boxed{C}", gold=gold))
    assert result.strict_boxed_answer == result.normalized_answer == "C"
    assert result.strict_boxed_reward == result.normalized_reward == float(gold == "C")


async def test_aggregation_uses_each_metrics_own_extracted_answers():
    instance = server()
    rows = []
    for index, text in enumerate(["B", r"\boxed{C}", "Answer: B"]):
        row = (await instance.verify(request(text))).model_dump()
        row.update({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: index})
        rows.append(row)
    result = await instance.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert result.agent_metrics["pass@1/strict_boxed_accuracy"] == 0
    assert result.agent_metrics["pass@1/normalized_accuracy"] == pytest.approx(200 / 3)
    assert result.agent_metrics["majority@3/strict_boxed_accuracy"] == 0
    assert result.agent_metrics["majority@3/normalized_accuracy"] == 100
    assert result.agent_metrics["pass@1/strict_boxed_no_answer"] == pytest.approx(200 / 3)
    assert result.agent_metrics["pass@1/normalized_no_answer"] == 0
    assert result.key_metrics["pass@1/normalized_accuracy"] == pytest.approx(200 / 3)
    assert result.key_metrics["pass@1/strict_boxed_accuracy"] == 0
    with pytest.raises(ValueError, match="reverify all rows"):
        instance.compute_metrics([[rows[0], {"reward": 1.0, "extracted_answer": "B"}]])


@pytest.mark.parametrize("text, strict", [("B", None), (r"\boxed{B}", "B")])
def test_verify_endpoint_returns_both_reward_components(text, strict):
    with TestClient(server().setup_webserver()) as client:
        response = client.post("/verify", json=request(text).model_dump(mode="json"))
    assert response.status_code == 200
    data = response.json()
    assert data["strict_boxed_answer"] == data["extracted_answer"] == strict
    assert data["normalized_answer"] == "B"
    assert data["strict_boxed_reward"] == data["reward"] == float(strict == "B")
    assert data["normalized_reward"] == 1
    assert data["reward_components"] == {"strict_boxed": float(strict == "B"), "normalized": 1}
