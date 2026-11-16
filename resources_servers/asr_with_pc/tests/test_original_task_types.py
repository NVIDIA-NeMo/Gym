# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The original task types score and aggregate exactly as before.

The expected values were produced by the server as it was before the OpenASR and Multilingual-ASR
task types were added. They pin the response fields (including the zeroed
PC variants of the non-PC task types) and the aggregate metrics.
"""

import threading
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.core import NeMoGymResponse
from resources_servers.asr_with_pc.app import ASRWithPCConfig, ASRWithPCResourcesServer, ASRWithPCVerifyRequest


ZEROED = {"wer_c": 0.0, "wer_pc": 0.0, "per": 0.0, "ref_pc_tok": "", "hyp_pc_tok": "", "ref_c": "", "hyp_c": ""}
CASES = [
    pytest.param(
        "ASR-PC",
        "Hello, world! It's 5 o'clock.",
        "hello world its five oclock",
        {},
        {
            "reward": 0.0,
            "text": "hello world it is 50 clock",
            "pred_text": "hello world its 5 oclock",
            "wer": 0.6666666666666666,
            "wer_c": 0.6,
            "wer_pc": 0.9166666666666666,
            "per": 1.0,
            "is_correct": False,
            "ref_pc_tok": "Hello , world ! It ' s 5 o ' clock .",
            "hyp_pc_tok": "hello world its five oclock",
            "ref_c": "Hello world Its 5 oclock",
            "hyp_c": "hello world its five oclock",
        },
        id="asr-pc-wrong",
    ),
    pytest.param(
        "ASR",
        "Hello, World. Dr. Smith has 2 cats.",
        "hello world doctor smith has two cats",
        {},
        {
            "reward": 1.0,
            "text": "hello world doctor smith has 2 cats",
            "pred_text": "hello world doctor smith has 2 cats",
            "wer": 0.0,
            "is_correct": True,
            **ZEROED,
        },
        id="asr-correct",
    ),
    pytest.param(
        "ASR",
        "",
        "something",
        {},
        {"reward": 0.0, "text": "", "pred_text": "something", "wer": 0.0, "is_correct": False, **ZEROED},
        id="asr-empty-reference",
    ),
    pytest.param(
        "ASR",
        "hello world",
        "",
        {},
        {"reward": 0.0, "text": "hello world", "pred_text": "empty", "wer": 1.0, "is_correct": False, **ZEROED},
        id="asr-empty-hypothesis",
    ),
    pytest.param(
        "Hallucination",
        "",
        "a" * 400,
        {"audio_duration": 10.0},
        {
            "reward": 0.0,
            "text": "",
            "pred_text": "a" * 400,
            "wer": 0.0,
            "is_correct": False,
            "hallucination_rate": 1.0,
            "char_rate": 2400.0,
            **ZEROED,
        },
        id="hallucination-flagged",
    ),
    pytest.param(
        "ASR_LEADERBOARD",
        "hello world",
        "totally unrelated transcription",
        {"reference_fields": ["text_alt"], "text_alt": "hello world"},
        {
            "reward": 0.0,
            "text": "hello world",
            "pred_text": "totally unrelated transcription",
            "wer": 1.5,
            "is_correct": False,
            "wer_alt": 1.5,
            "is_correct_alt": False,
            **ZEROED,
        },
        id="asr-leaderboard-wrong",
    ),
]


def _server(task_type: str = "ASR-PC") -> ASRWithPCResourcesServer:
    config = ASRWithPCConfig(host="0.0.0.0", port=8080, entrypoint="", name="", task_type=task_type)
    return ASRWithPCResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def _verify(task_type: str, reference: str, hypothesis: str, **extra) -> dict:
    response = NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": hypothesis, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    body = ASRWithPCVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "test"}], "parallel_tool_calls": True},
        response=response,
        expected_answer=reference,
        task_type=task_type,
        **extra,
    )
    return (await _server().verify(body)).model_dump()


@pytest.mark.parametrize("task_type,reference,hypothesis,extra,expected", CASES)
async def test_verify_output(task_type, reference, hypothesis, extra, expected) -> None:
    dump = await _verify(task_type, reference, hypothesis, **extra)
    assert {key: dump[key] for key in expected} == expected


async def test_aggregate_metrics_endpoint_scores_in_a_worker_thread(monkeypatch) -> None:
    """Corpus scoring must not block the event loop, and the endpoint reports the same key metrics."""
    rows = [
        await _verify("ASR", "hello world", "hello world") | {"_ng_task_index": 0, "_ng_rollout_index": 0},
        await _verify("ASR", "good night", "good day") | {"_ng_task_index": 1, "_ng_rollout_index": 0},
    ]
    server = _server()
    scoring_threads: list[int] = []
    compute_metrics = ASRWithPCResourcesServer.compute_metrics

    def spy(self, tasks):
        scoring_threads.append(threading.get_ident())
        return compute_metrics(self, tasks)

    monkeypatch.setattr(ASRWithPCResourcesServer, "compute_metrics", spy)
    aggregated = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert scoring_threads and threading.get_ident() not in scoring_threads
    assert aggregated.key_metrics["wer"] == pytest.approx(100.0 * 1 / 4)


async def test_aggregate_metrics() -> None:
    server = _server()
    pc = [
        [await _verify("ASR-PC", "Hello, world! It's 5 o'clock.", "hello world its five oclock")],
        [await _verify("ASR-PC", "Hello, world! It's 5 o'clock.", "Hello, world! It's 5 o'clock.")],
    ]
    metrics = server.compute_metrics(pc)
    assert metrics["corpus_wer@k=1"] == pytest.approx(33.33333333333333)
    assert metrics["wer_c@k=1"] == pytest.approx(30.0)
    assert metrics["wer_pc@k=1"] == pytest.approx(45.83333333333333)
    assert metrics["per@k=1"] == pytest.approx(50.0)
    assert metrics["pass@1[avg-of-1]/accuracy"] == pytest.approx(50.0)
    assert server.get_key_metrics(metrics)["wer"] == pytest.approx(33.33333333333333)

    leaderboard = [
        [
            await _verify(
                "ASR_LEADERBOARD",
                "one hundred dollars",
                "One hundred dollars",
                reference_fields=["text_tn", "text_itn"],
                text_tn="$100",
                text_itn="one hundred dollars",
            )
        ],
        [
            await _verify(
                "ASR_LEADERBOARD",
                "hello world",
                "totally unrelated transcription",
                reference_fields=["text_alt"],
                text_alt="hello world",
            )
        ],
    ]
    metrics = server.compute_metrics(leaderboard)
    assert metrics["corpus_wer@k=1"] == pytest.approx(100.0)
    assert metrics["wer_tn@k=1"] == pytest.approx(0.0)
    assert metrics["wer_itn@k=1"] == pytest.approx(0.0)
    assert metrics["wer_alt@k=1"] == pytest.approx(150.0)
    assert metrics["wer_c@k=1"] == pytest.approx(0.0)
    key = server.get_key_metrics(metrics)
    assert (key["wer"], key["wer_alt"], key["wer_tn"]) == pytest.approx((100.0, 150.0, 0.0))


def test_rows_without_pc_variants_stay_out_of_the_means() -> None:
    """Only rows that report ``None`` for the PC variants (the edit-count task types) are left out of their means."""
    row = {"is_correct": True, "pred_text": "x", "text": "x", "wer_c": None, "wer_pc": None, "per": None}
    metrics = _server().compute_metrics([[row]])
    assert not {"wer_c@k=1", "wer_pc@k=1", "per@k=1", "pass@1[avg-of-1]/per"} & metrics.keys()


def test_hallucination_rows_with_empty_reference_aggregate() -> None:
    rows = [[{"is_correct": True, "pred_text": "short", "text": "", "hallucination_rate": 0.0}]]
    metrics = _server("Hallucination").compute_metrics(rows)
    assert "corpus_wer@k=1" not in metrics
    assert metrics["hallucination_rate@k=1"] == 0.0
