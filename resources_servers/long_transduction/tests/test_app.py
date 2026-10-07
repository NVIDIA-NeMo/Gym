# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise verifier dispatch, partial credit, reasoning, and aggregate metrics."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.long_transduction.app import (
    LongTransductionConfig,
    LongTransductionServer,
    LongTransductionVerifyRequest,
    _strip_reasoning,
)


@pytest.fixture
def server() -> LongTransductionServer:
    return LongTransductionServer(
        config=LongTransductionConfig(host="127.0.0.1", port=8080, name="long_transduction", entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )


def make_request(payload: dict[str, object], output: str) -> LongTransductionVerifyRequest:
    return LongTransductionVerifyRequest.model_validate(
        {
            **payload,
            "responses_create_params": {"input": [{"role": "user", "content": "Test task"}]},
            "response": {
                "id": "test-response",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "test-message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": output, "annotations": []}],
                    }
                ],
            },
        }
    )


CASES = [
    ("unnumbered_streaming_sum", {"expressions": [{"expr": "1+2", "answer": 3}]}, "1+2=3"),
    ("streaming_sum", {"expressions": [{"expr": "1+2", "answer": 3}]}, "[1]1+2=3"),
    ("shuffled_streaming_sum", {"expressions": [{"expr": "1+2", "answer": 3}]}, "[1]1+2=3"),
    ("unnumbered_uuid_sort", {"uuid_lines": [["bbbbbbbb", "aaaaaaaa"]]}, "aaaaaaaa,bbbbbbbb"),
    ("streaming_uuid_sort", {"uuid_lines": [["bbbbbbbb", "aaaaaaaa"]]}, "[1]aaaaaaaa,bbbbbbbb"),
    ("shuffled_streaming_uuid_sort", {"uuid_lines": [["bbbbbbbb", "aaaaaaaa"]]}, "[1]aaaaaaaa,bbbbbbbb"),
    ("unnumbered_var_expand", {"expressions": [{"expr": "abc+def", "answer": "big dog"}]}, "big dog"),
    ("streaming_var_expand", {"expressions": [{"expr": "abc+def", "answer": "big dog"}]}, "[1]big dog"),
    ("shuffled_streaming_var_expand", {"expressions": [{"expr": "abc+def", "answer": "big dog"}]}, "[1]big dog"),
    *[
        (kind, {"expected_output": ",[C0]\n[R0],big dog", "n_rows": 1, "n_cols": 1}, ",[C0]\n[R0],big dog")
        for kind in ("csv_permutation_homogeneous", "csv_permutation_heterogeneous", "csv_kv_lookup")
    ],
]


@pytest.mark.parametrize("kind,payload,gold", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("response_kind", ["correct", "empty", "wrong", "reasoning"])
async def test_verify_all_task_types(
    server: LongTransductionServer, kind: str, payload: dict[str, object], gold: str, response_kind: str
) -> None:
    output = {"correct": gold, "empty": "", "wrong": "garbage", "reasoning": f"<think>reason</think>\n{gold}"}[
        response_kind
    ]
    result = await server.verify(make_request({"type": kind, **payload}, output))
    expected = float(response_kind in {"correct", "reasoning"})
    assert result.reward == expected
    assert result.answer_correct == expected
    assert result.n_items_scored == 1
    assert result.item_scores == [[bool(expected)] * 3]
    assert result.n_items_copy_correct == int(expected)
    if kind.startswith("csv"):
        assert result.cell_accuracy == result.row_accuracy == result.col_accuracy == expected
    else:
        assert result.cell_accuracy is None


async def test_partial_arithmetic_credit_and_default_type(server: LongTransductionServer) -> None:
    result = await server.verify(
        make_request({"expressions": [{"expr": "1+2", "answer": 3}, {"expr": "2+3", "answer": 5}]}, "1+2=3\n2+3=9")
    )
    assert result.reward == 0.5
    assert result.n_items_copy_correct == 2
    assert result.item_scores == [[True, True, True], [True, False, False]]


async def test_partial_csv_credit(server: LongTransductionServer) -> None:
    result = await server.verify(
        make_request(
            {"type": "csv_kv_lookup", "expected_output": ",[C0],[C1]\n[R0],a,b\n[R1],c,d", "n_rows": 2, "n_cols": 2},
            ",[C0],[C1]\n[R0],a,b\n[R1],c,wrong",
        )
    )
    assert result.reward == result.cell_accuracy == 0.75
    assert result.row_accuracy == result.col_accuracy == 0.5
    assert result.n_items_scored == 4
    assert result.n_items_copy_correct == 3


@pytest.mark.parametrize("kind", ["streaming_sum", "streaming_uuid_sort", "streaming_var_expand", "csv_kv_lookup"])
async def test_no_items_scores_zero(server: LongTransductionServer, kind: str) -> None:
    result = await server.verify(make_request({"type": kind}, ""))
    assert result.reward == 0.0
    assert result.n_items_scored == 0


async def test_no_output_message(server: LongTransductionServer) -> None:
    request = make_request({"expressions": [{"expr": "1+2", "answer": 3}]}, "")
    request.response.output = []
    assert (await server.verify(request)).reward == 0.0


async def test_unknown_type_is_actionable(server: LongTransductionServer) -> None:
    with pytest.raises(ValueError, match="Unsupported long_transduction sample type: 'unknown'"):
        await server.verify(make_request({"type": "unknown"}, ""))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("answer", "answer"),
        ("<think>unfinished", ""),
        ("reason</think>\nanswer", "answer"),
        ("<think>one</think><think>two</think>\nanswer", "answer"),
        ("<|channel|>analysis<|message|>unfinished", ""),
        ("<|channel|>analysis<|message|>reason<|channel|>final<|message|>\nanswer", "answer"),
    ],
)
def test_reasoning_formats(text: str, expected: str) -> None:
    assert _strip_reasoning(text) == expected


async def test_reasoning_stripping_can_be_disabled(server: LongTransductionServer) -> None:
    server.config.strip_reasoning = False
    request = make_request(
        {"type": "unnumbered_var_expand", "expressions": [{"answer": "big dog"}]}, "<think>x</think>\nbig dog"
    )
    assert (await server.verify(request)).reward == 0.0


def test_metrics_group_types_and_difficulty(server: LongTransductionServer) -> None:
    tasks = [
        [{"answer_correct": 1.0, "max_operands": 2}, {"answer_correct": 0.0, "max_operands": 2}],
        [{"type": "streaming_uuid_sort", "answer_correct": 0.5, "uuids_per_line": 4}],
        [{"type": "csv_permutation_homogeneous", "answer_correct": 0.25, "perm_fraction": 0.2}],
        [{"type": "csv_kv_lookup", "answer_correct": 0.75, "vocab_fraction": 0.5}],
        [{"type": "streaming_var_expand", "answer_correct": 1.0, "n_variables": 8}],
        [{"type": "unnumbered_var_expand", "answer_correct": 0.0}],
        [{"reward": 1.0}, {"answer_correct": None}],
    ]
    metrics = server.compute_metrics(tasks)
    assert metrics["overall_accuracy"] == 0.5
    assert metrics["difficulty_2"] == {"accuracy": 0.5, "n": 2}
    assert metrics["type_unnumbered_streaming_sum"] == {"accuracy": 0.5, "n": 2}
    assert metrics["type_streaming_uuid_sort_difficulty_4"] == {"accuracy": 0.5, "n": 1}
    assert metrics["difficulty_n/a"] == {"accuracy": 0.0, "n": 1}
    assert metrics["type_csv_permutation_homogeneous_difficulty_0.2"]["accuracy"] == 0.25
    assert metrics["type_csv_kv_lookup_difficulty_0.5"]["accuracy"] == 0.75
    assert metrics["type_streaming_var_expand_difficulty_8"]["accuracy"] == 1.0
    key_metrics = server.get_key_metrics(metrics)
    assert key_metrics["type_streaming_uuid_sort_difficulty_4"] == 0.5
    assert "overall_accuracy" not in key_metrics
    assert server.compute_metrics([]) == {}
    assert server.get_key_metrics({"empty": {"accuracy": None}}) == {}


async def test_committed_real_rollouts_reproduce_rewards(server: LongTransductionServer) -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    examples = [json.loads(line) for line in (data_dir / "example.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in (data_dir / "example_rollouts.jsonl").read_text().splitlines()]
    assert len(examples) == len(rollouts) == 5
    for example, rollout in zip(examples, rollouts):
        result = await server.verify(LongTransductionVerifyRequest(**example, response=rollout["response"]))
        assert result.reward == pytest.approx(rollout["reward"])
        assert result.item_scores == rollout["item_scores"]
