# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.core import NeMoGymResponse
from nemo_gym.verification import judge_failsafe
from resources_servers.voicebench.app import (
    VoiceBenchConfig,
    VoiceBenchServer,
    VoiceBenchVerifyRequest,
    _ensure_nltk,
    final_answer,
)
from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.scorers.open_judge import score_outputs
from resources_servers.voicebench.tests.test_judge import client, judge_settings


def candidate(answer="answer", *, status="completed", output=None):
    return NeMoGymResponse.model_validate(
        {
            "id": "candidate",
            "created_at": 0,
            "model": "audio-model",
            "object": "response",
            "output": output
            if output is not None
            else [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": answer, "annotations": []}],
                }
            ],
            "status": status,
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        }
    )


def request(sample_id="test_0", *, response=None, subset="alpacaeval_full", **metadata):
    return VoiceBenchVerifyRequest.model_validate(
        {
            "id": f"voicebench/{subset}/{sample_id}",
            "responses_create_params": {
                "input": [{"role": "user", "content": "Please provide a response to the audio prompt."}],
                "metadata": {"audio_ref": '{"archive":"alpaca.tar","member":"audio/test_0.wav"}'},
            },
            "verifier_metadata": {"subset": subset, "sample_id": sample_id, "prompt": "spoken transcript", **metadata},
            "response": (response or candidate()).model_dump(),
        }
    )


def server(tmp_path, *, expected=1, ids=None, subset="alpacaeval_full", scoring_type="open", **overrides):
    values = judge_settings(tmp_path).model_dump() if scoring_type in ("open", "reference_qa") else {}
    if ids is not None:
        path = tmp_path / "ids.json"
        path.write_text(json.dumps(ids))
        values["expected_sample_ids_file"] = str(path)
    config = VoiceBenchConfig(
        name="voicebench_alpaca",
        host="127.0.0.1",
        port=8104,
        entrypoint="app.py",
        subset=subset,
        scoring_type=scoring_type,
        expected_num_rows=expected,
        **(values | overrides),
    )
    return VoiceBenchServer(config=config, server_client=client())


def test_types_reject_wrong_subset_scorer_and_missing_reference(tmp_path):
    with pytest.raises(ValidationError):
        server(tmp_path, scoring_type="mcq")
    with pytest.raises(ValidationError):
        request(subset="openbookqa")
    with pytest.raises(ValidationError):
        request(subset="ifeval", instruction_id_list=["x"], kwargs=[])
    with pytest.raises(ValidationError):
        request(subset="ifeval", instruction_id_list=["x"], kwargs=[None])
    with pytest.raises(ValidationError):
        server(tmp_path, num_workers=2)


def test_nltk_cache_existing_permissions_are_preserved_and_default_is_private(tmp_path):
    existing = tmp_path / "shared"
    existing.mkdir(mode=0o755)
    fake_nltk = SimpleNamespace(data=SimpleNamespace(path=[], find=MagicMock()), download=MagicMock())
    with patch.dict("sys.modules", {"nltk": fake_nltk}):
        _ensure_nltk(str(existing))
        assert existing.stat().st_mode & 0o777 == 0o755
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "home", return_value=tmp_path):
            _ensure_nltk(None)
    assert (tmp_path / ".voicebench_nltk_data").stat().st_mode & 0o777 == 0o700
    fake_nltk.download.assert_not_called()


def test_only_last_assistant_final_text_and_thinking_is_preserved(tmp_path):
    original = candidate().output[0].model_dump()
    last = candidate("last final").output[0].model_dump()
    response = candidate(
        output=[
            original,
            {"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": "private reasoning"}]},
            last,
        ]
    )
    assert final_answer(response) == "last final"
    result = asyncio.run(server(tmp_path).verify(request(response=response)))
    assert result.candidate_answer == "last final" and result.candidate_thinking == ["private reasoning"]
    assert result.id == "voicebench/alpacaeval_full/test_0"
    assert result.responses_create_params.metadata["audio_ref"].startswith('{"archive"')
    assert result.voicebench_score == 4 and result.reward == 0.8


@pytest.mark.parametrize(
    "answer, reasoning, expected",
    [
        ("\nC", "private reasoning", "C"),
        ("\n\nA", "private reasoning", "A"),
        (" C \t\n", "private reasoning", " C"),
        ("\nD\n", "private reasoning", "D"),
        ("\r\nC", "private reasoning", "\r\nC"),
        ("  answer\n", None, "answer"),
        ("<think>reason</think>\nanswer\n", None, "answer"),
        ("<think>reason</think> answer \n", None, " answer"),
        ("<think>first</think>A</think>\nB\n", "private reasoning", "B"),
        ("<think>reason</think>\n", "private reasoning", ""),
        ("", "  reasoning answer\n", "reasoning answer"),
        (" \t\n", "<think>reason</think>\n answer \n", " answer"),
        ("", "<think>reason</think>\n", ""),
        ("", None, ""),
        (" \t\n", " \t\n", ""),
    ],
)
def test_legacy_final_answer_normalizes_assembled_prediction(answer, reasoning, expected):
    output = [candidate(answer).output[0].model_dump()]
    if reasoning is not None:
        output.insert(0, {"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    response = candidate(output=output)
    original = response.model_dump(mode="json")
    assert final_answer(response) == expected
    assert response.model_dump(mode="json") == original


@pytest.mark.parametrize("subset", ["openbookqa", "mmsu"])
def test_mcq_scores_parsed_final_with_leading_lf_and_preserves_raw_response(tmp_path, subset):
    response = candidate(
        output=[
            {"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": "private reasoning"}]},
            candidate("\nC\n").output[0].model_dump(),
        ]
    )
    original = response.model_dump(mode="json")
    svc = server(tmp_path, subset=subset, scoring_type="mcq")
    result = asyncio.run(svc.verify(request(subset=subset, response=response, reference="C")))
    assert result.candidate_answer == result.scoring_answer == result.prediction == "C"
    assert result.reward == 1 and result.score_valid and result.candidate_valid
    assert not result.parse_failure and not result.legacy_random_fallback
    assert result.response.model_dump(mode="json") == original
    assert result.candidate_thinking == ["private reasoning"]
    svc.server_client.post.assert_not_called()


def test_incomplete_response_keeps_raw_score_but_zero_status_safe(tmp_path):
    result = asyncio.run(server(tmp_path).verify(request(response=candidate(status="incomplete"))))
    assert result.score_valid and result.voicebench_score == 4
    assert not result.candidate_valid and result.status_safe_reward == 0
    assert result.candidate_incomplete_details == {"reason": "max_output_tokens"}


def test_subset_runtime_mismatch_rejected_before_judge(tmp_path):
    svc = server(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(svc.verify(request(subset="commoneval")))
    svc.server_client.post.assert_not_called()


def test_rule_scorer_has_no_judge_or_nltk_setup(tmp_path):
    svc = server(tmp_path, subset="openbookqa", scoring_type="mcq")
    rule = MagicMock()
    rule.score.return_value = ScoringResult(1, {"correct": 1, "total": 1}, {"parse_failure": False})
    with patch("resources_servers.voicebench.app.importlib.import_module", return_value=rule):
        result = asyncio.run(svc.verify(request(subset="openbookqa", reference="A")))
    svc.server_client.post.assert_not_called()
    assert result.reward == 1 and result.score_valid


def test_expected_distinct_ids_guard_duplicate_missing_and_unexpected(tmp_path):
    svc = server(tmp_path, expected=2, ids=["test_0", "test_1"])
    row0 = asyncio.run(svc.verify(request("test_0"))).model_dump()
    duplicate = svc.compute_metrics([[row0], [row0]])
    assert not duplicate["voicebench/complete"]
    assert duplicate["voicebench/duplicate_rows"] == duplicate["voicebench/missing_rows"] == 1
    row1 = asyncio.run(svc.verify(request("test_1"))).model_dump()
    metrics = svc.compute_metrics([[row0], [row1]])
    assert metrics["voicebench/complete"] and metrics["voicebench/distinct_sample_ids"] == 2
    bad = row1 | {"sample_id": "unexpected"}
    metrics = svc.compute_metrics([[row0], [bad]])
    assert not metrics["voicebench/complete"]
    assert metrics["voicebench/missing_rows"] == metrics["voicebench/unexpected_sample_ids"] == 1


def test_cached_judges_aggregate_choice_weighted_and_actual_gym_hook(tmp_path):
    svc = server(tmp_path, expected=2)
    rows = []
    for sample_id, outputs in (("test_0", ["5"]), ("test_1", ["1", "1", "1"])):
        scored = score_outputs(outputs)
        rows.append(
            scored.details
            | {
                "sample_id": sample_id,
                "score_valid": scored.valid,
                "scoring_metrics": scored.metrics,
                "candidate_valid": True,
                "reward": scored.reward,
                "status_safe_reward": scored.reward,
                "_ng_task_index": len(rows),
                "_ng_rollout_index": 0,
            }
        )
    aggregated = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))
    assert aggregated.key_metrics["voicebench/raw_mean_valid"] == 2
    assert aggregated.key_metrics["voicebench/normalized_mean_valid"] == 40
    assert aggregated.key_metrics["voicebench/normalized_score_all_rows"] == 60
    assert aggregated.key_metrics["voicebench/complete"]


def test_judge_transport_failure_uses_masked_failsafe(tmp_path):
    svc = server(tmp_path)
    svc.server_client.post.side_effect = ConnectionError("judge unavailable")
    response = asyncio.run(judge_failsafe(svc.verify)(body=request()))
    result = json.loads(response.body)
    assert result["_ng_failure_class"] == "judge_failed" and result["mask_sample"]
    assert result["response"]["id"] == "candidate"


def test_native_aggregate_counts_masked_transport_rows_in_coverage_and_safe_score(tmp_path):
    svc = server(tmp_path, expected=2, ids=["test_0", "test_1"])
    valid = asyncio.run(svc.verify(request())).model_dump(mode="json") | {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
    }
    svc.server_client.post.side_effect = ConnectionError("judge unavailable")
    failed = asyncio.run(judge_failsafe(svc.verify)(body=request("test_1")))
    masked = json.loads(failed.body) | {"_ng_task_index": 1, "_ng_rollout_index": 0}
    aggregate = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=[valid, masked])))
    metrics = aggregate.key_metrics
    assert metrics["voicebench/count"] == metrics["voicebench/distinct_sample_ids"] == 2
    assert metrics["voicebench/valid_scores"] == metrics["voicebench/invalid_scores"] == 1
    assert metrics["voicebench/judge_coverage"] == 0.5
    assert metrics["voicebench/normalized_mean_valid"] == 80
    assert metrics["voicebench/status_safe_normalized_score"] == 40
    assert metrics["voicebench/normalized_score_all_rows"] == 40
    assert metrics["voicebench/missing_rows"] == 0
    assert not metrics["voicebench/complete"] and not metrics["voicebench/sample_complete"]


def test_native_aggregate_all_transport_failures_has_coverage_without_invented_score(tmp_path):
    svc = server(tmp_path)
    svc.server_client.post.side_effect = ConnectionError("judge unavailable")
    failed = asyncio.run(judge_failsafe(svc.verify)(body=request()))
    aggregate = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=[json.loads(failed.body)])))
    metrics = aggregate.key_metrics
    assert metrics["voicebench/count"] == metrics["voicebench/invalid_scores"] == 1
    assert metrics["voicebench/raw_mean_valid"] is None
    assert metrics["voicebench/status_safe_normalized_score"] == 0
    assert not metrics["voicebench/complete"]


def test_ifeval_aggregation_uses_four_denominator_aware_ratios(tmp_path):
    with patch("resources_servers.voicebench.app._ensure_nltk"):
        svc = server(tmp_path, expected=2, subset="ifeval", scoring_type="ifeval")
    metrics0 = {
        "strict_prompt_correct": 1,
        "loose_prompt_correct": 1,
        "prompt_total": 1,
        "strict_instruction_correct": 1,
        "loose_instruction_correct": 1,
        "instruction_total": 1,
    }
    metrics1 = {
        "strict_prompt_correct": 0,
        "loose_prompt_correct": 0,
        "prompt_total": 1,
        "strict_instruction_correct": 0,
        "loose_instruction_correct": 0,
        "instruction_total": 3,
    }
    rows = [
        {
            "sample_id": str(i),
            "score_valid": True,
            "candidate_valid": True,
            "scoring_metrics": metrics,
            "reward": reward,
            "status_safe_reward": reward,
        }
        for i, (metrics, reward) in enumerate(((metrics0, 1), (metrics1, 0)))
    ]
    aggregated = svc.compute_metrics([[row] for row in rows])
    assert aggregated["voicebench/legacy/final"] == 0.375  # mean(.5,.5,.25,.25), not row reward mean(.5)
    assert aggregated["voicebench/status_safe_normalized_score"] == 37.5


def test_legacy_blank_prediction_and_reasoning_fallback(tmp_path):
    svc = server(tmp_path, subset="advbench", scoring_type="refusal")
    blank = asyncio.run(svc.verify(request(subset="advbench", response=candidate(""))))
    assert blank.scoring_answer == "nan" and blank.reward == 0
    assert not blank.candidate_valid
    assert final_answer(candidate("  answer\n")) == "answer"
    assert final_answer(candidate("<think>reason</think>\nanswer\n")) == "answer"
    response = candidate(
        output=[{"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": "reasoning answer"}]}]
    )
    assert final_answer(response) == "reasoning answer"


def test_unparseable_judge_prevents_partial_legacy_aggregation(tmp_path):
    svc = server(tmp_path)
    svc.server_client = client(["garbage"])
    svc._judge.server_client = svc.server_client
    row = asyncio.run(svc.verify(request()))
    assert not row.score_valid and row.legacy_scoring_error
    with pytest.raises(ValueError, match="legacy VoiceBench evaluation failed"):
        svc.compute_metrics([[row.model_dump(mode="json")]])


@pytest.mark.parametrize("outputs, expected", [(["NaN"], "NaN"), ([], "NaN"), (["6"], 120.0)])
def test_nonfinite_legacy_metrics_survive_json_transport(tmp_path, outputs, expected):
    svc = server(tmp_path)
    svc._judge.server_client = client(outputs)
    row = asyncio.run(svc.verify(request()))
    saved = json.loads(row.model_dump_json())
    assert saved["score_valid"]
    restored = type(row).model_validate(saved)
    assert restored.reward == (1.2 if outputs == ["6"] else 0.0)
    metrics = svc.compute_metrics([[saved]])
    assert metrics["voicebench/legacy/normalized_score"] == expected
