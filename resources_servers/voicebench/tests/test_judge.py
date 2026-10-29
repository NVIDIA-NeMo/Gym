# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import math
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from nemo_gym.client import ServerClient
from nemo_gym.judge import JudgeError
from resources_servers.voicebench.scorers.open_judge import JudgeSettings, OpenJudgeScorer, parse_score, score_outputs
from resources_servers.voicebench.scorers.reference_qa import ReferenceQAScorer


def http_response(contents, *, finish="stop"):
    payload = {
        "id": "judge",
        "created": 0,
        "model": "judge-model",
        "object": "chat.completion",
        "choices": [
            {"index": i, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
            for i, text in enumerate(contents)
        ],
    }
    response = MagicMock()
    response.status = 200
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    response.content.read = AsyncMock(return_value=json.dumps(payload).encode())
    return response


def judge_settings(tmp_path, **overrides):
    prompt = tmp_path / "judge.yaml"
    prompt.write_text(yaml.safe_dump({"system": "judge system", "user": "{prompt}\n{reference}\n{response}"}))
    return JudgeSettings(
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_prompt_file=str(prompt),
        **overrides,
    )


def client(contents=("4",), **kwargs):
    result = MagicMock(spec=ServerClient)
    result.post = AsyncMock(return_value=http_response(contents, **kwargs))
    return result


@pytest.mark.parametrize("value,expected", [("4.5", 4.5), ("[[4]]", 4), (r"\boxed{ 5 }", 5)])
def test_legacy_formats(value, expected):
    assert parse_score(value) == expected


@pytest.mark.parametrize("value", ["garbage", "[[4.5]]"])
def test_legacy_unparseable_grades_raise(value):
    with pytest.raises(NotImplementedError):
        parse_score(value)


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf", "0", "6"])
def test_legacy_accepts_nonfinite_and_out_of_range(value):
    actual = parse_score(value)
    expected = float(value)
    assert math.isnan(actual) if math.isnan(expected) else actual == expected
    scored = score_outputs([value])
    assert scored.valid and not scored.details["judge_numeric_valid"]
    assert math.isfinite(scored.reward)


def test_legacy_missing_content_fails():
    with pytest.raises(AttributeError):
        parse_score(None)


def test_pure_cached_choices_no_http_and_explicit_mismatch():
    scored = score_outputs(["4"], requested_n=3)
    assert scored.valid and scored.reward == 0.8
    assert scored.details["judge_choice_count_mismatch"]
    assert scored.metrics == {"score_sum": 4.0, "score_count": 1.0}
    with pytest.raises(NotImplementedError):
        score_outputs(["4", "bad"])
    empty = score_outputs([])
    assert math.isnan(empty.details["voicebench_score"]) and empty.reward == 0
    assert score_outputs(["4"], finish_reasons=["length"]).reward == 0.8


def test_exact_text_only_judge_transport_and_requested_settings(tmp_path):
    network = client()
    scorer = OpenJudgeScorer(judge_settings(tmp_path), network)
    result = asyncio.run(scorer.score("answer {prompt}", {"prompt": "question {response}", "reference": "gold"}))
    params = network.post.call_args.kwargs["json"].model_dump(exclude_unset=True)
    assert params["messages"] == [
        {"role": "system", "content": "judge system"},
        {"role": "user", "content": "question answer {prompt}\ngold\nanswer {prompt}"},
    ]
    assert params["temperature"] == 0.5 and params["top_p"] == 0.95
    assert params["n"] == 3 and params["max_tokens"] == 1024
    assert params["frequency_penalty"] == params["presence_penalty"] == 0 and params["stop"] is None
    assert result.valid and result.details["judge_actual_n"] == 1
    assert result.details["judge_call_seconds"] >= 0 and result.details["judge_queue_seconds"] >= 0
    assert result.details["judge_prompt_sha256"] == scorer.prompt_sha256
    assert "audio_ref" not in json.dumps(params)


@pytest.mark.parametrize("limit,requests", [(2, 8), (40, 48)])
def test_shared_semaphore_limits_active_calls_and_reports_peak(tmp_path, limit, requests):
    active = 0
    peak = 0
    network = client()

    async def post(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return http_response(["4"])

    network.post.side_effect = post
    scorer = OpenJudgeScorer(judge_settings(tmp_path, judge_max_concurrency=limit), network)

    async def run():
        return await asyncio.gather(*(scorer.score("answer", {"prompt": "question"}) for _ in range(requests)))

    results = asyncio.run(run())
    assert peak == scorer.peak_inflight == limit
    assert scorer.inflight == 0 and scorer.total_calls == requests
    assert all(result.valid for result in results)
    assert max(result.details["judge_inflight_at_start"] for result in results) == limit
    assert all(result.details["judge_configured_concurrency"] == limit for result in results)
    assert any(result.details["judge_queue_seconds"] > 0.005 for result in results)


def test_transport_failure_remains_judge_error_and_releases_slot(tmp_path):
    network = client()
    network.post.side_effect = ConnectionError("judge unreachable")
    scorer = OpenJudgeScorer(judge_settings(tmp_path), network)
    with pytest.raises(JudgeError):
        asyncio.run(scorer.score("answer", {"prompt": "question"}))
    assert scorer.inflight == 0


def test_reference_qa_uses_reference_and_majority_without_open_scale(tmp_path):
    scorer = ReferenceQAScorer(judge_settings(tmp_path), client(["Yes", "No", "Yes"]))
    result = asyncio.run(scorer.score("candidate", {"prompt": "question", "reference": "gold"}))
    assert result.valid and result.reward == 1
    assert result.metrics == {"correct": 1, "total": 1}
    tie = ReferenceQAScorer(judge_settings(tmp_path), client(["Yes", "No"]))
    result = asyncio.run(tie.score("candidate", {"prompt": "question", "reference": "gold"}))
    assert result.valid and result.reward == float(max(set(["yes", "no"]), key=["yes", "no"].count) == "yes")
    assert result.details["qa_vote_tie"] and result.details["qa_vote_tie_policy"] == "legacy_hash_order"


@pytest.mark.parametrize("outputs", [["Yes", "Yes", "garbage"], ["No", "unexpected", "unexpected"]])
def test_reference_qa_votes_over_unexpected_strings(tmp_path, outputs):
    scorer = ReferenceQAScorer(judge_settings(tmp_path), client(outputs, finish="length"))
    result = asyncio.run(scorer.score("candidate", {"prompt": "question", "reference": "gold"}))
    lowered = [value.lower() for value in outputs]
    expected = max(set(lowered), key=lowered.count)
    assert result.valid and result.reward == float(expected == "yes")
    assert result.details["qa_vote_winner"] == expected


def test_reference_template_is_validated_before_scoring(tmp_path):
    settings = judge_settings(tmp_path)
    from pathlib import Path

    Path(settings.judge_prompt_file).write_text(yaml.safe_dump({"user": "{prompt}\n{response}"}))
    with pytest.raises(ValueError, match="reference"):
        ReferenceQAScorer(settings, client())
