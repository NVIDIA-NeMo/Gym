# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pytest

from benchmarks.vials.tests.test_harbor_adapter import grader


def fixture(tmp_path: Path):
    answer = tmp_path / "answer.txt"
    answer.write_text("<think>ANSWER: secret</think>\nANSWER: twelve")
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "question": "How many cells?",
                "expected_answer": "12",
                "answer_lower_bound": 11,
                "answer_upper_bound": 13,
                "images": ["images/image_1.png", "images/image_2.png"],
            }
        )
    )
    images = tmp_path / "images"
    images.mkdir()
    (images / "image_1.png").write_bytes(b"a")
    (images / "image_2.png").write_bytes(b"b")
    return {
        "answer_path": answer,
        "input_path": payload,
        "output_dir": tmp_path / "logs",
        "base_url": "https://judge.example/v1",
        "api_key": "TEST_PLACEHOLDER",
        "model": "gpt-5-mini",
    }


def reply(equivalent=True, reasoning="Equivalent."):
    return io.BytesIO(
        json.dumps(
            {
                "model": "gpt-5-mini",
                "choices": [{"message": {"content": json.dumps({"equivalent": equivalent, "reasoning": reasoning})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode()
    )


@pytest.mark.parametrize("correct", [True, False])
def test_judge_protocol_bounds_and_reward(tmp_path: Path, correct: bool) -> None:
    module = grader()
    options = fixture(tmp_path)
    calls = []

    def judge(request, **kwargs):
        calls.append(request)
        body = json.loads(request.data)
        assert request.full_url == "https://judge.example/v1/chat/completions"
        assert body["model"] == "gpt-5-mini"
        content = body["messages"][1]["content"]
        assert "lower=11" in content and "upper=13" in content
        assert "twelve" in content and "secret" not in content
        assert body["response_format"]["json_schema"]["strict"]
        return reply(correct)

    with patch.object(module, "urlopen", side_effect=judge):
        result = module.grade(**options)
    assert result["reward"] == float(correct)
    assert result["candidate_answer"] == "twelve"
    assert json.loads((options["output_dir"] / "reward.json").read_text()) == {"reward": float(correct)}
    assert len(calls) == 1


def test_multimodal_fallback_keeps_panel_order_and_combines_usage(tmp_path: Path) -> None:
    module = grader()
    options = fixture(tmp_path)
    captured = []

    def judge(request, **kwargs):
        body = json.loads(request.data)
        captured.append(body)
        return reply(False, "UNVERIFIABLE: label mapping") if len(captured) == 1 else reply(True)

    with patch.object(module, "urlopen", side_effect=judge):
        result = module.grade(**options)
    parts = captured[1]["messages"][1]["content"]
    assert [p["image_url"]["url"] for p in parts[:-1]] == ["data:image/png;base64,YQ==", "data:image/png;base64,Yg=="]
    assert result["judge_multimodal"] is True and result["reward"] == 1
    assert result["judge_input_tokens"] == 20 and result["judge_output_tokens"] == 10


@pytest.mark.parametrize("answer", [None, "", "No marker", "<think>ANSWER: twelve</think>"])
def test_missing_visible_answer_scores_zero_without_judge(tmp_path: Path, answer: str | None) -> None:
    module = grader()
    options = fixture(tmp_path)
    if answer is None:
        options["answer_path"].unlink()
    else:
        options["answer_path"].write_text(answer)
    with patch.object(module, "urlopen") as call:
        assert module.grade(**options)["reward"] == 0
    call.assert_not_called()


def test_transport_failure_is_a_trial_error_without_stale_reward(tmp_path: Path) -> None:
    module = grader()
    options = fixture(tmp_path)
    options["output_dir"].mkdir()
    (options["output_dir"] / "reward.json").write_text('{"reward":1}')
    with patch.object(module, "urlopen", side_effect=URLError("unavailable")), patch.object(module.time, "sleep"):
        with pytest.raises(RuntimeError, match="requests exhausted"):
            module.grade(**options)
    assert not (options["output_dir"] / "reward.json").exists()


def test_malformed_verdict_retried_then_scores_zero(tmp_path: Path) -> None:
    module = grader()
    options = fixture(tmp_path)

    def malformed(*args, **kwargs):
        return io.BytesIO(b'{"choices":[{"message":{"content":"not JSON"}}]}')

    with patch.object(module, "urlopen", side_effect=malformed) as calls, patch.object(module.time, "sleep"):
        result = module.grade(**options)
    assert calls.call_count == 3
    assert result["judge_unparseable"] and result["reward"] == 0
