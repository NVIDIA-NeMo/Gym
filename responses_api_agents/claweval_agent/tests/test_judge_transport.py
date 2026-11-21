# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest


native = pytest.importorskip("claw_eval.graders.llm_judge", reason="Optional native ClawEval source required")

from responses_api_agents.claweval_agent.judge_transport import (  # noqa: E402
    NvidiaInferenceGeminiJudge,
    create_judge,
)


def judge() -> NvidiaInferenceGeminiJudge:
    return NvidiaInferenceGeminiJudge(
        model_id="gcp/google/gemini-3-flash-preview",
        api_key="unit-test-key",
        base_url="https://inference-api.nvidia.com/v1",
    )


def response(*, thought: bool = False) -> httpx.Response:
    parts = [{"text": '{"score":0.75,"reasoning":"native parsing"}'}]
    if thought:
        parts.insert(0, {"thought": True, "text": "Not an answer"})
    return httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": parts}}]},
        request=httpx.Request("POST", "https://example.test"),
    )


@pytest.mark.parametrize(
    "content",
    [
        "original task evidence",
        [
            {"type": "text", "text": "rubric"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ],
    ],
)
def test_hub_changes_only_route_and_preserves_native_request_payload(monkeypatch, content) -> None:
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs))
        return response()

    monkeypatch.setattr(httpx, "post", post)
    original = native.LLMJudge(
        model_id="gcp/google/gemini-3-flash-preview",
        api_key="unit-test-key",
        base_url="https://inference-api.nvidia.com/v1",
    )
    assert original._call_gemini("original system rubric", content) == judge()._call_gemini(
        "original system rubric", content
    )
    assert requests[0][1] == requests[1][1]
    assert requests[0][0].endswith("/models/gemini-3-flash-preview:generateContent")
    assert (
        requests[1][0]
        == "https://inference-api.nvidia.com/v1beta/models/gcp/google/gemini-3-flash-preview:generateContent"
    )


def test_thought_text_is_excluded_from_native_score_json(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response(thought=True))
    assert json.loads(judge()._call_gemini("rubric", "evidence"))["score"] == 0.75


@pytest.mark.parametrize(
    "method,args",
    [
        ("evaluate", ("task", "conversation", "actions", "rubric")),
        ("evaluate_actions", ("task", "artifacts", "rubric")),
        ("evaluate_visual", ("rubric", ["AA=="], ["AQ=="])),
    ],
)
def test_actual_native_evaluation_and_call_log_are_preserved(monkeypatch, method, args) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response(thought=True))
    instance = judge()
    result = getattr(instance, method)(*args)
    assert isinstance(result, native.JudgeResult) and result.score == 0.75
    assert instance.get_call_log()[0]["method"] == method


@pytest.mark.parametrize(
    "base_url,model",
    [
        ("https://openrouter.ai/api/v1", "google/gemini-2.5-flash"),
        ("https://inference-api.nvidia.com/v1", "other-model"),
    ],
)
def test_other_native_judge_transports_are_preserved(base_url, model) -> None:
    assert type(create_judge(model_id=model, api_key="unit-test-key", base_url=base_url)) is native.LLMJudge


@pytest.mark.parametrize(
    "base_url",
    [
        "http://inference-api.nvidia.com/v1",
        "https://inference-api.nvidia.com/other",
        "https://inference-api.nvidia.com/v1?secret=value",
    ],
)
def test_invalid_hub_base_fails_before_a_request(base_url) -> None:
    with pytest.raises(ValueError, match="root or /v1"):
        NvidiaInferenceGeminiJudge(
            model_id="gcp/google/gemini-3-flash-preview", api_key="unit-test-key", base_url=base_url
        )


def test_http_error_propagates_from_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(403, request=httpx.Request("POST", "https://example.test")),
    )
    with pytest.raises(httpx.HTTPStatusError):
        judge()._call_gemini("rubric", "evidence")
