# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Returned history under training-token capture.

With capture on, the model server withholds token arrays from the agent, so the
exact-trace output is unavailable. Accepted returned history uses the content
the model server served; complete evidence also retains rejected attempts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

import pytest

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.models.token_capture.completion import completion_items, output_item_fingerprint
from responses_api_agents.osworld_agent.app import (
    OSWorldRunRequest,
    _build_messages_model_fn,
    _build_response,
)


_WRAP = ChatCompletionsConverter(return_token_id_information=False, uses_reasoning_parser=True)


def _served(reasoning: str, content: str) -> str:
    """Chat content exactly as the Gym vLLM model server serves it."""

    return _WRAP._wrap_reasoning_in_think_tags([reasoning]) + content


def _call(parse_attempt: int, response: Dict[str, Any] | None, *, accepted: bool) -> Dict[str, Any]:
    return {
        "parse_attempt": parse_attempt,
        "prompt_messages": [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
        "response": response,
        "accepted": accepted,
        "parse_error": None if accepted else "bad action",
        "parsed_actions": [],
    }


_SERVED = {
    "chatcmpl-1": _served("Open the menu.", "## Action:\nclick\n## Code:\n```python\npyautogui.click(1, 2)\n```"),
    "chatcmpl-2": _served("Retry.", "no code block"),
    "chatcmpl-3": _served("Retry again.", "## Action:\nclick\n## Code:\n```python\npyautogui.click(3, 4)\n```"),
    "chatcmpl-5": _served("Done.", "## Action:\nfinish\n## Code:\n```code\nDONE\n```"),
}


def _response(response_id: str) -> Dict[str, Any]:
    raw = _SERVED[response_id]
    return {
        "content": raw.split("</think>", 1)[1],
        "reasoning_content": "stripped by the agent",
        "raw_content": raw,
        "finish_reason": "stop",
        "response_id": response_id,
    }


def _capture_on_result() -> Dict[str, Any]:
    """Three env steps, one parse retry, one context-overflow stand-in, no token arrays."""

    steps = [
        {
            "step": 0,
            "model_text": "pyautogui.click(1, 2)",
            "actions": ["pyautogui.click(1, 2)"],
            "reward": 0.0,
            "done": False,
            "info": {"agent": {"model_calls": [_call(1, _response("chatcmpl-1"), accepted=True)]}},
        },
        {
            "step": 1,
            "model_text": "pyautogui.click(3, 4)",
            "actions": ["pyautogui.click(3, 4)"],
            "reward": 0.0,
            "done": False,
            "info": {
                "agent": {
                    "model_calls": [
                        _call(1, _response("chatcmpl-2"), accepted=False),
                        _call(2, _response("chatcmpl-3"), accepted=True),
                    ]
                }
            },
        },
        {
            "step": 2,
            "model_text": "DONE",
            "actions": ["DONE"],
            "reward": 1.0,
            "done": True,
            "info": {
                "agent": {
                    "model_calls": [
                        # The model server's local context-overflow reply: nothing generated or captured.
                        _call(
                            1,
                            {"content": "", "raw_content": "", "finish_reason": "length", "response_id": "local-4"},
                            accepted=False,
                        ),
                        # No observed response can supply a returned-content witness.
                        _call(2, None, accepted=False),
                        _call(3, _response("chatcmpl-5"), accepted=True),
                    ]
                }
            },
        },
    ]
    return {
        "reward": 1.0,
        "score": 1.0,
        "finished": True,
        "horizon_reached": False,
        "evaluation_completed": True,
        "runtime_eligible": True,
        "runtime_admission_reason": "valid_evaluated_outcome",
        "runtime_admission_policy_id": "osworld-runtime-admission-v1",
        "mask_sample": False,
        "error": None,
        "artifact_dir": None,
        "steps": steps,
    }


def _request() -> OSWorldRunRequest:
    return OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={"task_id": "task-001", "domain": "chrome"},
    )


def _captured_fingerprint(response_id: str) -> str:
    """The content witness token capture records for one served Chat completion."""

    payload = {
        "id": response_id,
        "object": "chat.completion",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": _SERVED[response_id]}}
        ],
    }
    (item,) = completion_items(payload)
    return output_item_fingerprint(item)


def test_capture_on_output_returns_only_accepted_generations_verbatim() -> None:
    result = _capture_on_result()
    response = _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True)
    dumped = response.model_dump()

    output = dumped["response"]["output"]
    assert [item["id"] for item in output] == ["msg-call-1", "msg-call-3", "msg-call-6"]
    assert [item["content"][0]["text"] for item in output] == [
        _SERVED[rid] for rid in ("chatcmpl-1", "chatcmpl-3", "chatcmpl-5")
    ]
    # Each returned item carries the same content witness capture recorded.
    assert [output_item_fingerprint(item) for item in output] == [
        _captured_fingerprint(rid) for rid in ("chatcmpl-1", "chatcmpl-3", "chatcmpl-5")
    ]
    assert dumped["terminal_response_id"] == "chatcmpl-5"
    calls = dumped["response"]["trajectory_model_calls"]
    assert len(calls) == 6
    assert calls[1]["accepted"] is False
    assert calls[1]["action"]["raw_completion"] == _SERVED["chatcmpl-2"]
    assert dumped["ng_trajectory"] == {
        "turns": [
            {
                "model_calls": [
                    {"response_id": "chatcmpl-1"},
                    {"response_id": "chatcmpl-3"},
                    {"response_id": "chatcmpl-5"},
                ]
            }
        ]
    }
    # Semantic per-step evidence is unchanged.
    transitions = dumped["response"]["trajectory_transitions"]
    assert [t["action"]["raw_completion"] for t in transitions] == [
        "pyautogui.click(1, 2)",
        "pyautogui.click(3, 4)",
        "DONE",
    ]
    assert dumped["response"]["trajectory_contract"]["capabilities"]["exact_model_call_evidence"] is False


def test_rejected_terminal_is_not_reattributed_to_an_earlier_accepted_call() -> None:
    result = _capture_on_result()
    result["steps"][-1]["info"]["agent"]["model_calls"][-1]["accepted"] = False
    response = _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True).model_dump()
    assert [item["id"] for item in response["response"]["output"]] == ["msg-call-1", "msg-call-3"]
    assert response["terminal_response_id"] == "chatcmpl-5"
    assert response["reward"] == 1.0
    assert response["mask_sample"] is False


@pytest.mark.parametrize(
    ("reason", "failure_kind"),
    [("evaluator_error", "infrastructure_error"), ("model_timeout", "model_policy_response_timeout")],
)
def test_capture_history_keeps_runtime_failure_fields(reason: str, failure_kind: str) -> None:
    result = _capture_on_result()
    result.update(
        reward=0.0,
        score=0.0,
        finished=False,
        evaluation_completed=False,
        runtime_eligible=False,
        runtime_admission_reason=reason,
        mask_sample=True,
        error="TimeoutError: request timed out",
    )

    response = _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True).model_dump()

    assert response["terminal_response_id"] == "chatcmpl-5"
    assert response["failure_kind"] == failure_kind
    assert response["_ng_failure_class"] == failure_kind
    assert response["mask_sample"] is True
    assert [item["content"][0]["text"] for item in response["response"]["output"]] == [
        _SERVED[rid] for rid in ("chatcmpl-1", "chatcmpl-3", "chatcmpl-5")
    ]


@pytest.mark.parametrize("kind", ["rejected", "empty", "missing"])
def test_capture_with_no_accepted_output_never_falls_back_to_step_text(kind: str) -> None:
    result = _capture_on_result()
    result["steps"] = [result["steps"][0]]
    call = result["steps"][0]["info"]["agent"]["model_calls"][0]
    call["accepted"] = False
    if kind == "empty":
        call["response"]["raw_content"] = ""
    elif kind == "missing":
        call["response"] = None
    response = _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True).model_dump()
    assert response["response"]["output"] == []
    if kind == "rejected":
        assert response["terminal_response_id"] == "chatcmpl-1"
    else:
        assert "terminal_response_id" not in response


def test_capture_does_not_reuse_an_earlier_terminal_id_when_the_last_id_is_missing() -> None:
    result = _capture_on_result()
    result["steps"][-1]["info"]["agent"]["model_calls"][-1]["response"].pop("response_id")
    response = _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True).model_dump()
    assert "terminal_response_id" not in response


def test_member_without_a_served_id_omits_call_references() -> None:
    result = _capture_on_result()
    result["steps"][0]["info"]["agent"]["model_calls"][0]["response"].pop("response_id")
    response = _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True).model_dump()
    assert response["terminal_response_id"] == "chatcmpl-5"
    assert "ng_trajectory" not in response


def test_rejected_terminal_without_a_served_id_fails_closed() -> None:
    result = _capture_on_result()
    terminal = result["steps"][-1]["info"]["agent"]["model_calls"][-1]
    terminal["accepted"] = False
    terminal["response"].pop("response_id")
    with pytest.raises(ValueError, match="Rejected terminal call has no served response_id"):
        _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True)


def test_capture_off_output_keeps_step_messages() -> None:
    response = _build_response(_request(), _capture_on_result(), "policy", 0.6, 0.95)
    dumped = response.model_dump()

    assert [item["id"] for item in dumped["response"]["output"]] == ["msg-step-0", "msg-step-1", "msg-step-2"]
    assert "terminal_response_id" not in dumped


def test_capture_on_without_call_evidence_keeps_step_messages() -> None:
    result = _capture_on_result()
    for step in result["steps"]:
        step["info"] = {}
    response = _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True)
    dumped = response.model_dump()

    assert [item["id"] for item in dumped["response"]["output"]] == ["msg-step-0", "msg-step-1", "msg-step-2"]
    assert "terminal_response_id" not in dumped


def test_capture_on_with_no_accepted_calls_returns_empty_membership() -> None:
    result = _capture_on_result()
    for step in result["steps"]:
        for call in step["info"]["agent"]["model_calls"]:
            call["accepted"] = False
            call["parse_error"] = "bad action"

    response = _build_response(_request(), result, "policy", 0.6, 0.95, served_call_output=True)
    dumped = response.model_dump()

    assert dumped["response"]["output"] == []
    assert dumped["terminal_response_id"] == "chatcmpl-5"
    assert "ng_trajectory" not in dumped
    model_calls = dumped["response"]["trajectory_model_calls"]
    assert len(model_calls) == 6
    assert all(call["accepted"] is False for call in model_calls)


@patch("openai.DefaultHttpxClient")
@patch("openai.OpenAI")
def test_structured_model_call_keeps_served_response_id(mock_openai, _mock_http_client, monkeypatch) -> None:
    monkeypatch.delenv("OSWORLD_MODEL_IO_LOG", raising=False)
    message = SimpleNamespace(content=_SERVED["chatcmpl-1"], tool_calls=[], model_extra={})
    mock_openai.return_value.chat.completions.create.return_value = SimpleNamespace(
        id="chatcmpl-1",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
    )
    call = _build_messages_model_fn(base_url="http://policy/v1", model_name="policy", api_key="k")
    messages = [{"role": "user", "content": "inspect"}]

    normalized = call(
        messages,
        {
            "model": "policy",
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0.6,
            "_nemo_gym_return_message": True,
        },
    )

    assert normalized["response_id"] == "chatcmpl-1"
    assert normalized["raw_content"] == _SERVED["chatcmpl-1"]
