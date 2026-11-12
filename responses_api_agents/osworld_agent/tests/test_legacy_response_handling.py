# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in response-handling switches."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models.chat_completions_converter import ChatCompletionsConverter
from nemo_gym.models.token_capture.completion import completion_items, output_item_fingerprint
from responses_api_agents.osworld_agent.adapter_agents import (
    NemotronV3NanoOmniAgent,
    normalize_nemotron_agent_options,
)
from responses_api_agents.osworld_agent.app import OSWorldRunRequest, _build_response


GOOD = "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"
SYNTAX_ERROR = "## Action:\nClick.\n## Code:\n```python\nif True\n    pyautogui.click(0.5, 0.5)\n```"
NO_CODE = "I keep thinking about the next step"
_WRAP = ChatCompletionsConverter(return_token_id_information=False, uses_reasoning_parser=True)


def _agent(**kwargs: Any) -> NemotronV3NanoOmniAgent:
    options: Dict[str, Any] = {"model": "policy-under-test", "max_steps": 20, "parse_retries": 1}
    options.update(kwargs)
    return NemotronV3NanoOmniAgent(**options)


def _predict(agent: NemotronV3NanoOmniAgent, replies: List[Dict[str, Any]]):
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return replies[len(payloads) - 1]

    agent.call_llm = call_llm  # type: ignore[method-assign]
    _, actions, info = agent.predict("Complete the task.", {"screenshot": b"png-1"})
    return payloads, actions, info


def _reply(content: str, finish_reason: str = "stop", response_id: str = "chatcmpl-1") -> Dict[str, Any]:
    return {
        "content": content,
        "reasoning_content": "Thought",
        "raw_content": f"<think>\nThought\n</think>\n{content}",
        "finish_reason": finish_reason,
        "response_id": response_id,
    }


def test_defaults_keep_the_contract_and_the_strict_checks() -> None:
    options = normalize_nemotron_agent_options({"unterminated_episode_policy": "fail"})
    assert not {"validate_python_syntax", "require_clean_finish", "train_invalid_terminal_call"} & set(options)

    payloads, actions, info = _predict(_agent(unterminated_episode_policy="fail"), [_reply(SYNTAX_ERROR)])
    assert payloads[0]["_nemo_gym_require_stop"] is True
    assert actions == ["FAIL"]
    assert info["forced_terminal"] == "model_response_invalid"
    assert "loss_member" not in info["model_calls"][-1]

    _, actions, _ = _predict(_agent(unterminated_episode_policy="fail"), [_reply(GOOD, finish_reason="length")])
    assert actions == ["FAIL"]


def test_non_default_switches_enter_the_contract() -> None:
    options = normalize_nemotron_agent_options(
        {
            "unterminated_episode_policy": "fail",
            "validate_python_syntax": False,
            "require_clean_finish": False,
            "train_invalid_terminal_call": True,
        }
    )
    assert options["validate_python_syntax"] is False
    assert options["require_clean_finish"] is False
    assert options["train_invalid_terminal_call"] is True


def test_train_invalid_terminal_call_requires_the_fail_policy() -> None:
    with pytest.raises(ValueError, match="requires unterminated_episode_policy='fail'"):
        normalize_nemotron_agent_options({"train_invalid_terminal_call": True})


def test_switches_reject_non_boolean_values() -> None:
    with pytest.raises(ValueError, match="validate_python_syntax must be a boolean"):
        normalize_nemotron_agent_options({"validate_python_syntax": "false"})


def test_syntax_check_can_be_disabled_like_the_reference_parser() -> None:
    _, actions, info = _predict(_agent(validate_python_syntax=False), [_reply(SYNTAX_ERROR)])
    assert actions != ["FAIL"]
    assert "if True" in actions[0]
    assert info["model_calls"][-1]["accepted"] is True


def test_truncated_text_is_parsed_when_clean_finish_is_not_required() -> None:
    payloads, actions, info = _predict(_agent(require_clean_finish=False), [_reply(GOOD, finish_reason="length")])
    assert "_nemo_gym_require_stop" not in payloads[0]
    assert actions == ["pyautogui.click(960, 540)"]
    assert info["model_calls"][-1]["accepted"] is True


def test_context_overflow_stand_in_stays_invalid_without_clean_finish() -> None:
    stand_in = {"content": "", "raw_content": "", "finish_reason": "length", "response_id": "local-1"}
    _, actions, info = _predict(_agent(require_clean_finish=False, unterminated_episode_policy="fail"), [stand_in])
    assert actions == ["FAIL"]
    assert info["model_calls"][-1]["accepted"] is False


def test_invalid_terminal_call_becomes_a_loss_member() -> None:
    agent = _agent(unterminated_episode_policy="fail", train_invalid_terminal_call=True)
    _, actions, info = _predict(agent, [_reply(NO_CODE, finish_reason="length")])
    assert actions == ["FAIL"]
    assert info["forced_terminal"] == "model_response_invalid"
    assert info["model_calls"][-1]["accepted"] is False
    assert info["model_calls"][-1]["loss_member"] is True


def test_only_the_terminal_retry_becomes_a_loss_member() -> None:
    agent = _agent(unterminated_episode_policy="fail", train_invalid_terminal_call=True, parse_retries=2)
    _, _, info = _predict(agent, [_reply(NO_CODE), _reply(NO_CODE, response_id="chatcmpl-2")])
    assert ["loss_member" in call for call in info["model_calls"]] == [False, True]


def _call(response: Dict[str, Any], *, accepted: bool, loss_member: bool = False) -> Dict[str, Any]:
    call = {
        "parse_attempt": 1,
        "prompt_messages": [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
        "response": response,
        "accepted": accepted,
        "parse_error": None if accepted else "no code",
        "parsed_actions": [],
    }
    if loss_member:
        call["loss_member"] = True
    return call


def _served(reasoning: str, content: str) -> str:
    return _WRAP._wrap_reasoning_in_think_tags([reasoning]) + content


def _served_response(response_id: str, raw: str, finish_reason: str) -> Dict[str, Any]:
    return {
        "content": raw.split("</think>", 1)[1],
        "reasoning_content": "stripped",
        "raw_content": raw,
        "finish_reason": finish_reason,
        "response_id": response_id,
    }


def _result(*, terminal_loss_member: bool) -> Dict[str, Any]:
    first = _served("Open.", "## Action:\nclick\n## Code:\n```python\npyautogui.click(1, 2)\n```")
    runaway = _served("Thinking " * 8, "")
    return {
        "reward": 0.0,
        "score": 0.0,
        "finished": True,
        "horizon_reached": False,
        "evaluation_completed": True,
        "runtime_eligible": True,
        "runtime_admission_reason": "valid_evaluated_outcome",
        "runtime_admission_policy_id": "osworld-runtime-admission-v1",
        "mask_sample": False,
        "error": None,
        "artifact_dir": None,
        "steps": [
            {
                "step": 0,
                "model_text": "pyautogui.click(1, 2)",
                "actions": ["pyautogui.click(1, 2)"],
                "reward": 0.0,
                "done": False,
                "info": {
                    "agent": {"model_calls": [_call(_served_response("chatcmpl-1", first, "stop"), accepted=True)]}
                },
            },
            {
                "step": 1,
                "model_text": "no code",
                "actions": ["FAIL"],
                "reward": 0.0,
                "done": True,
                "info": {
                    "agent": {
                        "forced_terminal": "model_response_invalid",
                        "model_calls": [
                            _call(
                                _served_response("chatcmpl-2", runaway, "length"),
                                accepted=False,
                                loss_member=terminal_loss_member,
                            )
                        ],
                    }
                },
            },
        ],
    }, runaway


def _request() -> OSWorldRunRequest:
    return OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={"task_id": "task-001", "domain": "chrome"},
    )


def test_terminal_loss_member_is_returned_for_capture_membership() -> None:
    result, runaway = _result(terminal_loss_member=True)
    dumped = _build_response(_request(), result, "policy", 1.0, 1.0, served_call_output=True).model_dump()

    output = dumped["response"]["output"]
    assert [item["content"][0]["text"] for item in output][-1] == runaway
    assert dumped["terminal_response_id"] == "chatcmpl-2"
    assert dumped["ng_trajectory"] == {
        "turns": [{"model_calls": [{"response_id": "chatcmpl-1"}, {"response_id": "chatcmpl-2"}]}]
    }
    captured = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": runaway}}],
    }
    (item,) = completion_items(captured)
    assert output_item_fingerprint(output[-1]) == output_item_fingerprint(item)


INLINE_CODE_HEADING = 'Click on "View".## Code:\n```python\npyautogui.click(0.82543, 0.208526)\n```'


def _legacy_agent(**kwargs: Any) -> NemotronV3NanoOmniAgent:
    return _agent(
        unterminated_episode_policy="fail",
        validate_python_syntax=False,
        require_clean_finish=False,
        train_invalid_terminal_call=True,
        legacy_response_parser=True,
        **kwargs,
    )


def _served_reply(raw: str) -> Dict[str, Any]:
    content = raw.split("</think>", 1)[1] if "</think>" in raw else raw
    return {"content": content.strip(), "reasoning_content": "x", "raw_content": raw, "finish_reason": "stop"}


def test_legacy_parser_enters_the_contract_only_when_enabled() -> None:
    assert "legacy_response_parser" not in normalize_nemotron_agent_options({})
    assert normalize_nemotron_agent_options({"legacy_response_parser": True})["legacy_response_parser"] is True


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (GOOD, "pyautogui.click(960, 540)"),
        (
            'Click on "View".\n## Code:\n```python\npyautogui.click(0.82543, 0.208526)\n```',
            "pyautogui.click(1585, 225)",
        ),
        # No Action section: the parser's FAIL is executed as a policy action.
        (INLINE_CODE_HEADING, "FAIL"),
    ],
)
def test_legacy_parser_accepts_what_the_old_agent_executed(content: str, expected: str) -> None:
    _, actions, info = _predict(_legacy_agent(), [_served_reply(f"<think>Thought</think>{content}")])
    assert info["model_calls"][-1]["accepted"] is True
    assert actions == [expected]


def test_legacy_parser_reads_the_engine_reasoning_split() -> None:
    # A stray close marker inside the reasoning leaves no content after </think>.
    raw = "<think>Plan.\u25c1/think\u25b7## Action:\nClick.\n## Code:\n```python\npyautogui.click(1, 2)\n```</think>"
    _, actions, info = _predict(_legacy_agent(), [_served_reply(raw)])
    assert actions == ["FAIL"]
    assert info["forced_terminal"] == "model_response_invalid"
    assert info["model_calls"][-1]["loss_member"] is True


def test_legacy_parser_keeps_policy_terminate_semantics() -> None:
    done = "## Action:\nDone.\n## Code:\n```code\ncomputer.terminate(status='success')\n```"
    _, actions, info = _predict(_legacy_agent(), [_served_reply(f"<think>T</think>{done}")])
    assert actions == ["DONE"]
    assert info["model_calls"][-1]["accepted"] is True


def test_rejected_terminal_call_stays_out_of_membership_by_default() -> None:
    result, _ = _result(terminal_loss_member=False)
    dumped = _build_response(_request(), result, "policy", 1.0, 1.0, served_call_output=True).model_dump()

    # The terminal still names the scored (rejected) call; only membership excludes it.
    assert dumped["terminal_response_id"] == "chatcmpl-2"
    assert dumped["ng_trajectory"] == {"turns": [{"model_calls": [{"response_id": "chatcmpl-1"}]}]}


def _over_guard_reply() -> Dict[str, Any]:
    return {**_served_reply(f"<think>T</think>{GOOD}"), "prompt_tokens": 38001, "completion_tokens": 50}


def test_context_limit_stays_a_trained_fail_by_default() -> None:
    agent = _legacy_agent(context_overflow_policy="stop", context_prompt_token_limit=38000)
    _, actions, info = _predict(agent, [_over_guard_reply()])
    assert actions == ["FAIL"]
    assert info["forced_terminal"] == "context_limit"
    assert "loss_member" not in info["model_calls"][-1]


def test_masked_context_limit_ends_the_rollout_without_a_terminal_action() -> None:
    agent = _legacy_agent(context_overflow_policy="stop", context_prompt_token_limit=38000, mask_context_limit=True)
    _, actions, info = _predict(agent, [_over_guard_reply()])
    assert actions == []
    assert "forced_terminal" not in info
    assert info["agent_outcome"] == "context_limit_reached"
    assert info["stop_rollout"] is True


def test_masked_context_limit_leaves_invalid_responses_trained() -> None:
    agent = _legacy_agent(context_overflow_policy="stop", context_prompt_token_limit=38000, mask_context_limit=True)
    _, actions, info = _predict(agent, [_served_reply(NO_CODE)])
    assert actions == ["FAIL"]
    assert info["forced_terminal"] == "model_response_invalid"


def test_mask_context_limit_requires_the_stop_policy() -> None:
    assert "mask_context_limit" not in normalize_nemotron_agent_options({})
    with pytest.raises(ValueError, match="requires context_overflow_policy='stop'"):
        normalize_nemotron_agent_options({"mask_context_limit": True})
