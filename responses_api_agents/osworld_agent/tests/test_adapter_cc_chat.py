# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cc_chat history layout: chat turns with a rolling 3-to-10 screenshot window."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent.adapter_agents import (
    CC_OMITTED_IMAGE_MARKER,
    NemotronV3NanoOmniAgent,
    normalize_nemotron_agent_options,
)


def _agent(**overrides: Any) -> NemotronV3NanoOmniAgent:
    kwargs: Dict[str, Any] = dict(
        model="m",
        max_steps=200,
        assistant_history_replay="raw_full",
        context_overflow_policy="stop",
        unterminated_episode_policy="fail",
        history_layout="cc_chat",
        parse_retries=1,
    )
    kwargs.update(overrides)
    return NemotronV3NanoOmniAgent(**kwargs)


def _reasoning(step: int) -> str:
    return f"Thought {step}.\n"


def _content(step: int) -> str:
    return f"\n## Action:\nClick {step}.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"


def _run(agent: NemotronV3NanoOmniAgent, steps: int) -> List[List[Dict[str, Any]]]:
    prompts: List[List[Dict[str, Any]]] = []

    def call_llm(payload: Dict[str, Any], _model: Any = None) -> Dict[str, Any]:
        step = len(prompts)
        prompts.append(payload["messages"])
        return {
            "content": _content(step).strip(),
            "reasoning_content": _reasoning(step).strip(),
            "raw_content": f"<think>{_reasoning(step)}</think>{_content(step)}",
            "finish_reason": "stop",
        }

    agent.call_llm = call_llm
    agent.reset()
    for step in range(steps):
        agent.predict("Do the task.", {"screenshot": bytes([step])})
    return prompts


def _layout(messages: List[Dict[str, Any]]) -> str:
    marks = []
    for message in messages:
        if message["role"] != "user":
            continue
        kinds = [part["type"] for part in message["content"]]
        texts = [part.get("text") for part in message["content"]]
        marks.append("I" if "image_url" in kinds else "M" if CC_OMITTED_IMAGE_MARKER in texts else ".")
    return "".join(marks)


def test_cc_chat_keeps_initial_image_and_rebases_after_ten_images() -> None:
    prompts = _run(_agent(), 20)

    assert _layout(prompts[9]) == "I" * 10
    # Eleven images would exceed the guard: re-base to initial + last two.
    assert _layout(prompts[10]) == "IM" + "." * 7 + "II"
    assert _layout(prompts[17]) == "IM" + "." * 7 + "I" * 9
    assert _layout(prompts[18]) == "IM" + "." * 15 + "II"


def test_cc_chat_replays_every_turn_as_template_rendered_reasoning() -> None:
    prompts = _run(_agent(), 3)
    messages = prompts[2]

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"] == f"<think>\n{_reasoning(0)}\n</think>\n{_content(0)}"
    assert messages[-1]["content"][-1]["text"].endswith("You are currently on Step 3.\n")


def test_cc_chat_chunk_turns_as_generated_only_inside_the_chunk() -> None:
    prompts = _run(_agent(cc_chunk_turns_as_generated=True), 12)
    messages = prompts[11]
    assistants = [message["content"] for message in messages if message["role"] == "assistant"]

    # The chunk opened at step 11 (index 10): older turns are re-rendered,
    # in-chunk turns replay the served message so capture can splice tokens.
    assert assistants[9] == f"<think>\n{_reasoning(9)}\n</think>\n{_content(9)}"
    assert assistants[10] == f"<think>{_reasoning(10)}</think>{_content(10)}"


def _run_with_usage(agent: NemotronV3NanoOmniAgent, steps: int, *, first: int, growth: int, completion: int):
    prompts: List[List[Dict[str, Any]]] = []
    expected_prompt = [first]

    def call_llm(payload: Dict[str, Any], _model: Any = None) -> Dict[str, Any]:
        step = len(prompts)
        prompts.append(payload["messages"])
        prompt_tokens = expected_prompt[0]
        if agent.snapshot_window.get("snapshot_compaction_triggered"):
            prompt_tokens = first + step * 100
        expected_prompt[0] = prompt_tokens + completion + growth
        return {
            "content": _content(step).strip(),
            "reasoning_content": _reasoning(step).strip(),
            "raw_content": f"<think>{_reasoning(step)}</think>{_content(step)}",
            "finish_reason": "stop",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
        }

    agent.call_llm = call_llm
    agent.reset()
    for step in range(steps):
        agent.predict("Do the task.", {"screenshot": bytes([step])})
    return prompts


def test_cc_chat_prompt_guard_rebases_before_crossing_the_limit() -> None:
    # 1000 + k * 300 crosses 2500 at the sixth call (index 5): 1000 + 5 * 300.
    agent = _agent(cc_prompt_token_guard=2400)
    prompts = _run_with_usage(agent, 8, first=1000, growth=200, completion=100)

    assert _layout(prompts[4]) == "I" * 5
    assert _layout(prompts[5]) == "IM" + "." * 2 + "II"
    assert agent.cc_guard_rebases == 1


def test_cc_chat_prompt_guard_is_off_by_default() -> None:
    agent = _agent()
    prompts = _run_with_usage(agent, 8, first=1000, growth=200, completion=100)

    assert _layout(prompts[7]) == "I" * 8
    assert agent.cc_guard_rebases == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_overflow_policy": "shrink"},
        {"assistant_history_replay": "normalized"},
        {"cc_keep_last_groups": 10, "cc_max_active_images": 10},
    ],
)
def test_cc_chat_rejects_incompatible_options(overrides: Dict[str, Any]) -> None:
    options = {
        "assistant_history_replay": "raw_full",
        "context_overflow_policy": "stop",
        "history_layout": "cc_chat",
        **overrides,
    }
    with pytest.raises(ValueError):
        normalize_nemotron_agent_options(options)


def test_osworld_layout_ignores_cc_options() -> None:
    base = {"assistant_history_replay": "raw_full", "context_overflow_policy": "stop"}
    with_cc = {**base, "history_layout": "osworld", "cc_keep_last_groups": 3, "cc_chunk_turns_as_generated": True}

    assert normalize_nemotron_agent_options(with_cc) == normalize_nemotron_agent_options(base)
