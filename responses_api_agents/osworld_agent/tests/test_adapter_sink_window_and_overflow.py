# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter behaviour for a sink policy and for a rejected over-long prompt."""

from __future__ import annotations

import base64
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent.adapter_agents import NemotronV3NanoOmniAgent

GOOD = "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"


def _reply(index: int) -> Dict[str, Any]:
    return {
        "content": GOOD,
        "reasoning_content": f"Thought {index + 1}",
        "finish_reason": "stop",
    }


def _image_turns(payload: Dict[str, Any]) -> List[int]:
    """Recover which screenshots a rendered prompt actually carries."""

    turns = []
    for message in payload["messages"]:
        for part in message.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "image_url":
                blob = part["image_url"]["url"].split(",", 1)[1]
                turns.append(int(base64.b64decode(blob).decode().removeprefix("png-")))
    return turns


def _drive(agent: NemotronV3NanoOmniAgent, steps: int) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return _reply(len(payloads) - 1)

    agent.call_llm = call_llm  # type: ignore[method-assign]
    for index in range(steps):
        _, actions, _ = agent.predict("Complete the task.", {"screenshot": f"png-{index + 1}".encode()})
        assert actions == ["pyautogui.click(960, 540)"]
    return payloads


def test_sink_window_prompt_keeps_the_first_screenshot_and_slides_the_rest() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=20,
        history_policy={
            "schema_version": 1,
            "name": "sink_window",
            "params": {"sink": 1, "low_water": 4, "high_water": 4},
        },
        parse_retries=1,
    )
    payloads = _drive(agent, 11)

    # The requested shape, read straight off the wire.
    assert _image_turns(payloads[8]) == [1, 7, 8, 9]
    assert _image_turns(payloads[9]) == [1, 8, 9, 10]
    assert _image_turns(payloads[10]) == [1, 9, 10, 11]
    # The sink screenshot survives every compaction.
    assert all(_image_turns(payload)[0] == 1 for payload in payloads)


def test_sink_window_text_summary_sits_between_the_sink_and_the_recent_window() -> None:
    """The folded turns must be described after the sink, before the window.

    Attaching the summary to whichever image comes first -- what the previous
    renderer did -- would place a description of turns 2..6 before turn 1.
    """

    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=20,
        history_policy={
            "schema_version": 1,
            "name": "sink_window",
            "params": {"sink": 1, "low_water": 4, "high_water": 4},
        },
        parse_retries=1,
    )
    payloads = _drive(agent, 9)
    messages = payloads[8]["messages"]

    texts = [
        part["text"]
        for message in messages
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    carriers = [index for index, text in enumerate(texts) if "# Previous History Actions" in text]
    assert len(carriers) == 1, "exactly one summary block for one contiguous folded run"

    summary = texts[carriers[0]]
    # Folded turns are 2..6 (1-based); they must all appear in the summary.
    for step in range(2, 7):
        assert f"Step {step}" in summary
    # The block rides on the first *recent-window* image (turn 7), not the sink.
    assert "You are currently on Step 7." in summary
    assert "# Previous History Actions" not in texts[0], "the sink image must not carry the summary"
    assert "You are currently on Step 1." in texts[0]


def test_sink_window_summary_ordering_matches_the_image_it_precedes() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=20,
        history_policy={
            "schema_version": 1,
            "name": "sink_window",
            "params": {"sink": 2, "low_water": 4, "high_water": 4},
        },
        parse_retries=1,
    )
    payloads = _drive(agent, 10)
    assert _image_turns(payloads[9]) == [1, 2, 9, 10]

    # Each user message carries its screenshot and then its text, so the
    # question is which screenshot's message the summary rides with.
    carriers: List[tuple[str, bool]] = []
    for message in payloads[9]["messages"]:
        parts = [part for part in (message.get("content") or []) if isinstance(part, dict)]
        images = [part for part in parts if part.get("type") == "image_url"]
        if not images:
            continue
        blob = images[0]["image_url"]["url"].split(",", 1)[1]
        text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
        carriers.append((base64.b64decode(blob).decode(), "# Previous History Actions" in text))

    # The summary belongs to the first recent-window turn, not to the sink:
    # it describes turns 3..8, which come after the sink and before turn 9.
    assert carriers == [
        ("png-1", False),
        ("png-2", False),
        ("png-9", True),
        ("png-10", False),
    ]


def test_fixed_three_rendering_is_unchanged_by_the_new_renderer() -> None:
    """Regression guard for every already-published benchmark release."""

    agent = NemotronV3NanoOmniAgent(model="policy-under-test", max_steps=20, parse_retries=1)
    payloads = _drive(agent, 6)

    assert _image_turns(payloads[5]) == [4, 5, 6]
    texts = [
        part["text"]
        for message in payloads[5]["messages"]
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    # One summary, carried by the first (and only) leading image, as before.
    assert sum("# Previous History Actions" in text for text in texts) == 1
    assert "# Previous History Actions" in texts[0]
    assert "You are currently on Step 4." in texts[0]


def test_context_overflow_shrinks_the_prompt_instead_of_resending_it() -> None:
    """The defect b9 hit: five identical over-long requests, five 400s."""

    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=30,
        max_image_history_length=3,
        max_live_images=10,
        parse_retries=5,
    )
    payloads: List[Dict[str, Any]] = []
    reject_until = {"images": 3}

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        images = len(_image_turns(payload))
        if images > reject_until["images"]:
            raise ValueError(
                f"Error code: 400 - Input length ({images * 6000}) exceeds "
                "model's maximum context length (64000)."
            )
        return _reply(len(payloads) - 1)

    agent.call_llm = call_llm  # type: ignore[method-assign]

    for index in range(9):
        _, actions, info = agent.predict(
            "Complete the task.", {"screenshot": f"png-{index + 1}".encode()}
        )
        assert actions == ["pyautogui.click(960, 540)"], f"step {index + 1} must still act"

    # Steps 1..3 fit; from step 4 the window would exceed the limit and the
    # agent must walk it back down rather than burn its retries.
    sizes = [len(_image_turns(payload)) for payload in payloads]
    assert max(sizes) <= 4, f"never sent a prompt more than one step past the limit: {sizes}"
    assert sizes.count(4) >= 1, "the overflow must actually have been provoked"
    assert all(size <= 3 for size in sizes if size != 4)

    # Every step ended in an accepted call, and no step exhausted its retries.
    assert len(payloads) < 9 * 5, "shrinking must be cheaper than exhausting the retry budget"


def test_context_overflow_records_the_shrink_and_the_specific_outcome() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=30,
        max_image_history_length=3,
        max_live_images=10,
        parse_retries=3,
    )

    def always_reject(_payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        raise ValueError("Error code: 400 - This model's maximum context length is 64000 tokens")

    agent.call_llm = always_reject  # type: ignore[method-assign]
    _, actions, info = agent.predict("Complete the task.", {"screenshot": b"png-1"})

    assert actions == []
    assert info["agent_outcome"] == "model_context_overflow"
    assert info["agent_outcome_family"] == "model_response_invalid"
    failure = info["parse_failure"]
    assert failure["last_failure_kind"] == "context_overflow"
    assert failure["failure_kind_counts"] == {"context_overflow": 3}
    assert failure["completed_model_call_count"] == 0


def test_overflow_retries_do_not_advance_the_history_policy_state() -> None:
    """Re-planning must restart from the entry state, or replay would drift."""

    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=30,
        max_image_history_length=3,
        max_live_images=10,
        parse_retries=4,
    )
    calls = {"n": 0}

    def reject_twice(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ValueError("Error code: 400 - maximum context length exceeded")
        return _reply(calls["n"])

    agent.call_llm = reject_twice  # type: ignore[method-assign]
    before = agent.history_policy_state
    _, actions, _ = agent.predict("Complete the task.", {"screenshot": b"png-1"})

    assert actions == ["pyautogui.click(960, 540)"]
    # One step advanced the epoch at most once, not once per rejected attempt.
    assert agent.history_policy_state.compaction_epoch - before.compaction_epoch <= 1


def test_parse_failures_still_retry_normally_and_do_not_shrink() -> None:
    """Only context overflow changes the prompt; a bad parse just resamples."""

    agent = NemotronV3NanoOmniAgent(model="policy-under-test", max_steps=5, parse_retries=3)
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(]\n```",
                "reasoning_content": "not repairable",
                "finish_reason": "stop",
            }
        return _reply(len(payloads) - 1)

    agent.call_llm = call_llm  # type: ignore[method-assign]
    _, actions, info = agent.predict("Complete the task.", {"screenshot": b"png-1"})

    assert actions == ["pyautogui.click(960, 540)"]
    assert len(payloads) == 2
    assert info["model_calls"][0]["failure_kind"] == "unparseable"
    assert _image_turns(payloads[0]) == _image_turns(payloads[1]), "prompt unchanged on a parse retry"


def test_overflow_shrinking_terminates_when_even_one_image_is_rejected() -> None:
    """The shrink must bottom out, not spin or drop the current observation."""

    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=30,
        max_image_history_length=3,
        max_live_images=10,
        parse_retries=6,
    )
    payloads: List[Dict[str, Any]] = []

    def always_too_long(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        raise ValueError("Error code: 400 - Input length (99999) exceeds model's maximum context length (64000).")

    agent.call_llm = always_too_long  # type: ignore[method-assign]
    _, actions, info = agent.predict("Complete the task.", {"screenshot": b"png-1"})

    assert actions == []
    assert len(payloads) == 6, "exactly parse_retries attempts, no more and no spinning"
    assert all(len(_image_turns(payload)) >= 1 for payload in payloads), "never sends zero images"
    assert info["agent_outcome"] == "model_context_overflow"
    assert info["parse_failure"]["failure_kind_counts"] == {"context_overflow": 6}


def test_overflow_shrink_under_a_sink_policy_keeps_the_sink() -> None:
    """Budget pressure may shrink the recent window; it must not drop the sink."""

    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=30,
        history_policy={
            "schema_version": 1,
            "name": "sink_window",
            "params": {"sink": 2, "low_water": 5, "high_water": 8},
        },
        parse_retries=4,
    )
    payloads: List[Dict[str, Any]] = []
    limit = {"images": 3}

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        if len(_image_turns(payload)) > limit["images"]:
            raise ValueError("Error code: 400 - maximum context length exceeded")
        return _reply(len(payloads) - 1)

    agent.call_llm = call_llm  # type: ignore[method-assign]
    for index in range(10):
        _, actions, _ = agent.predict(
            "Complete the task.", {"screenshot": f"png-{index + 1}".encode()}
        )
        assert actions == ["pyautogui.click(960, 540)"], f"step {index + 1} must still act"

    accepted = [payload for payload in payloads if len(_image_turns(payload)) <= limit["images"]]
    for payload in accepted:
        turns = _image_turns(payload)
        assert turns[0] == 1, "the sink screenshot survives every shrink"
        assert turns[-1] == max(turns), "the current observation is always last and present"


def test_shrink_events_are_recorded_on_the_step_that_recovered() -> None:
    """A recovered step must still say it was rejected and shrunk.

    Recording the evidence only on rollouts that died is how twenty HTTP 400s
    left no trace in a whole benchmark release.
    """

    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=30,
        max_image_history_length=3,
        max_live_images=10,
        parse_retries=5,
    )
    calls = {"n": 0}
    reject_at_call = 4  # step 4, where the window first carries four images

    def reject_once(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == reject_at_call:
            raise ValueError(
                "Error code: 400 - Input length (70000) exceeds model's maximum context length (64000)."
            )
        return _reply(calls["n"])

    agent.call_llm = reject_once  # type: ignore[method-assign]
    infos = []
    for index in range(5):
        _, actions, info = agent.predict(
            "Complete the task.", {"screenshot": f"png-{index + 1}".encode()}
        )
        assert actions == ["pyautogui.click(960, 540)"], f"step {index + 1} recovered"
        infos.append(info)

    rejected_step = infos[3]
    assert rejected_step["prompt_shrink_events"] == [
        {"parse_attempt": 1, "from_images": 4, "to_images": 3, "reason": "context_overflow"}
    ]
    assert rejected_step["failure_kind_counts"] == {"context_overflow": 1}
    # Steps that never hit a rejection stay byte-clean: no extra keys at all.
    for index, info in enumerate(infos):
        if index == 3:
            continue
        assert "prompt_shrink_events" not in info, f"step {index + 1} should carry no shrink evidence"
        assert "failure_kind_counts" not in info


def test_sink_window_agent_exposes_its_policy_in_the_agent_contract() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=10,
        history_policy={
            "schema_version": 1,
            "name": "sink_window",
            "params": {"sink": 1, "low_water": 3, "high_water": 10},
        },
        parse_retries=1,
    )
    contract = agent.agent_contract["history_policy"]

    assert contract["name"] == "sink_window"
    assert contract["params"] == {"sink": 1, "low_water": 3, "high_water": 10}
    assert contract["supports_append_stable_intervals"] is True
    assert contract["history_policy_id"].startswith("osworld-history-policy-")
    # A sliding sink window is not append stable, and must say so.
    sliding = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=10,
        history_policy={
            "schema_version": 1,
            "name": "sink_window",
            "params": {"sink": 1, "low_water": 4, "high_water": 4},
        },
        parse_retries=1,
    )
    assert sliding.agent_contract["history_policy"]["supports_append_stable_intervals"] is False


def test_sink_window_telemetry_reaches_the_model_call_records() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=20,
        history_policy={
            "schema_version": 1,
            "name": "sink_window",
            "params": {"sink": 1, "low_water": 4, "high_water": 4},
        },
        parse_retries=1,
    )
    infos = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        return _reply(0)

    agent.call_llm = call_llm  # type: ignore[method-assign]
    for index in range(9):
        _, _, info = agent.predict("Complete the task.", {"screenshot": f"png-{index + 1}".encode()})
        infos.append(info["model_calls"][0])

    last = infos[-1]
    assert last["prompt_snapshot_count"] == 4
    assert last["snapshot_sink_size"] == 1
    assert last["snapshot_image_intervals"] == [[0, 1], [6, 9]]
    assert last["history_policy_name"] == "sink_window"
