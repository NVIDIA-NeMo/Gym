# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""With context resets off (max_reset_count: 0), every model call's input extends the previous call's input and output.

Training token capture links each call to the one before it by that prefix, so a loop that rewrites earlier turns
(a reset, a re-rendered progress board, a nudge on a tool result the model already saw) leaves later calls without a
parent. This drives the host loop with the training settings (progress board, step nudges, a prompt over the reset
threshold on every call) and checks the prefix property call by call, then shows the same run with resets on breaks it.
"""

import json
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.browsecomp_agent.app import BrowsecompAgent
from responses_api_agents.browsecomp_agent.tests.test_progress_tracking import (
    SYSTEM,
    USER,
    _make_agent,
    _make_fn_call,
    _make_msg,
    _model_response,
    _progress_call,
    _run,
)


def _items(body) -> list:
    return [json.dumps(i.model_dump() if hasattr(i, "model_dump") else i, sort_keys=True) for i in body.input]


def _turns(over_threshold_tokens: int) -> list:
    """Six tool-calling turns (search, board, browse, bash, search, search), then the answer."""
    calls = [
        _make_fn_call("search", call_id="c1", args={"queries": ["q1"]}),
        _progress_call("constraints: a, b", call_id="c2"),
        _make_fn_call("browse", call_id="c3", args={"urls": ["https://example.com"], "goal": "g"}),
        _make_fn_call("bash_command", call_id="c4", args={"keystrokes": "ls"}),
        _make_fn_call("search", call_id="c5", args={"queries": ["q2"]}),
        _make_fn_call("search", call_id="c6", args={"queries": ["q3"]}),
    ]
    turns = [
        _model_response([c], input_tokens=over_threshold_tokens, response_id=f"r{i}") for i, c in enumerate(calls)
    ]
    return turns + [
        _model_response([_make_msg("Exact Answer: X")], input_tokens=over_threshold_tokens, response_id="r9")
    ]


def _prefix_breaks(bodies: list) -> list[int]:
    """Indices i where call i+1's input does not start with call i's input."""
    return [i for i in range(len(bodies) - 1) if _items(bodies[i + 1])[: len(_items(bodies[i]))] != _items(bodies[i])]


@pytest.mark.parametrize("tokenize", [False, True])
async def test_history_only_grows_with_resets_off(tokenize, monkeypatch) -> None:
    # The training settings: board on, step nudges on (max_steps 8 puts the quarter / half / near-end notes at steps
    # 2, 4 and 7), every prompt far over the reset threshold, resets capped at 0.
    agent, fake = _make_agent(
        _turns(over_threshold_tokens=120_000),
        progress=True,
        nudge_steps=True,
        max_steps=8,
        max_context_tokens=131_072,
        context_reset_tokens=0,
        max_reset_count=0,
        save_model_call_using_vllm_tokenize_endpoint=tokenize,
    )
    monkeypatch.setattr(BrowsecompAgent, "_count_prompt_tokens", AsyncMock(return_value=120_000))
    result = await _run(agent, [SYSTEM, USER])

    assert len(fake.model_bodies) == 7
    assert _prefix_breaks(fake.model_bodies) == []
    assert getattr(result, "reset_count", 0) == 0
    # Each call's input is the previous input, then the previous call's output, then its tool results.
    for i in range(len(fake.model_bodies) - 1):
        assert len(_items(fake.model_bodies[i + 1])) == len(_items(fake.model_bodies[i])) + 2
    # The nudges went out on the newest tool result of their step, inside the prefix of every later call.
    last = "\n".join(_items(fake.model_bodies[-1]))
    assert "You have used 2 out of 8 turns" in last and "halfway through your budget" in last


async def test_resets_on_rewrite_the_history() -> None:
    # The same run with resets uncapped: the reset drops earlier turns, so some call no longer extends the one before.
    agent, fake = _make_agent(
        _turns(over_threshold_tokens=120_000),
        progress=True,
        nudge_steps=True,
        max_steps=8,
        max_context_tokens=131_072,
        context_reset_tokens=0,
        max_reset_count=None,
    )
    await _run(agent, [SYSTEM, USER])

    assert _prefix_breaks(fake.model_bodies) != []
