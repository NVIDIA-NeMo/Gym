# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.kuhn_poker.app import (
    KuhnPokerConfig,
    KuhnPokerEnvironment,
    KuhnPokerState,
    parse_action,
    payoff,
)


def _environment(invalid_retries: int = 1) -> KuhnPokerEnvironment:
    config = KuhnPokerConfig(
        host="",
        port=0,
        entrypoint="",
        name="kuhn_poker",
        invalid_retries=invalid_retries,
    )
    return KuhnPokerEnvironment(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    ("history", "cards", "expected"),
    [
        ("check-check", ("K", "Q"), 1),
        ("check-check", ("J", "Q"), -1),
        ("bet-fold", ("J", "K"), 1),
        ("check-bet-fold", ("K", "J"), -1),
        ("bet-call", ("K", "Q"), 2),
        ("bet-call", ("J", "Q"), -2),
        ("check-bet-call", ("K", "J"), 2),
        ("check-bet-call", ("J", "K"), -2),
    ],
)
def test_payoff(history: str, cards: tuple[str, str], expected: int) -> None:
    assert payoff(history, cards) == expected


@pytest.mark.parametrize(
    ("reply", "legal", "expected"),
    [
        ("[check]", ("check", "bet"), "check"),
        ("I choose [BET].", ("check", "bet"), "bet"),
        ("bet", ("check", "bet"), None),
        ("[fold]", ("check", "bet"), None),
        ("[check], though [bet] is tempting", ("check", "bet"), None),
        ("[raise] then [bet]", ("check", "bet"), "bet"),
    ],
)
def test_parse_action(reply: str, legal: tuple[str, ...], expected: str | None) -> None:
    assert parse_action(reply, legal) == expected


@pytest.mark.asyncio
async def test_seeded_reset_is_reproducible_and_private() -> None:
    environment = _environment()
    first = await environment.reset({"verifier_metadata": {"seed": 7}}, "first")
    await environment.reset({"verifier_metadata": {"seed": 7}}, "second")
    first_state = environment.session_state["first"]
    second_state = environment.session_state["second"]

    assert first_state.cards == second_state.cards
    assert first.active_agent == "player0"
    assert first.info["seed"] == 7
    assert "seed" not in first.info["view"]
    assert "seed" not in environment._latest_public_view
    assert all(card is None for card in environment._latest_public_view["cards"].values())
    assert f"Your private card: {first_state.cards[0]}" in first.observation
    assert f"Your private card: {first_state.cards[1]}" not in first.observation


@pytest.mark.asyncio
async def test_full_checked_hand_routes_private_observations_and_rewards() -> None:
    environment = _environment()
    environment.session_state["sid"] = KuhnPokerState(seed=0, cards=("K", "Q"))

    after_player0 = await environment.step("player0", "[check]", {}, "sid")
    assert after_player0.active_agent == "player1"
    assert "Player 0 chose [check]" in after_player0.observation
    assert "Your private card: Q" in after_player0.observation
    assert "Your private card: K" not in after_player0.observation

    terminal = await environment.step("player1", "[check]", {}, "sid")
    assert terminal.terminated is True
    assert terminal.rewards == {"player0": 1.0, "player1": -1.0}
    assert terminal.info["history"] == "check-check"


@pytest.mark.asyncio
async def test_invalid_action_retries_then_forfeits() -> None:
    environment = _environment(invalid_retries=1)
    environment.session_state["sid"] = KuhnPokerState(seed=0, cards=("J", "K"))

    retry = await environment.step("player0", "check", {}, "sid")
    assert retry.terminated is False
    assert retry.active_agent == "player0"
    assert retry.info["retries_remaining"] == 1

    terminal = await environment.step("player0", "[check] and [bet]", {}, "sid")
    assert terminal.terminated is True
    assert terminal.rewards == {"player0": -1.0, "player1": 1.0}
    assert terminal.info["forfeited"] == 0


@pytest.mark.asyncio
async def test_rejects_action_from_inactive_agent() -> None:
    environment = _environment()
    environment.session_state["sid"] = KuhnPokerState(seed=0, cards=("J", "K"))

    with pytest.raises(HTTPException, match="player0's turn"):
        await environment.step("player1", "[check]", {}, "sid")


@pytest.mark.asyncio
async def test_sessions_are_isolated() -> None:
    environment = _environment()
    await environment.reset({"verifier_metadata": {"seed": 0}}, "a")
    await environment.reset({"verifier_metadata": {"seed": 1}}, "b")
    await environment.step("player0", "[bet]", {}, "a")

    assert environment.session_state["a"].history == ["bet"]
    assert environment.session_state["b"].history == []


def test_http_session_runs_and_cleans_up_terminal_hand() -> None:
    environment = _environment()
    client = TestClient(environment.setup_webserver())
    task = {
        "responses_create_params": {"input": "play"},
        "verifier_metadata": {"seed": 0},
    }

    reset = client.post("/reset", json=task)
    assert reset.status_code == 200
    assert reset.json()["active_agent"] == "player0"

    first_step = client.post("/step", json=task | {"agent_id": "player0", "action": "[check]"})
    assert first_step.status_code == 200
    assert first_step.json()["active_agent"] == "player1"

    terminal = client.post("/step", json=task | {"agent_id": "player1", "action": "[check]"})
    assert terminal.status_code == 200
    assert terminal.json()["terminated"] is True
    assert environment.session_state == {}
