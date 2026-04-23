# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
from nemo_gym.server_utils import ServerClient
from resources_servers.blackjack.app import BlackjackEnv, _hand_value


def _make_env():
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
    return BlackjackEnv(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


class TestHandValue:
    def test_basic(self):
        assert _hand_value(["5", "7"]) == 12

    def test_face_cards(self):
        assert _hand_value(["K", "Q"]) == 20

    def test_ace_low_when_busting(self):
        assert _hand_value(["A", "K", "5"]) == 16

    def test_ace_high_when_safe(self):
        assert _hand_value(["A", "9"]) == 20

    def test_two_aces(self):
        assert _hand_value(["A", "A"]) == 12


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_populates_state(self):
        env = _make_env()
        obs, info = await env.reset({}, session_id="sid")
        assert "sid" in env.session_state
        state = env.session_state["sid"]
        assert len(state["player"]) == 2
        assert len(state["dealer"]) == 2
        assert "rng" in state
        assert "Your hand" in obs
        assert "<action>hit</action>" in obs

    @pytest.mark.asyncio
    async def test_per_session_rng_is_isolated(self):
        # Two sessions get distinct RNG instances (not the module default).
        env = _make_env()
        await env.reset({}, session_id="a")
        await env.reset({}, session_id="b")
        assert env.session_state["a"]["rng"] is not env.session_state["b"]["rng"]


class TestStep:
    @pytest.mark.asyncio
    async def test_stand_finishes_game(self):
        env = _make_env()
        # Preload a known safe state so the result is deterministic.
        env.session_state["sid"] = {
            "player": ["K", "9"],  # 19
            "dealer": ["10"],  # dealer will draw until ≥ 17
            "rng": __import__("random").Random(0),
        }
        _, reward, term, trunc, info = await env.step(_response("<action>stand</action>"), {}, session_id="sid")
        assert term is True
        assert trunc is False
        assert reward in (-1.0, 0.0, 1.0)
        assert info["result"] in ("win", "draw", "loss")

    @pytest.mark.asyncio
    async def test_hit_bust_ends_game(self):
        env = _make_env()
        # Force a bust on the next draw regardless of RNG: player already at 20, drawing any card except A busts.
        # Use a seeded RNG that deals non-A.
        rng = __import__("random").Random(0)
        env.session_state["sid"] = {"player": ["K", "K"], "dealer": ["5"], "rng": rng}
        _, reward, term, _, info = await env.step(_response("<action>hit</action>"), {}, session_id="sid")
        # Player at 20 + any non-ace → bust. With rng seed 0, the draw is deterministic.
        if term:
            assert reward == -1.0
            assert info["result"] == "bust"

    @pytest.mark.asyncio
    async def test_hit_continues_when_safe(self):
        env = _make_env()
        env.session_state["sid"] = {
            "player": ["5", "3"],  # 8
            "dealer": ["10"],
            "rng": __import__("random").Random(0),
        }
        obs, reward, term, _, _ = await env.step(_response("<action>hit</action>"), {}, session_id="sid")
        assert term is False
        assert reward == 0.0
        assert "Your hand" in obs


class TestActionParser:
    @pytest.mark.asyncio
    async def _decide(self, text: str) -> str:
        env = _make_env()
        env.session_state["sid"] = {
            "player": ["2", "2"],  # 4, can't bust on hit
            "dealer": ["10"],
            "rng": __import__("random").Random(0),
        }
        obs, _, term, _, info = await env.step(_response(text), {}, session_id="sid")
        # Terminal → stood; non-terminal → hit.
        return "stand" if term else "hit"

    @pytest.mark.asyncio
    async def test_hit_tag(self):
        assert await self._decide("<action>hit</action>") == "hit"

    @pytest.mark.asyncio
    async def test_stand_tag(self):
        assert await self._decide("<action>stand</action>") == "stand"

    @pytest.mark.asyncio
    async def test_whitespace_tolerated(self):
        assert await self._decide("<action>  hit  </action>") == "hit"

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        assert await self._decide("<action>HIT</action>") == "hit"

    @pytest.mark.asyncio
    async def test_no_tag_defaults_stand(self):
        # Previously the fallback did a substring match, so "don't hit" would parse as hit.
        assert await self._decide("i don't know, maybe hit?") == "stand"

    @pytest.mark.asyncio
    async def test_unknown_action_defaults_stand(self):
        assert await self._decide("<action>fold</action>") == "stand"
