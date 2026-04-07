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

"""Blackjack environment.

Multi-step episode: the model hits or stands until the hand ends.
Reward: +1 win, 0 draw, -1 loss/bust.

No verifier_metadata required. The JSONL just needs a system prompt.
"""

import random
import re
from typing import Dict, Optional

from pydantic import Field

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer


_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]


def _deal():
    return random.choice(_RANKS)


def _hand_value(hand: list[str]) -> int:
    total = sum(10 if r in ("J", "Q", "K") else 11 if r == "A" else int(r) for r in hand)
    aces = hand.count("A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def _fmt(hand: list[str]) -> str:
    return "[" + ", ".join(hand) + "]"


def _extract_text(response: NeMoGymResponse) -> str:
    for item in response.output:
        if item.type == "message":
            content = item.content
            if isinstance(content, str):
                return content
            for c in content:
                if c.type == "output_text":
                    return c.text
    return ""


class BlackjackEnv(GymnasiumServer):
    session_state: Dict[str, dict] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        player = [_deal(), _deal()]
        dealer = [_deal(), _deal()]
        self.session_state[session_id] = {"player": player, "dealer": dealer}
        obs = (
            f"Your hand: {_fmt(player)} = {_hand_value(player)}\n"
            f"Dealer shows: {dealer[0]}\n"
            f"Respond with <action>hit</action> or <action>stand</action>."
        )
        return obs, {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        state = self.session_state.get(session_id, {})
        player = state.get("player", [])
        dealer = state.get("dealer", [])
        text = _extract_text(action)
        m = re.search(r"<action>\s*(hit|stand)\s*</action>", text, re.IGNORECASE)
        decision = m.group(1).lower() if m else ("hit" if "hit" in text.lower() else "stand")

        if decision == "hit":
            player.append(_deal())
            val = _hand_value(player)
            if val > 21:
                return None, -1.0, True, False, {"result": "bust", "player": _fmt(player), "value": val}
            obs = (
                f"Your hand: {_fmt(player)} = {val}\n"
                f"Dealer shows: {dealer[0]}\n"
                f"Respond with <action>hit</action> or <action>stand</action>."
            )
            return obs, 0.0, False, False, {}

        # stand: dealer plays out
        while _hand_value(dealer) < 17:
            dealer.append(_deal())

        pv, dv = _hand_value(player), _hand_value(dealer)
        if dv > 21 or pv > dv:
            reward, result = 1.0, "win"
        elif pv == dv:
            reward, result = 0.0, "draw"
        else:
            reward, result = -1.0, "loss"

        return (
            None,
            reward,
            True,
            False,
            {
                "result": result,
                "player": _fmt(player),
                "player_value": pv,
                "dealer": _fmt(dealer),
                "dealer_value": dv,
            },
        )


if __name__ == "__main__":
    BlackjackEnv.run_webserver()
