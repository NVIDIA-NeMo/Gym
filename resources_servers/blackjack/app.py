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

"""
Blackjack environment in Gymnasium API style.

Multi-step: the model hits or stands until the hand ends.
Reward: +1 win, 0 draw, -1 loss.
"""

import random
import re
from typing import ClassVar, Optional

from fastapi import HTTPException
from pydantic import JsonValue

from nemo_gym._checkpoint.resources import ResourcesCheckpointMode
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer, extract_text


_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]


def _deal(rng: random.Random) -> str:
    return rng.choice(_RANKS)


def _hand_value(hand: list[str]) -> int:
    total = sum(10 if r in ("J", "Q", "K") else 11 if r == "A" else int(r) for r in hand)
    aces = hand.count("A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def _fmt(hand: list[str]) -> str:
    return "[" + ", ".join(hand) + "]"


class BlackjackEnv(GymnasiumServer):
    ray_enabled = False
    # The rest of a game depends only on the two hands and the session's generator, which are all exported.
    checkpoint_mode: ClassVar[ResourcesCheckpointMode] = "exported"

    def export_env_state(self, state: dict) -> JsonValue:
        version, internal_state, gauss_next = state["rng"].getstate()
        return {
            "player": list(state["player"]),
            "dealer": list(state["dealer"]),
            "rng_state": [version, list(internal_state), gauss_next],
        }

    def restore_env_state(self, state: JsonValue) -> dict:
        try:
            player, dealer = state["player"], state["dealer"]
            if not all(isinstance(hand, list) and set(hand) <= set(_RANKS) for hand in (player, dealer)):
                raise ValueError("hands must be lists of card ranks")
            version, internal_state, gauss_next = state["rng_state"]
            rng = random.Random()
            rng.setstate((version, tuple(internal_state), gauss_next))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid blackjack session state: {error}") from error
        return {"player": list(player), "dealer": list(dealer), "rng": rng}

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        rng = random.Random()
        player = [_deal(rng), _deal(rng)]
        dealer = [_deal(rng), _deal(rng)]
        self.session_state[session_id] = {"player": player, "dealer": dealer, "rng": rng}
        obs = (
            f"Your hand: {_fmt(player)} = {_hand_value(player)}\n"
            f"Dealer shows: {dealer[0]}\n"
            f"Respond with <action>hit</action> or <action>stand</action>."
        )
        return obs, {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        if session_id not in self.session_state:
            # Never dealt, or released, for example by a checkpoint retire: there is no game to continue.
            raise HTTPException(status_code=404, detail="no Blackjack game for this session; call /reset first")
        state = self.session_state[session_id]
        player = state.get("player", [])
        dealer = state.get("dealer", [])
        rng = state.get("rng") or random.Random()
        text = extract_text(action)
        m = re.search(r"<action>\s*(hit|stand)\s*</action>", text, re.IGNORECASE)
        decision = m.group(1).lower() if m else "stand"

        if decision == "hit":
            player.append(_deal(rng))
            val = _hand_value(player)
            if val > 21:
                return None, -1.0, True, False, {"result": "bust", "player": _fmt(player), "value": val}
            obs = (
                f"Your hand: {_fmt(player)} = {val}\n"
                f"Dealer shows: {dealer[0]}\n"
                f"Respond with <action>hit</action> or <action>stand</action>."
            )
            return obs, 0.0, False, False, {}

        while _hand_value(dealer) < 17:
            dealer.append(_deal(rng))

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
