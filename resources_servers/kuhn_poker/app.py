# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Seeded, two-player Kuhn Poker as a turn-based multi-agent environment."""

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.multi_agent import MultiAgentResetResponse, MultiAgentResourcesServer, MultiAgentStepResponse


STATIC_DIR = Path(__file__).parent / "static"
RANKS = {"J": 0, "Q": 1, "K": 2}
TO_ACT = {"": 0, "check": 1, "bet": 1, "check-bet": 0}
LEGAL = {
    "": ("check", "bet"),
    "check": ("check", "bet"),
    "bet": ("fold", "call"),
    "check-bet": ("fold", "call"),
}

RULES = """You are playing Kuhn poker as {agent_id} (Player {seat}).
- The deck is J, Q, K; K beats Q and Q beats J.
- Both players ante 1 chip and receive one private card.
- Player 0 acts first with [check] or [bet].
- After a check, Player 1 may [check] or [bet].
- After a bet, the other player may [fold] or [call].

Your private card: {card}.
Reply with exactly one legal action in square brackets. Reasoning outside the brackets is allowed."""


def payoff(history: str, cards: tuple[str, str]) -> int:
    """Return Player 0's terminal net chips."""
    showdown = 1 if RANKS[cards[0]] > RANKS[cards[1]] else -1
    return {
        "check-check": showdown,
        "bet-fold": 1,
        "check-bet-fold": -1,
        "bet-call": 2 * showdown,
        "check-bet-call": 2 * showdown,
    }[history]


def parse_action(reply: str, legal: tuple[str, ...]) -> Optional[str]:
    """Return the sole legal bracketed action, or None for an invalid reply."""
    found = [action for action in re.findall(r"\[(\w+)\]", reply.lower()) if action in legal]
    return found[0] if len(found) == 1 else None


@dataclass
class KuhnPokerState:
    seed: int
    cards: tuple[str, str]
    history: list[str] = field(default_factory=list)
    invalid_attempts: int = 0


class KuhnPokerConfig(BaseResourcesServerConfig):
    invalid_retries: int = Field(1, ge=0)
    agent_ids: tuple[str, str] = ("player0", "player1")


class KuhnPokerEnvironment(MultiAgentResourcesServer):
    config: KuhnPokerConfig
    _latest_public_view: Optional[dict[str, Any]] = PrivateAttr(default=None)
    _spectators: set[asyncio.Queue[dict[str, Any]]] = PrivateAttr(default_factory=set)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/play", include_in_schema=False)(self.play)
        app.get("/events", include_in_schema=False)(self.events)
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="kuhn-poker-static")
        return app

    async def play(self) -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    async def events(self, request: Request) -> StreamingResponse:
        return StreamingResponse(
            self._event_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> MultiAgentResetResponse:
        if session_id is None:
            raise HTTPException(status_code=400, detail="Missing session id.")

        verifier_metadata = metadata.get("verifier_metadata") or {}
        seed = int(verifier_metadata.get("seed", metadata.get("seed", 0)))
        deal = random.Random(seed).sample(list(RANKS), 2)
        state = KuhnPokerState(seed=seed, cards=(deal[0], deal[1]))
        self.session_state[session_id] = state
        private_view = self._view(state, active_seat=0, private_seat=0)
        self._publish(self._view(state, active_seat=0))
        return MultiAgentResetResponse(
            active_agent=self.config.agent_ids[0],
            observation=self._observation(state, 0),
            info={"seed": seed, "view": private_view},
        )

    async def step(
        self,
        agent_id: str,
        action: str,
        metadata: dict,
        session_id: Optional[str] = None,
    ) -> MultiAgentStepResponse:
        state = self._state(session_id)
        history_key = "-".join(state.history)
        seat = TO_ACT[history_key]
        expected_agent = self.config.agent_ids[seat]
        if agent_id != expected_agent:
            raise HTTPException(
                status_code=409,
                detail=f"It is {expected_agent}'s turn, not {agent_id}'s.",
            )

        legal = LEGAL[history_key]
        parsed = parse_action(action, legal)
        if parsed is None:
            state.invalid_attempts += 1
            if state.invalid_attempts <= self.config.invalid_retries:
                choices = " or ".join(f"[{item}]" for item in legal)
                message = f"That was not a legal move. Choose exactly one of {choices}."
                private_view = self._view(
                    state,
                    active_seat=seat,
                    private_seat=seat,
                    message=message,
                )
                self._publish(self._view(state, active_seat=seat, message=message))
                return MultiAgentStepResponse(
                    active_agent=expected_agent,
                    observation=f"That was not a legal move. Reply with exactly one of {choices}.",
                    info={
                        "invalid_move": True,
                        "retries_remaining": self.config.invalid_retries - state.invalid_attempts + 1,
                        "view": private_view,
                    },
                )
            return self._terminal(state, forfeited=seat)

        state.invalid_attempts = 0
        state.history.append(parsed)
        history_key = "-".join(state.history)
        if history_key not in TO_ACT:
            return self._terminal(state)

        next_seat = TO_ACT[history_key]
        private_view = self._view(state, active_seat=next_seat, private_seat=next_seat)
        self._publish(self._view(state, active_seat=next_seat))
        return MultiAgentStepResponse(
            active_agent=self.config.agent_ids[next_seat],
            observation=self._observation(state, next_seat),
            info={"history": history_key, "view": private_view},
        )

    def _state(self, session_id: Optional[str]) -> KuhnPokerState:
        state = self.session_state.get(session_id)
        if not isinstance(state, KuhnPokerState):
            raise HTTPException(status_code=400, detail="Session not initialized. Call /reset first.")
        return state

    def _observation(self, state: KuhnPokerState, seat: int) -> str:
        history_key = "-".join(state.history)
        if state.history:
            events = []
            for index, action in enumerate(state.history):
                prior_history = "-".join(state.history[:index])
                events.append(f"Player {TO_ACT[prior_history]} chose [{action}]")
            betting = ", ".join(events)
        else:
            betting = "nothing yet"
        legal = " or ".join(f"[{action}]" for action in LEGAL[history_key])
        return (
            RULES.format(agent_id=self.config.agent_ids[seat], seat=seat, card=state.cards[seat])
            + f"\n\nBetting so far: {betting}.\nYour legal actions: {legal}."
        )

    def _view(
        self,
        state: KuhnPokerState,
        active_seat: Optional[int],
        private_seat: Optional[int] = None,
        rewards: Optional[dict[str, float]] = None,
        forfeited: Optional[int] = None,
        message: Optional[str] = None,
    ) -> dict[str, Any]:
        history_key = "-".join(state.history)
        terminal = active_seat is None
        history = []
        for index, action in enumerate(state.history):
            prior_history = "-".join(state.history[:index])
            seat = TO_ACT[prior_history]
            history.append(
                {
                    "agent": self.config.agent_ids[seat],
                    "seat": seat,
                    "action": action,
                }
            )

        cards = {
            agent_id: state.cards[seat] if terminal or seat == private_seat else None
            for seat, agent_id in enumerate(self.config.agent_ids)
        }
        return {
            "status": "finished" if terminal else "playing",
            "active_agent": self.config.agent_ids[active_seat] if active_seat is not None else None,
            "active_seat": active_seat,
            "cards": cards,
            "history": history,
            "pot": 2 + sum(action in {"bet", "call"} for action in state.history),
            "legal_actions": list(LEGAL[history_key]) if not terminal else [],
            "rewards": rewards or {},
            "forfeited": forfeited,
            "message": message,
        }

    def _publish(self, view: dict[str, Any]) -> None:
        self._latest_public_view = view
        for queue in tuple(self._spectators):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(view)

    async def _event_stream(self, request: Request) -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._spectators.add(queue)
        try:
            if self._latest_public_view is not None:
                yield f"data: {json.dumps(self._latest_public_view)}\n\n"
            while not await request.is_disconnected():
                try:
                    view = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(view)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            self._spectators.discard(queue)

    def _terminal(self, state: KuhnPokerState, forfeited: Optional[int] = None) -> MultiAgentStepResponse:
        history = "-".join(state.history)
        net0 = payoff(history, state.cards) if forfeited is None else (1 if forfeited == 1 else -1)
        rewards = {
            self.config.agent_ids[0]: float(net0),
            self.config.agent_ids[1]: float(-net0),
        }
        agent_info = {
            self.config.agent_ids[seat]: {
                "seat": seat,
                "card": state.cards[seat],
                "history": history,
                "seed": state.seed,
                "forfeited": forfeited,
            }
            for seat in (0, 1)
        }
        view = self._view(state, active_seat=None, rewards=rewards, forfeited=forfeited)
        self._publish(view)
        return MultiAgentStepResponse(
            rewards=rewards,
            terminated=True,
            info={
                "kuhn": agent_info,
                "history": history,
                "forfeited": forfeited,
                "payoff_player0": net0,
                "view": view,
            },
        )


if __name__ == "__main__":
    KuhnPokerEnvironment.run_webserver()
