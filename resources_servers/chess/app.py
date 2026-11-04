# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chess against a built-in opponent, driven by gymnasium_agent.

The model moves with the ``make_move`` tool in UCI notation; the final /step scores the game from its side (+1 win,
0 draw, -1 loss). An invalid action changes nothing and is reported back; more than ``max_consecutive_invalid_actions``
in a row forfeits.
"""

import json
import random
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from nemo_gym.server_runtime import BaseResourcesServerConfig
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.chess.chess_rules import (
    STARTING_FEN,
    ChessGame,
    Color,
    IllegalMoveError,
    InvalidFenError,
    Move,
    Position,
    piece_color,
    square_name,
)
from resources_servers.chess.task_data import TaskData
from resources_servers.gymnasium import GymnasiumServer


MOVE_TOOL_NAME = "make_move"
RECENT_MOVES_SHOWN = 10
PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
MATE_SCORE = 1000


class ChessResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MakeMoveRequest(BaseModel):
    move: str


@dataclass
class ChessSession:
    game: ChessGame
    task: TaskData
    rng: random.Random
    agent_moves: int = 0
    invalid_actions: int = 0
    consecutive_invalid_actions: int = 0
    forfeited: bool = False

    def is_over(self) -> bool:
        return self.forfeited or self.game.outcome() is not None


def choose_opponent_move(position: Position, rng: random.Random, opponent: str) -> Move:
    """A random choice among the opponent's best-scoring moves, in legal-move order so a seeded ``rng`` replays games.

    ``random`` scores every move the same. ``greedy`` takes a mate in one, else the most valuable capture.
    ``lookahead`` searches two plies: the material balance after the reply that is worst for it, or a mate in one.
    """
    candidates = list(position.legal_moves)
    if opponent != "random":
        score = _greedy_score if opponent == "greedy" else _lookahead_score
        scores = {move: score(position, move) for move in candidates}
        best = max(scores.values())
        candidates = [move for move in candidates if scores[move] == best]
    return rng.choice(candidates)


def _greedy_score(position: Position, move: Move) -> int:
    if _is_checkmate(position.apply(move)):
        return MATE_SCORE
    captured = position.captured_piece(move)
    return PIECE_VALUES[captured.lower()] if captured else 0


def _lookahead_score(position: Position, move: Move) -> int:
    after = position.apply(move)
    if not after.legal_moves:
        return MATE_SCORE if after.in_check() else 0
    return min(_material_balance(after.apply(reply), position.turn) for reply in after.legal_moves)


def _is_checkmate(position: Position) -> bool:
    return position.in_check() and not position.legal_moves


def _material_balance(position: Position, color: Color) -> int:
    return sum(
        PIECE_VALUES[piece.lower()] * (1 if piece_color(piece) == color else -1)
        for piece in position.board
        if piece is not None and piece.lower() in PIECE_VALUES
    )


class ChessResourcesServer(GymnasiumServer):
    config: ChessResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Advertised at reset, so gymnasium_agent releases games it stops early, such as at max_steps.
        app.post("/close")(self._close_endpoint)
        return app

    async def _close_endpoint(self, request: Request) -> dict:
        await self.close_session(request.session.get(SESSION_ID_KEY))
        self.checkpoint_session_ended(request)
        return {}

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        try:
            task = TaskData.model_validate(metadata)
            game = ChessGame(task.fen or STARTING_FEN)
        except (ValidationError, InvalidFenError) as error:
            raise HTTPException(status_code=400, detail=f"Invalid chess task: {error}") from error
        outcome = game.outcome()
        if outcome is not None:
            raise HTTPException(
                status_code=400, detail=f"Invalid chess task: the game is already over ({outcome.termination})"
            )

        session = ChessSession(game=game, task=task, rng=random.Random(task.seed))
        self.session_state[session_id] = session
        observation = {}
        if game.position.turn != task.agent_color:
            observation["opponent_move"] = self._play_opponent(session)
        observation |= self._observe(session)
        return json.dumps(observation), {"supports_explicit_close": True}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        session = self.session_state.get(session_id)
        if session is None:
            # Never reset, or already released: there is no game to continue.
            raise HTTPException(status_code=404, detail="No chess game for this session; call /reset first.")

        calls = [item for item in action.output if item.type == "function_call"]
        # A game can end before the model's first move, when the opponent moved first; this step then reports it.
        if not session.is_over():
            if not calls:
                result = self._reject(session, "No move was made. Call make_move with your move in UCI notation.")
                if not session.is_over():
                    return json.dumps(result), 0.0, False, False, {}
            else:
                tool_outputs = [self.tool_output(calls[0], self._take_turn(session, calls[0]))]
                tool_outputs += [
                    self.tool_output(call, {"error": "Only one move is played per turn; this call was ignored."})
                    for call in calls[1:]
                ]
                if not session.is_over():
                    return None, 0.0, False, False, {"tool_outputs": tool_outputs}
        reward, info = self._final_result(session)
        return None, reward, True, False, info

    def _take_turn(self, session: ChessSession, call: NeMoGymResponseFunctionToolCall) -> dict:
        """Play the model's move and the opponent's reply, returning the tool output for the call."""
        if call.name != MOVE_TOOL_NAME:
            return self._reject(session, f"Unknown tool {call.name!r}; the only tool is {MOVE_TOOL_NAME}.")
        try:
            request = MakeMoveRequest.model_validate_json(call.arguments or "{}")
            move = session.game.play(request.move)
        except ValidationError as error:
            return self._reject(session, f"Invalid arguments for {MOVE_TOOL_NAME}: {error}")
        except IllegalMoveError as error:
            return self._reject(session, str(error))
        session.agent_moves += 1
        session.consecutive_invalid_actions = 0
        result = {"your_move": move.uci()}
        if session.game.outcome() is None:
            result["opponent_move"] = self._play_opponent(session)
        return result | self._observe(session)

    def _reject(self, session: ChessSession, error: str) -> dict:
        """Count an invalid action, forfeiting once too many come in a row, and describe it to the model."""
        session.invalid_actions += 1
        session.consecutive_invalid_actions += 1
        remaining = session.task.max_consecutive_invalid_actions - session.consecutive_invalid_actions
        error = error if error.endswith(".") else error + "."
        if remaining < 0:
            session.forfeited = True
            error += " That is too many invalid actions in a row, so you forfeit the game."
        else:
            error += f" Invalid actions in a row left before you forfeit: {remaining}."
        return {"error": error} | self._observe(session)

    def _play_opponent(self, session: ChessSession) -> str:
        move = choose_opponent_move(session.game.position, session.rng, session.task.opponent)
        return session.game.play(move.uci()).uci()

    def _observe(self, session: ChessSession) -> dict:
        game, agent_color = session.game, session.task.agent_color
        position = game.position
        outcome = game.outcome()
        your_turn = not session.is_over() and position.turn == agent_color
        return {
            "status": "forfeit" if session.forfeited else outcome.termination if outcome else "ongoing",
            "you_play": agent_color,
            "turn": position.turn,
            "in_check": position.in_check(),
            "fen": position.fen(),
            "board": position.board_rows(),
            "castling_rights": position.castling_rights or "-",
            "en_passant": square_name(position.en_passant) if position.en_passant is not None else None,
            "halfmove_clock": position.halfmove_clock,
            "fullmove_number": position.fullmove_number,
            "recent_moves": game.numbered_moves()[-RECENT_MOVES_SHOWN:],
            "legal_moves": sorted(move.uci() for move in position.legal_moves) if your_turn else [],
        }

    def _final_result(self, session: ChessSession) -> tuple[float, dict]:
        """The reward from the model's side and the episode's info for a game that is over."""
        outcome = session.game.outcome()
        if session.forfeited:
            result, termination = "loss", "forfeit"
        else:
            assert outcome is not None, "only a finished game is scored"
            termination = outcome.termination
            if outcome.winner is None:
                result = "draw"
            else:
                result = "win" if outcome.winner == session.task.agent_color else "loss"
        info = {
            "result": result,
            "termination": termination,
            "agent_moves": session.agent_moves,
            "invalid_actions": session.invalid_actions,
            "final_fen": session.game.position.fen(),
        }
        return {"win": 1.0, "draw": 0.0, "loss": -1.0}[result], info


if __name__ == "__main__":
    ChessResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = ChessResourcesServer.run_webserver()  # noqa: F401
