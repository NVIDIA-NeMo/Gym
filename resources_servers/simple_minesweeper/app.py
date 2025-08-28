from typing import Any, Dict, Tuple, List
import json

from pydantic import BaseModel
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)

from minesweeper_logic import MinesweeperEnv


# --------------------------------------------------------------------------- #
#  Pydantic models
# --------------------------------------------------------------------------- #
class MinesweeperResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetInitialBoardRequest(BaseModel):
    rows: int = 8
    cols: int = 8
    num_mines: int = 10
    max_turns: int = 20


class GetInitialBoardResponse(BaseModel):
    instructions: str
    board_text: str
    game_state: Dict[str, Any]


class MakeMoveRequest(BaseModel):
    game_state: Dict[str, Any]
    row: int
    col: int
    action_type: str  # "reveal" | "flag"


class MakeMoveResponse(BaseModel):
    success: bool
    message: str
    game_state: Dict[str, Any]
    board_text: str
    is_complete: bool
    move_reward: float


class MinesweeperRunRequest(BaseRunRequest):
    rows: int = 8
    cols: int = 8
    num_mines: int = 10
    max_turns: int = 20


class MinesweeperVerifyRequest(MinesweeperRunRequest, BaseVerifyRequest):
    reward: float = 0.0
    total_moves: int = 0
    is_complete: bool = False


class MinesweeperVerifyResponse(BaseVerifyResponse):
    total_moves: int = 0
    is_complete: bool = False


# --------------------------------------------------------------------------- #
#  Helper functions to (de)serialise MinesweeperEnv
# --------------------------------------------------------------------------- #
ENV_STATE_KEYS = (
    "rows", "cols", "num_mines", "max_turns",
    "grid", "revealed", "flags",
    "first_reveal", "turn_count"
)


def env_to_state(env: MinesweeperEnv) -> Dict[str, Any]:
    return {k: getattr(env, k) for k in ENV_STATE_KEYS}


def state_to_env(state: Dict[str, Any]) -> MinesweeperEnv:
    env = MinesweeperEnv(
        rows=state["rows"],
        cols=state["cols"],
        num_mines=state["num_mines"],
        max_turns=state["max_turns"],
    )
    # overwrite internal attrs so env matches previous state
    for k in ENV_STATE_KEYS:
        setattr(env, k, state[k])
    return env


# --------------------------------------------------------------------------- #
#  The FastAPI resources server
# --------------------------------------------------------------------------- #
class MinesweeperResourcesServer(SimpleResourcesServer):
    config: MinesweeperResourcesServerConfig

    # -------------- fastapi wiring ----------------
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/get_initial_board")(self.get_initial_board)
        app.post("/make_move")(self.make_move)

        return app

    # -------------- endpoint handlers -------------
    async def get_initial_board(
        self, body: GetInitialBoardRequest
    ) -> GetInitialBoardResponse:
        env = MinesweeperEnv(
            rows=body.rows,
            cols=body.cols,
            num_mines=body.num_mines,
            max_turns=body.max_turns,
        )
        instructions, _ = env.reset()
        board_text = env._render_board(is_start=True)

        game_state = env_to_state(env)

        return GetInitialBoardResponse(
            instructions=instructions,
            board_text=board_text,
            game_state=game_state,
        )

    async def make_move(self, body: MakeMoveRequest) -> MakeMoveResponse:
        env = state_to_env(body.game_state)

        # Compose boxed action string the env expects, e.g.  \boxed{reveal 3 2}
        boxed_action = f"\\boxed{{{body.action_type} {body.row} {body.col}}}"
        message, reward, terminated, _, _ = env.step(boxed_action)

        game_state = env_to_state(env)
        board_text = env._render_board()

        # Fix success logic: unsuccessful only if it's an invalid action or format error
        # From minesweeper_logic.py: invalid_action_reward = 0.0, format_error_reward = -0.1
        success = reward >= 0.0  # Accept 0 reward (flagging) and positive rewards

        return MakeMoveResponse(
            success=success,
            message=message,
            game_state=game_state,
            board_text=board_text,
            is_complete=terminated,
            move_reward=reward,
        )

    # -------------- verify (trivial) --------------
    async def verify(self, body: MinesweeperVerifyRequest) -> MinesweeperVerifyResponse:
        return MinesweeperVerifyResponse(**body.model_dump())


# --------------------------------------------------------------------------- #
#  Entry-point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    MinesweeperResourcesServer.run_webserver()