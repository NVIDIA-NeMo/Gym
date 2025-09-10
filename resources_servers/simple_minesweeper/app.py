import random
from collections import deque
from typing import Any, Dict

from pydantic import BaseModel
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)


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
#  The FastAPI resources server with embedded game logic
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
        # Initialize empty board state
        game_state = {
            "rows": body.rows,
            "cols": body.cols,
            "num_mines": body.num_mines,
            "max_turns": body.max_turns,
            "grid": [[0 for _ in range(body.cols)] for _ in range(body.rows)],
            "revealed": [[False for _ in range(body.cols)] for _ in range(body.rows)],
            "flags": [[False for _ in range(body.cols)] for _ in range(body.rows)],
            "first_reveal": True,
            "turn_count": 0,
        }

        instructions = self._get_instructions(body.rows, body.cols)
        # Enforce tool calling in the prompt
        instructions += (
            "\nImportant: Submit each move by calling the make_move tool with parameters "
            '{"row": <0-based>, "col": <0-based>, "action_type": "reveal" or "flag"}. '
            "Do not print moves like \\boxed{...}; always call the tool. "
            "When you make a move, respond only with a function call and no extra text."
        )
        board_text = self._render_board(game_state, is_start=True)

        return GetInitialBoardResponse(
            instructions=instructions,
            board_text=board_text,
            game_state=game_state,
        )

    async def make_move(self, body: MakeMoveRequest) -> MakeMoveResponse:
        game_state = body.game_state.copy()
        game_state["turn_count"] += 1

        # Parse action (similar to minesweeper_logic step method)
        action_type = body.action_type.lower() if body.action_type else None
        row = body.row
        col = body.col

        rows = game_state["rows"]
        cols = game_state["cols"]
        grid = game_state["grid"]
        revealed = game_state["revealed"]
        flags = game_state["flags"]
        first_reveal = game_state["first_reveal"]
        turn_count = game_state["turn_count"]
        max_turns = game_state["max_turns"]

        # Validate input
        if action_type is None or row is None or col is None:
            message = f"At turn {turn_count}, you did not provide a valid guess."
            return self._create_move_response(
                game_state, message, success=False, terminated=True, binary_reward=0.0
            )

        if not (0 <= row < rows and 0 <= col < cols):
            message = f"At turn {turn_count}, you chose cell ({row}, {col}), which is outside the bounds of the grid."
            return self._create_move_response(
                game_state, message, success=False, terminated=False, binary_reward=0.0
            )

        # Handle reveal action
        if action_type == "reveal":
            if first_reveal:
                self._setup_mines(game_state, row, col)
                game_state["first_reveal"] = False

            if grid[row][col] == -1:
                message = f"Game over! You hit a mine at ({row}, {col})"
                return self._create_move_response(
                    game_state,
                    message,
                    success=False,
                    terminated=True,
                    binary_reward=0.0,
                )
            elif revealed[row][col] or flags[row][col]:
                message = f"At turn {turn_count}, you chose to reveal cell ({row}, {col}), which has already been revealed."
                return self._create_move_response(
                    game_state,
                    message,
                    success=False,
                    terminated=False,
                    binary_reward=0.0,
                )
            else:
                # Valid reveal
                self._update_grid(game_state, row, col)
                message = f"At turn {turn_count}, you successfully revealed cell ({row}, {col})."

                if self._is_solved(game_state):
                    message = "Congratulations! You have successfully cleared the Minesweeper board."
                    return self._create_move_response(
                        game_state,
                        message,
                        success=True,
                        terminated=True,
                        binary_reward=1.0,
                    )
                else:
                    return self._create_move_response(
                        game_state,
                        message,
                        success=True,
                        terminated=False,
                        binary_reward=0.0,
                    )

        # Handle flag action
        elif action_type == "flag":
            if revealed[row][col]:
                message = f"At turn {turn_count}, you chose to flag cell ({row}, {col}), which has already been revealed."
                return self._create_move_response(
                    game_state,
                    message,
                    success=False,
                    terminated=False,
                    binary_reward=0.0,
                )
            else:
                flags[row][col] = not flags[row][col]
                action_word = "added" if flags[row][col] else "removed"
                message = f"At turn {turn_count}, you {action_word} a flag on cell ({row}, {col})."
                return self._create_move_response(
                    game_state,
                    message,
                    success=True,
                    terminated=False,
                    binary_reward=0.0,
                )

        else:
            message = f"At turn {turn_count}, you chose an invalid action '{action_type}'. Valid actions are 'reveal' or 'flag'."
            return self._create_move_response(
                game_state, message, success=False, terminated=False, binary_reward=0.0
            )

        # Check max turns
        if turn_count >= max_turns:
            message = "You have reached the maximum number of turns."
            return self._create_move_response(
                game_state, message, success=False, terminated=True, binary_reward=0.0
            )

    def _create_move_response(
        self,
        game_state: Dict[str, Any],
        message: str,
        success: bool,
        terminated: bool,
        binary_reward: float,
    ) -> MakeMoveResponse:
        # Enforce max_turns termination here as well
        if not terminated and game_state["turn_count"] >= game_state["max_turns"]:
            terminated = True
            message = "You have reached the maximum number of turns."
            binary_reward = 0.0

        board_text = self._render_board(game_state)
        return MakeMoveResponse(
            success=success,
            message=message,
            game_state=game_state,
            board_text=board_text,
            is_complete=terminated,
            move_reward=binary_reward,
        )

    # -------------- game logic methods -------------
    def _get_instructions(self, rows: int, cols: int) -> str:
        example_row_1 = random.randint(0, rows - 1)
        example_col_1 = random.randint(0, cols - 1)
        example_row_2 = random.randint(0, rows - 1)
        example_col_2 = random.randint(0, cols - 1)

        return (
            f"You are playing the Minesweeper game.\n"
            "The objective of the game is to reveal all cells that do not contain mines.\n"
            "Use the make_move tool to submit a move with parameters: row (0-based), col (0-based), and action_type ('reveal' or 'flag').\n"
            'For example, call make_move with {"row": '
            f'{example_row_1}, "col": {example_col_1}, "action_type": "reveal".\n'
            'Or call make_move with {"row": '
            f'{example_row_2}, "col": {example_col_2}, "action_type": "flag".\n'
            "Cells that are unrevealed are '.', revealed numbers show adjacent mine counts, and flagged cells are 'F'.\n"
            "Avoid revealing mines and use logic to deduce safe cells.\n"
        )

    def _setup_mines(self, game_state: Dict[str, Any], safe_row: int, safe_col: int):
        rows = game_state["rows"]
        cols = game_state["cols"]
        num_mines = game_state["num_mines"]
        grid = game_state["grid"]

        mines = set()
        while len(mines) < num_mines:
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            # Avoid placing mines in the safe zone
            if (r, c) not in mines and (
                r < safe_row - 1
                or r > safe_row + 1
                or c < safe_col - 1
                or c > safe_col + 1
            ):
                mines.add((r, c))
                grid[r][c] = -1  # -1 represents a mine
        self._calculate_adjacent_numbers(game_state)

    def _calculate_adjacent_numbers(self, game_state: Dict[str, Any]):
        rows = game_state["rows"]
        cols = game_state["cols"]
        grid = game_state["grid"]

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == -1:
                    continue
                mine_count = sum(
                    (
                        0 <= r + dr < rows
                        and 0 <= c + dc < cols
                        and grid[r + dr][c + dc] == -1
                    )
                    for dr, dc in directions
                )
                grid[r][c] = mine_count

    def _update_grid(self, game_state: Dict[str, Any], row: int, col: int):
        rows = game_state["rows"]
        cols = game_state["cols"]
        grid = game_state["grid"]
        revealed = game_state["revealed"]
        flags = game_state["flags"]

        queue = deque([(row, col)])
        revealed[row][col] = True
        num_newly_revealed = 1

        while queue:
            current_row, current_col = queue.popleft()

            # If the cell has no adjacent mines, add its neighbors to the queue
            if grid[current_row][current_col] == 0:
                for dr, dc in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]:
                    neighbor_row, neighbor_col = current_row + dr, current_col + dc
                    # Only add to the queue if within bounds and not revealed or flagged
                    if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                        if (
                            not revealed[neighbor_row][neighbor_col]
                            and not flags[neighbor_row][neighbor_col]
                        ):
                            revealed[neighbor_row][neighbor_col] = True
                            num_newly_revealed += 1
                            queue.append((neighbor_row, neighbor_col))
        return num_newly_revealed

    def _is_solved(self, game_state: Dict[str, Any]) -> bool:
        """Check if the board is in a solved state."""
        rows = game_state["rows"]
        cols = game_state["cols"]
        grid = game_state["grid"]
        revealed = game_state["revealed"]
        flags = game_state["flags"]

        return all(
            (grid[r][c] == -1 and flags[r][c]) or (grid[r][c] != -1 and revealed[r][c])
            for r in range(rows)
            for c in range(cols)
        )

    def _render_board(self, game_state: Dict[str, Any], is_start: bool = False) -> str:
        """Render the game board."""
        rows = game_state["rows"]
        cols = game_state["cols"]
        grid = game_state["grid"]
        revealed = game_state["revealed"]
        flags = game_state["flags"]

        board_str = "   " + " ".join([str(c).rjust(2) for c in range(cols)]) + "\n"
        for r in range(rows):
            row_str = f"{r:2} "
            for c in range(cols):
                if is_start:
                    row_str += " . "
                else:
                    if revealed[r][c]:
                        if grid[r][c] == -1:
                            row_str += " * "
                        else:
                            row_str += f" {grid[r][c]} "
                    elif flags[r][c]:
                        row_str += " F "
                    else:
                        row_str += " . "
            board_str += row_str + "\n"
        return board_str

    # -------------- verify (trivial) --------------
    async def verify(self, body: MinesweeperVerifyRequest) -> MinesweeperVerifyResponse:
        return MinesweeperVerifyResponse(**body.model_dump())


# --------------------------------------------------------------------------- #
#  Entry-point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    MinesweeperResourcesServer.run_webserver()
