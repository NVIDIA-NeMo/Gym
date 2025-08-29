import copy
import random
import re
import json
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)


class SudokuResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetInitialBoardRequest(BaseModel):
    clues: int = 30
    scale: int = 4  # 4 for easy version


class GetInitialBoardResponse(BaseModel):
    board_text: str
    instructions: str
    game_state: dict  # Store the current board state


class MakeMoveRequest(BaseModel):
    game_state: dict
    row: int  # 1-based row index
    col: int  # 1-based column index
    number: int  # value to place


class MakeMoveResponse(BaseModel):
    success: bool
    message: str
    game_state: dict
    board_text: str
    is_complete: bool
    move_reward: float


class SudokuRunRequest(BaseRunRequest):
    clues: int = 30
    scale: int = 9


class SudokuVerifyRequest(SudokuRunRequest, BaseVerifyRequest):
    reward: float = 0.0
    total_moves: int = 0
    is_complete: bool = False


class SudokuVerifyResponse(BaseVerifyResponse):
    total_moves: int = 0
    is_complete: bool = False


class SudokuResourcesServer(SimpleResourcesServer):
    config: SudokuResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/get_initial_board")(self.get_initial_board)
        app.post("/make_move")(self.make_move)

        return app

    async def get_initial_board(
        self, body: GetInitialBoardRequest
    ) -> GetInitialBoardResponse:
        # Generate initial sudoku puzzle
        full_grid, puzzle_grid = self._generate_board(body.clues, body.scale)

        game_state = {
            "current_board": puzzle_grid,
            "solution": full_grid,
            "scale": body.scale,
            "initial_empty_count": sum(row.count(0) for row in puzzle_grid),
            "moves_made": 0,
            "correct_moves": 0,
        }

        board_text = self._render_board(puzzle_grid, body.scale)
        instructions = self._get_instructions(body.scale)

        return GetInitialBoardResponse(
            board_text=board_text, instructions=instructions, game_state=game_state
        )

    async def make_move(self, body: MakeMoveRequest) -> MakeMoveResponse:
        game_state = body.game_state
        current_board = game_state["current_board"]
        solution = game_state["solution"]
        scale = game_state["scale"]

        row = body.row
        col = body.col
        guess_num = body.number

        game_state["moves_made"] += 1

        # Validate move format
        if row is None or col is None or guess_num is None:
            return MakeMoveResponse(
                success=False,
                message="Invalid move format. Use \\boxed{row column number}, e.g. \\boxed{1 1 5}",
                game_state=game_state,
                board_text=self._render_board(current_board, scale),
                is_complete=False,
                move_reward=-0.1,
            )

        # Validate bounds
        if not (1 <= row <= scale and 1 <= col <= scale and 1 <= guess_num <= scale):
            return MakeMoveResponse(
                success=False,
                message=f"Move out of bounds: R{row} C{col} with value {guess_num}",
                game_state=game_state,
                board_text=self._render_board(current_board, scale),
                is_complete=False,
                move_reward=-0.1,
            )

        row_idx, col_idx = row - 1, col - 1

        # Check if cell is already filled
        if current_board[row_idx][col_idx] != 0:
            return MakeMoveResponse(
                success=False,
                message=f"Cell R{row} C{col} is already filled. You cannot overwrite pre-filled cells.",
                game_state=game_state,
                board_text=self._render_board(current_board, scale),
                is_complete=False,
                move_reward=-0.1,
            )

        # Check if move is correct
        if solution[row_idx][col_idx] == guess_num:
            # Correct move
            current_board[row_idx][col_idx] = guess_num
            game_state["correct_moves"] += 1

            # Check if puzzle is complete
            is_complete = self._is_puzzle_complete(current_board)
            reward = (
                1.0 / game_state["initial_empty_count"]
            )  # Reward proportional to contribution

            if is_complete:
                message = f"Correct move! R{row} C{col} = {guess_num}. Congratulations! Puzzle completed!"
                reward += 1.0  # Bonus for completion
            else:
                message = f"Correct move! R{row} C{col} = {guess_num}"

            return MakeMoveResponse(
                success=True,
                message=message,
                game_state=game_state,
                board_text=self._render_board(current_board, scale),
                is_complete=is_complete,
                move_reward=reward,
            )
        else:
            # Incorrect move
            return MakeMoveResponse(
                success=False,
                message=f"Incorrect move: R{row} C{col} = {guess_num} violates Sudoku rules",
                game_state=game_state,
                board_text=self._render_board(current_board, scale),
                is_complete=False,
                move_reward=-0.1,
            )

    async def verify(self, body: SudokuVerifyRequest) -> SudokuVerifyResponse:
        return SudokuVerifyResponse(**body.model_dump())

    def _get_instructions(self, scale: int) -> str:
        prompt = (
            f"You are playing {'a simple version of' if scale == 4 else ''} Sudoku.\n"
            f"Each row is numbered from 1 to {scale}, and each column is also numbered from 1 to {scale}.\n"
            f"Empty cells are represented by '.', and pre-filled cells contain digits from 1 to {scale}.\n\n"
        )

        sub_scale = int(scale**0.5)
        prompt += (
            f"Your objective is to fill the empty cells in the {scale}x{scale} grid with digits from 1 to {scale} such that:\n"
            f"1. Each row contains all digits from 1 to {scale} without repetition.\n"
            f"2. Each column contains all digits from 1 to {scale} without repetition.\n"
            f"3. Each {sub_scale}x{sub_scale} subgrid contains all digits from 1 to {scale} without repetition.\n\n"
            "Rules and Instructions:\n"
            "1. **Do not overwrite** the initial numbers provided in the grid.\n"
            "2. **Only fill** empty cells represented by '.'.\n"
            "3. You must respond with the format '\\boxed{row column number}', e.g. \\boxed{1 1 5}.\n"
            "4. **Ensure** that your move does not violate Sudoku rules. Invalid moves will result in penalties.\n"
            "Use the make_move function to submit your moves. Good luck!\n\n"
        )
        return prompt

    def _render_board(self, board: List[List[int]], scale: int) -> str:
        """Render the board as a formatted string with row and column indices."""
        sub_scale = int(scale**0.5)
        header = "   " + " ".join(
            [
                f"C{j + 1}" + ("  " if (j + 1) % sub_scale == 0 else "")
                for j in range(scale)
            ]
        )

        lines = [header]
        for i, row in enumerate(board):
            row_str = f"R{i + 1} "
            for j, num in enumerate(row):
                cell = str(num) if num != 0 else "."
                row_str += f" {cell} "
                if (j + 1) % sub_scale == 0 and j < (scale - 1):
                    row_str += "| "
            lines.append(row_str.strip())
            if (i + 1) % sub_scale == 0 and i < (scale - 1):
                lines.append("   " + "- " * 2 * (scale - 1))

        return "\n".join(lines)

    def _generate_board(
        self, clues: int, scale: int
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate a complete sudoku grid and a puzzle with given clues."""
        full_grid = self._generate_full_grid(scale)
        puzzle_grid = self._remove_cells(full_grid, clues, scale)
        return full_grid, puzzle_grid

    def _generate_full_grid(self, scale: int) -> List[List[int]]:
        """Generate a complete valid sudoku grid."""
        grid = [[0 for _ in range(scale)] for _ in range(scale)]
        self._fill_grid(grid, scale)
        return grid

    def _fill_grid(self, grid: List[List[int]], scale: int) -> bool:
        """Fill the grid using backtracking."""
        empty = self._find_empty(grid, scale)
        if not empty:
            return True
        row, col = empty

        numbers = list(range(1, scale + 1))
        random.shuffle(numbers)

        for num in numbers:
            if self._is_safe(grid, row, col, num, scale):
                grid[row][col] = num
                if self._fill_grid(grid, scale):
                    return True
                grid[row][col] = 0
        return False

    def _find_empty(
        self, grid: List[List[int]], scale: int
    ) -> Optional[Tuple[int, int]]:
        """Find an empty cell in the grid."""
        for i in range(scale):
            for j in range(scale):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def _is_safe(
        self, grid: List[List[int]], row: int, col: int, num: int, scale: int
    ) -> bool:
        """Check if it's safe to place a number in the given cell."""
        # Check row
        if num in grid[row]:
            return False

        # Check column
        if num in [grid[i][col] for i in range(scale)]:
            return False

        # Check subgrid
        n = int(scale**0.5)
        start_row, start_col = n * (row // n), n * (col // n)
        for i in range(start_row, start_row + n):
            for j in range(start_col, start_col + n):
                if grid[i][j] == num:
                    return False
        return True

    def _remove_cells(
        self, grid: List[List[int]], clues: int, scale: int
    ) -> List[List[int]]:
        """Remove cells from full grid to create puzzle, maintaining unique solution."""
        puzzle = copy.deepcopy(grid)
        cells = [(i, j) for i in range(scale) for j in range(scale)]
        random.shuffle(cells)

        while len(cells) > ((scale**2) - clues):
            row, col = cells.pop()
            removed = puzzle[row][col]
            puzzle[row][col] = 0

        return puzzle

    def _is_puzzle_complete(self, board: List[List[int]]) -> bool:
        """Check if the puzzle is completely filled with no empty cells."""
        for row in board:
            if 0 in row:
                return False
        return True


if __name__ == "__main__":
    SudokuResourcesServer.run_webserver()
