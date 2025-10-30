"""Standalone Tetris environment implementation for the GRL Tetris resource server.

This module adapts the environment logic from ``GRL/grl/agents/tetrisAgent/env.py``
and removes dependencies on the upstream GRL repository so it can run entirely
within the NeMo Gym project.
"""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

import gymnasium as gym
import numpy as np


# ─────────────────────────── utilities ────────────────────────────


@contextmanager
def all_seed(seed: int | None) -> Iterator[None]:
    """Temporarily set ``random`` and ``numpy`` seeds within a context."""
    random_state = random.getstate()
    numpy_state = np.random.get_state()

    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


class BaseEnv:
    """Minimal base class mirroring the GRL interface."""

    def __init__(self, config: Dict[str, Any] | None = None, **_kwargs: Any) -> None:
        self.config: Dict[str, Any] = config or {}

    def reset(self, seed: int | None = None, **_kwargs: Any) -> Any:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def render(self, mode: str = "text") -> Any:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def is_occupied(shape: List[Tuple[int, int]], anchor: Tuple[int, int], board: np.ndarray) -> bool:
    """Return True when ``shape`` anchored at ``anchor`` collides with the board."""
    for dx, dy in shape:
        x, y = anchor[0] + dx, anchor[1] + dy
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False


class TetrisEnv(BaseEnv):
    """Self-contained Tetris environment modelled after the GRL training environment."""

    def __init__(self, config: Dict[str, Any] | None = None, **_kwargs: Any) -> None:
        super().__init__(config=config or {})
        self.width = self.config.get("dim_x", 4)
        self.height = self.config.get("dim_y", 4)
        self.board = np.zeros((self.width, self.height), dtype=np.bool_)
        self.GRID_LOOKUP = self.config.get("grid_lookup", {0: "_", 1: "#", 2: "X"})
        self.ACTION_LOOKUP = self.config.get(
            "action_lookup",
            {0: "Left", 1: "Right", 2: "Down"},
        )
        self.ACTION_SPACE = gym.spaces.Discrete(3, start=0)
        self.render_mode = self.config.get("render_mode", "text")

        box_type = self.config.get("box_type", 1)
        if box_type == 2:
            self.shapes = {
                "I": [(0, 0), (0, -1)],
                "-": [(0, 0), (-1, 0)],
            }
            self.shape_names = ["I", "-"]
        elif box_type == 3:
            self.shapes = {
                "I": [(0, 0), (0, -1)],
                "-": [(0, 0), (-1, 0)],
                "O": [(0, 0), (-1, 0), (0, -1), (-1, -1)],
            }
            self.shape_names = ["I", "-", "O"]
        else:
            self.shapes = {"O": [(0, 0)]}
            self.shape_names = ["O"]

        self.actions = {0: self._left, 1: self._right, 2: self._soft_drop}

        self.time = 0
        self.score = 0
        self.anchor: Tuple[int, int] | None = None
        self.shape: List[Tuple[int, int]] | None = None
        self.n_deaths = 0
        self._shape_counts = [0] * len(self.shapes)

        self.pre_generated_pieces: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = []
        self.current_piece_index = 0

        self.reset()

    # ─────────────────────────── core helpers ────────────────────────────

    def _choose_shape(self) -> List[Tuple[int, int]]:
        max_count = max(self._shape_counts)
        weights = [5 + max_count - count for count in self._shape_counts]
        r = random.randint(1, sum(weights))
        for i, weight in enumerate(weights):
            r -= weight
            if r <= 0:
                self._shape_counts[i] += 1
                return self.shapes[self.shape_names[i]]
        return self.shapes[self.shape_names[0]]

    def _generate_piece(self) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        shape = self._choose_shape()
        if (-1, 0) in shape:
            anchor = (random.randint(1, self.width - 1), 0)
        else:
            anchor = (random.randint(0, self.width - 1), 0)
        return anchor, shape

    def _new_piece(self) -> None:
        if self.current_piece_index < len(self.pre_generated_pieces):
            self.anchor, self.shape = self.pre_generated_pieces[self.current_piece_index]
            self.current_piece_index += 1
        else:
            self.anchor, self.shape = self._generate_piece()

    def _has_dropped(self) -> bool:
        assert self.shape is not None and self.anchor is not None
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self) -> int:
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        write_idx = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, write_idx] = self.board[:, i]
                write_idx -= 1
        lines_cleared = sum(can_clear)
        self.score += lines_cleared
        self.board = new_board
        return lines_cleared

    def _set_piece(self, on: bool = False) -> None:
        assert self.shape is not None and self.anchor is not None
        for dx, dy in self.shape:
            x, y = self.anchor[0] + dx, self.anchor[1] + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.board[x, y] = on

    def _left(self) -> None:
        assert self.shape is not None and self.anchor is not None
        new_anchor = (self.anchor[0] - 1, self.anchor[1])
        if not is_occupied(self.shape, new_anchor, self.board):
            self.anchor = new_anchor

    def _right(self) -> None:
        assert self.shape is not None and self.anchor is not None
        new_anchor = (self.anchor[0] + 1, self.anchor[1])
        if not is_occupied(self.shape, new_anchor, self.board):
            self.anchor = new_anchor

    def _soft_drop(self) -> None:
        assert self.shape is not None and self.anchor is not None
        new_anchor = (self.anchor[0], self.anchor[1] + 1)
        if not is_occupied(self.shape, new_anchor, self.board):
            self.anchor = new_anchor

    def _idle(self) -> None:
        pass

    # ───────────────────────────── API ──────────────────────────────

    def reset(self, seed: int | None = None, **_kwargs: Any) -> Any:
        """Reset the environment to its initial state."""
        try:
            with all_seed(seed):
                self.time = 0
                self.score = 0
                self.board = np.zeros((self.width, self.height), dtype=np.bool_)

                self.pre_generated_pieces = []
                self.current_piece_index = 0
                num_pieces_to_generate = self.width * self.height + 1
                for _ in range(num_pieces_to_generate):
                    self.pre_generated_pieces.append(self._generate_piece())

                self._new_piece()
                return self.render()
        except (RuntimeError, RuntimeWarning):
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if action not in self.actions:
            return self.render(), 0.0, True, {"error": "Invalid action"}

        previous_pos = copy.deepcopy(self.anchor)
        self.actions[action]()

        self.time += 1
        reward = -0.1
        done = False
        dropped = False
        info: Dict[str, Any] = {}

        lines_cleared = 0
        if self._has_dropped():
            dropped = True
            self._set_piece(True)
            lines_cleared = self._clear_lines()
            reward += lines_cleared * 10

            if np.any(self.board[:, 0]):
                done = True
            else:
                self._new_piece()

        self._set_piece(True)
        state = self.render()
        self._set_piece(False)

        action_effective = previous_pos is not None and previous_pos != self.anchor
        info["action_is_effective"] = action_effective
        info["action_is_valid"] = True
        info["success"] = lines_cleared > 0
        info["dropped"] = dropped

        if lines_cleared > 0:
            done = True

        return state, reward, done, info

    def render(self, mode: str = "text") -> Any:
        if mode != "text":
            return self.board.copy()

        board_str = "\n".join("".join("#" if cell else "_" for cell in row) for row in self.board.T)

        self._set_piece(True)
        assert self.shape is not None and self.anchor is not None
        positions = [(self.anchor[0] + dx, self.anchor[1] + dy) for dx, dy in self.shape]
        self._set_piece(False)

        lines = board_str.split("\n")
        for x, y in positions:
            if 0 <= y < len(lines) and 0 <= x < len(lines[0]):
                line = list(lines[y])
                line[x] = "X"
                lines[y] = "".join(line)
        return "\n".join(lines)

    def get_all_actions(self) -> List[int]:
        return list(self.actions.keys())

    def close(self) -> None:
        self.board = None
        self.anchor = None
        self.shape = None
