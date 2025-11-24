"""Local Sokoban environment implementation for the NeMo Gym server.

This module adapts the environment used in the GRL repository while keeping all
runtime dependencies inside the Gym project. It relies on ``gym_sokoban`` for
core Sokoban mechanics and provides deterministic room generation utilities
vendored into this directory.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv

from .generation import generate_room


class SokobanEnv(GymSokobanEnv):
    """Self-contained Sokoban environment used by the resource server."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        self.config = config
        self.GRID_LOOKUP = self.config.get("grid_lookup", {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"})
        self.ACTION_LOOKUP = self.config.get("action_lookup", {1: "Up", 2: "Down", 3: "Left", 4: "Right"})
        self.search_depth = self.config.get("search_depth", 300)
        self.ACTION_SPACE = gym.spaces.Discrete(4, start=1)
        self.render_mode = self.config.get("render_mode", "text")

        super().__init__(
            dim_room=self.config.get("dim_room", (6, 6)),
            max_steps=self.config.get("max_steps", 100),
            num_boxes=self.config.get("num_boxes", 1),
            **kwargs,
        )

    def reset(self, seed: int | None = None):  # type: ignore[override]
        python_state = None
        numpy_state = None
        if seed is not None:
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed)

        try:
            (
                self.room_fixed,
                self.room_state,
                self.box_mapping,
                _action_sequence,
            ) = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                search_depth=self.search_depth,
            )
        except (RuntimeError, RuntimeWarning):  # pragma: no cover - rare fallback
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed)
        finally:
            if seed is not None and python_state is not None and numpy_state is not None:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.player_position = np.argwhere(self.room_state == 5)[0]

        return self.render()

    def step(self, action: int):  # type: ignore[override]
        previous_pos = self.player_position.copy()
        _, reward, done, _ = super().step(action)
        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)
        info = {
            "action_is_effective": action_effective,
            "action_is_valid": True,
            "success": self.boxes_on_target == self.num_boxes,
        }
        return next_obs, reward, done, info

    def render(self, mode: str | None = None):  # type: ignore[override]
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == "text":
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return "\n".join("".join(self.GRID_LOOKUP.get(int(cell), "?") for cell in row) for row in room.tolist())
        if render_mode == "rgb_array":
            return self.get_image(mode="rgb_array", scale=1)
        raise ValueError(f"Invalid render mode: {render_mode}")

    def get_all_actions(self):
        return list(self.ACTION_LOOKUP.keys())

    def close(self):  # type: ignore[override]
        self.render_cache = None
        super().close()
