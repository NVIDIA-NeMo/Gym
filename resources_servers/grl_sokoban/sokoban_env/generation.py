"""Sokoban level generation utilities.

This module is adapted from the ``gym_sokoban`` project and GRL's fork of the
same utilities. It produces solvable single-box Sokoban rooms suitable for
text-based rendering.
"""

from __future__ import annotations

import marshal
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np


# Constants for room generation
TYPE_LOOKUP = {
    0: "wall",
    1: "empty space",
    2: "box target",
    3: "box on target",
    4: "box not on target",
    5: "player",
}

ACTION_LOOKUP_INTERNAL = {
    0: "push up",
    1: "push down",
    2: "push left",
    3: "push right",
    4: "move up",
    5: "move down",
    6: "move left",
    7: "move right",
}

# Moves are mapped to coordinate changes as follows
# 0: Move up, 1: Move down, 2: Move left, 3: Move right
CHANGE_COORDINATES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def generate_room(
    dim: Tuple[int, int] = (13, 13),
    p_change_directions: float = 0.35,
    num_steps: int = 25,
    num_boxes: int = 3,
    tries: int = 4,
    second_player: bool = False,
    search_depth: int = 100,
):
    """Generate a Sokoban room represented as integer matrices."""

    room_state = np.zeros(shape=dim)
    room_structure = np.zeros(shape=dim)

    for _ in range(tries):
        room = room_topology_generation(dim, p_change_directions, num_steps)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player)

        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1

        room_state = room.copy()
        room_state[room_state == 2] = 4

        room_state, box_mapping, action_sequence = reverse_playing(room_state, room_structure, search_depth)
        room_state[room_state == 3] = 4

        if box_displacement_score(box_mapping) > 0:
            break

    if box_displacement_score(box_mapping) == 0:
        raise RuntimeWarning("Generated Model with score == 0")

    move_probability = 0.8 if box_displacement_score(box_mapping) == 1 else 0.5
    room_state = add_random_player_movement(
        room_state,
        room_structure,
        move_probability=move_probability,
        continue_probability=0.5,
        max_steps=3,
    )

    return room_structure, room_state, box_mapping, action_sequence


def room_topology_generation(dim: Tuple[int, int] = (10, 10), p_change_directions: float = 0.35, num_steps: int = 15):
    dim_x, dim_y = dim

    masks = [
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
    ]

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = random.sample(directions, 1)[0]

    position = np.array([random.randint(1, dim_x - 1), random.randint(1, dim_y - 1)])

    level = np.zeros(dim, dtype=int)

    for _ in range(num_steps):
        if random.random() < p_change_directions:
            direction = random.sample(directions, 1)[0]

        position = position + direction
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)

        mask = random.sample(masks, 1)[0]
        mask_start = position - 1
        level[mask_start[0] : mask_start[0] + 3, mask_start[1] : mask_start[1] + 3] += mask

    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0

    return level


def place_boxes_and_player(room: np.ndarray, num_boxes: int, second_player: bool):
    possible_positions = np.where(room == 1)
    num_possible_positions = possible_positions[0].shape[0]
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError(
            "Not enough free spots ({}) to place {} player(s) and {} boxes.".format(
                num_possible_positions, num_players, num_boxes
            )
        )

    ind = np.random.randint(num_possible_positions)
    player_position = possible_positions[0][ind], possible_positions[1][ind]
    room[player_position] = 5

    if second_player:
        ind = np.random.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        room[player_position] = 5

    for _ in range(num_boxes):
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]

        ind = np.random.randint(num_possible_positions)
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        room[box_position] = 2

    return room


def add_random_player_movement(
    room_state: np.ndarray,
    room_structure: np.ndarray,
    move_probability: float = 0.5,
    continue_probability: float = 0.5,
    max_steps: int = 3,
):
    if random.random() > move_probability:
        return room_state

    player_pos = np.where(room_state == 5)
    player_pos = np.array([player_pos[0][0], player_pos[1][0]])

    previous_positions = [tuple(player_pos)]
    steps_taken = 0

    while steps_taken < max_steps:
        valid_moves = []
        for action in range(4):
            change = CHANGE_COORDINATES[action]
            next_pos = player_pos + change

            if room_state[next_pos[0], next_pos[1]] in [1, 2] and tuple(next_pos) not in previous_positions:
                valid_moves.append((action, next_pos))

        if not valid_moves:
            break

        _, next_pos = random.choice(valid_moves)

        room_state[player_pos[0], player_pos[1]] = room_structure[player_pos[0], player_pos[1]]
        room_state[next_pos[0], next_pos[1]] = 5

        player_pos = next_pos
        previous_positions.append(tuple(player_pos))

        steps_taken += 1

        if steps_taken >= max_steps or random.random() > continue_probability:
            break

    return room_state


def reverse_playing(room_state: np.ndarray, room_structure: np.ndarray, search_depth: int = 100):
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for idx in range(num_boxes):
        box = (box_locations[0][idx], box_locations[1][idx])
        box_mapping[box] = box

    explored_states: set[bytes] = set()
    best_room_score = -1
    best_room = room_state.copy()
    best_box_mapping = box_mapping.copy()
    best_action_sequence: List[int] = []

    stack: deque = deque(
        [
            (
                room_state.copy(),
                box_mapping.copy(),
                0,
                (-1, -1),
                search_depth,
                [],
            )
        ]
    )

    while stack:
        state, mapping, box_swaps, last_pull, ttl, action_sequence = stack.pop()
        ttl -= 1
        if ttl <= 0 or len(explored_states) >= 300000:
            continue

        state_hash = marshal.dumps(state)
        if state_hash in explored_states:
            continue

        room_score = box_swaps * box_displacement_score(mapping)
        if np.where(state == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best_room_score:
            best_room = state.copy()
            best_room_score = room_score
            best_box_mapping = mapping.copy()
            best_action_sequence = action_sequence.copy()

        explored_states.add(state_hash)

        for action in ACTION_LOOKUP_INTERNAL.keys():
            if action >= 4:
                continue

            state_next = state.copy()
            mapping_next = mapping.copy()

            state_next, mapping_next, last_pull_next = reverse_move(
                state_next, room_structure, mapping_next, last_pull, action
            )

            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1

            action_sequence_next = action_sequence + [action]
            stack.append(
                (
                    state_next,
                    mapping_next,
                    box_swaps_next,
                    last_pull_next,
                    ttl,
                    action_sequence_next,
                )
            )

    return best_room, best_box_mapping, best_action_sequence


def reverse_move(
    room_state: np.ndarray,
    room_structure: np.ndarray,
    box_mapping: Dict[Tuple[int, int], Tuple[int, int]],
    last_pull: Tuple[int, int],
    action: int,
):
    player_position = np.where(room_state == 5)
    player_position = np.array([player_position[0][0], player_position[1][0]])

    change = CHANGE_COORDINATES[action % 4]
    next_position = player_position + change

    if room_state[next_position[0], next_position[1]] in [1, 2]:
        room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
        room_state[next_position[0], next_position[1]] = 5

        if action < 4:
            possible_box_location = change[0] * -1, change[1] * -1
            possible_box_location = (
                possible_box_location[0] + player_position[0],
                possible_box_location[1] + player_position[1],
            )

            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                room_state[player_position[0], player_position[1]] = 3
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]
                ]

                for key in list(box_mapping.keys()):
                    if box_mapping[key] == (
                        possible_box_location[0],
                        possible_box_location[1],
                    ):
                        box_mapping[key] = (player_position[0], player_position[1])
                        last_pull = key

    return room_state, box_mapping, last_pull


def box_displacement_score(box_mapping: Dict[Tuple[int, int], Tuple[int, int]]):
    score = 0
    for box_target, location in box_mapping.items():
        box_location = np.array(location)
        box_target_arr = np.array(box_target)
        dist = np.sum(np.abs(box_location - box_target_arr))
        score += dist
    return score
