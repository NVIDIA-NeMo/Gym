# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write chess task rows: the committed examples, or train and validation splits.

``--tasks games`` (the default) starts every game from the initial position, cycling through both colors and all
opponents with disjoint seeds per split. ``--tasks mate1`` writes positions where the side to move can checkmate at
once, a short-horizon task for training.

Usage:
    python environments/chess/prepare.py --split train --size 1000
    python environments/chess/prepare.py --split validation --size 100
    python environments/chess/prepare.py --split train --tasks mate1 --size 3000
    python environments/chess/prepare.py --split example
"""

import argparse
import json
import random
import zlib
from pathlib import Path
from typing import Optional

from resources_servers.chess.app import choose_opponent_move
from resources_servers.chess.chess_rules import ChessGame, Position


AGENT_NAME = "chess_gymnasium_agent"
SPLIT_SEED_OFFSETS = {"train": 0, "validation": 1_000_000}

SYSTEM_PROMPT = (
    "You are playing chess against an opponent, and your goal is to win by checkmating the opponent's king. "
    "Every reply you give must call the make_move tool with exactly "
    "one move, until the game is over: describing a move in text does not play it, and you cannot end the game "
    "yourself. Write the move in UCI "
    "notation: the from-square followed by the to-square, such as e2e4 or g8f6. Castle by moving the king two "
    "squares (e1g1, e1c1, e8g8 or e8c8). When a pawn reaches the last rank, add the piece it promotes to: "
    "q, r, b or n, such as e7e8q.\n\n"
    "The position is given as JSON: the board (uppercase letters are White's pieces, lowercase are Black's, '.' "
    "is an empty square), the FEN, whose turn it is, whether you are in check, the recent moves, and the legal "
    "moves available to you. Each make_move result shows your move, the opponent's reply, and the new position.\n\n"
    "An illegal or malformed move, or a turn without a move, is an invalid action: the position does not change "
    "and the result says what was wrong. Too many invalid actions in a row forfeit the game. Threefold repetition and the "
    "fifty-move rule end the game in a draw automatically. A win scores 1, a draw 0, and a loss or forfeit -1."
)

MAKE_MOVE_TOOL = {
    "type": "function",
    "name": "make_move",
    "description": "Play one move for your side. Returns the opponent's reply and the new position.",
    "parameters": {
        "type": "object",
        "properties": {
            "move": {
                "type": "string",
                "description": "The move in UCI notation, such as e2e4, e1g1 (castling) or e7e8q (promotion).",
            }
        },
        "required": ["move"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Representative tasks: full games as either color against each opponent, a mate in one, a promotion, and a
# king-and-queen versus king ending that has to be converted before the fifty-move rule.
EXAMPLE_TASKS: list[dict] = [
    {"agent_color": "white", "opponent": "random", "seed": 0},
    {"agent_color": "black", "opponent": "greedy", "seed": 1},
    {"fen": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "agent_color": "white", "opponent": "greedy", "seed": 2},
    {"fen": "8/P7/8/8/8/8/k7/4K3 w - - 0 1", "agent_color": "white", "opponent": "random", "seed": 3},
    {"fen": "8/8/8/4k3/8/8/8/3QK3 w - - 0 1", "agent_color": "white", "opponent": "random", "seed": 4},
]


def build_row(
    *,
    agent_color: str,
    opponent: str,
    seed: int,
    fen: Optional[str] = None,
    max_consecutive_invalid_actions: int = 3,
) -> dict:
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"You play {agent_color}. Here is the current position."},
            ],
            "tools": [MAKE_MOVE_TOOL],
            "parallel_tool_calls": False,
        },
        "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
    }
    if fen is not None:
        row["fen"] = fen
    return row | {
        "agent_color": agent_color,
        "opponent": opponent,
        "seed": seed,
        "max_consecutive_invalid_actions": max_consecutive_invalid_actions,
    }


def generated_tasks(split: str, size: int) -> list[dict]:
    offset = SPLIT_SEED_OFFSETS[split]
    return [
        {
            "agent_color": ("white", "black")[index % 2],
            "opponent": ("random", "greedy", "lookahead")[index // 2 % 3],
            "seed": offset + index,
        }
        for index in range(size)
    ]


def mate_in_one_tasks(split: str, size: int) -> list[dict]:
    """Positions from seeded random and greedy self-play where the side to move can checkmate at once.

    A position belongs to validation when the CRC32 of its FEN is divisible by 10 and to train otherwise, so the
    splits never share a position.
    """
    rng = random.Random(SPLIT_SEED_OFFSETS[split])
    tasks: list[dict] = []
    seen: set[str] = set()
    while len(tasks) < size:
        game = ChessGame()
        players = {color: rng.choice(["random", "greedy"]) for color in ("white", "black")}
        first_ply = rng.randrange(10, 120)
        while game.outcome() is None and len(game.moves) < 200:
            position = game.position
            fen = position.fen()
            in_split = (zlib.crc32(fen.encode()) % 10 == 0) == (split == "validation")
            if len(game.moves) >= first_ply and in_split and fen not in seen and _has_mate_in_one(position):
                seen.add(fen)
                tasks.append({"fen": fen, "agent_color": position.turn, "opponent": "greedy", "seed": len(tasks)})
                break
            game.play(choose_opponent_move(position, rng, players[position.turn]).uci())
    return tasks


def _has_mate_in_one(position: Position) -> bool:
    return any((after := position.apply(move)).in_check() and not after.legal_moves for move in position.legal_moves)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["example", *SPLIT_SEED_OFFSETS], default="train")
    parser.add_argument(
        "--tasks", choices=["games", "mate1"], default="games", help="Task type for train or validation."
    )
    parser.add_argument("--size", type=int, default=1000, help="Number of rows for train or validation.")
    args = parser.parse_args()

    if args.split == "example":
        tasks = EXAMPLE_TASKS
    elif args.tasks == "mate1":
        tasks = mate_in_one_tasks(args.split, args.size)
    else:
        tasks = generated_tasks(args.split, args.size)
    output_path = Path(__file__).parent / "data" / f"{args.split}.jsonl"
    with output_path.open("w") as f:
        for task in tasks:
            f.write(json.dumps(build_row(**task)) + "\n")
    print(f"Wrote {len(tasks)} rows to {output_path}")


if __name__ == "__main__":
    main()
