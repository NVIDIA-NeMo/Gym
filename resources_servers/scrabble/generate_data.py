#!/usr/bin/env python3
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
"""Download the word list and generate Scrabble move-finding puzzles.

Each puzzle is a position reached by seeded self-play: two simulated players alternate, each picking one
of the top-k scoring moves with k drawn per move, so boards range from greedy to loosely played. The
player to move's rack becomes the puzzle, and exhaustive move generation gives its ``best_score``.

Usage:
    python resources_servers/scrabble/generate_data.py
    python resources_servers/scrabble/generate_data.py --train_samples 20000 --workers 32
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# Run as a script from the repository root: make `resources_servers` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resources_servers.scrabble import game  # noqa: E402


DATA_DIR = Path(__file__).parent / "data"
WORDS_PATH = DATA_DIR / "words.txt"

# ENABLE (enable1.txt) is in the public domain. Same pinned mirror as resources_servers/wordle.
ENABLE_URL = "https://raw.githubusercontent.com/dolph/dictionary/c65f04b0b5b27a981f437b940cf62fe71320d5ec/enable1.txt"
ENABLE_WORD_COUNT = 168_551  # Words of 2-15 letters; guards against a changed download.

# Seeds for the two splits never overlap.
TRAIN_SEED_BASE = 0
VALIDATION_SEED_BASE = 10_000_000

MAX_PLIES = 16
OPENING_FRACTION = 0.05
TOP_K_CHOICES = (1, 1, 2, 3, 5, 10)

SYSTEM_PROMPT = f"""You are an expert Scrabble player. You are shown a board and your rack. Find the single highest-scoring legal move and play it with the play_move tool.

Rules:
- Place tiles in one row (across) or one column (down). Together with any tiles already in that line, they must form one continuous word.
- Every word formed must be valid, including words formed perpendicular to yours where your tiles touch existing ones. The dictionary is the ENABLE word list.
- The move must connect to tiles already on the board. The first move of a game must cover the center square H8, marked *.
- Each letter scores its value: {", ".join(f"{letter}={value}" for letter, value in game.LETTER_VALUES.items())}.
- Premium squares count only for tiles you place this turn: 2L/3L double/triple that letter, 2W/3W double/triple every word that tile is part of.
- Every word you form is scored, and placing all 7 rack tiles earns a 50-point bonus.
- ? is a blank: it can be any letter and scores 0. Lowercase letters on the board are blanks played earlier and score 0.

Call play_move with:
- word: the complete word along your line, including letters already on the board,
- row (1-15) and col (A-O): the square of the word's first letter,
- direction: "across" (left to right) or "down" (top to bottom).
Blanks are assigned for you. An illegal move is rejected with the reason and you may try again, but each illegal move lowers your reward. Your reward is your move's score divided by the best possible score."""

TOOLS = [
    {
        "type": "function",
        "name": "play_move",
        "description": "Play a move. Ends the puzzle if the move is legal; otherwise returns why it is illegal.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The complete word formed along the line, including letters already on the board.",
                },
                "row": {"type": "integer", "description": "Row of the word's first letter, 1-15."},
                "col": {
                    "type": "string",
                    "enum": list(game.COLUMNS),
                    "description": "Column of the word's first letter, A-O.",
                },
                "direction": {"type": "string", "enum": ["across", "down"]},
            },
            "required": ["word", "row", "col", "direction"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def user_prompt(board: list[str], rack: str) -> str:
    return (
        f"Board:\n```\n{game.render_board(board)}\n```\n"
        f"Your rack: {game.render_rack(rack)}\n\n"
        "Find the highest-scoring legal move and play it."
    )


def make_row(position_id: str, board: list[str], rack: str, best: game.GeneratedMove) -> dict:
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(board, rack)},
            ],
            "tools": TOOLS,
            "parallel_tool_calls": False,
            "temperature": 1.0,
        },
        "agent_ref": {"type": "responses_api_agents", "name": "scrabble_gymnasium_agent"},
        "position_id": position_id,
        "board": board,
        "rack": rack,
        "best_score": best.score,
        "best_move": best.to_json(),
        "max_attempts": 3,
    }


def download_words() -> list[str]:
    with urllib.request.urlopen(ENABLE_URL) as response:
        text = response.read().decode()
    words = sorted({w.upper() for w in text.split() if 2 <= len(w) <= game.BOARD_SIZE and w.isalpha()})
    if len(words) != ENABLE_WORD_COUNT:
        raise RuntimeError(f"expected {ENABLE_WORD_COUNT} ENABLE words, downloaded {len(words)}")
    return words


_WORKER: dict = {}


def _init_worker(words: list[str]) -> None:
    _WORKER["dictionary"] = frozenset(words)
    _WORKER["trie"] = game.build_trie(words)


def make_position(seed: int) -> tuple[list[str], str, game.GeneratedMove] | None:
    """Play a seeded game for a random number of plies; return (board, rack, best move), or None if stuck."""
    trie, dictionary = _WORKER["trie"], _WORKER["dictionary"]
    rng = random.Random(seed)
    bag = game.full_bag()
    board = game.empty_board()
    racks = [game.draw(bag, "", rng), game.draw(bag, "", rng)]
    plies = 0 if rng.random() < OPENING_FRACTION else rng.randint(1, MAX_PLIES)

    for ply in range(plies):
        player = ply % 2
        moves = game.generate_moves(board, racks[player], trie, dictionary)
        if not moves:
            # Exchange the whole rack, as a stuck player would.
            if len(bag) < game.RACK_SIZE:
                return None
            old = racks[player]
            racks[player] = game.draw(bag, "", rng)
            bag.extend(old)
            continue
        moves.sort(key=lambda m: m.score, reverse=True)
        move = rng.choice(moves[: rng.choice(TOP_K_CHOICES)])
        board = game.apply_move(board, move.placed)
        racks[player] = game.draw(bag, game.remove_from_rack(racks[player], move.placed), rng)
        if not racks[player]:
            return None  # The game ended.

    rack = racks[plies % 2]
    moves = game.generate_moves(board, rack, trie, dictionary)
    if not moves:
        return None
    best = max(moves, key=lambda m: (m.score, m.direction == "across"))
    return board, rack, best


def generate_split(name: str, seed_base: int, count: int, pool: ProcessPoolExecutor) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    seed = seed_base
    while len(rows) < count:
        seeds = range(seed, seed + 2 * (count - len(rows)) + 8)
        seed = seeds.stop
        for position in pool.map(make_position, seeds, chunksize=16):
            if position is None or len(rows) == count:
                continue
            board, rack, best = position
            key = (*board, rack)
            if key in seen:
                continue
            seen.add(key)
            rows.append(make_row(f"{name}-{len(rows):06d}", board, rack, best))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the ENABLE word list and generate Scrabble puzzles")
    parser.add_argument("--train_samples", type=int, default=10_000)
    parser.add_argument("--validation_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    words = download_words()
    WORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORDS_PATH.write_text("".join(w + "\n" for w in words))
    print(f"Wrote {len(words)} words to {WORDS_PATH}")

    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(words,)) as pool:
        validation = generate_split("validation", VALIDATION_SEED_BASE, args.validation_samples, pool)
        train = generate_split("train", TRAIN_SEED_BASE, args.train_samples, pool)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "example.jsonl", validation[:5])


if __name__ == "__main__":
    main()
