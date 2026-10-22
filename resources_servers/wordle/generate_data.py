#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"

# Valid guesses are the 5-letter words of ENABLE (enable1.txt), which is in the public domain.
ENABLE_URL = "https://raw.githubusercontent.com/dolph/dictionary/c65f04b0b5b27a981f437b940cf62fe71320d5ec/enable1.txt"

# Targets are the SCOWL/ESDB hunspell en_US base forms that are also in ENABLE, minus taboo words.
# The SCOWL/ESDB license requires this notice in all copies of lists created from it.
SCOWL_URL = "https://github.com/en-wl/wordlist/releases/download/rel-2026.02.25/hunspell-en_US-2026.02.25.zip"
SCOWL_NOTICE = """Copyright 2000-2026 by Kevin Atkinson

Permission to use, copy, modify, distribute, and sell any part of the English
Speller Database (ESDB, previously known as SCOWLv2), or word lists
created from it, is hereby granted without fee, provided that the above
copyright notice appears in all copies and that both the above copyright
notice and this notice appear in supporting documentation.  Kevin Atkinson
makes no representations about the suitability of this database for any
purpose.  It is provided "as is" without express or implied warranty."""

SYSTEM_PROMPT = """You are playing Wordle, a word-guessing game. Your goal is to guess a secret 5-letter word in 6 attempts or fewer.

After each guess, you'll receive feedback:
- G (Green): Letter is correct and in the right position
- Y (Yellow): Letter is in the word but in the wrong position
- _ (Gray): Letter is not in the word

Strategy tips:
- Start with words containing common letters (E, A, R, T, O, I, N, S)
- Use the feedback to narrow down possibilities
- Never repeat a guess
- Place confirmed green letters in their positions
- Include yellow letters in different positions
- Avoid gray (eliminated) letters

IMPORTANT: Always respond with a tool call. Never reply with plain text. After receiving feedback, immediately call submit_guess with your next guess. When you see "won": true or "game_over": true in a response, the game is over — do not make any more tool calls."""

TOOLS = [
    {
        "type": "function",
        "name": "submit_guess",
        "description": "Submit a 5-letter word guess. Returns feedback for each letter: G (green) = correct position, Y (yellow) = wrong position but in word, _ (gray) = not in word.",
        "parameters": {
            "type": "object",
            "properties": {"guess": {"type": "string", "description": "A 5-letter English word to guess"}},
            "required": ["guess"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "check_word_validity",
        "description": "Check if a word is valid before guessing. This is optional and informational only - it won't affect your game.",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string", "description": "A word to check for validity"}},
            "required": ["word"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_game_state",
        "description": "Get the current game state including guesses made, feedback received, and accumulated knowledge about the target word.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
]


def download(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def build_word_lists() -> tuple[list[str], list[str]]:
    guesses = sorted({w for w in download(ENABLE_URL).decode().split() if len(w) == 5 and w.isalpha()})
    with zipfile.ZipFile(io.BytesIO(download(SCOWL_URL))) as z:
        dic = z.read("en_US.dic").decode().splitlines()[1:]
    guess_set = set(guesses)
    # The "!" flag marks taboo words. ENABLE is lowercase, so proper nouns drop out.
    entries = (line.partition("/") for line in dic)
    targets = sorted({w for w, _, flags in entries if w in guess_set and "!" not in flags})
    # The train/validation split depends on these exact lists.
    assert (len(targets), len(guesses)) == (3088, 8636), (len(targets), len(guesses))
    return targets, guesses


def split_targets(targets: list[str]) -> tuple[list[str], list[str]]:
    shuffled = list(targets)
    random.Random(42).shuffle(shuffled)
    num_validation = round(0.15 * len(shuffled))
    return shuffled[num_validation:], shuffled[:num_validation]


def make_row(target: str) -> dict:
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Make your first guess."},
            ],
            "tools": TOOLS,
            "parallel_tool_calls": False,
            "temperature": 1.0,
        },
        "word_length": 5,
        "max_turns": 6,
        "agent_ref": {"type": "responses_api_agents", "name": "wordle_gymnasium_agent"},
        "custom_target": target,
    }


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines))
    print(f"Wrote {len(lines)} lines to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Wordle word lists and generate train/validation data")
    parser.add_argument("--train_samples", type=int, default=1000, help="Cycles through the training targets")
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=886)
    args = parser.parse_args()

    targets, guesses = build_word_lists()
    notice = ["# " + line if line else "#" for line in SCOWL_NOTICE.splitlines()]
    write_lines(DATA_DIR / "targets.txt", notice + targets)
    write_lines(DATA_DIR / "guesses.txt", guesses)

    train_words, validation_words = split_targets(targets)
    random.Random(args.seed).shuffle(train_words)
    train = [make_row(train_words[i % len(train_words)]) for i in range(args.train_samples)]
    validation = [make_row(w) for w in validation_words[:100]]
    example = random.Random(args.seed).sample(validation, 5)
    for name, rows in [("train", train), ("validation", validation), ("example", example)]:
        write_lines(args.output_dir / f"{name}.jsonl", [json.dumps(row) for row in rows])


if __name__ == "__main__":
    main()
